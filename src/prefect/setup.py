import os
import time
from contextlib import contextmanager

import httpx
from prefect.filesystems import LocalFileSystem, RemoteFileSystem
from prefect_ray import RayTaskRunner
from pydantic import BaseModel

from src.utils.bart_toolbox import toolbox_path

# Prefect blocks holding the MinIO (S3) connection, one per bucket. They are created by the
# driver (main.py -> setup_storage_blocks) so Ray workers only need to load them by name.
CACHING_STORAGE_NAME = "minio-caching-storage"
CACHING_BUCKET = "prefect-results"
MODELS_STORAGE_NAME = "ml-models"
MODELS_BUCKET = "ml-models"


def _retry_transient(fn, *, exceptions, attempts: int = 6, delay_seconds: float = 3.0):
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except exceptions:
            if attempt == attempts:
                raise
            time.sleep(delay_seconds)


def _load_block_with_retry(load_fn):
    return _retry_transient(load_fn, exceptions=(httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout))


def retry_s3_transfer(fn, *, attempts: int = 6, delay_seconds: float = 3.0):
    import botocore.exceptions

    return _retry_transient(fn, exceptions=(botocore.exceptions.ConnectionError,), attempts=attempts, delay_seconds=delay_seconds)


def setup_storage_blocks(minio_cfg) -> None:
    """Save (or update) the MinIO storage blocks and create their buckets if missing."""
    for storage_name, bucket in ((CACHING_STORAGE_NAME, CACHING_BUCKET), (MODELS_STORAGE_NAME, MODELS_BUCKET)):
        storage = RemoteFileSystem(
            basepath=f"s3://{bucket}/",
            settings={
                "key": minio_cfg.access_key,
                "secret": minio_cfg.secret_key,
                "client_kwargs": {"endpoint_url": minio_cfg.endpoint_url},
                "use_ssl": str(minio_cfg.endpoint_url).startswith("https"),
            },
        )
        if not storage.filesystem.exists(bucket):
            storage.filesystem.mkdir(bucket)
        storage.save(storage_name, overwrite=True)


def load_local_storage(storage_name: str, storage_path: str) -> LocalFileSystem:
    try:
        return _load_block_with_retry(lambda: LocalFileSystem.load(storage_name))
    except ValueError:
        local_storage = LocalFileSystem(basepath=storage_path)
        local_storage.save(storage_name)
        return local_storage


def load_remote_storage(storage_name: str, bucket_name: str) -> RemoteFileSystem:
    try:
        return _load_block_with_retry(lambda: RemoteFileSystem.load(storage_name))
    except ValueError as e:
        raise ValueError(
            f"Prefect storage block '{storage_name}' (bucket '{bucket_name}') does not exist. "
            "It is created by main.py from the `minio` section of hydra/config.yaml."
        ) from e


def get_caching_storage():
    return load_remote_storage(CACHING_STORAGE_NAME, CACHING_BUCKET)


def _ray_env_vars(prefect_api_url: str) -> dict:
    """Environment of the Ray workers: fixed defaults, plus a few variables forwarded from the driver if set."""
    env_vars = {
        "PREFECT_API_URL": prefect_api_url,
        "PREFECT_RESULTS_PERSIST_BY_DEFAULT": "true",
        "MKL_THREADING_LAYER": "GNU",
        "MKL_SERVICE_FORCE_INTEL": "0",
        "OMP_NUM_THREADS": "1",
    }
    for key in ("HF_HOME", "WANDB_API_KEY", "TORCH_CUDA_ARCH_LIST"):
        if os.environ.get(key):
            env_vars[key] = os.environ[key]
    # libs/ is gitignored, so Ray does not ship it with the working_dir: pass the driver's absolute BART path.
    bart_path = toolbox_path()
    if bart_path:
        env_vars["TOOLBOX_PATH"] = bart_path
    return env_vars


def get_and_setup_task_runner(name: str, ray_address: str | None, prefect_api_url: str):
    """`ray`: run tasks on a Ray cluster (ray_address=None starts a local Ray instance). `local`: Prefect's default runner."""
    if name == "ray":
        return RayTaskRunner(
            address=ray_address,
            init_kwargs={
                "log_to_driver": True,
                "runtime_env": {
                    "working_dir": "./",
                    "excludes": ["*.pyc", "*.pyo", "__pycache__", ".git", "*.ipynb", "*.safetensors", "*.pdf", "models-*/**"],
                    "env_vars": _ray_env_vars(prefect_api_url),
                },
            },
        )
    elif name == "local":
        return None
    else:
        raise ValueError(f"Task runner {name} is not defined (use 'ray' or 'local').")


class StorageSettings(BaseModel):
    storage_name: str
    bucket_or_base_path: str
    is_remote: bool


def load_file_system(settings: StorageSettings):
    if settings.is_remote:
        return load_remote_storage(settings.storage_name, bucket_name=settings.bucket_or_base_path)
    else:
        return load_local_storage(settings.storage_name, storage_path=settings.bucket_or_base_path)


class StoragePath(BaseModel):
    storage_path: str
    storage_settings: StorageSettings


@contextmanager
def download_directory_to_temp_on_enter(filesystem: LocalFileSystem | RemoteFileSystem,
        temp_folder: str, fs_folder: str, ex_subfolder: str):

    temp_subfolder = os.path.join(temp_folder, ex_subfolder)
    fs_subfolder = os.path.join(fs_folder, ex_subfolder)
    os.makedirs(temp_subfolder, exist_ok=True)

    retry_s3_transfer(lambda: filesystem.get_directory(fs_subfolder, temp_subfolder))

    cwd_saved = os.getcwd()
    os.chdir(temp_subfolder)
    try:
        yield
    finally:
        os.chdir(cwd_saved)
