"""Download or validate every raw dataset a flow lists in `required_datasets` before it starts.

Names refer to hydra/task/download_dataset/<name>.yaml; idempotent, so safe to run on every start.
"""

from typing import List

from hydra import compose
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize
from prefect import flow, get_run_logger

from src.prefect.caching import hydra_config_to_cacheable_dict
from src.prefect.task_options import task_options as remote_options
from src.tasks.download_dataset_task import download_dataset
from src.utils.data_roots import DataRoots


@flow(name="ensure_datasets_flow")
def ensure_datasets_flow(
    required_datasets: List[str],
    data_roots: DataRoots,
    download_num_gpus: int = 0,
) -> List[str]:
    logger = get_run_logger()
    futures = []
    for name in required_datasets:
        GlobalHydra.instance().clear()
        with initialize(config_path="../../hydra/task/download_dataset", version_base="1.2"):
            cfg = compose(config_name=name)
        logger.info(f"Ensuring dataset '{name}' ({cfg.mode}: {cfg.target_dir})")
        with remote_options(num_gpus=download_num_gpus):
            futures.append((name, download_dataset.submit(**hydra_config_to_cacheable_dict(cfg), data_roots=data_roots)))
    paths = []
    for name, fut in futures:
        paths.append(fut.result())
        logger.info(f"Dataset '{name}' ready at {paths[-1]}")
    return paths
