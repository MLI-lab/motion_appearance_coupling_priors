import hydra
from omegaconf import DictConfig
from prefect.settings import PREFECT_API_URL, PREFECT_RESULTS_PERSIST_BY_DEFAULT, temporary_settings

from src.flows.flow_resolver import resolve_flow
from src.prefect.setup import get_and_setup_task_runner, get_caching_storage, setup_storage_blocks
from src.utils.data_roots import DataRoots


@hydra.main(config_path='hydra', config_name='config', version_base='1.2')
def coordinator(cfg: DictConfig) -> None:
    data_roots = DataRoots(cache=str(cfg.data_cache_root), manual=str(cfg.data_manual_root))
    print(f"[main] data_cache_root={data_roots.cache} data_manual_root={data_roots.manual}")

    with temporary_settings(updates={PREFECT_API_URL: cfg.prefect.api_url, PREFECT_RESULTS_PERSIST_BY_DEFAULT: True}):
        setup_storage_blocks(cfg.minio)
        options = {
            "task_runner": get_and_setup_task_runner(cfg.task_runner, ray_address=cfg.ray_address, prefect_api_url=cfg.prefect.api_url),
            "result_storage": get_caching_storage(),
        }
        flow_kwargs = dict(cfg.flow)

        required_datasets = list(flow_kwargs.pop("required_datasets", None) or [])
        if required_datasets:
            from src.flows.ensure_datasets_flow import ensure_datasets_flow
            ensure_datasets_flow.with_options(**options)(required_datasets=required_datasets, data_roots=data_roots)

        resolve_flow(
            options=options,
            wandb_cfg=cfg.wandb,
            data_roots=data_roots,
            **flow_kwargs,
        )()


if __name__ == "__main__":
    coordinator()
