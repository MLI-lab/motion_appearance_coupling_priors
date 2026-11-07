from prefect import flow

from src.tasks.preprocess_video_dataset_for_pretraining_task import (
    preprocess_video_dataset_for_pretraining_task,
)
from src.tasks.convert_vimeo_to_video_dataset_task import (
    convert_vimeo_to_video_dataset,
)
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose
from typing import List, Dict, Any, cast
from itertools import product
from hydra.core.global_hydra import GlobalHydra
import os
from src.prefect.sweeping import get_hydra_sweep_combos
from src.prefect.task_options import task_options as remote_options

from src.prefect.sweeping import (
    get_hydra_overwrite_list,
    get_hydra_sweep_combos,
    get_hydra_overwrite_str,
    get_pandas_dataframe_from_sweep_combos,
)

from src.tasks.plot_sweep_results import seaborn_plot_sweep_results_auto

from prefect.logging import get_run_logger

from src.utils.wandb_utils import (
    wandb_kwargs_for_prefect_task,
    gather_flow_infos_for_wandb,
    flatten_hydra_config,
)
from src.prefect.caching import hydra_config_to_cacheable_dict

from src.tasks.visualize_vdm_training_dataset import (
    visualize_vdm_training_dataset_samples,
)

from src.tasks.train_probabilistic_coupling_model import train_probabilistic_coupling_model

from prefect import task
from prefect.futures import PrefectFuture
from prefect.cache_policies import TASK_SOURCE, INPUTS, NONE
from src.prefect.caching import CacheableDict, CacheableDictConfig

from src.tasks.recon_tasks import recon_task

from src.flows.flow_utils.dm_pretraining_part import dm_pretraining_flow_part
from src.flows.flow_utils.tuning_flow import tuning_flow
from src.flows.flow_utils.plot_flow_part import plot_flow_part
from typing import Tuple, Optional

from src.flows.flow_utils.recon_part import recon_part
from src.prefect.setup import StoragePath, StorageSettings
from src.utils.data_roots import DataRoots

@flow(name="recon_with_pretrained_dms")
def recon_with_pretrained_dms(
    # flow part 1 (dm pretraining)
    # conversion task -> resolved
    convert_to_video_dataset_task_name: str,
    convert_to_video_dataset_base_config_path: str,
    convert_to_video_dataset_config_name: str,
    convert_to_video_dataset_base_overrides: List[str],
    convert_to_video_dataset_task_overrides: Dict[str, List[Any]],
    convert_to_video_dataset_cache_refresh: bool,
    convert_to_video_dataset_num_gpus: int,
    # preprocessing
    preprocess_video_dataset_base_config_path: str,
    preprocess_video_dataset_train_config_name: str,
    preprocess_video_dataset_val_config_name: str,
    preprocess_video_dataset_base_overrides: List[str],
    preprocess_video_dataset_task_overrides: Dict[str, List[Any]],
    preprocess_video_dataset_cache_refresh: bool,
    preprocess_video_dataset_num_gpus: int,
    # visualization
    visualize_base_config_path: str,
    visualize_base_config_name: str,
    visualize_base_overrides: List[str],
    visualize_task_overrides: Dict[str, List[Any]],
    visualize_num_gpus: int,
    # training
    train_base_config_path: str,
    train_base_config_name: str,
    train_task_name: str,
    train_overrides: List[str],
    train_task_overrides: Dict[str, List[Any]],
    train_post_overrides : List[str],
    train_cache_refresh: bool,
    train_num_gpus: int,
    # reconstructing
    recon_task_name : str,
    recon_base_config_path: str,
    recon_base_config_name: str,
    recon_base_overrides: List[str],
    recon_task_overrides: Dict[str, List[Any]],
    recon_post_overrides: List[str],
    recon_cache_refresh: bool,
    recon_num_gpus: int,
    # plotting
    plot_sweep_config_path: str,
    plot_sweep_config_name: str,
    plot_sweep_overrides: List[str],
    plot_sweep_num_gpus: int,
    plot_visual_config_path: str,
    plot_visual_config_name: str,
    plot_visual_overrides: List[str],
    plot_visual_num_gpus: int,
    wandb_cfg: DictConfig,
    data_roots: DataRoots,
    # trained model path in the ml-models bucket; skips pretraining
    pretrained_model_path_override: Optional[str] = None,
    **recon_and_baseline_kwargs
) -> None:

    logger = get_run_logger()
    logger.info("Starting preprocess train vdm recon eval flow")
    wandb_params = gather_flow_infos_for_wandb()

    if pretrained_model_path_override is not None:
        logger.info(f"Skipping pretraining, using pretrained model {pretrained_model_path_override}")
        train_task_results = [(
            StoragePath(
                storage_path=pretrained_model_path_override,
                storage_settings=StorageSettings(storage_name="ml-models", bucket_or_base_path="ml-models", is_remote=True),
            ),
            [],
        )]
    else:
        # this first part trains one (or a set of) diffusion models
        train_task_results = cast(
        List[Tuple[PrefectFuture | StoragePath | None, List[Tuple[str, Any]]]],
        dm_pretraining_flow_part(
        convert_to_video_dataset_task_name=convert_to_video_dataset_task_name,
        convert_to_video_dataset_base_config_path=convert_to_video_dataset_base_config_path,
        convert_to_video_dataset_config_name=convert_to_video_dataset_config_name,
        convert_to_video_dataset_base_overrides=convert_to_video_dataset_base_overrides,
        convert_to_video_dataset_task_overrides=convert_to_video_dataset_task_overrides,
        convert_to_video_dataset_cache_refresh=convert_to_video_dataset_cache_refresh,
        convert_to_video_dataset_num_gpus=convert_to_video_dataset_num_gpus,
        preprocess_video_dataset_base_config_path=preprocess_video_dataset_base_config_path,
        preprocess_video_dataset_train_config_name=preprocess_video_dataset_train_config_name,
        preprocess_video_dataset_val_config_name=preprocess_video_dataset_val_config_name,
        preprocess_video_dataset_base_overrides=preprocess_video_dataset_base_overrides,
        preprocess_video_dataset_task_overrides=preprocess_video_dataset_task_overrides,
        preprocess_video_dataset_cache_refresh=preprocess_video_dataset_cache_refresh,
        preprocess_video_dataset_num_gpus=preprocess_video_dataset_num_gpus,
        visualize_base_config_path=visualize_base_config_path,
        visualize_base_config_name=visualize_base_config_name,
        visualize_base_overrides=visualize_base_overrides,
        visualize_task_overrides=visualize_task_overrides,
        visualize_num_gpus=visualize_num_gpus,
        train_base_config_path=train_base_config_path,
        train_base_config_name=train_base_config_name,
        train_task_name=train_task_name,
        train_overrides=train_overrides,
        train_task_overrides=train_task_overrides,
        train_post_overrides=train_post_overrides,
        train_cache_refresh=train_cache_refresh,
        train_num_gpus=train_num_gpus,
        wandb_cfg=wandb_cfg,
        data_roots=data_roots,
        wandb_params=wandb_params,
        logger=logger
        )
        )

    recon_task_results = recon_part(
        train_task_results=train_task_results,
        recon_task_name=recon_task_name,
        recon_base_config_path=recon_base_config_path,
        recon_base_config_name=recon_base_config_name,
        recon_base_overrides=recon_base_overrides,
        recon_task_overrides=recon_task_overrides,
        recon_post_overrides=recon_post_overrides,
        recon_cache_refresh=recon_cache_refresh,
        recon_num_gpus=recon_num_gpus,
        wandb_cfg=wandb_cfg,
        wandb_params=wandb_params,
        data_roots=data_roots
    )

    plot_flow_part(
        all_task_results=recon_task_results,
        plot_config_path=plot_sweep_config_path,
        plot_config_name=plot_sweep_config_name,
        plot_overrides=plot_sweep_overrides,
        plot_num_gpus=plot_sweep_num_gpus,
        visual_plot_config_path=plot_visual_config_path,
        visual_plot_config_name=plot_visual_config_name,
        visual_plot_overrides=plot_visual_overrides,
        visual_plot_num_gpus=plot_visual_num_gpus,
        wandb_cfg=wandb_cfg,
        wandb_params=wandb_params,
        logger=logger,
    )
    