from prefect import flow

from src.tasks.preprocess_video_dataset_for_pretraining_task import (
    preprocess_video_dataset_for_pretraining_task,
)
from src.tasks.convert_vimeo_to_video_dataset_task import (
    convert_vimeo_to_video_dataset,
)
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose
from typing import List, Dict, Any
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

from src.flows.flow_utils.tuning_flow import tune_recon_and_baselines
from src.flows.flow_utils.hp_sharing import resolve_hp_share_references
from src.utils.data_roots import DataRoots
from src.prefect.setup import StoragePath, StorageSettings

@flow(name="recon_and_baseline_tuning_with_pretrained_dms_flow")
def recon_and_baseline_tuning_with_pretrained_dms_flow(
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
    # training (coupling model)
    train_cpl_base_config_path: str,
    train_cpl_base_config_name: str,
    train_cpl_task_name: str,
    train_cpl_overrides: List[str],
    train_cpl_task_overrides: Dict[str, List[Any]],
    train_cpl_post_overrides : List[str],
    train_cpl_cache_refresh: bool,
    train_cpl_num_gpus: int,
    plot_config_path: str,
    plot_config_name: str,
    plot_overrides: List[str],
    plot_num_gpus: int,
    plot_visual_enabled: bool,
    plot_visual_config_path: str,
    plot_visual_config_name: str,
    plot_visual_overrides: List[str],
    plot_visual_num_gpus: int,
    wandb_cfg: DictConfig,
    data_roots: DataRoots,
    # CPU counts per stage (Ray num_cpus); None -> Ray default.
    convert_to_video_dataset_cpus_per_task: Optional[int] = None,
    preprocess_video_dataset_cpus_per_task: Optional[int] = None,
    visualize_cpus_per_task: Optional[int] = None,
    train_cpl_cpus_per_task: Optional[int] = None,
    plot_cpus_per_task: Optional[int] = None,
    plot_visual_cpus_per_task: Optional[int] = None,
    # train_cpl_enabled=False skips pretraining (pretrained_model_path=None, needs skip_recon=True)
    train_cpl_enabled: bool = True,
    skip_recon: bool = False,
    # optional dataset download before pretraining
    download_task_name: Optional[str] = None,
    download_config_path: Optional[str] = None,
    download_config_name: Optional[str] = None,
    download_overrides: List[str] = [],
    download_cache_refresh: bool = False,
    download_num_gpus: int = 0,
    download_cpus_per_task: Optional[int] = None,
    # recon HP sharing: models differing only in these per-stage keys reuse their reference's tuned HPs
    recon_hp_share_over: Optional[Dict[str, List[str]]] = None,
    recon_hp_share_reference: Optional[Dict[str, Dict[str, Any]]] = None,
    # run model-independent baselines once (with the first model) instead of once per model
    baseline_model_independent: bool = False,
    # trained coupling model path in the ml-models bucket; skips pretraining (single model only)
    pretrained_model_path_override: Optional[str] = None,
    **recon_and_baseline_kwargs
) -> None:

    logger = get_run_logger()
    logger.info("Starting preprocess train vdm recon eval flow")
    wandb_params = gather_flow_infos_for_wandb()

    # this first part trains one (or a set of) diffusion models
    task_results : List[Tuple[Optional[PrefectFuture | StoragePath], List[Tuple[str, Any]]]]
    if pretrained_model_path_override is not None:
        if not train_cpl_enabled:
            raise ValueError("pretrained_model_path_override requires train_cpl_enabled=True")
        if train_cpl_task_overrides or recon_hp_share_over:
            raise ValueError("pretrained_model_path_override replaces a single model: no train_cpl_task_overrides sweep / recon_hp_share_over")
        logger.info(f"Skipping pretraining, using pretrained coupling model {pretrained_model_path_override}")
        task_results = [(
            StoragePath(
                storage_path=pretrained_model_path_override,
                storage_settings=StorageSettings(storage_name="ml-models", bucket_or_base_path="ml-models", is_remote=True),
            ),
            [],
        )]
    elif not train_cpl_enabled:
        logger.info("train_cpl_enabled=False: skipping dataset preparation and coupling-model training; pretrained_model_path=None")
        task_results = [(None, [])]
    else:
        task_results = dm_pretraining_flow_part(
            convert_to_video_dataset_task_name=convert_to_video_dataset_task_name,
            convert_to_video_dataset_base_config_path=convert_to_video_dataset_base_config_path,
            convert_to_video_dataset_config_name=convert_to_video_dataset_config_name,
            convert_to_video_dataset_base_overrides=convert_to_video_dataset_base_overrides,
            convert_to_video_dataset_task_overrides=convert_to_video_dataset_task_overrides,
            convert_to_video_dataset_cache_refresh=convert_to_video_dataset_cache_refresh,
            convert_to_video_dataset_num_gpus=convert_to_video_dataset_num_gpus,
            convert_to_video_dataset_cpus_per_task=convert_to_video_dataset_cpus_per_task,
            preprocess_video_dataset_base_config_path=preprocess_video_dataset_base_config_path,
            preprocess_video_dataset_train_config_name=preprocess_video_dataset_train_config_name,
            preprocess_video_dataset_val_config_name=preprocess_video_dataset_val_config_name,
            preprocess_video_dataset_base_overrides=preprocess_video_dataset_base_overrides,
            preprocess_video_dataset_task_overrides=preprocess_video_dataset_task_overrides,
            preprocess_video_dataset_cache_refresh=preprocess_video_dataset_cache_refresh,
            preprocess_video_dataset_num_gpus=preprocess_video_dataset_num_gpus,
            preprocess_video_dataset_cpus_per_task=preprocess_video_dataset_cpus_per_task,
            visualize_base_config_path=visualize_base_config_path,
            visualize_base_config_name=visualize_base_config_name,
            visualize_base_overrides=visualize_base_overrides,
            visualize_task_overrides=visualize_task_overrides,
            visualize_num_gpus=visualize_num_gpus,
            visualize_cpus_per_task=visualize_cpus_per_task,
            train_base_config_path=train_cpl_base_config_path,
            train_base_config_name=train_cpl_base_config_name,
            train_task_name=train_cpl_task_name,
            train_overrides=train_cpl_overrides,
            train_task_overrides=train_cpl_task_overrides,
            train_post_overrides=train_cpl_post_overrides,
            train_cache_refresh=train_cpl_cache_refresh,
            train_num_gpus=train_cpl_num_gpus,
            train_cpus_per_task=train_cpl_cpus_per_task,
            wandb_cfg=wandb_cfg,
            data_roots=data_roots,
            wandb_params=wandb_params,
            logger=logger,
            download_task_name=download_task_name,
            download_config_path=download_config_path,
            download_config_name=download_config_name,
            download_overrides=download_overrides,
            download_cache_refresh=download_cache_refresh,
            download_num_gpus=download_num_gpus,
            download_cpus_per_task=download_cpus_per_task,
        )

    combined_all_task_results: List[Tuple[PrefectFuture, List[Tuple[str, Any]]]] = []
    combined_visual_task_results: List[Tuple[PrefectFuture, List[Tuple[str, Any]]]] = []

    if recon_hp_share_over and (skip_recon or not train_cpl_enabled):
        raise ValueError("recon_hp_share_over requires skip_recon=False and train_cpl_enabled=True")
    # index of the model whose tuned recon HPs model i uses (i itself when it is tuned)
    recon_hp_ref_of = resolve_hp_share_references(
        model_labels=[task_info for _, task_info in task_results],
        # same order in which dm_pretraining_flow_part concatenates the stage labels
        stage_sweeps={
            "convert_to_video_dataset": convert_to_video_dataset_task_overrides,
            "preprocess_video_dataset": preprocess_video_dataset_task_overrides,
            "train_cpl": train_cpl_task_overrides,
        },
        share_over=recon_hp_share_over,
        reference=recon_hp_share_reference,
        name="recon",
        logger=logger,
    )
    skip_baseline = recon_and_baseline_kwargs.pop("skip_baseline", False)
    if baseline_model_independent and skip_recon:
        raise ValueError("baseline_model_independent requires skip_recon=False")
    recon_hp_choices: Dict[int, List[Tuple[PrefectFuture, List[Tuple[str, Any]]]]] = {}
    per_model_results: Dict[int, Tuple[List[Any], List[Any]]] = {}

    # HP-sharing references first; results are collected back in original order
    model_order = sorted(range(len(task_results)), key=lambda j: recon_hp_ref_of[j] != j)
    for i in model_order:
        task_result, task_info = task_results[i]
        if isinstance(task_result, PrefectFuture):
            logger.info(f"Finished DM pretraining task with info {task_info}, result: {task_result.result()}")

        task_to_tune_recon_inputs : Dict[str, Any] = {
            "pretrained_model_path": task_result,
        }
        task_to_tune_baseline_inputs : Dict[str, Any] = {
            "pretrained_model_path": task_result,
        }

        if baseline_model_independent:
            task_to_tune_baseline_inputs = {}

        is_recon_hp_ref = recon_hp_ref_of[i] == i
        if is_recon_hp_ref:
            recon_hp_choices[i] = []
        iter_all_task_results, iter_visual_task_results = tune_recon_and_baselines(
            wandb_cfg=wandb_cfg,
            data_roots=data_roots,
            logger=logger,
            wandb_params=wandb_params,
            task_to_tune_recon_input=task_to_tune_recon_inputs,
            task_to_tune_baseline_input=task_to_tune_baseline_inputs,
            extra_combo_labels=task_info,
            skip_recon=skip_recon,
            skip_baseline=skip_baseline or (baseline_model_independent and i != model_order[0]),
            recon_shared_hp_choices=None if is_recon_hp_ref else recon_hp_choices[recon_hp_ref_of[i]],
            recon_hp_choices_out=recon_hp_choices[i] if is_recon_hp_ref and recon_hp_share_over else None,
            **recon_and_baseline_kwargs,
        )
        per_model_results[i] = (iter_all_task_results, iter_visual_task_results or [])

    for i in range(len(task_results)):
        combined_all_task_results.extend(per_model_results[i][0])
        combined_visual_task_results.extend(per_model_results[i][1])

    # plot once across all training sweep configs
    plot_flow_part(
        all_task_results=combined_all_task_results,
        plot_config_path=plot_config_path,
        plot_config_name=plot_config_name,
        plot_overrides=plot_overrides,
        plot_num_gpus=plot_num_gpus,
        plot_cpus_per_task=plot_cpus_per_task,
        wandb_cfg=wandb_cfg,
        wandb_params=wandb_params,
        logger=logger,
        visual_plot_config_path=plot_visual_config_path if plot_visual_enabled else None,
        visual_plot_config_name=plot_visual_config_name if plot_visual_enabled else None,
        visual_plot_overrides=plot_visual_overrides if plot_visual_enabled else None,
        visual_plot_num_gpus=plot_visual_num_gpus if plot_visual_enabled else None,
        visual_plot_cpus_per_task=plot_visual_cpus_per_task if plot_visual_enabled else None,
        visual_task_results=combined_visual_task_results if combined_visual_task_results else None,
    )
    

    
