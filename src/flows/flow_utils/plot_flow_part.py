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
from typing import cast, Tuple

from src.prefect.sweeping import (
    get_hydra_overwrite_list,
    get_hydra_sweep_combos,
    get_hydra_overwrite_str,
    get_pandas_dataframe_from_sweep_combos,
)

from src.tasks.plot_sweep_results import seaborn_plot_sweep_results_auto
from src.tasks.plot_visual_examples import plot_visual_examples

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

from src.utils.wandb_utils import WandbParamsTask
import inspect


def _filter_task_kwargs(task_obj, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    fn = task_obj.fn if hasattr(task_obj, "fn") else task_obj
    signature = inspect.signature(fn)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}

def plot_flow_part(
    all_task_results : List[Tuple[PrefectFuture, List[Tuple[str, Any]]]],
    plot_config_path: str,
    plot_config_name: str,
    plot_overrides: List[str],
    plot_num_gpus: int,
    wandb_cfg: DictConfig,
    wandb_params : WandbParamsTask,
    logger,
    visual_plot_config_path: str | None = None,
    visual_plot_config_name: str | None = None,
    visual_plot_overrides: List[str] | None = None,
    visual_plot_num_gpus: int | None = None,
    visual_task_results: List[Tuple[PrefectFuture, List[Tuple[str, Any]]]] | None = None,
    plot_cpus_per_task: int | None = None,
    visual_plot_cpus_per_task: int | None = None,
) -> None:

    result_futures = [result[0] for result in all_task_results]
    df_sweep_combos = get_pandas_dataframe_from_sweep_combos(
        [result[1] for result in all_task_results]
    )

    GlobalHydra.instance().clear()
    with initialize(
        config_path=os.path.join("../../../hydra", plot_config_path), version_base="1.2"
    ):
        sweep_cfg = compose(config_name=plot_config_name, overrides=plot_overrides)

    sweep_wandb_params_task = wandb_params.model_copy(
        update={"name_aux": "", "config": flatten_hydra_config(sweep_cfg), **cast(Dict[str, Any], wandb_cfg)}
    )
    sweep_task_kwargs = dict(
        results=result_futures,
        df_sweep=df_sweep_combos,
        wandb_params_task=sweep_wandb_params_task,
        **hydra_config_to_cacheable_dict(sweep_cfg),
    )

    with remote_options(num_gpus=cast(Any, plot_num_gpus), cpus_per_task=plot_cpus_per_task):
        seaborn_plot_sweep_results_auto.submit(
            **_filter_task_kwargs(seaborn_plot_sweep_results_auto, sweep_task_kwargs),
        ).wait()

    if visual_plot_config_path is not None and visual_plot_config_name is not None:
        visual_result_source = all_task_results if visual_task_results is None else visual_task_results
        visual_result_futures = [result[0] for result in visual_result_source]
        visual_df_sweep_combos = get_pandas_dataframe_from_sweep_combos(
            [result[1] for result in visual_result_source]
        )
        visual_plot_overrides = visual_plot_overrides or []
        GlobalHydra.instance().clear()
        with initialize(
            config_path=os.path.join("../../../hydra", visual_plot_config_path), version_base="1.2"
        ):
            visual_cfg = compose(config_name=visual_plot_config_name, overrides=visual_plot_overrides)

        visual_wandb_params_task = wandb_params.model_copy(
            update={"name_aux": "", "config": flatten_hydra_config(visual_cfg), **cast(Dict[str, Any], wandb_cfg)}
        )
        visual_task_kwargs = dict(
            results=visual_result_futures,
            df_sweep=visual_df_sweep_combos,
            wandb_params_task=visual_wandb_params_task,
            **hydra_config_to_cacheable_dict(visual_cfg),
        )

        with remote_options(num_gpus=cast(Any, visual_plot_num_gpus or plot_num_gpus), cpus_per_task=visual_plot_cpus_per_task or plot_cpus_per_task):
            plot_visual_examples.submit(
                **_filter_task_kwargs(plot_visual_examples, visual_task_kwargs),
            ).wait()
    elif bool(getattr(sweep_cfg, "visual_examples_enabled", False)):
        with remote_options(num_gpus=cast(Any, plot_num_gpus), cpus_per_task=plot_cpus_per_task):
            plot_visual_examples.submit(
                **_filter_task_kwargs(plot_visual_examples, sweep_task_kwargs),
            ).wait()

    logger.info("All tasks completed successfully.")