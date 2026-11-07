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
from typing import cast, Optional, Tuple

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

from src.utils.wandb_utils import WandbParamsTask
import inspect
from src.utils.data_roots import DataRoots

def recon_part(
    train_task_results : List[Tuple[Optional[PrefectFuture], List[Tuple[str, Any]]]],
    recon_task_name : str,
    recon_base_config_path: str,
    recon_base_config_name: str,
    recon_base_overrides: List[str],
    recon_task_overrides: Dict[str, List[Any]],
    recon_post_overrides: List[str],
    recon_cache_refresh: bool,
    recon_num_gpus: int,
    wandb_cfg: DictConfig,
    wandb_params : WandbParamsTask,
    data_roots: DataRoots
) -> List[Tuple[PrefectFuture, List[Tuple[str, Any]]]]:

    from src.tasks.task_resolver import resolve_task
    recon_task = resolve_task(recon_task_name)

    GlobalHydra.instance().clear()
    recon_task_results = []
    if len(train_task_results) == 0:
        train_task_results = [(None, [])]

    recon_task_fn = recon_task.fn if hasattr(recon_task, "fn") else recon_task
    recon_task_signature = inspect.signature(recon_task_fn)

    with initialize(
        config_path=os.path.join("../../../hydra", recon_base_config_path),
        version_base="1.2",
    ):
        # perform recon for all sweep combinations
        for sweep_combo_recon in get_hydra_sweep_combos(recon_task_overrides):
            cfg = compose(
                config_name=recon_base_config_name,
                overrides=recon_base_overrides
                + get_hydra_overwrite_list(sweep_combo_recon)
                + recon_post_overrides,
            )

            for train_task_result, sweep_combo in train_task_results:

                name_parts = []
                recon_name_part = get_hydra_overwrite_str(sweep_combo_recon)
                prev_name_part = get_hydra_overwrite_str(sweep_combo)
                if recon_name_part:
                    name_parts.append(recon_name_part)
                if prev_name_part:
                    name_parts.append(prev_name_part)

                wandb_params_task = wandb_params.model_copy(
                    update={
                        "name_aux": "_".join(name_parts),
                        "config": flatten_hydra_config(cfg),
                        **cast(Dict[str, Any], wandb_cfg),
                    }
                )

                with remote_options(num_gpus=cast(Any, recon_num_gpus)):

                    submit_kwargs = {
                        **hydra_config_to_cacheable_dict(cfg),
                        "wandb_params_task": wandb_params_task,
                        "data_roots": data_roots,
                    }
                    if "pretrained_model_path" in recon_task_signature.parameters and train_task_result is not None:
                        submit_kwargs["pretrained_model_path"] = train_task_result

                    # run the recon task to obtain the deformation model (if necessary)
                    recon_task_results.append(
                        (
                            recon_task.with_options(
                                refresh_cache=recon_cache_refresh
                            ).submit(**submit_kwargs),
                            sweep_combo + sweep_combo_recon,
                        )
                    )
    
        return recon_task_results