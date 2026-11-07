from prefect import flow

from src.tasks.preprocess_video_dataset_for_pretraining_task import (
    preprocess_video_dataset_for_pretraining_task,
)
from src.tasks.convert_vimeo_to_video_dataset_task import (
    convert_vimeo_to_video_dataset,
)
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose
from typing import List, Dict, Any, Optional
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

from prefect import task
from prefect.futures import PrefectFuture
from prefect.cache_policies import TASK_SOURCE, INPUTS, NONE
from src.prefect.caching import CacheableDict, CacheableDictConfig

from src.utils.wandb_utils import WandbParamsTask

from src.tasks.recon_tasks import recon_task
from src.utils.data_roots import DataRoots
def dm_pretraining_flow_part(
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
    wandb_cfg: DictConfig,
    data_roots: DataRoots,
    wandb_params : WandbParamsTask,
    logger,
    # CPU counts per stage (Ray num_cpus); None -> Ray default.
    convert_to_video_dataset_cpus_per_task: Optional[int] = None,
    preprocess_video_dataset_cpus_per_task: Optional[int] = None,
    visualize_cpus_per_task: Optional[int] = None,
    train_cpus_per_task: Optional[int] = None,
    # optional dataset download, run before conversion
    download_task_name: Optional[str] = None,
    download_config_path: Optional[str] = None,
    download_config_name: Optional[str] = None,
    download_overrides: Optional[List[str]] = None,
    download_cache_refresh: bool = False,
    download_num_gpus: int = 0,
    download_cpus_per_task: Optional[int] = None,
) -> List[Tuple[PrefectFuture, List[Tuple[str, Any]]]]:

    import src.tasks.task_resolver as task_resolver
    convert_to_video_dataset = task_resolver.resolve_task(convert_to_video_dataset_task_name)
    train_task = task_resolver.resolve_task(train_task_name)

    if download_task_name is not None:
        download_task = task_resolver.resolve_task(download_task_name)
        GlobalHydra.instance().clear()
        with initialize(
            config_path=os.path.join("../../../hydra", download_config_path),
            version_base="1.2",
        ):
            download_cfg = compose(
                config_name=download_config_name,
                overrides=download_overrides or [],
            )
        with remote_options(num_gpus=download_num_gpus, cpus_per_task=download_cpus_per_task):
            # block until the dataset is downloaded
            download_task.with_options(refresh_cache=download_cache_refresh).submit(
                **hydra_config_to_cacheable_dict(download_cfg),
                data_roots=data_roots,
            ).result()

    # First we start with the raw dataset preprocessing (vimeox to video dataset)
    GlobalHydra.instance().clear()
    preprocess_vimeo_to_video_results = (
        []
    )  # list of base directories where the datasets are stored
    with initialize(
        config_path=os.path.join(
            "../../../hydra", convert_to_video_dataset_base_config_path
        ),
        version_base="1.2",
    ):
        for sweep_combo_preprocess_vimeo in get_hydra_sweep_combos(
            convert_to_video_dataset_task_overrides
        ):

            cfg = compose(
                config_name=convert_to_video_dataset_config_name,
                overrides=convert_to_video_dataset_base_overrides
                + get_hydra_overwrite_list(sweep_combo_preprocess_vimeo),
            )

            wandb_params_task = wandb_params.model_copy(
                update={
                    "config": flatten_hydra_config(cfg),
                    **cast(Dict[str, Any], wandb_cfg),
                    "name_aux": get_hydra_overwrite_str(sweep_combo_preprocess_vimeo),
                }
            )

            with remote_options(num_gpus=convert_to_video_dataset_num_gpus, cpus_per_task=convert_to_video_dataset_cpus_per_task):
                preprocess_vimeo_to_video_results.append(
                    (
                        convert_to_video_dataset.with_options(
                            refresh_cache=convert_to_video_dataset_cache_refresh
                        ).submit(
                            **hydra_config_to_cacheable_dict(cfg),
                            wandb_params_task=wandb_params_task,
                            data_roots=data_roots,
                        ),
                        sweep_combo_preprocess_vimeo,
                    )
                )

    # Now prepare the different training and validation datasets, e.g. doing motion estimation and storing the deformation model parameters
    GlobalHydra.instance().clear()
    train_val_preprocess_results = []
    with initialize(
        config_path=os.path.join(
            "../../../hydra", preprocess_video_dataset_base_config_path
        ),
        version_base="1.2",
    ):
        # perform recon for all sweep combinations

        for sweep_combo_preprocess_video in get_hydra_sweep_combos(
            preprocess_video_dataset_task_overrides
        ):

            for (
                preprocess_vimeo_to_video_result,
                sweep_combo_preprocess_vimeo,
            ) in preprocess_vimeo_to_video_results:

                train_cfg = compose(
                    config_name=preprocess_video_dataset_train_config_name,
                    overrides=preprocess_video_dataset_base_overrides
                    + get_hydra_overwrite_list(sweep_combo_preprocess_video),
                )
                train_wandb_params_task = wandb_params.model_copy(
                    update={
                        "config": flatten_hydra_config(train_cfg),
                        **cast(Dict[str, Any], wandb_cfg),
                        "name_aux": get_hydra_overwrite_str(sweep_combo_preprocess_vimeo)
                        + "_"
                        + get_hydra_overwrite_str(sweep_combo_preprocess_video),
                    }
                )
                val_cfg = compose(
                    config_name=preprocess_video_dataset_val_config_name,
                    overrides=preprocess_video_dataset_base_overrides
                    + get_hydra_overwrite_list(sweep_combo_preprocess_video),
                )
                val_wandb_params_task = wandb_params.model_copy(
                    update={
                        "config": flatten_hydra_config(val_cfg),
                        **cast(Dict[str, Any], wandb_cfg),
                        "name_aux": get_hydra_overwrite_str(sweep_combo_preprocess_vimeo)
                        + "_"
                        + get_hydra_overwrite_str(sweep_combo_preprocess_video),
                    }
                )

                with remote_options(num_gpus=preprocess_video_dataset_num_gpus, cpus_per_task=preprocess_video_dataset_cpus_per_task):
                    train_val_preprocess_results.append(
                        (
                            preprocess_video_dataset_for_pretraining_task.with_options(
                                refresh_cache=preprocess_video_dataset_cache_refresh
                            ).submit(
                                **hydra_config_to_cacheable_dict(train_cfg),
                                wandb_params_task=train_wandb_params_task,
                                base_input_dir=preprocess_vimeo_to_video_result,
                                data_roots=data_roots,
                            ),
                            preprocess_video_dataset_for_pretraining_task.with_options(
                                refresh_cache=preprocess_video_dataset_cache_refresh
                            ).submit(
                                **hydra_config_to_cacheable_dict(val_cfg),
                                wandb_params_task=val_wandb_params_task,
                                base_input_dir=preprocess_vimeo_to_video_result,
                                data_roots=data_roots,
                            ),
                            sweep_combo_preprocess_vimeo + sweep_combo_preprocess_video,
                        )
                    )

    # We visualize some samples of the validation dataset
    GlobalHydra.instance().clear()
    visualize_tasks = []
    with initialize(
        config_path=os.path.join("../../../hydra", visualize_base_config_path),
        version_base="1.2",
    ):

        visualize_combos = get_hydra_sweep_combos(visualize_task_overrides)

        for _, val_preprocess_result, sweep_combo in train_val_preprocess_results:
            for vis_combo in visualize_combos:

                visualize_cfg = compose(
                    config_name=visualize_base_config_name,
                    overrides=visualize_base_overrides
                    + get_hydra_overwrite_list(vis_combo),
                )
                visualize_wandb_params_task = wandb_params.model_copy(
                    update={
                        "config": flatten_hydra_config(visualize_cfg),
                        **cast(Dict[str, Any], wandb_cfg),
                        "name_aux": f"{get_hydra_overwrite_str(sweep_combo)}_{get_hydra_overwrite_str(vis_combo)}",
                    }
                )

                with remote_options(num_gpus=visualize_num_gpus, cpus_per_task=visualize_cpus_per_task):
                    visualize_tasks.append(
                        visualize_vdm_training_dataset_samples.submit(
                            val_preprocess_result,
                            **hydra_config_to_cacheable_dict(visualize_cfg),
                            wandb_params_task=visualize_wandb_params_task, data_roots=data_roots
                        )
                    )

    # We train the video diffusion models
    GlobalHydra.instance().clear()
    train_task_results = []
    with initialize(
        config_path=os.path.join("../../../hydra", train_base_config_path),
        version_base="1.2",
    ):

        for sweep_combo_train in get_hydra_sweep_combos(train_task_overrides):

            train_cfg = compose(
                config_name=train_base_config_name,
                overrides=train_overrides + get_hydra_overwrite_list(sweep_combo_train) + train_post_overrides,
            )

            for (
                train_preprocess_result,
                val_preprocess_result,
                sweep_combo,
            ) in train_val_preprocess_results:

                train_wandb_params_task = wandb_params.model_copy(
                    update={
                        "config": flatten_hydra_config(train_cfg),
                        **cast(Dict[str, Any], wandb_cfg),
                        "name_aux": get_hydra_overwrite_str(sweep_combo)
                        + "_"
                        + get_hydra_overwrite_str(sweep_combo_train),
                    }
                )

                with remote_options(num_gpus=train_num_gpus, cpus_per_task=train_cpus_per_task):
                    train_task_results.append(
                        (
                            train_task.with_options(
                                refresh_cache=train_cache_refresh
                            ).submit(
                                **hydra_config_to_cacheable_dict(train_cfg),
                                dataset_train_path=train_preprocess_result,
                                dataset_val_path=val_preprocess_result,
                                wandb_params_task=train_wandb_params_task,
                                data_roots=data_roots
                            ),
                            sweep_combo + sweep_combo_train,
                        )
                    )

    return train_task_results
