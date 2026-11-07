import hydra
import logging
import torch
import wandb
import tqdm
from omegaconf import DictConfig, OmegaConf
import random
import numpy as np
import socket
import pickle
from functools import partial



from src.utils.compute_stats import ReconComputeStats
from src.prefect.caching import CacheableDict, CacheableDictConfig

from typing import Dict, Optional, Tuple, cast

from prefect import flow, task
from prefect.cache_policies import TASK_SOURCE, INPUTS, NONE, CachePolicy
from dataclasses import dataclass
import logging
from functools import partial
from src.utils.wandb_utils import wandb_kwargs_for_prefect_task, WandbParamsTask

import tempfile
import accelerate
from src.prefect.setup import StoragePath
from datetime import datetime
from src.prefect.setup import load_remote_storage, StorageSettings, StoragePath, load_file_system
import os
from pathlib import Path
from tqdm import tqdm
from src.recon_baselines.recon_baseline_resolver import resolve_recon_baseline, BaseReconBaseline, BASELINES_CONSUMING_PRETRAINED_MODELS
from src.regularizers.baseline_regularizer_wrapper import BaselineRegularizerWrapper
from src.utils.data_roots import DataRoots


def _to_python_primitives(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_python_primitives(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python_primitives(v) for v in value]
    return value

_PRETRAINED_MODEL_INPUTS = ("pretrained_model_path", "pretrained_model_path2", "pretrained_vdm_model_path")


def baseline_consumes_pretrained_models(baseline_params_cfg, regularizer_wrapper_cfg) -> bool:
    """Whether a recon_baseline_task call can read any of the pretrained_* inputs: either the
    baseline itself does (BASELINES_CONSUMING_PRETRAINED_MODELS) or a regularizer wrapper is
    attached (its regularizer_fn gets pretrained_model_path/_path2)."""
    if regularizer_wrapper_cfg is not None:
        return True
    try:
        name = baseline_params_cfg.cfg.name
    except AttributeError:
        return True  # unknown shape -> be conservative and keep the paths in the key
    return name in BASELINES_CONSUMING_PRETRAINED_MODELS


@dataclass
class _BaselineCachePolicy(CachePolicy):
    """`inner` cache policy with the pretrained_* inputs nulled for baselines that never read them.

    Set below the task, not in the decorator (editing the decorator would change TASK_SOURCE)."""
    inner: CachePolicy = None  # type: ignore[assignment]

    def compute_key(self, task_ctx, inputs, flow_parameters, **kwargs):
        if not baseline_consumes_pretrained_models(inputs.get("baseline_params_cfg"), inputs.get("regularizer_wrapper_cfg")):
            inputs = {k: (None if k in _PRETRAINED_MODEL_INPUTS else v) for k, v in inputs.items()}
        return self.inner.compute_key(task_ctx=task_ctx, inputs=inputs, flow_parameters=flow_parameters, **kwargs)


@task(cache_policy=TASK_SOURCE + INPUTS - "wandb_params_task" - "sweep_cfg" - "data_roots", name="Recon Baseline Task", tags=["recon"], version="1.0", retries=0)
def recon_baseline_task(
    seed : int,
    save_result_cfg: CacheableDictConfig,
    domain_specific_setup_cfg : CacheableDictConfig,
    animate_cfg : CacheableDictConfig,
    storage_settings_cfg: CacheableDictConfig,
    data_roots: DataRoots,
    baseline_params_cfg: CacheableDictConfig,
    wandb_params_task : WandbParamsTask,
    regularizer_wrapper_cfg : Optional[CacheableDictConfig] = None,
    sweep_cfg : Optional[CacheableDictConfig] = None,
    pretrained_model_path : Optional[str] = None,
    pretrained_model_path2 : Optional[str] = None,
    pretrained_vdm_model_path : Optional[str] = None,
) -> Tuple[CacheableDictConfig, Optional[StoragePath]]:


    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = "cuda"
    wandb_kwargs = wandb_kwargs_for_prefect_task(wandb_params_task)
    with wandb.init(**wandb_kwargs):
        # dataset loading (done multiple times per gpu, if necessary)
        path_resolver = data_roots.resolver()
        from src.domain_specific_setups.base_domain_specific_setup import resolve_base_domain_specific_setup, BaseDomainSpecificSetup

        domain_specific_setup : BaseDomainSpecificSetup = resolve_base_domain_specific_setup(**domain_specific_setup_cfg.cfg, path_resolver=path_resolver, device=device)

        domain_specific_setup.cache_datasets_on_gpu(device)

        regularizer_wrapper : Optional[BaselineRegularizerWrapper] = None
        if regularizer_wrapper_cfg is not None:
            from src.regularizers.regularizers import regularizer_fn_resolver
            regularizer_fn = regularizer_fn_resolver(**regularizer_wrapper_cfg.cfg.regularizer_fn, **domain_specific_setup.get_meshes_dict(), device=device, path_resolver=path_resolver, target_image=None, frame_processor_for_visualization=domain_specific_setup.get_video_processor_for_visualization(), accelerator=None, pretrained_model_path=pretrained_model_path, pretrained_model_path2=pretrained_model_path2)
            
            regularizer_wrapper = BaselineRegularizerWrapper(regularizer_fn, **domain_specific_setup.get_meshes_dict(), device=device, **regularizer_wrapper_cfg.cfg.wrapper_params)

        # log the gt for reference
        if animate_cfg.cfg.init:
            wandb.log({
                "target" : wandb.Video(
                    domain_specific_setup.get_video_processor_for_visualization()(domain_specific_setup.get_reference_video())
                ),
                "pseudoinv" : wandb.Video(
                    domain_specific_setup.get_video_processor_for_visualization()(domain_specific_setup.get_train_dataset())
                ),
                "target_std" : domain_specific_setup.get_reference_video().std(),
                "target_mean" : domain_specific_setup.get_reference_video().mean(),
                "pseudorec_std" : domain_specific_setup.get_train_dataset().std(),
                "pseudorec_mean" : domain_specific_setup.get_train_dataset().mean()
            })

        ### until here all baseline implementations are the same
        recon_baseline : BaseReconBaseline = resolve_recon_baseline(path_resolver=path_resolver,device=device,pretrained_model_path=pretrained_model_path,pretrained_model_path2=pretrained_model_path2,pretrained_vdm_model_path=pretrained_vdm_model_path,**baseline_params_cfg.cfg)

        # Flow-sensitivity ablation: seed RAFT flow estimation from a differently-degraded
        # video (oracle target, or a pseudoinverse at another undersampling factor) instead
        # of the baseline's own pseudoinverse. Mirrors what NaturalVideoSetup.fit() feeds the
        # flowdist regularizer on the recon_task side; None (the base-class default, and the
        # case for BinnedMRISetup/UnbinnedMRISetup) keeps the baseline's own pseudoinverse.
        flow_reference_video: Optional[torch.Tensor] = domain_specific_setup.get_flow_reference_video(scaled=False)
        if flow_reference_video is not None:
            flow_reference_video = flow_reference_video.movedim(-1, 1)  # (T,H,W,C) -> (T,C,H,W)

        # Runtime / peak GPU memory of the reconstruction itself (run() only: baseline construction
        # incl. checkpoint loading, animations and the final evaluation are excluded). Same
        # ReconComputeStats metrics as recon_task_inner, so both are comparable in the sweep plots.
        compute_stats = ReconComputeStats(device)
        with compute_stats.segment("run"):
            result = recon_baseline.run(
                measurement=domain_specific_setup.get_measurement(scaled=False), # we also pass the reference video here, since some baselines might want to use it (e.g. for self-supervised losses or scaling)
                forward_model=domain_specific_setup.get_forward_model(),
                evaluator=partial(domain_specific_setup.evaluate_video, scaled=False),
                regularizer_wrapper=regularizer_wrapper,
                gt_flow_fwd=domain_specific_setup.get_ground_truth_flow_pair()[0],
                gt_flow_bwd=domain_specific_setup.get_ground_truth_flow_pair()[1],
                flow_reference_video=flow_reference_video,
            ) # here we assume that the result has shape (T, X, Y, C)

        ### from here on baseline implementations are the same again

        if animate_cfg.cfg.final:
            from src.utils.wandb_utils import wandb_video_animation
            print(f"Creating final animation, test, test")
            from src.representations.videovoxel_representation import VideoVoxelRepresentation
            model = VideoVoxelRepresentation(
                video=result.unsqueeze(1), # (T, C=1, X, Y, C2=2)
                spatial_mesh=domain_specific_setup.get_spatial_mesh(),
                time_mesh=domain_specific_setup.get_time_mesh(),
                device=device,
                learnable=False,
            )
            wandb.log({
                "animation": wandb_video_animation(
                    model,
                    domain_specific_setup.get_spatial_mesh(),
                    domain_specific_setup.get_time_mesh(),
                    frame_processor=domain_specific_setup.get_video_processor_for_visualization(),
                    fps=domain_specific_setup.get_fps(),
                    **animate_cfg.cfg.params,
                    device=device),
                "global_step": 1
            })

            if animate_cfg.cfg.warp_animation:
                wandb.log(
                    model.wandb_animation(
                        quiver_cfg=animate_cfg.cfg.warp_animation_quiver_cfg,
                        quiver_enable=animate_cfg.cfg.warp_animation_quiver_enable,
                        color_cfg=animate_cfg.cfg.warp_animation_color_cfg,
                        color_enable=animate_cfg.cfg.warp_animation_color_enable,
                        overlay_cfg=animate_cfg.cfg.warp_animation_overlay_cfg,
                        overlay_enable=animate_cfg.cfg.warp_animation_overlay_enable,
                        reference_video=domain_specific_setup.get_reference_video()
                    )
                )

        storage_path: Optional[StoragePath] = None
        if save_result_cfg.cfg.enable:
            output_dir_in_fs = os.path.join(
                wandb_kwargs["config"]["flow_name"] + "_" + wandb_kwargs["config"]["flow_run_name"],
                str(datetime.now().strftime('%Y%m%d_%H%M%S')) + "_" + wandb_kwargs["name"]
            )
            storage_settings_dict = OmegaConf.to_container(storage_settings_cfg.cfg, resolve=True)
            storage_settings = StorageSettings(**storage_settings_dict)
            filesystem = load_file_system(storage_settings)
            with tempfile.TemporaryDirectory() as tmp_dir:
                file_name = "wandb_kwargs.yaml"
                OmegaConf.save(config=wandb_params_task.config, f=os.path.join(tmp_dir, file_name))
                result_for_save = result.to(torch.float16) if save_result_cfg.cfg.save_in_fp16 else result
                torch.save(result_for_save, os.path.join(tmp_dir, "recon.pt"))
                torch.save(domain_specific_setup.get_reference_video(), os.path.join(tmp_dir, "reference_video.pt"))
                # get_train_dataset() is in the measurement-scaled domain; undo the scaling so the
                # stored pseudoinverse matches reference_video.pt / recon.pt (as recon_tasks.py does).
                pseudoinv_video = domain_specific_setup.get_train_dataset()
                if isinstance(pseudoinv_video, torch.Tensor):
                    pseudoinv_video = domain_specific_setup.unscale_video(pseudoinv_video.detach()).cpu()
                torch.save(pseudoinv_video, os.path.join(tmp_dir, "pseudoinv_video.pt"))
                filesystem.put_directory(tmp_dir, output_dir_in_fs)

            storage_path = StoragePath(
                storage_path=output_dir_in_fs,
                storage_settings=storage_settings,
            )

        eval_results = domain_specific_setup.evaluate_video(
            result, extensive_eval=True, scaled=False
        ) | compute_stats.metrics()
        wandb.log({
            **eval_results,
            "global_step": 1
        })
        # wait until wandb has finished logging
        wandb.finish()
        eval_payload = _to_python_primitives(eval_results | {"save_results": save_result_cfg.cfg.enable})
        eval_cfg = cast(DictConfig, OmegaConf.create(eval_payload))
        return (
            CacheableDictConfig(cfg=eval_cfg),
            storage_path,
        )
    raise RuntimeError("Should not reach here")


# see _BaselineCachePolicy
recon_baseline_task.cache_policy = _BaselineCachePolicy(inner=recon_baseline_task.cache_policy)
