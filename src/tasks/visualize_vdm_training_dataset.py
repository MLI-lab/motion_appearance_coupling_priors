import hydra
from omegaconf import DictConfig
from prefect import flow, task
import logging
import os
from accelerate import notebook_launcher, Accelerator
from diffusers import DDPMScheduler
import torch
from prefect import flow, task
from prefect.cache_policies import TASK_SOURCE, INPUTS, NONE
from src.prefect.caching import CacheableDict, CacheableDictConfig
from src.utils.wandb_utils import wandb_kwargs_for_prefect_task, WandbParamsTask
from functools import partial
import socket
import wandb
import numpy as np
from src.dataset.cmr_ocmr_dataset import H5FileDataset
from tqdm import tqdm
import math
from src.utils.state_dict_h5 import torch_module_from_h5dict

from src.utils.utils import clamp_upper_lower, scale_to_unit_interval

from src.utils.wandb_utils import wandb_quiver_animation
from src.warp_methods.base_discrete_warp_method import BaseDiscreteWarpMethod
from src.utils.data_roots import DataRoots

def single_channel_video_to_wandb(v, label, caption, clamp_lower, clamp_upper, fps):
    assert v.shape[1] == 1, "Input video must have a single channel"
    v = v.repeat(1, 3, 1, 1)
    v = clamp_upper_lower(v, clamp_lower, clamp_upper)
    return {label : wandb.Video((scale_to_unit_interval(v) * 255).to(torch.uint8).cpu().numpy(), fps=math.ceil(fps), format="gif", caption=caption)}

def _magnitude_to_unit_interval(v: torch.Tensor) -> torch.Tensor:
    """Robust percentile-based [0,1] stretch for MRI magnitude images/videos.

    Mean +/- k*std clamping fails for right-skewed magnitude (renders as flat gray).
    """
    flat = v.flatten()
    # torch.quantile is capped at 2**24 elements; subsample above that
    max_elements = 2**24 - 1
    if flat.numel() > max_elements:
        idx = torch.randperm(flat.numel(), device=flat.device)[:max_elements]
        flat = flat[idx]
    lo = torch.quantile(flat, 0.01)
    hi = torch.quantile(flat, 0.99)
    return ((v - lo) / (hi - lo + 1e-8)).clamp(0.0, 1.0)

def complex_video_to_wandb(v, label_mag, caption_mag, label_angle, caption_angle, clamp_lower, clamp_upper, fps):
    # assume input shape (_, 2, _, _)
    video_mag = torch.view_as_complex(v.movedim(1, -1).contiguous()).abs()[:, None]
    video_angle = torch.view_as_complex(v.movedim(1, -1).contiguous()).angle()[:, None]

    video_mag_disp = _magnitude_to_unit_interval(video_mag).repeat(1, 3, 1, 1)
    return {
        label_mag: wandb.Video((video_mag_disp * 255).to(torch.uint8).cpu().numpy(), fps=math.ceil(fps), format="gif", caption=caption_mag),
        **single_channel_video_to_wandb(video_angle, label_angle, caption_angle, clamp_lower, clamp_upper, fps)
    }

def three_channel_video_to_wandb(v, label, caption, clamp_lower, clamp_upper, fps):
    assert v.shape[1] == 3, "Input video must have three channels"
    v = clamp_upper_lower(v, clamp_lower, clamp_upper)
    return {label : wandb.Video((scale_to_unit_interval(v) * 255).to(torch.uint8).cpu().numpy(), fps=math.ceil(fps), format="gif", caption=caption)}

@task(cache_policy=TASK_SOURCE + INPUTS - "wandb_params_task" - "data_roots", name="Visualize VDM Training dataset", tags=["visualize"], version="1.0", retries=0)
def visualize_vdm_training_dataset_samples(
        dataset_path : str,
        seed : int,
        sample_nr : int,
        sample_random : bool,
        animation_cfg : CacheableDictConfig,
        clamp_lower : float,
        clamp_upper : float,
        noise_warp_upsample_factor : int,
        fps_slowdown_factor : float,
        wandb_params_task: WandbParamsTask,
        data_roots: DataRoots,
    ):

    dataset_path = data_roots.resolve(dataset_path)  # upstream tasks return cache-root-relative paths
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda"
    # Close any previous wandb session if active
    if wandb.run is not None:
        wandb.finish()

    with wandb.init(**wandb_kwargs_for_prefect_task(wandb_params_task)):

        dataset = H5FileDataset(dataset_path, key_filter=None)

        if sample_nr > len(dataset):
            sample_nr = len(dataset)
            print(f"Requested more samples than available in dataset, reducing to {sample_nr}.")

        if sample_random:
            indices = np.random.choice(len(dataset), size=sample_nr, replace=False)
        else:
            indices = np.arange(min(sample_nr, len(dataset)))

        full_dataset = torch.utils.data.Subset(dataset, indices)

        for data in full_dataset:

            video = data["video"]
            warped_noise_sample = data["warped_noise_sample"]
            attrs = data["attrs"]

            fps = attrs["fps"] * fps_slowdown_factor
            ms_per_frame = 1000.0 / fps

            first_frame = video[[0]] # shape is (1, C, X, Y)

            _, C, X, Y = video.shape
            from src.dataset.cinemri_dataset import get_spatial_mesh_for_matrix_size
            spatial_mesh = get_spatial_mesh_for_matrix_size((X, Y), device=device)  # (X, Y, 2)
            time_mesh = torch.tensor([i*ms_per_frame / 1000 for i in range(video.shape[0])], device=device)  # (T,)

            # video shape is F, X, Y, C, but should be F, C, X, Y
            F, C, X, Y = video.shape
            log_dict_warp_method = {}
            try:

                from src.utils.state_dict_h5 import load_module_h5_auto_prefix
                key_prefix = "warp_parameters"
                warp_method = load_module_h5_auto_prefix(h5_dict=data, key_prefix=key_prefix, spatial_mesh=spatial_mesh, base_tensor=first_frame.movedim(1, -1).to(device), time_mesh=time_mesh, device=device, video=video.movedim(1,-1)[:,None]) # video shape is (F, C, X, Y) -> (F, 1, X, Y, C)

                video_regen = warp_method.full_warp().movedim(-1, 1)  # (F, C, X, Y)
                _, C, _, _ = video_regen.shape
                if C == 1:
                    log_dict_warp_method |= single_channel_video_to_wandb(video_regen, "video_regen", "Regenerated video", clamp_lower, clamp_upper, fps)
                elif C == 2:
                    log_dict_warp_method |= complex_video_to_wandb(video_regen, "video_regen_mag", "Regenerated video magnitude", "video_regen_angle", "Regenerated video phase", clamp_lower, clamp_upper, fps)
                elif C == 3:
                    log_dict_warp_method |= three_channel_video_to_wandb(video_regen, "video_regen_rgb", "Regenerated video RGB", clamp_lower, clamp_upper, fps)
                else:
                    raise ValueError("Cannot visualize videos with more than 2 channels.")

                random_sample = torch.randn( (1, noise_warp_upsample_factor*X, noise_warp_upsample_factor*Y, C), device=device)
                warped_noise = warp_method.full_warp(base_tensor_replacement=random_sample, spatial_mesh_replacement=spatial_mesh, noise_correction=True).movedim(-1, 1)
                if C == 1:
                    log_dict_warp_method |= single_channel_video_to_wandb(warped_noise, "warped_noise_new", "Generated video with warped noise and base image", clamp_lower, clamp_upper, fps)
                elif C == 2:
                    log_dict_warp_method |= complex_video_to_wandb(warped_noise, "warped_noise_new_mag", "Generated video with warped noise magnitude", "warped_noise_new_angle", "Generated video with warped noise phase", clamp_lower, clamp_upper, fps)
                elif C == 3:
                    log_dict_warp_method |= three_channel_video_to_wandb(warped_noise, "warped_noise_new_rgb", "Generated video with warped noise RGB", clamp_lower, clamp_upper, fps)
                else:
                    raise ValueError("Cannot visualize videos with more than 2 channels.")

                psnr_video_regen = -10.0 * torch.log10( (video - video_regen).square().mean() )

                log_dict_warp_method |= {
                    "psnr_video_regen": psnr_video_regen.item(),
                    **warp_method.wandb_animation(
                        quiver_enable=False,
                        quiver_cfg=animation_cfg.cfg.quiver_cfg,
                        color_enable=True,
                        color_cfg=animation_cfg.cfg.color_cfg,
                        overlay_enable=True,
                        overlay_cfg=animation_cfg.cfg.overlay_cfg,
                        reference_video=video.movedim(1,-1),
                    ),
                }

                if isinstance(warp_method, BaseDiscreteWarpMethod):
                    log_dict_warp_method |= {
                        "spatial_param_std": warp_method.spatial_param.std().item(),
                        "temporal_param_std": warp_method.temporal_param.std().item(),
                        # "temporal_param_std": warp_method.deformation_model.get_temporal_param().std().item(),
                    }

            except KeyError as e:
                print(f"Could not load warp method or warp video: {e}, warping methods might be incompatible")
                warped_noise = torch.randn_like(video)

            if C == 1:
                log_dict_warp_method |= {
                    **single_channel_video_to_wandb(video, "video", "Video", clamp_lower, clamp_upper, fps),
                    **single_channel_video_to_wandb(warped_noise_sample, "warped_noise", "Warped noise", clamp_lower, clamp_upper, fps)
                }
            elif C == 2:
                log_dict_warp_method |= {
                    **complex_video_to_wandb(video, "video_mag", "Video magnitude", "video_angle", "Video phase", clamp_lower, clamp_upper, fps),
                    **complex_video_to_wandb(warped_noise_sample, "warped_noise_mag", "Warped noise magnitude", "warped_noise_angle", "Warped noise phase", clamp_lower, clamp_upper, fps)
                }
            elif C == 3:
                log_dict_warp_method |= {
                    **three_channel_video_to_wandb(video, "video_rgb", "Video RGB", clamp_lower, clamp_upper, fps),
                    **three_channel_video_to_wandb(warped_noise_sample, "warped_noise_rgb", "Warped noise RGB", clamp_lower, clamp_upper, fps)
                }
            else:
                raise ValueError("Cannot visualize videos with more than 2 channels.")

            wandb.log({
                "video_std" : video.std(),
                "video_regen_std" : video_regen.std() if 'video_regen' in locals() else None,
                "example_base_tensor": wandb.Image(first_frame[0].norm(dim=0).cpu().numpy(), caption="Example base tensor"),
                **log_dict_warp_method
            })
