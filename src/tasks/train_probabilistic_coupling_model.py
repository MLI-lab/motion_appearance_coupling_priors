import numpy as np
from typing import Dict, List, Tuple, Union
import hydra
from omegaconf import DictConfig
from prefect import flow, task
import logging
import os
from accelerate import Accelerator
from diffusers import DDPMScheduler, DDIMScheduler, UNetMotionModel
import torch
from prefect import flow, task
from prefect.cache_policies import TASK_SOURCE, INPUTS, NONE
from src.prefect.caching import CacheableDict, CacheableDictConfig
from src.utils.wandb_utils import wandb_kwargs_for_prefect_task, WandbParamsTask
from functools import partial
import socket
from src.dataset.cmr_ocmr_dataset import ImagesFromH5VideoFileDataset

from src.prefect.setup import load_remote_storage, StorageSettings, StoragePath, load_file_system
import tempfile
from src.warp_methods.base_warp_method import BaseWarpMethod, displacement_field_to_velocity_field, velocity_field_to_displacement_field
from src.utils.state_dict_h5 import load_module_h5_auto_prefix
from typing import Optional

from diffusers.training_utils import EMAModel

import wandb
from datetime import datetime
import yaml

import socket
import os

from src.dataset.cmr_ocmr_dataset import CachedH5FileDatasetNew
import torch.nn.functional as F
from src.utils.diffusion_parameterization import (
    get_prediction_type,
    get_training_target,
    predict_clean_from_model_output as _predict_clean_from_model_output,
)

def collate_dict_of_tensors(batch):
    """Stack list of dicts with tensors into a single dict with batched tensors."""
    if not batch:
        return {}
    result = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if all(torch.is_tensor(v) for v in values):
            result[key] = torch.cat(values)
        else:
            # keep non-tensor metadata (e.g. attrs/coupling_norm dicts) as a list
            result[key] = values
    return result


def _pixel_grid_to_normalized_grid(
    grid_px: torch.Tensor,
    h: int,
    w: int,
    align_corners: bool = True,
) -> torch.Tensor:
    if align_corners:
        x = 2.0 * grid_px[..., 0] / max(w - 1, 1) - 1.0
        y = 2.0 * grid_px[..., 1] / max(h - 1, 1) - 1.0
    else:
        x = 2.0 * (grid_px[..., 0] + 0.5) / max(w, 1) - 1.0
        y = 2.0 * (grid_px[..., 1] + 0.5) / max(h, 1) - 1.0
    return torch.stack([x, y], dim=-1)


def _grid_sample_pixels(
    x: torch.Tensor,
    grid_px: torch.Tensor,
    *,
    align_corners: bool = True,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    _, _, h, w = x.shape
    grid_norm = _pixel_grid_to_normalized_grid(
        grid_px,
        h=h,
        w=w,
        align_corners=align_corners,
    )
    return F.grid_sample(
        x,
        grid_norm,
        align_corners=align_corners,
        mode=mode,
        padding_mode=padding_mode,
    )

def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def compute_residual_statistics(
    dataset,
    max_samples: int = 500,
    per_channel: bool = False,
) -> Dict[str, Union[float, List[float]]]:
    """Mean/std of residuals ('image' key) over dataset samples, used to normalize residuals.

    per_channel=True returns per-channel lists, otherwise global scalars."""
    residuals_list = []
    count = 0

    for sample in dataset:
        residual = sample.get("image")
        if residual is not None:
            arr = residual.cpu().numpy() if isinstance(residual, torch.Tensor) else np.asarray(residual)
            residuals_list.append(arr)
            count += 1
            if count >= max_samples:
                break

    if not residuals_list:
        return {"residual_mean": 0.0, "residual_std": 1.0}

    if per_channel:
        # Normalise each element to (1, C, H, W) so we can stack along the batch dim
        normalised = []
        for arr in residuals_list:
            if arr.ndim == 4:       # (B, C, H, W)
                normalised.append(arr)
            elif arr.ndim == 3:     # (C, H, W)
                normalised.append(arr[np.newaxis])
            else:
                normalised.append(arr.reshape(1, 1, -1))
        all_arr = np.concatenate(normalised, axis=0)  # (N, C, ...)
        C = all_arr.shape[1]
        flat = all_arr.transpose(1, 0, *range(2, all_arr.ndim)).reshape(C, -1)  # (C, N*H*W)
        ch_mean = flat.mean(axis=1).tolist()
        ch_std  = [max(float(s), 1e-3) for s in flat.std(axis=1).tolist()]
        print(f"[Train CPL] Per-channel residual statistics from {count} samples (C={C}):")
        for c, (m, s) in enumerate(zip(ch_mean, ch_std)):
            print(f"  ch{c}: mean={m:.6f}  std={s:.6f}")
        return {"residual_mean": ch_mean, "residual_std": ch_std}
    else:
        all_residuals = np.concatenate([a.flatten() for a in residuals_list])
        residual_mean = float(np.mean(all_residuals))
        residual_std  = float(np.std(all_residuals))
        print(f"[Train CPL] Global residual statistics from {count} samples:")
        print(f"  - Mean: {residual_mean:.6f}")
        print(f"  - Std:  {residual_std:.6f}")
        return {"residual_mean": residual_mean, "residual_std": max(residual_std, 1e-3)}


def _as_broadcastable(
    val: Union[float, List[float]],
    device,
) -> Union[float, "torch.Tensor"]:
    """
    Convert a scalar or per-channel list/array to a tensor shaped (1, C, 1, 1)
    that broadcasts cleanly over (B, C, H, W).  A plain scalar is returned as-is.
    """
    if isinstance(val, (int, float)):
        return float(val)
    arr = np.asarray(val, dtype=np.float32)
    if arr.ndim == 0:
        return float(arr)
    return torch.from_numpy(arr).to(device).view(1, arr.size, 1, 1)


def h5_motion_to_pairwise_flow(
    raw: torch.Tensor,
    total_frames: int,
    key_name: str,
) -> torch.Tensor:
    """Convert H5 motion to pairwise standard (x, y) flow.

    Stored tensors are expected as (1, 2, H, W, K), with K either T-1
    (pairwise velocity already) or T (displacement over time).
    """
    if raw.ndim != 5 or raw.shape[0] != 1 or raw.shape[1] != 2:
        raise ValueError(
            f"[full_video_eval] Unexpected motion tensor shape for {key_name}: {tuple(raw.shape)}"
        )

    k = int(raw.shape[-1])
    if k == total_frames - 1:
        pairwise = raw[0].permute(3, 0, 1, 2)
    elif k == total_frames:
        disp = raw[0].permute(3, 0, 1, 2)  # (T,2,H,W)
        pairwise = disp[1:] - disp[:-1]
        print(f"[full_video_eval] Converted displacement motion to pairwise for {key_name} (K=T).")
    else:
        raise ValueError(
            f"[full_video_eval] Motion time dim mismatch for {key_name}: got K={k}, expected T={total_frames} or T-1={total_frames-1}"
        )
    if pairwise.shape[0] != total_frames - 1:
        raise ValueError(
            f"[full_video_eval] Pairwise motion length mismatch for {key_name}: got {pairwise.shape[0]}, expected {total_frames - 1}"
        )
    return pairwise


def predict_clean_from_model_output(
    noisy_images: torch.Tensor,
    model_output: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler: DDPMScheduler,
) -> torch.Tensor:
    """Recover predicted clean sample x0 from the network output.

    Honours the scheduler's ``prediction_type`` (epsilon / v_prediction / sample).
    """
    return _predict_clean_from_model_output(
        noise_scheduler=noise_scheduler,
        model_output=model_output,
        noisy_sample=noisy_images,
        timesteps=timesteps,
    )


def compute_residual_prior_loss(
    pred_clean_scaled: torch.Tensor,
    residual_mean: Union[float, torch.Tensor],
    residual_std: Union[float, torch.Tensor],
    prior_type: str,
    prior_delta: float,
    prior_blur_sigma: float = 0.0,
    prior_blur_kernel_size: int = 0,
    target_clean_scaled: Optional[torch.Tensor] = None,
    condition: Optional[torch.Tensor] = None,
    coupling_model: Optional["LearntCouplingModel"] = None,
) -> torch.Tensor:
    """Compute a configurable prior on the predicted residual in raw residual units."""

    def _gaussian_blur_bchw(x: torch.Tensor, sigma: float, kernel_size: int = 0) -> torch.Tensor:
        if sigma <= 0:
            return x
        if kernel_size is None or kernel_size <= 0:
            radius = max(1, int(np.ceil(3.0 * float(sigma))))
            kernel_size = 2 * radius + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        coords = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - (kernel_size - 1) / 2
        g1 = torch.exp(-(coords * coords) / (2.0 * float(sigma) * float(sigma)))
        g1 = g1 / g1.sum().clamp_min(1e-12)
        g2 = torch.outer(g1, g1)
        g2 = g2 / g2.sum().clamp_min(1e-12)
        kernel = g2.view(1, 1, kernel_size, kernel_size).repeat(x.shape[1], 1, 1, 1)
        pad = kernel_size // 2
        x_pad = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="replicate")
        return torch.nn.functional.conv2d(x_pad, kernel, groups=x.shape[1])

    if prior_type == "none":
        return pred_clean_scaled.new_zeros(())

    pred_clean = pred_clean_scaled * residual_std + residual_mean
    pred_for_prior = _gaussian_blur_bchw(pred_clean, sigma=float(prior_blur_sigma), kernel_size=int(prior_blur_kernel_size))

    if prior_type == "l1":
        return pred_for_prior.abs().mean()
    if prior_type == "huber":
        return torch.nn.functional.huber_loss(
            pred_for_prior,
            torch.zeros_like(pred_for_prior),
            delta=float(prior_delta),
            reduction="mean",
        )
    if prior_type == "blurred_reconstruction":
        if target_clean_scaled is None or condition is None or coupling_model is None:
            raise ValueError(
                "prior_type='blurred_reconstruction' requires target_clean_scaled, condition, and coupling_model."
            )
        target_clean = target_clean_scaled * residual_std + residual_mean
        pred_next = coupling_model.reconstruct_next_frame_from_residual_and_condition(
            residual=pred_clean,
            condition=condition,
        )
        target_next = coupling_model.reconstruct_next_frame_from_residual_and_condition(
            residual=target_clean,
            condition=condition,
        )
        pred_next_blur = _gaussian_blur_bchw(
            pred_next,
            sigma=float(prior_blur_sigma),
            kernel_size=int(prior_blur_kernel_size),
        )
        target_next_blur = _gaussian_blur_bchw(
            target_next,
            sigma=float(prior_blur_sigma),
            kernel_size=int(prior_blur_kernel_size),
        )
        return torch.nn.functional.mse_loss(pred_next_blur, target_next_blur)
    raise ValueError(f"Unknown residual_prior_type: {prior_type!r}")

from src.utils.utils import scale_to_unit_interval
def pipe_image_sampler_coupling(sample, forward_fn, sample_scheduler : DDIMScheduler, condition : torch.Tensor):
    # sample shape (B, C, H, W)
    # condition (B, C+2, H, W)
    result = sample.clone()
    for t in sample_scheduler.timesteps:
        # step() interprets the output via the scheduler's prediction_type
        model_output = forward_fn(result, t, condition)
        result = sample_scheduler.step(model_output, t, result).prev_sample
    return result

def _cat_condition_parts(parts: List[torch.Tensor], ref: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """torch.cat over condition channels; returns a zero-width tensor if parts is empty (unconditional variant)."""
    if not parts:
        shape = list(ref.shape)
        shape[dim] = 0
        return ref.new_zeros(shape)
    return torch.cat(parts, dim=dim)


class LearntCouplingModel:

    def __init__(
        self,
        condition_include_motion: bool,
        condition_include_warped_nframe: bool,
        condition_include_confidence: bool,
        condition_include_nframe: Optional[bool] = None,
        condition_include_frame_distance: bool = False,
        frame_distance_normalize: float = 1.0,
        frame_distance_conditioning_mode: str = "spatial",
        max_frame_distance: int = 1,
        confidence_tau_pix: float = 1.0,
        confidence_floor: float = 0.05,
        motion_unit_scale: bool = False,
        training_branches: str = "both",
        branch_stacking_mode: str = "batch",
        warp_padding_mode: str = "zeros",
        **kwargs,
    ):
        # "zeros" leaves image content in the residual's border strip; "border"/"reflection" keep it small
        if warp_padding_mode not in ("zeros", "border", "reflection"):
            raise ValueError(f"warp_padding_mode must be zeros|border|reflection, got {warp_padding_mode!r}")
        self.warp_padding_mode = warp_padding_mode
        # Accept legacy bool learn_fwd_bwd kwarg and map to training_branches
        if "learn_fwd_bwd" in kwargs:
            _legacy = kwargs.pop("learn_fwd_bwd")
            if isinstance(_legacy, bool):
                training_branches = "both" if _legacy else "r0"
        # Legacy option kept only for backwards-compatible config loading.
        kwargs.pop("invert_motion_switch_xy", None)
        prediction_target = kwargs.pop("prediction_target", "residual")
        if training_branches not in ("r0", "r1", "both"):
            raise ValueError(f"training_branches must be one of 'r0', 'r1', 'both', got {training_branches!r}")
        if prediction_target not in ("residual", "full_frame"):
            raise ValueError(f"prediction_target must be 'residual' or 'full_frame', got {prediction_target!r}")
        if branch_stacking_mode not in ("batch", "channel"):
            raise ValueError(f"branch_stacking_mode must be 'batch' or 'channel', got {branch_stacking_mode!r}")
        if branch_stacking_mode == "channel" and training_branches != "both":
            raise ValueError("branch_stacking_mode='channel' requires training_branches='both'")
        if frame_distance_conditioning_mode not in ("none", "spatial", "film"):
            raise ValueError(
                f"frame_distance_conditioning_mode must be 'none', 'spatial', or 'film', "
                f"got {frame_distance_conditioning_mode!r}"
            )
        self.prediction_target = prediction_target
        self.condition_include_motion = condition_include_motion
        self.condition_include_warped_nframe = condition_include_warped_nframe
        if condition_include_nframe is None:
            self.condition_include_nframe = not condition_include_warped_nframe
        else:
            self.condition_include_nframe = bool(condition_include_nframe)
        self.motion_unit_scale = motion_unit_scale
        self.training_branches = training_branches
        self.branch_stacking_mode = branch_stacking_mode
        self.condition_include_confidence = condition_include_confidence
        self.confidence_tau_pix = confidence_tau_pix
        self.confidence_floor = confidence_floor
        self.condition_include_frame_distance = bool(condition_include_frame_distance)
        self.frame_distance_normalize = float(frame_distance_normalize) if frame_distance_normalize else 1.0
        self.frame_distance_conditioning_mode = str(frame_distance_conditioning_mode)
        self.max_frame_distance = int(max_frame_distance)

    def sample_recon_frame_distance(self, recon_max_frame_distance: Optional[int] = 1) -> int:
        """Sample frame distance d ~ U{1..K} for one recon-time CPL evaluation.

        K = recon_max_frame_distance, or the checkpoint's training max_frame_distance if None."""
        _mfd = self.max_frame_distance if recon_max_frame_distance is None else int(recon_max_frame_distance)
        return int(torch.randint(1, _mfd + 1, (1,)).item()) if _mfd > 1 else 1

    def get_extra_input_channels(self, default_input_channels):
        extra_channels = 0
        if self.condition_include_motion:
            extra_channels += 2
        if self.condition_include_warped_nframe:
            extra_channels += default_input_channels
        if self.condition_include_nframe:
            extra_channels += default_input_channels
        if self.condition_include_confidence:
            extra_channels += 1
        if self.condition_include_frame_distance and self.frame_distance_conditioning_mode == "spatial":
            # only "spatial" adds a UNet input channel; "film" strips it in the wrapper
            extra_channels += 1
        return extra_channels

    def warp_frame_with_motion(
        self,
        frame_anchor: torch.Tensor,
        motion_step: torch.Tensor,
        base_grid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Warp frame_anchor with a per-step motion field.

        frame_anchor: (B, C, H, W)
        motion_step:  (B, 2, H, W)
        base_grid:    optional (1|B, H, W, 2)
        """
        b, _, h, w = frame_anchor.shape
        if base_grid is None:
            base_grid = make_coords_grid(batch_size=b, h=h, w=w, device=str(frame_anchor.device)).movedim(1, -1)
        if base_grid.shape[0] == 1 and b > 1:
            base_grid = base_grid.expand(b, -1, -1, -1)
        grid = base_grid + motion_step.movedim(1, -1)
        return grid_sample(frame_anchor, grid, align_corners=True)

    def build_condition_from_anchor_and_motion(
        self,
        frame_anchor: torch.Tensor,
        motion_step: torch.Tensor,
        base_grid: Optional[torch.Tensor] = None,
        motion_step_opposite: Optional[torch.Tensor] = None,
        frame_distance: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build model condition from anchor frame and one motion step.

        motion_step          : flow warping frame_anchor to the target (bwd for r1, fwd for r0).
        motion_step_opposite : optional reverse flow for the FB-consistency confidence map;
                               confidence is all-ones when None.

        Returns:
            condition : (B, C_cond, H, W)
            warped    : (B, C, H, W) — frame_anchor warped by motion_step
        """
        b, _, h, w = frame_anchor.shape
        if base_grid is None:
            base_grid = make_coords_grid(batch_size=b, h=h, w=w, device=str(frame_anchor.device)).movedim(1, -1)
        if base_grid.shape[0] == 1 and b > 1:
            base_grid = base_grid.expand(b, -1, -1, -1)

        # Build sampling grid and warp once; reuse grid for confidence below.
        grid = base_grid + motion_step.movedim(1, -1)
        warped = grid_sample(frame_anchor, grid, align_corners=True)

        cond_parts: List[torch.Tensor] = []
        if self.condition_include_warped_nframe:
            cond_parts.append(warped)
        if self.condition_include_motion:
            if self.motion_unit_scale:
                cond_parts.append(normalize_motion_grid(motion_step.movedim(1, -1), h, w).movedim(-1, 1))
            else:
                cond_parts.append(motion_step)
        if self.condition_include_nframe:
            cond_parts.append(frame_anchor)
        if self.condition_include_confidence:
            if motion_step_opposite is not None:
                # fb_error = ||motion_step + warp(motion_step_opposite, grid)||, as in training
                flow_opp_warped = grid_sample(motion_step_opposite, grid, align_corners=True, mode="bilinear")
                fb_error = torch.linalg.norm(motion_step + flow_opp_warped, dim=1)  # (B, H, W)
                conf = (
                    torch.exp(-fb_error.square() / (2.0 * self.confidence_tau_pix ** 2))
                    * (1.0 - self.confidence_floor)
                    + self.confidence_floor
                )
                # Rescale [floor, 1] → [-1, 1] to match image-channel range.
                conf = 2.0 * conf - 1.0
                cond_parts.append(conf.unsqueeze(1))
            else:
                # No opposite flow available — use max confidence: 1 mapped to +1.
                cond_parts.append(torch.ones((b, 1, h, w), device=frame_anchor.device, dtype=frame_anchor.dtype))

        if self.condition_include_frame_distance:
            fd_val = float(frame_distance) / self.frame_distance_normalize if frame_distance is not None else 0.0
            cond_parts.append(torch.full((b, 1, h, w), fd_val, device=frame_anchor.device, dtype=frame_anchor.dtype))

        condition = _cat_condition_parts(cond_parts, frame_anchor, dim=1)
        return condition, warped

    def predict_adjacent_frame(
        self,
        frame_anchor: torch.Tensor,
        motion_step: torch.Tensor,
        pipe,
        residual_mean: Union[float, torch.Tensor],
        residual_std: Union[float, torch.Tensor],
        noise: Optional[torch.Tensor] = None,
        base_grid: Optional[torch.Tensor] = None,
        motion_step_opposite: Optional[torch.Tensor] = None,
        frame_distance: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict adjacent frame from an anchor frame and one motion step.

        adjacent_pred = warped + pred for prediction_target="residual", else pred.

        Returns:
            adjacent_pred : predicted next frame (B, C, H, W)
            warped        : pure geometric warp, no residual (B, C, H, W)
        """
        condition, warped = self.build_condition_from_anchor_and_motion(
            frame_anchor=frame_anchor,
            motion_step=motion_step,
            base_grid=base_grid,
            motion_step_opposite=motion_step_opposite,
            frame_distance=frame_distance,
        )
        if noise is None:
            noise = torch.randn_like(frame_anchor)
        else:
            noise = noise.to(device=frame_anchor.device, dtype=frame_anchor.dtype)

        pred_scaled = pipe(sample=noise, condition=condition)
        pred = pred_scaled * residual_std + residual_mean
        if self.prediction_target == "full_frame":
            # Model directly predicts the next frame; no warp-correction needed.
            return pred, warped
        return warped + pred, warped

    def reconstruct_next_frame_from_residual_and_condition(
        self,
        residual: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct next-frame image from a residual and model condition.

        This mirrors the coupling model conditioning layout:
          [warped_nframe?][motion?][nframe?][confidence?]
        and returns: next_frame = warped_anchor + residual.
        """
        b, c, h, w = residual.shape
        if condition.ndim != 4 or condition.shape[0] != b:
            raise ValueError(
                f"Expected condition shape (B,C,H,W) with B={b}, got {tuple(condition.shape)}"
            )

        start = 0
        warped = None
        if self.condition_include_warped_nframe:
            if condition.shape[1] < start + c:
                raise ValueError(
                    f"Condition has too few channels for warped frame: needed >= {start + c}, got {condition.shape[1]}"
                )
            warped = condition[:, start : start + c]
            start += c

        motion = None
        if self.condition_include_motion:
            if condition.shape[1] < start + 2:
                raise ValueError(
                    f"Condition has too few channels for motion: needed >= {start + 2}, got {condition.shape[1]}"
                )
            motion = condition[:, start : start + 2]
            start += 2

        frame_anchor = None
        if self.condition_include_nframe:
            if condition.shape[1] < start + c:
                raise ValueError(
                    f"Condition has too few channels for anchor frame: needed >= {start + c}, got {condition.shape[1]}"
                )
            frame_anchor = condition[:, start : start + c]

        if warped is None:
            if frame_anchor is None or motion is None:
                raise ValueError(
                    "Cannot reconstruct next frame without warped_nframe in condition unless both nframe and motion are included."
                )
            if self.motion_unit_scale:
                raise ValueError(
                    "Cannot reconstruct next frame from normalized motion condition when motion_unit_scale=True without explicit warped_nframe."
                )
            base_grid = make_coords_grid(batch_size=b, h=h, w=w, device=str(residual.device)).movedim(1, -1)
            warped = self.warp_frame_with_motion(
                frame_anchor=frame_anchor,
                motion_step=motion,
                base_grid=base_grid,
            )

        if self.prediction_target == "full_frame":
            # Model predicts the full frame, not a residual correction.
            return residual
        return warped + residual

    def video_voxel_repr_to_training_data(self, video_param, motion_param, motion_bwd_param, return_aux_params=False, frame_distances=None, frame_distance: int = 1, domain_transform=None):
        """
        :param video: Tensor of shape (T, C1, H, W, C2)
        :param motion: Tensor of shape (1, 2, H, W,T-1)
        :param motion_bwd: Tensor of shape (1, 2, H, W,T-1)
        :param frame_distances: Optional tensor of shape (T-1,) with signed distances.
                                Defaults to all-ones (consecutive pairs) when None.
        :param frame_distance: Step size d. When >1, flows for steps t..t+d-1 are
                               additively composed so that motion has shape (T-d, 2, H, W).

        Returns: Tuple of (target,condition) with shapes (2*(T-d), C, H, W) and (2*(T-d), C+2, H, W) respectively.
        """
        motion_steps     = motion_param.squeeze(0).movedim(-1, 0)      # (T-1, 2, H, W)
        motion_bwd_steps = motion_bwd_param.squeeze(0).movedim(-1, 0)  # (T-1, 2, H, W)
        T_video = motion_steps.shape[0] + 1  # number of frames
        if frame_distance == 1:
            motion_composed     = motion_steps      # (T-1, 2, H, W)
            motion_bwd_composed = motion_bwd_steps  # (T-1, 2, H, W)
        else:
            d = frame_distance
            # Additive composition: flow(t->t+d) ≈ sum of single-step flows
            motion_composed = torch.stack(
                [motion_steps[t:t+d].sum(0) for t in range(T_video - d)], dim=0
            )  # (T-d, 2, H, W)
            motion_bwd_composed = torch.stack(
                [motion_bwd_steps[t:t+d].sum(0) for t in range(T_video - d)], dim=0
            )  # (T-d, 2, H, W)
        video_reshaped = video_param.squeeze(dim=1).movedim(-1, 1)  # (T, C, H, W)
        if domain_transform is not None:
            # e.g. FrozenLatentEncoder.encode_video_and_motion (VAE latents + latent-grid motion)
            video_reshaped, motion_composed, motion_bwd_composed = domain_transform(
                video_reshaped, motion_composed, motion_bwd_composed
            )
        return self.video_and_motion_to_training_data(
            video=video_reshaped,
            motion=motion_composed,
            motion_bwd=motion_bwd_composed,
            return_aux_params=return_aux_params,
            frame_distances=frame_distances,
            frame_distance=frame_distance,
        )

    def video_and_motion_to_training_data(self, video, motion, motion_bwd, return_aux_params=False, frame_distances=None, frame_distance: int = 1):
        """
        Build training (target, condition) pairs from a video and its motion fields.

        :param video: Tensor of shape (T, C, H, W)
        :param motion: Tensor of shape (T-d, 2, H, W) — forward flow (t -> t+d)
        :param motion_bwd: Tensor of shape (T-d, 2, H, W) — backward flow (t+d -> t)
        :param frame_distances: Optional tensor of shape (T-d,) giving the signed
                                frame distance for each pair.  Positive = forward.
                                When *None*, defaults to all *frame_distance*-ones (consecutive).
        :param frame_distance: Step size d used to select frame pairs (anchors=video[:-d],
                               targets=video[d:]).  Defaults to 1 (consecutive).

        Returns: Tuple of (target, condition) with shapes
                 (2*(T-d), C, H, W) and (2*(T-d), C_cond, H, W) respectively.
        """
        d = frame_distance
        T, C, H, W = video.shape
        n_pairs = T - d
        base_grid = make_coords_grid(batch_size=n_pairs, h=H, w=W, device=str(video.device)).movedim(1,-1) # (n_pairs, H, W, 2)
        grid_fwd = base_grid + motion.movedim(1,-1)
        grid_bwd = base_grid + motion_bwd.movedim(1,-1)
        _pad = getattr(self, "warp_padding_mode", "zeros")
        vid_fwd = _grid_sample_pixels(video[d:],  grid_fwd, align_corners=True, padding_mode=_pad)  # warp target frame back to anchor
        vid_bwd = _grid_sample_pixels(video[:-d], grid_bwd, align_corners=True, padding_mode=_pad)  # warp anchor frame forward to target
        # validity: warp source inside the frame (False in the border strip of width |flow|)
        valid_fwd = ((grid_fwd[..., 0] >= 0) & (grid_fwd[..., 0] <= W - 1) & (grid_fwd[..., 1] >= 0) & (grid_fwd[..., 1] <= H - 1))
        valid_bwd = ((grid_bwd[..., 0] >= 0) & (grid_bwd[..., 0] <= W - 1) & (grid_bwd[..., 1] >= 0) & (grid_bwd[..., 1] <= H - 1))
        mc_r0 = video[:-d] - vid_fwd  # motion-compensation (warp) residuals, (n_pairs, C, H, W)
        mc_r1 = video[d:]  - vid_bwd
        if self.prediction_target == "full_frame":
            r0 = video[:-d]   # predict x_t   given Warp(x_{t+d}, m_fwd)
            r1 = video[d:]    # predict x_{t+d} given Warp(x_t, m_bwd)
        else:
            r0 = mc_r0
            r1 = mc_r1

        flow_fwd_warped = grid_sample(motion, grid_bwd, align_corners=True, mode="bilinear")
        flow_bwd_warped = grid_sample(motion_bwd, grid_fwd, align_corners=True, mode="bilinear")
        fb_err = torch.linalg.norm(motion + flow_bwd_warped, dim=1)  # (n_pairs,H,W) pixels
        fb_bwd_error = torch.linalg.norm(motion_bwd + flow_fwd_warped, dim=1)  # (n_pairs,H,W) pixels

        # confidence masking / occlusion masking
        conf_fwd = torch.exp( -fb_err.square() / (2 * self.confidence_tau_pix * self.confidence_tau_pix) )*(1-self.confidence_floor) + self.confidence_floor  # (n_pairs,H,W)
        conf_bwd = torch.exp( -fb_bwd_error.square() / (2 * self.confidence_tau_pix * self.confidence_tau_pix) )*(1-self.confidence_floor) + self.confidence_floor  # (n_pairs,H,W)
        # Rescale from [floor, 1] → [-1, 1] so the conditioning channel
        # matches the image-channel range (typically times2_minus1).
        conf_fwd = 2.0 * conf_fwd - 1.0
        conf_bwd = 2.0 * conf_bwd - 1.0

        r0_condition_list = []
        r1_condition_list = []
        if self.condition_include_warped_nframe:
            r0_condition_list.append(vid_fwd)
            r1_condition_list.append(vid_bwd)
        if self.condition_include_motion:
            if self.motion_unit_scale:
                r0_condition_list.append(
                    normalize_motion_grid(motion.movedim(1, -1), H, W).movedim(-1, 1)
                )
                r1_condition_list.append(
                    normalize_motion_grid(motion_bwd.movedim(1, -1), H, W).movedim(
                        -1, 1
                    )
                )
            else:
                r0_condition_list.append(motion)
                r1_condition_list.append(motion_bwd)
        if self.condition_include_nframe:
            r0_condition_list.append(video[d:])
            r1_condition_list.append(video[:-d])
        if self.condition_include_confidence:
            r0_condition_list.append(conf_fwd.unsqueeze(1))
            r1_condition_list.append(conf_bwd.unsqueeze(1))

        if self.condition_include_frame_distance:
            # frame_distances: (n_pairs,) signed distances.  Default to +frame_distance (d-step).
            if frame_distances is None:
                fd = torch.full((n_pairs,), fill_value=float(d), device=video.device, dtype=video.dtype)
            else:
                fd = frame_distances.to(device=video.device, dtype=video.dtype)
            fd_norm = fd / self.frame_distance_normalize
            # r0 branch predicts frame_t given frame_{t+d}: direction from anchor to target is *negative*
            r0_fd = (-fd_norm).view(n_pairs, 1, 1, 1).expand(n_pairs, 1, H, W)
            # r1 branch predicts frame_{t+d} given frame_t: direction is *positive*
            r1_fd = fd_norm.view(n_pairs, 1, 1, 1).expand(n_pairs, 1, H, W)
            r0_condition_list.append(r0_fd)
            r1_condition_list.append(r1_fd)

        r0_condition_stack = _cat_condition_parts(r0_condition_list, r0, dim=1) # (n_pairs, C+2, H, W)
        r1_condition_stack = _cat_condition_parts(r1_condition_list, r1, dim=1) # (n_pairs, C+2, H, W)

        def _stack_branches(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
            if self.training_branches == "both":
                # channel mode: (n_pairs, 2*C, H, W); batch mode: (2*n_pairs, C, H, W)
                return torch.cat([x0, x1], dim=1 if self.branch_stacking_mode == "channel" else 0)
            return x1 if self.training_branches == "r1" else x0

        target = _stack_branches(r0, r1)
        condition = _stack_branches(r0_condition_stack, r1_condition_stack)

        if not return_aux_params:
            return target, condition
        else:
            aux_params = {
                "conf_fwd" : conf_fwd,
                "conf_bwd" : conf_bwd,
                "valid_fwd" : valid_fwd,   # (n_pairs, H, W) bool
                "valid_bwd" : valid_bwd,
                # warp residuals laid out like `target`, independent of prediction_target and of
                # which condition channels are included (the classical l1 prior of the regularizers)
                "mc_residual" : _stack_branches(mc_r0, mc_r1),
            }
            return target, condition, aux_params

from torchvision.models.optical_flow._utils import make_coords_grid, grid_sample
from src.utils.utils import scale_to_unit_interval
class CouplingModelPreprocessingTransform:

    def __init__(
        self,
        size_before_downscale: tuple,
        size_after_downscale: tuple,
        enable_crop_and_downscale : bool,
        coupling_model : LearntCouplingModel,
        normalization: Union[str, bool] = "none",
        resample_noise: bool = False,
        append_all: bool = False,
        constant_scaling_factor : Optional[float] = None,
        latent_space: Optional[dict] = None,
        path_resolver=None,
        **kwargs,
    ):

        print(f"Unused kwargs in CouplingModelPreprocessingTransform: {kwargs}")

        if isinstance(size_after_downscale, int):
            size_after_downscale = (size_after_downscale, size_after_downscale)
        else:
            size_after_downscale = tuple(size_after_downscale)

        if isinstance(size_before_downscale, int):
            size_before_downscale = (size_before_downscale, size_before_downscale)
        else:
            size_before_downscale = tuple(size_before_downscale)

        self.size_before_downscale = size_before_downscale  # (height, width) 
        self.size_after_downscale = size_after_downscale  # (height, width)
        if isinstance(normalization, bool):
            # old configs passed bools; True means gaussian z-score
            normalization = "gaussian" if normalization else "none"
            print(f"[Train CPL] Coupling transform normalization bool mapped to: {normalization}")
        self.normalization = str(normalization)
        self.resample_noise = resample_noise
        self.append_all = append_all
        self.enable_crop_and_downscale = enable_crop_and_downscale
        self.constant_scaling_factor = constant_scaling_factor
        self.coupling_model = coupling_model

        # Optional latent-space training: frames are encoded by a frozen SD VAE and motion moved
        # to the latent grid. The encoder is built lazily (per DataLoader worker without dataset cache).
        self._latent_cfg = dict(latent_space) if latent_space and bool(latent_space.get("enabled", False)) else None
        self._path_resolver = path_resolver
        self._latent_encoder = None

    def _get_latent_encoder(self):
        if self._latent_encoder is None and self._latent_cfg is not None:
            from src.regularizers.diffusers_cpl_latent_regularizers import FrozenLatentEncoder
            self._latent_encoder = FrozenLatentEncoder(
                pretrained_model_dir=self._latent_cfg.get("pretrained_model_dir", "model_cache/sd15"),
                device=self._latent_cfg.get("device", "cuda"),
                vae_subfolder=self._latent_cfg.get("vae_subfolder", "vae"),
                assume_pixel_range=self._latent_cfg.get("assume_pixel_range", "0_1"),
                path_resolver=self._path_resolver,
            )
            print(f"[Train CPL] latent-space training: VAE f={self._latent_encoder.downsample_factor} "
                  f"c={self._latent_encoder.latent_channels} on {self._latent_encoder.device}")
        return self._latent_encoder

    def __call__(self, video, attrs, **data):
        f, c, h, w = video.shape # same shape for the warped_noise_sample
        
        assert f == 2, f"Expected 2 frames in the input video, got {f}"

        motion = data["warp_parameters_motion_param"] # shape (1,2,H,W)
        motion_bwd = data["warp_parameters_motion_bwd_param"] # shape (1,2,H,W)

        # first we center crop to the target size
        if self.enable_crop_and_downscale:
            target_h_shape, target_w_shape = self.size_before_downscale
            h1 = min(max(0, h//2 - target_h_shape//2), h-1)
            h2 = max(min(h-1, h//2 + target_h_shape//2), 0)
            w1 = min(max(0, w//2 - target_w_shape//2), w-1)
            w2 = max(min(w-1, w//2 + target_w_shape//2), 0)
            video_cropped = video[:, :, h1:h2, w1:w2]
            motion_cropped = motion[:, :, h1:h2, w1:w2]
            motion_bwd_cropped = motion_bwd[:, :, h1:h2, w1:w2]
        else:
            video_cropped = video
            motion_cropped = motion
            motion_bwd_cropped = motion_bwd

        if self.normalization == "gaussian":
            # normalize to zero mean and unit variance
            video_cropped = (video_cropped - video_cropped.mean()) / (video_cropped.std() + 1e-8)
        elif self.normalization == "uniform_-1_1":
            # normalize to [-1, 1]
            video_cropped = 2 * (video_cropped - video_cropped.min()) / (video_cropped.max() - video_cropped.min() + 1e-8) - 1
        elif self.normalization == "uniform_0_1":
            # normalize to [0, 1]
            video_cropped = (video_cropped - video_cropped.min()) / (video_cropped.max() - video_cropped.min() + 1e-8)
        elif self.normalization in ("none", "frame0_channel_mean_div_std", "video_div_std"):
            pass  # handled in per-clip block below when needed
        else:
            raise ValueError(f"Unknown normalization type: {self.normalization}")
        _clip_norm_dict = None
        if self.normalization in ("frame0_channel_mean_div_std", "video_div_std"):
            from src.utils.video_normalization import VideoNormalization as _VN
            _fit_video = video_cropped[0:1]
            _clip_norm = _VN.fit_fchw(mode=self.normalization, video_fchw=_fit_video)
            video_cropped = _clip_norm.normalize_fchw(video_cropped)
            _clip_norm_dict = _clip_norm.to_dict()

        # now we apply downsampling using torch.nn interpolate with nearest neighbor
        if self.size_before_downscale != self.size_after_downscale and self.enable_crop_and_downscale:
            video_processed = torch.nn.functional.interpolate(video_cropped, size=self.size_after_downscale, mode='nearest')
            _h_in, _w_in = motion_cropped.shape[-2], motion_cropped.shape[-1]
            _h_out, _w_out = self.size_after_downscale
            motion_processed = torch.nn.functional.interpolate(motion_cropped, size=self.size_after_downscale, mode='nearest')
            motion_processed = motion_processed * torch.tensor(
                [_w_out / _w_in, _h_out / _h_in], dtype=motion_processed.dtype, device=motion_processed.device
            ).view(1, 2, 1, 1)
            motion_bwd_processed = torch.nn.functional.interpolate(motion_bwd_cropped, size=self.size_after_downscale, mode='nearest')
            motion_bwd_processed = motion_bwd_processed * torch.tensor(
                [_w_out / _w_in, _h_out / _h_in], dtype=motion_bwd_processed.dtype, device=motion_bwd_processed.device
            ).view(1, 2, 1, 1)
        else:
            video_processed = video_cropped
            motion_processed = motion_cropped
            motion_bwd_processed = motion_bwd_cropped

        if self.constant_scaling_factor is not None:
            video_processed = video_processed * self.constant_scaling_factor

        if self._latent_cfg is not None:
            enc = self._get_latent_encoder()
            if bool(self._latent_cfg.get("denormalize_from_attrs", True)):
                # undo clip normalization so the VAE sees [0,1] frames
                from src.utils.video_normalization import VideoNormalization as _VN
                video_processed = _VN.from_h5_attrs(dict(attrs)).denormalize_fchw(video_processed)
            with torch.no_grad():
                video_processed, motion_processed, motion_bwd_processed = enc.encode_video_and_motion(
                    video_processed.float(), motion_processed.float(), motion_bwd_processed.float()
                )
            video_processed, motion_processed, motion_bwd_processed = (
                video_processed.cpu(), motion_processed.cpu(), motion_bwd_processed.cpu()
            )

        # Extract frame distance (emitted by ImagesFromH5VideoFileDataset).
        frame_distance_val = data.pop("__frame_distance__", None)
        if frame_distance_val is not None:
            # Single scalar tensor → (1,) tensor for a 2-frame video (T-1 = 1 pair).
            _fd = torch.as_tensor(frame_distance_val).reshape(1).float()
        else:
            _fd = None

        target, condition = self.coupling_model.video_and_motion_to_training_data(
            video_processed, motion_processed, motion_bwd_processed,
            frame_distances=_fd,
        )
        base_ret = {"image": target, "condition": condition}
        if _clip_norm_dict is not None:
            base_ret["coupling_norm"] = _clip_norm_dict

        if self.append_all:
            base_ret["attrs"] = attrs
            base_ret.update(data) # might be need for validation etc
        
        return base_ret

from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

def normalize_motion_grid(absolute_grid, h, w):
    # from torchvision optical flow utils
    xgrid, ygrid = absolute_grid.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (w - 1) - 1
    # Adding condition if h > 1 to enable this function be reused in raft-stereo
    if h > 1:
        ygrid = 2 * ygrid / (h - 1) - 1
    return torch.cat([xgrid, ygrid], dim=-1)


def get_diffuser_coupling_model_and_forward_fun(
    name: str,
    device: str,
    set_default_attn_processor: bool,
    enable_forward_chunking: bool,
    forward_chunking_chunk_size: int,
    use_xformers_memory_efficient_attention: bool,
    coupling_model : LearntCouplingModel,
    conditioning_mode: str = "concat",
    **params,
):
    # input shape is (batch, frames, channel, height, width), some models want to have different formats
    if name == "UNet2DCondition":
        if conditioning_mode not in ("concat", "cross_attention"):
            raise ValueError(f"conditioning_mode must be 'concat' or 'cross_attention', got {conditioning_mode!r}")

        in_channels_default = params.get("in_channels", 1)
        extra_input_channels = coupling_model.get_extra_input_channels(in_channels_default)
        _ch_stack = getattr(coupling_model, "branch_stacking_mode", "batch") == "channel"

        if conditioning_mode == "concat":
            if _ch_stack:
                # channel stacking: in_channels = 2*(C + C_cond), out_channels doubled
                out_channels_default = params.get("out_channels", in_channels_default)
                total_in = 2 * (in_channels_default + extra_input_channels)
                print(
                    f"[Train CPL] Coupling conditioning mode=concat + channel-stacking: "
                    f"in_channels {in_channels_default}→{total_in}, "
                    f"out_channels {out_channels_default}→{2 * out_channels_default}."
                )
                params.update({
                    "in_channels": total_in,
                    "out_channels": 2 * out_channels_default,
                })
            else:
                print(
                    f"[Train CPL] Coupling conditioning mode=concat: add {extra_input_channels} channels to input "
                    f"(base {in_channels_default} -> total {in_channels_default + extra_input_channels})."
                )
                params.update({
                    "in_channels" : extra_input_channels + in_channels_default,
                })
        else:
            if _ch_stack:
                # Channel-stacking + cross-attention: noisy input doubles, condition also doubles.
                out_channels_default = params.get("out_channels", in_channels_default)
                print(
                    f"[Train CPL] Coupling conditioning mode=cross_attention + channel-stacking: "
                    f"in_channels {in_channels_default}→{2 * in_channels_default}, "
                    f"out_channels {out_channels_default}→{2 * out_channels_default}, "
                    f"condition channels {extra_input_channels}→{2 * extra_input_channels}."
                )
                params.update({
                    "in_channels": 2 * in_channels_default,
                    "out_channels": 2 * out_channels_default,
                })
                extra_input_channels = 2 * extra_input_channels  # used for cross_attention_dim validation below
            else:
                print(
                    f"[Train CPL] Coupling conditioning mode=cross_attention: keep input channels at {in_channels_default}, "
                    f"condition channels={extra_input_channels} routed through cross-attention context."
                )
                params.update({
                    "in_channels": in_channels_default,
                })

            down_block_types = list(params.get("down_block_types", []))
            up_block_types = list(params.get("up_block_types", []))
            params["down_block_types"] = [
                "CrossAttnDownBlock2D" if blk == "DownBlock2D" else blk
                for blk in down_block_types
            ]
            params["up_block_types"] = [
                "CrossAttnUpBlock2D" if blk == "UpBlock2D" else blk
                for blk in up_block_types
            ]

            _xad = params.get("cross_attention_dim", None)
            if isinstance(_xad, (list, tuple)):
                _vals = [v for v in _xad if v is not None]
                if len(_vals) == 0:
                    params["cross_attention_dim"] = [int(extra_input_channels)] * max(1, len(_xad))
                else:
                    _first = int(_vals[0])
                    if any(int(v) != _first for v in _vals):
                        raise ValueError(f"cross_attention_dim list must have equal non-null entries, got {_xad!r}")
                    if _first != int(extra_input_channels):
                        raise ValueError(
                            f"cross_attention_dim={_first} must match condition channels={extra_input_channels} "
                            "for conditioning_mode='cross_attention'."
                        )
            else:
                _dim = int(_xad) if _xad is not None else int(extra_input_channels)
                if _dim != int(extra_input_channels):
                    raise ValueError(
                        f"cross_attention_dim={_dim} must match condition channels={extra_input_channels} "
                        "for conditioning_mode='cross_attention'."
                    )
                params["cross_attention_dim"] = _dim

        model = UNet2DConditionModel(**params)
        # dummy encoder_hidden_states (batch, seq_len, cross_attention_dim) for concat mode
        zero_condition = torch.zeros((1, 1, 1), device=device)
        if use_xformers_memory_efficient_attention:
            print("Enabling xformers memory efficient attention for UNet2DConditionModel")
            model.enable_xformers_memory_efficient_attention()
        if enable_forward_chunking:
            print("Enabling forward chunking for UNet2DConditionModel")
            model.enable_forward_chunking(chunk_size=forward_chunking_chunk_size, dim=0)
        if set_default_attn_processor:
            print("Setting default attention processor for UNet2DConditionModel")
            model.set_default_attn_processor()  # disables custom attn processors
        disable_cross_attention = conditioning_mode == "concat"
        if disable_cross_attention:
            for name, module in model.named_modules():
                if hasattr(module, "transformer_blocks"):
                    for block in module.transformer_blocks:
                        if hasattr(block, "attn2"):
                            block.attn2 = None
        def _condition_to_cross_attn_tokens(condition: torch.Tensor) -> torch.Tensor:
            # Compact global token to avoid quadratic memory costs from H*W token sequences.
            return condition.mean(dim=(2, 3), keepdim=False).unsqueeze(1)

        return (
            model,
            (
                lambda sample, timestep, condition: model(
                    sample=torch.cat([sample, condition], dim=1),  # concatenate condition to the sample along the channel dimension
                    timestep=timestep,
                    encoder_hidden_states=zero_condition.repeat_interleave(
                        sample.shape[0], dim=0
                    ),
                ).sample
            )
            if conditioning_mode == "concat"
            else (
                lambda sample, timestep, condition: model(
                    sample=sample,
                    timestep=timestep,
                    encoder_hidden_states=_condition_to_cross_attn_tokens(condition),
                ).sample
            ),
        )
    elif name == "UNetMotionModel":
        return UNetMotionModel(**params)
    else:
        raise ValueError(f"Unknown model name: {name}")

from src.representations.videovoxel_representation import time_slice_selector, time_slice_selector_with_distance

@torch.no_grad()
def evaluate_validation_loss(
    forward_fn,
    val_dataloader: torch.utils.data.DataLoader,
    noise_scheduler: DDPMScheduler,
    use_random_noise : bool,
    residual_mean: Union[float, List[float]],
    residual_std: Union[float, List[float]],
    residual_prior_type: str,
    residual_prior_weight: float,
    residual_prior_delta: float,
    residual_prior_blur_sigma: float,
    residual_prior_blur_kernel_size: int,
    coupling_model: LearntCouplingModel,
    device : str,
) -> Dict[str, float]:

    """
        Calculate the training loss on the validation dataset and reports some statistics about it. 
    """

    assert use_random_noise

    val_losses = []
    val_mse_losses = []
    val_prior_losses = []
    clean_target_raw_std = []
    clean_target_raw_var = []
    clean_target_std = []
    clean_target_var = []
    condition_std = []
    condition_var = []
    noisy_images_std = []
    noisy_images_var = []
    model_input_std = []
    model_input_var = []
    _rm = _as_broadcastable(residual_mean, device)
    _rs = _as_broadcastable(residual_std, device)
    for val_batch in val_dataloader:
        image_raw = val_batch["image"].to(device, non_blocking=True)
        image = (image_raw - _rm) / _rs
        condition = val_batch["condition"].to(device, non_blocking=True)

        noise_sample = torch.randn_like(image)

        timesteps = torch.randint(
            0,
            noise_scheduler.num_train_timesteps,
            (image.shape[0],),
            device=image.device,
        )

        noisy_images = noise_scheduler.add_noise(image, noise_sample, timesteps)
        model_input = torch.cat([noisy_images, condition], dim=1)

        clean_target_raw_std.append(image_raw.std().item())
        clean_target_raw_var.append(image_raw.var().item())
        clean_target_std.append(image.std().item())
        clean_target_var.append(image.var().item())
        condition_std.append(condition.std().item())
        condition_var.append(condition.var().item())
        noisy_images_std.append(noisy_images.std().item())
        noisy_images_var.append(noisy_images.var().item())
        model_input_std.append(model_input.std().item())
        model_input_var.append(model_input.var().item())

        output = forward_fn(noisy_images, timestep=timesteps, condition=condition)
        diffusion_target = get_training_target(
            noise_scheduler=noise_scheduler,
            clean_sample=image,
            noise=noise_sample,
            timesteps=timesteps,
        )
        val_mse_loss = torch.nn.functional.mse_loss(output, diffusion_target)
        pred_clean_scaled = predict_clean_from_model_output(
            noisy_images=noisy_images,
            model_output=output,
            timesteps=timesteps,
            noise_scheduler=noise_scheduler,
        )
        val_prior_loss = compute_residual_prior_loss(
            pred_clean_scaled=pred_clean_scaled,
            residual_mean=_rm,
            residual_std=_rs,
            prior_type=residual_prior_type,
            prior_delta=residual_prior_delta,
            prior_blur_sigma=residual_prior_blur_sigma,
            prior_blur_kernel_size=residual_prior_blur_kernel_size,
            target_clean_scaled=image,
            condition=condition,
            coupling_model=coupling_model,
        )
        val_loss = val_mse_loss + float(residual_prior_weight) * val_prior_loss
        val_losses.append(val_loss.item())
        val_mse_losses.append(val_mse_loss.item())
        val_prior_losses.append(val_prior_loss.item())

    return {
        "val_loss_mean": sum(val_losses) / len(val_losses) if val_losses else 0.0,
        "val_loss_count": len(val_losses),
        "val_loss_std": float(np.std(val_losses)) if val_losses else 0.0,
        "val_loss_mse_mean": float(np.mean(val_mse_losses)) if val_mse_losses else 0.0,
        "val_loss_residual_prior_mean": float(np.mean(val_prior_losses)) if val_prior_losses else 0.0,
        "debug_val/clean_target_raw_std_mean": float(np.mean(clean_target_raw_std)) if clean_target_raw_std else 0.0,
        "debug_val/clean_target_raw_var_mean": float(np.mean(clean_target_raw_var)) if clean_target_raw_var else 0.0,
        "debug_val/clean_target_std_mean": float(np.mean(clean_target_std)) if clean_target_std else 0.0,
        "debug_val/clean_target_var_mean": float(np.mean(clean_target_var)) if clean_target_var else 0.0,
        "debug_val/condition_std_mean": float(np.mean(condition_std)) if condition_std else 0.0,
        "debug_val/condition_var_mean": float(np.mean(condition_var)) if condition_var else 0.0,
        "debug_val/noisy_images_std_mean": float(np.mean(noisy_images_std)) if noisy_images_std else 0.0,
        "debug_val/noisy_images_var_mean": float(np.mean(noisy_images_var)) if noisy_images_var else 0.0,
        "debug_val/model_input_std_mean": float(np.mean(model_input_std)) if model_input_std else 0.0,
        "debug_val/model_input_var_mean": float(np.mean(model_input_var)) if model_input_var else 0.0,
    }

from src.utils.utils import clamp_upper_lower, scale_to_unit_interval
from src.utils.data_roots import DataRoots


def _to_display_unit_interval(v: torch.Tensor) -> torch.Tensor:
    v = torch.nan_to_num(v)
    v_min = float(v.min().item())
    v_max = float(v.max().item())

    def _percentile_stretch(v: torch.Tensor) -> torch.Tensor:
        flat = v.flatten()
        lo = torch.quantile(flat, 0.01)
        hi = torch.quantile(flat, 0.99)
        return ((v - lo) / (hi - lo + 1e-8)).clamp(0.0, 1.0)

    # times2_minus1 range; must span both signs so non-negative MRI magnitude doesn't match
    if v_min <= -0.5 and v_max >= 0.5 and v_min >= -1.1 and v_max <= 1.1:
        v_disp = ((v + 1.0) / 2.0).clamp(0.0, 1.0)
    # already in [0,1] with a meaningful dynamic range
    elif v_min >= -0.05 and v_max <= 1.05 and (v_max - v_min) >= 0.1:
        v_disp = v.clamp(0.0, 1.0)
    else:
        v_disp = _percentile_stretch(v)

    # fall back to percentile stretch if the result is near-constant
    if float(v_disp.max().item() - v_disp.min().item()) < 0.05:
        v_disp = _percentile_stretch(v)

    return v_disp


def single_channel_image_to_wandb(v, label, caption, clamp_lower, clamp_upper):
    # Handle both (C,H,W) and (B,C,H,W) input shapes
    if v.ndim == 4:  # (B,C,H,W) - squeeze batch dimension
        v = v.squeeze(0)  # -> (C,H,W)
    assert v.shape[0] == 1, "Input video must have a single channel"
    v = v.repeat(3, 1, 1)

    v_disp = _to_display_unit_interval(v)
    if torch.isnan(v_disp).any() or torch.isinf(v_disp).any():
        v = clamp_upper_lower(v, clamp_lower, clamp_upper)
        v_disp = scale_to_unit_interval(v)

    return {
        label: wandb.Image((v_disp * 255).to(torch.uint8).cpu().movedim(0, -1).numpy(), caption=caption)
    }

def complex_image_to_wandb(v, label_mag, caption_mag, label_angle, caption_angle, clamp_lower, clamp_upper):
    image_mag = torch.view_as_complex(v.movedim(0, -1).contiguous()).abs()[None]
    image_angle = torch.view_as_complex(v.movedim(0, -1).contiguous()).angle()[None]
    return {
        **single_channel_image_to_wandb(image_mag, label_mag, caption_mag, clamp_lower, clamp_upper),
        **single_channel_image_to_wandb(image_angle, label_angle, caption_angle, clamp_lower, clamp_upper)
    }
def multi_channel_image_to_wandb(v, label, caption, clamp_lower, clamp_upper):
    """Log each channel as its own grayscale image (e.g. 4-channel SD-VAE latents)."""
    if v.ndim == 4:
        v = v.squeeze(0)
    return_dict = {}
    for c in range(v.shape[0]):
        return_dict.update(single_channel_image_to_wandb(
            v[c:c+1], label=f"{label}_ch{c}", caption=f"{caption} (channel {c})",
            clamp_lower=clamp_lower, clamp_upper=clamp_upper))
    return return_dict

def three_channel_image_to_wandb(v, label, caption, clamp_lower, clamp_upper):
    if v.ndim == 4:
        v = v.squeeze(0)
    assert v.shape[0] == 3, "Input video must have three channels"
    v_disp = _to_display_unit_interval(v)
    return {
        label: wandb.Image((v_disp * 255).to(torch.uint8).cpu().movedim(0, -1).numpy(), caption=caption)
    }

def _interior_mask_from_condition(cond: torch.Tensor, coupling_model, ch_stack: bool):
    """(H, W) bool mask of pixels whose warp source lies inside the frame, from the motion
    channels of a condition (C_cond, H, W); None if the layout doesn't carry raw motion."""
    try:
        if ch_stack or not (coupling_model.condition_include_warped_nframe and coupling_model.condition_include_motion):
            return None
        if getattr(coupling_model, "motion_unit_scale", False):
            return None
        H, W = cond.shape[-2:]
        m = cond[3:5]  # (2, H, W): dx, dy in pixels, following the warped frame's 3 channels
        base = make_coords_grid(batch_size=1, h=H, w=W, device=str(cond.device))[0]  # (2, H, W)
        g = base + m
        return (g[0] >= 0) & (g[0] <= W - 1) & (g[1] >= 0) & (g[1] <= H - 1)
    except Exception:
        return None


@torch.no_grad()
def evaluate_sample_based_losses(
    val_dataset: torch.utils.data.Dataset,
    pipe,
    coupling_model: LearntCouplingModel,
    nr_of_image_samples : int,
    nr_of_image_samples_uploaded_to_wandb : int,
    sample_unwarped_noise : bool,
    use_random_noise : bool,
    device : str = "cpu",
    use_quiver_anim_for_2_channel_images : bool = False,
    sample_show_quiver : bool = True,
    sample_show_rgb_flow : bool = True,
    clamp_lower = -6,
    clamp_upper = 6,
    residual_mean: Union[float, List[float]] = 0.0,
    residual_std: Union[float, List[float]] = 1.0,
    fixed_sample_indices: Optional[List[int]] = None,
) -> Dict[str, float]:

    """
        Sample-based validation metrics/visualizations, computed on DDIM samples.
    """

    # Log video sample every N epochs
    return_dict = {}
    _rm = _as_broadcastable(residual_mean, device)
    _rs = _as_broadcastable(residual_std, device)
    rgb_channel_drift_abs = []
    rgb_channel_drift_signed = []
    rgb_channel_drift_interior_abs = []
    rgb_channel_drift_interior_signed = []
    border_fraction = []


    def _denorm_for_viz_if_available(v_chw: torch.Tensor, attrs: Optional[Dict], coupling_norm: Optional[dict] = None) -> torch.Tensor:
        """De-normalize a (C, H, W) frame using per-clip coupling norm then H5 dataset norm."""
        from src.utils.video_normalization import VideoNormalization
        # Per-clip norm (frame0_channel_mean_div_std fitted at training-transform time) applied last → undo first
        if coupling_norm is not None:
            cn = VideoNormalization.from_dict(coupling_norm)
            if cn.mode != "none":
                v_chw = cn.denormalize_chw(v_chw)
        if not attrs:
            return v_chw
        norm = VideoNormalization.from_h5_attrs(attrs)
        if norm.mode == "none":
            return v_chw
        return norm.denormalize_chw(v_chw)

    for i in range(nr_of_image_samples):

        # pick a random sample of the validation dataset
        if fixed_sample_indices is not None and len(fixed_sample_indices) > 0:
            sample_idx = int(fixed_sample_indices[i % len(fixed_sample_indices)])
        else:
            sample_idx = int(torch.randint(0, len(val_dataset), (1,), device=device).item())
        sample = val_dataset[sample_idx]
        val_image = sample["image"].to(device)
        condition = sample["condition"].to(device)
        sample_attrs = sample.get("attrs", None)
        sample_coupling_norm = sample.get("coupling_norm", None)

        if "warped_noise_sample" in sample and not use_random_noise:
            val_sample = sample["warped_noise_sample"].to(device) # shape is (batch, channel, height width), where channel is usually 1
            val_sample = (val_sample - _rm) / _rs
        elif use_random_noise:
            val_sample = torch.randn_like(val_image)
        else:
            raise ValueError("warped_noise_sample not in batch and use_random_noise is False")

        image_frames_warped_scaled = pipe(sample=val_sample, condition=condition)
        image_frames_warped = image_frames_warped_scaled * _rs + _rm

        _ch_stack = getattr(coupling_model, "branch_stacking_mode", "batch") == "channel"

        if sample_unwarped_noise:
            noise = torch.randn_like(val_sample, device=device)
            image_frames_nowarp_scaled = pipe(sample=noise, condition=condition)
            image_frames_nowarp = image_frames_nowarp_scaled * _rs + _rm

        # Convert to numpy and log
        if i < nr_of_image_samples_uploaded_to_wandb:
            B, C, H, W = image_frames_warped.shape
            _base_C = C // 2 if _ch_stack else C
            for j in range(B):
                image_frame_warped = image_frames_warped[j].contiguous() # (C, H, W)
                if C == 3 or (_ch_stack and _base_C == 3):
                    _drift_target = image_frame_warped[:3] if _ch_stack else image_frame_warped
                    _drift_ref = val_image[j].contiguous()[:3] if _ch_stack else val_image[j].contiguous()
                    _diff = (_drift_target - _drift_ref).detach()
                    _drift_vec = _diff.mean(dim=(1, 2)).abs().cpu().numpy()
                    rgb_channel_drift_abs.append(_drift_vec)
                    rgb_channel_drift_signed.append(_diff.mean(dim=(1, 2)).cpu().numpy())
                    # interior-only variant: exclude the out-of-frame warp border strip
                    _vm = _interior_mask_from_condition(condition[j], coupling_model, _ch_stack)
                    if _vm is not None:
                        _w = _vm.to(_diff.dtype)
                        _int = (_diff * _w[None]).sum(dim=(1, 2)) / _w.sum().clamp_min(1.0)
                        rgb_channel_drift_interior_abs.append(_int.abs().cpu().numpy())
                        rgb_channel_drift_interior_signed.append(_int.cpu().numpy())
                        border_fraction.append(float(1.0 - _w.mean().item()))
                val_image_viz = _denorm_for_viz_if_available(val_image[j].contiguous(), sample_attrs, sample_coupling_norm)
                image_frame_warped_viz = _denorm_for_viz_if_available(image_frame_warped, sample_attrs, sample_coupling_norm)
                if sample_unwarped_noise:
                    image_frame_nowarp = image_frames_nowarp[j].contiguous() # (C, H, W)
                    image_frame_nowarp_viz = _denorm_for_viz_if_available(image_frame_nowarp, sample_attrs, sample_coupling_norm)

                if C == 1:
                    return_dict |= single_channel_image_to_wandb(
                        val_image_viz,
                        label=f"original_image_{i}_{j}/{B}",
                        caption=f"Original image sample {i}_{j}/{B}",
                        clamp_lower=clamp_lower,
                        clamp_upper=clamp_upper)

                    return_dict |= single_channel_image_to_wandb(
                        image_frame_warped_viz,
                        label=f"sampled_image_{i}_{j}/{B}",
                        caption=f"Sampled image sample {i}_{j}/{B}",
                        clamp_lower=clamp_lower,
                        clamp_upper=clamp_upper)

                    if sample_unwarped_noise:
                        return_dict |= single_channel_image_to_wandb(
                            image_frame_nowarp_viz,
                            label=f"sampled_image_nowarp_{i}_{j}{B}",
                            caption=f"Sampled image from unwarped noise sample {i}_{j}/{B}",
                            clamp_lower=clamp_lower,
                            clamp_upper=clamp_upper)

                elif C == 2:
                    if use_quiver_anim_for_2_channel_images:
                        raise NotImplementedError("Quiver animation for 2-channel videos not implemented yet")
                    else:
                        return_dict |= complex_image_to_wandb(
                            val_image_viz,
                            label_mag=f"original_image_{i}_{j}_mag",
                            label_angle=f"original_image_{i}_{j}_phase",
                            caption_mag=f"Original image magnitude sample {i}_{j}/{B}",
                            caption_angle=f"Original image phase sample {i}_{j}/{B}",
                            clamp_lower=clamp_lower,
                            clamp_upper=clamp_upper)

                        return_dict |= complex_image_to_wandb(
                            image_frame_warped_viz,
                            label_mag=f"sampled_image_{i}_{j}_mag",
                            label_angle=f"sampled_image_{i}_{j}_phase",
                            caption_mag=f"Sampled image magnitude sample {i}_{j}/{B}",
                            caption_angle=f"Sampled image phase sample {i}_{j}/{B}",
                            clamp_lower=clamp_lower,
                            clamp_upper=clamp_upper)
                elif C == 3:
                    return_dict |= three_channel_image_to_wandb(
                        val_image_viz,
                        label=f"original_image_{i}_{j}",
                        caption=f"Original image sample {i}_{j}/{B}",
                        clamp_lower=clamp_lower,
                        clamp_upper=clamp_upper)
                    return_dict |= three_channel_image_to_wandb(
                        image_frame_warped_viz,
                        label=f"sampled_image_{i}_{j}",
                        caption=f"Sampled image sample {i}_{j}/{B}",
                        clamp_lower=clamp_lower,
                        clamp_upper=clamp_upper)
                elif _ch_stack:
                    # Channel-stacking mode: split the 2*base_C channels back into two branches.
                    _log_fn = (
                        three_channel_image_to_wandb if _base_C == 3
                        else single_channel_image_to_wandb if _base_C == 1
                        else complex_image_to_wandb if _base_C == 2
                        else multi_channel_image_to_wandb
                    )
                    for _br_idx, _br_label in enumerate(("r0", "r1")):
                        _orig_br = val_image_viz[_br_idx * _base_C : (_br_idx + 1) * _base_C]
                        _samp_br = image_frame_warped_viz[_br_idx * _base_C : (_br_idx + 1) * _base_C]
                        if _base_C == 2:
                            return_dict |= _log_fn(
                                _orig_br,
                                label_mag=f"original_image_{i}_{j}_{_br_label}_mag",
                                label_angle=f"original_image_{i}_{j}_{_br_label}_phase",
                                caption_mag=f"Original image ({_br_label}) magnitude {i}_{j}/{B}",
                                caption_angle=f"Original image ({_br_label}) phase {i}_{j}/{B}",
                                clamp_lower=clamp_lower,
                                clamp_upper=clamp_upper)
                            return_dict |= _log_fn(
                                _samp_br,
                                label_mag=f"sampled_image_{i}_{j}_{_br_label}_mag",
                                label_angle=f"sampled_image_{i}_{j}_{_br_label}_phase",
                                caption_mag=f"Sampled image ({_br_label}) magnitude {i}_{j}/{B}",
                                caption_angle=f"Sampled image ({_br_label}) phase {i}_{j}/{B}",
                                clamp_lower=clamp_lower,
                                clamp_upper=clamp_upper)
                        else:
                            return_dict |= _log_fn(
                                _orig_br,
                                label=f"original_image_{i}_{j}_{_br_label}",
                                caption=f"Original image ({_br_label}) sample {i}_{j}/{B}",
                                clamp_lower=clamp_lower,
                                clamp_upper=clamp_upper)
                            return_dict |= _log_fn(
                                _samp_br,
                                label=f"sampled_image_{i}_{j}_{_br_label}",
                                caption=f"Sampled image ({_br_label}) sample {i}_{j}/{B}",
                                clamp_lower=clamp_lower,
                                clamp_upper=clamp_upper)
                else:
                    return_dict |= multi_channel_image_to_wandb(
                        val_image_viz, label=f"original_image_{i}_{j}",
                        caption=f"Original image sample {i}_{j}/{B}", clamp_lower=clamp_lower, clamp_upper=clamp_upper)
                    return_dict |= multi_channel_image_to_wandb(
                        image_frame_warped_viz, label=f"sampled_image_{i}_{j}",
                        caption=f"Sampled image sample {i}_{j}/{B}", clamp_lower=clamp_lower, clamp_upper=clamp_upper)

            # Reconstruct the next frame from the r1 branch (x_{t+1} = r1 + GS(x_t, motion_bwd)).
            # r1 is at batch index 1 for "both", 0 for "r1"; upper channel half in channel-stack mode.
            _has_r1 = coupling_model.training_branches in ("r1", "both")
            _r1_batch_idx = (1 if coupling_model.training_branches == "both" else 0) if not _ch_stack else 0
            _r1_ch_start = _base_C if _ch_stack else 0  # channel offset within a sample
            _r1_ch_end = 2 * _base_C if _ch_stack else _base_C
            _vis_C = _base_C  # channel count for visualization
            if (
                _vis_C == 3
                and _has_r1
                and coupling_model.condition_include_warped_nframe
                and B > _r1_batch_idx
            ):
                # warped frame from the (r1 half of the) condition
                _cond_C = condition.shape[1]
                _cond_base = _cond_C // 2 if _ch_stack else _cond_C
                _cond_r1_start = _cond_base if _ch_stack else 0
                warped_prev = condition[_r1_batch_idx, _cond_r1_start : _cond_r1_start + _vis_C].contiguous()
                if getattr(coupling_model, "prediction_target", "residual") == "full_frame":
                    gt_next = val_image[_r1_batch_idx, _r1_ch_start:_r1_ch_end].contiguous()
                    pred_next = image_frames_warped[_r1_batch_idx, _r1_ch_start:_r1_ch_end].contiguous()
                else:
                    gt_next = (val_image[_r1_batch_idx, _r1_ch_start:_r1_ch_end].contiguous() + warped_prev)
                    pred_next = (image_frames_warped[_r1_batch_idx, _r1_ch_start:_r1_ch_end].contiguous() + warped_prev)
                gt_next_viz = _denorm_for_viz_if_available(gt_next, sample_attrs, sample_coupling_norm)
                pred_next_viz = _denorm_for_viz_if_available(pred_next, sample_attrs, sample_coupling_norm)

                return_dict |= three_channel_image_to_wandb(
                    gt_next_viz,
                    label=f"gt_reconstructed_next_frame_{i}",
                    caption=f"GT reconstructed x_t+1 from residual branch sample {i}",
                    clamp_lower=clamp_lower,
                    clamp_upper=clamp_upper,
                )
                return_dict |= three_channel_image_to_wandb(
                    pred_next_viz,
                    label=f"sampled_reconstructed_next_frame_{i}",
                    caption=f"Sampled reconstructed x_t+1 from residual branch sample {i}",
                    clamp_lower=clamp_lower,
                    clamp_upper=clamp_upper,
                )

    if rgb_channel_drift_abs:
        drift_np = np.stack(rgb_channel_drift_abs, axis=0)  # (N, 3)
        return_dict["sample_color_drift/overall_abs_mean"] = float(np.mean(np.abs(drift_np)))
        sgn = np.stack(rgb_channel_drift_signed, axis=0).mean(axis=0)
        for ci, cn in enumerate(("r", "g", "b")):
            return_dict[f"sample_color_drift/signed_mean_{cn}"] = float(sgn[ci])
    if rgb_channel_drift_interior_abs:
        return_dict["sample_color_drift/interior_abs_mean"] = float(np.mean(np.abs(np.stack(rgb_channel_drift_interior_abs, axis=0))))
        sgn = np.stack(rgb_channel_drift_interior_signed, axis=0).mean(axis=0)
        for ci, cn in enumerate(("r", "g", "b")):
            return_dict[f"sample_color_drift/interior_signed_mean_{cn}"] = float(sgn[ci])
        return_dict["sample_color_drift/border_fraction"] = float(np.mean(border_fraction))

    return return_dict


# ---- shared helpers for PSNR / LPIPS / FID, used by evaluate_full_video_prediction ----

def _psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """GPU-safe PSNR with data_range = gt's peak-to-peak (inputs may be [0,1] RGB or std-normalized MRI)."""
    mse = (pred - gt).square().mean()
    if float(mse.item()) <= 1e-12:
        return float("inf")
    data_range = (gt.max() - gt.min()).clamp_min(1e-8)
    return float((20 * torch.log10(data_range) - 10 * torch.log10(mse)).item())


try:
    import lpips as _lpips_lib
except ImportError:
    _lpips_lib = None

_lpips_fn_cache: Dict[str, object] = {}


def _get_lpips_fn(device: str):
    if _lpips_lib is None:
        return None
    key = str(device)
    if key not in _lpips_fn_cache:
        _lpips_fn_cache[key] = _lpips_lib.LPIPS(net="alex").to(device).eval()
    return _lpips_fn_cache[key]


def _to_metric_display_channels(x: torch.Tensor) -> torch.Tensor:
    """Reduce channels to 1 or 3: magnitude for complex (C=2), channel-mean for C>3, else passthrough."""
    if x.shape[1] == 2:
        return torch.linalg.norm(x, dim=1, keepdim=True)
    if x.shape[1] not in (1, 3):
        return x.mean(dim=1, keepdim=True)
    return x


@torch.no_grad()
def _lpips_distance(pred: torch.Tensor, gt: torch.Tensor, device: str) -> Optional[float]:
    lpips_fn = _get_lpips_fn(device)
    if lpips_fn is None:
        return None
    pred_c = _to_metric_display_channels(pred)
    gt_c = _to_metric_display_channels(gt)
    if pred_c.shape[1] == 1:
        pred_c = pred_c.repeat(1, 3, 1, 1)
        gt_c = gt_c.repeat(1, 3, 1, 1)
    gt_min, gt_max = gt_c.amin(), gt_c.amax()
    data_range = (gt_max - gt_min).clamp_min(1e-8)
    pred_n = (2.0 * (pred_c - gt_min) / data_range - 1.0).clamp(-1.0, 1.0)
    gt_n = 2.0 * (gt_c - gt_min) / data_range - 1.0
    return float(lpips_fn(pred_n, gt_n).mean().item())


_inception_model_cache: Dict[str, torch.nn.Module] = {}


def _get_inception_model(device: str) -> torch.nn.Module:
    key = str(device)
    if key not in _inception_model_cache:
        from torchvision.models import inception_v3, Inception_V3_Weights
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        model.fc = torch.nn.Identity()
        model.eval().to(device)
        _inception_model_cache[key] = model
    return _inception_model_cache[key]


@torch.no_grad()
def _inception_features(frames: List[torch.Tensor], device: str, batch_size: int = 32) -> np.ndarray:
    """frames: list of (C,H,W) tensors at any scale; each is stretched to [0,1] before Inception."""
    model = _get_inception_model(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    feats = []
    for i in range(0, len(frames), batch_size):
        chunk = [
            _to_display_unit_interval(_to_metric_display_channels(f.unsqueeze(0))[0])
            for f in frames[i:i + batch_size]
        ]
        batch = torch.stack(chunk).to(device)
        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)
        batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
        batch = (batch - mean) / std
        feats.append(model(batch).cpu().numpy())
    return np.concatenate(feats, axis=0)


def _frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    from scipy.linalg import sqrtm
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


# sample_image.params keys not accepted by evaluate_sample_based_losses
_PARAMS_NOT_FOR_SAMPLE_LOSSES = {
    "sample_image_times_total",
    "nr_of_full_video_samples",
    "nr_random_range_samples_per_video",
    "random_range_max_k",
    "fid_every_n_sample_evals",
    "fid_nr_samples",
}


def _fid_between(frames_a: List[torch.Tensor], frames_b: List[torch.Tensor], device: str) -> Optional[float]:
    """FID between two frame pools; None for pools too small for a non-degenerate covariance."""
    if len(frames_a) < 8 or len(frames_b) < 8:
        return None
    feats_a = _inception_features(frames_a, device)
    feats_b = _inception_features(frames_b, device)
    mu_a, sigma_a = feats_a.mean(axis=0), np.cov(feats_a, rowvar=False)
    mu_b, sigma_b = feats_b.mean(axis=0), np.cov(feats_b, rowvar=False)
    try:
        return _frechet_distance(mu_a, sigma_a, mu_b, sigma_b)
    except Exception as e:
        print(f"[full_video_eval] FID computation failed, skipping: {e}")
        return None


@torch.no_grad()
def evaluate_full_video_prediction(
    dataset_val_path: str,
    pipe,
    coupling_model: LearntCouplingModel,
    nr_of_full_video_samples: int,
    device: str,
    clamp_lower: float = -6,
    clamp_upper: float = 6,
    residual_mean: Union[float, List[float]] = 0.0,
    residual_std: Union[float, List[float]] = 1.0,
    nr_random_range_samples_per_video: int = 2,
    random_range_max_k: int = 6,
    compute_fid: bool = False,
    fid_nr_samples: int = 64,
) -> Dict[str, float]:
    """
    Evaluate next-frame prediction on full videos from the validation set.

    Modes, reported per active branch (r0/r1):
      - sequential:   autoregressive rollout from the first frame (MSE/PSNR vs. cumulative warp-only).
      - single_step:  GT frame_t -> frame_t+1 (MSE).
      - random_range: GT frame_t -> frame_{t+k} in one shot, random k (LPIPS, optional FID).
    """
    import glob as _glob
    import h5py as _h5py
    _rm = _as_broadcastable(residual_mean, device)
    _rs = _as_broadcastable(residual_std, device)

    h5_files = sorted(_glob.glob(os.path.join(dataset_val_path, "*.h5")))
    if not h5_files:
        return {}

    sampled_files = h5_files[:nr_of_full_video_samples]
    if not sampled_files:
        return {}

    _eval_branches = [b for b in ("r0", "r1") if coupling_model.training_branches in (b, "both")]

    return_dict: Dict = {}
    all_seq_mse = {b: [] for b in _eval_branches}
    all_ss_mse = {b: [] for b in _eval_branches}
    all_seq_psnr = {b: [] for b in _eval_branches}
    all_seq_warp_psnr = {b: [] for b in _eval_branches}
    all_rr_lpips = {b: [] for b in _eval_branches}
    all_rr_warp_lpips = {b: [] for b in _eval_branches}
    all_seq_channel_bias = {b: [] for b in _eval_branches}
    all_ss_channel_bias = {b: [] for b in _eval_branches}
    fid_pred_frames = {b: [] for b in _eval_branches}
    fid_warp_frames = {b: [] for b in _eval_branches}
    fid_real_frames = {b: [] for b in _eval_branches}

    # on FID epochs, spread fid_nr_samples random-range draws across videos*branches
    _rr_draws_per_video_branch = (
        max(1, fid_nr_samples // max(1, len(sampled_files) * max(1, len(_eval_branches))))
        if compute_fid else nr_random_range_samples_per_video
    )

    for file_idx, fpath in enumerate(sampled_files):
        with _h5py.File(fpath, 'r') as f:
            all_keys = list(f.keys())
            # Locate video and motion keys
            video_key = "video"
            # Prefer exact keys from preprocessing; fall back to strict suffix matching.
            motion_fwd_key = (
                "warp_parameters_motion_param"
                if "warp_parameters_motion_param" in all_keys
                else next((k for k in all_keys if k.endswith("motion_param") and "bwd" not in k), None)
            )
            motion_bwd_key = (
                "warp_parameters_motion_bwd_param"
                if "warp_parameters_motion_bwd_param" in all_keys
                else next((k for k in all_keys if k.endswith("motion_bwd_param")), None)
            )

            if video_key not in f or motion_fwd_key is None or motion_bwd_key is None:
                print(f"[full_video_eval] Skipping {fpath}: missing video or motion keys (found: {all_keys})")
                continue

            video      = torch.from_numpy(f[video_key][:]).float().to(device)       # (T, C, H, W)
            # Stored as (1, 2, H, W, T-1)  – last dim is time step
            motion_raw     = torch.from_numpy(f[motion_fwd_key][:]).float().to(device)  # (1,2,H,W,T-1)
            motion_bwd_raw = torch.from_numpy(f[motion_bwd_key][:]).float().to(device)  # (1,2,H,W,T-1)

        T, C, H, W = video.shape
        if T < 2:
            continue

        motion = h5_motion_to_pairwise_flow(motion_raw, T, motion_fwd_key)
        motion_bwd = h5_motion_to_pairwise_flow(motion_bwd_raw, T, motion_bwd_key)

        # Pre-compute a single-frame coordinate grid
        base_grid_1 = make_coords_grid(batch_size=1, h=H, w=W, device=device).movedim(1, -1)  # (1,H,W,2)

        for _eval_branch in _eval_branches:
            _eval_gen = torch.Generator(device=device)
            _eval_gen.manual_seed(20260328 + file_idx + (0 if _eval_branch == "r1" else 500))

            seq_mse_per_frame: list = []
            ss_mse_per_frame:  list = []
            seq_warp_mse_per_frame: list = []
            ss_warp_mse_per_frame: list = []
            cumulative_warp_mse_per_frame: list = []
            seq_residual_mag_per_frame: list = []
            ss_residual_mag_per_frame: list = []
            flow_anchor_mismatch_per_frame: list = []
            seq_channel_bias_per_frame: list = []
            ss_channel_bias_per_frame: list = []
            seq_psnr_per_frame: list = []
            seq_warp_psnr_per_frame: list = []

            # r1: sequential goes forward (start at frame 0, predict 1, 2, …)
            # r0: sequential goes backward (start at frame T-1, predict T-2, T-3, …)
            if _eval_branch == "r1":
                frame_seq = video[0:1].clone()
                frame_cumulative_warp = video[0:1].clone()
                time_steps = list(range(T - 1))
            else:
                frame_seq = video[T-1:T].clone()
                frame_cumulative_warp = video[T-1:T].clone()
                time_steps = list(reversed(range(T - 1)))

            for t in time_steps:
                if _eval_branch == "r1":
                    gt_next  = video[t + 1: t + 2]
                    gt_curr  = video[t:     t + 1]
                    _step_tag = t + 1
                else:
                    gt_next  = video[t:     t + 1]       # the "next" to predict is actually earlier
                    gt_curr  = video[t + 1: t + 2]       # anchor is the later frame
                    _step_tag = t

                # --- cumulative-flow baseline from fixed initial GT anchor ---
                # r1: x_0 --bwd[0]--> x_1 --bwd[1]--> ...
                # r0: x_{T-1} --fwd[T-2]--> x_{T-2} --fwd[T-3]--> ...
                if _eval_branch == "r1":
                    m_step = motion_bwd[t:t+1]
                    m_step_opposite = motion[t:t+1]
                else:
                    m_step = motion[t:t+1]
                    m_step_opposite = motion_bwd[t:t+1]
                frame_cumulative_warp = coupling_model.warp_frame_with_motion(
                    frame_anchor=frame_cumulative_warp,
                    motion_step=m_step,
                    base_grid=base_grid_1,
                )
                cumulative_warp_mse = torch.nn.functional.mse_loss(frame_cumulative_warp, gt_next).item()
                cumulative_warp_mse_per_frame.append(cumulative_warp_mse)
                seq_warp_psnr_per_frame.append(_psnr(frame_cumulative_warp, gt_next))

                # --- sequential prediction ---
                # same noise for sequential and single-step so differences come from conditioning
                shared_noise = torch.randn((1, C, H, W), device=device, generator=_eval_gen)
                frame_seq_next, warped_seq = coupling_model.predict_adjacent_frame(
                    frame_anchor=frame_seq,
                    motion_step=m_step,
                    pipe=pipe,
                    residual_mean=_rm,
                    residual_std=_rs,
                    noise=shared_noise,
                    base_grid=base_grid_1,
                    motion_step_opposite=m_step_opposite,
                )
                seq_mse = torch.nn.functional.mse_loss(frame_seq_next, gt_next).item()
                seq_warp_mse = torch.nn.functional.mse_loss(warped_seq, gt_next).item()
                seq_residual_mag = (frame_seq_next - warped_seq).abs().mean().item()
                seq_mse_per_frame.append(seq_mse)
                seq_warp_mse_per_frame.append(seq_warp_mse)
                seq_residual_mag_per_frame.append(seq_residual_mag)
                seq_psnr_per_frame.append(_psnr(frame_seq_next, gt_next))
                seq_channel_bias = (frame_seq_next - gt_next).mean(dim=(0, 2, 3)).detach().cpu().numpy()
                seq_channel_bias_per_frame.append(seq_channel_bias)
                frame_seq = frame_seq_next

                # --- single-step (GT-conditioned) prediction ---
                frame_ss_next, warped_ss = coupling_model.predict_adjacent_frame(
                    frame_anchor=gt_curr,
                    motion_step=m_step,
                    pipe=pipe,
                    residual_mean=_rm,
                    residual_std=_rs,
                    noise=shared_noise,
                    base_grid=base_grid_1,
                    motion_step_opposite=m_step_opposite,
                )
                ss_mse = torch.nn.functional.mse_loss(frame_ss_next, gt_next).item()
                ss_warp_mse = torch.nn.functional.mse_loss(warped_ss, gt_next).item()
                ss_residual_mag = (frame_ss_next - warped_ss).abs().mean().item()
                flow_anchor_mismatch = torch.nn.functional.mse_loss(warped_seq, warped_ss).item()
                ss_mse_per_frame.append(ss_mse)
                ss_warp_mse_per_frame.append(ss_warp_mse)
                ss_residual_mag_per_frame.append(ss_residual_mag)
                flow_anchor_mismatch_per_frame.append(flow_anchor_mismatch)
                ss_channel_bias = (frame_ss_next - gt_next).mean(dim=(0, 2, 3)).detach().cpu().numpy()
                ss_channel_bias_per_frame.append(ss_channel_bias)

                # WandB image logging for first video, first 4 transitions
                if file_idx == 0 and t < 8:
                    tag = f"full_video/{_eval_branch}/{file_idx}"
                    if C == 1:
                        _img = single_channel_image_to_wandb
                        return_dict |= _img(gt_next[0],           f"{tag}/t{_step_tag}_gt",          f"GT frame {_step_tag}",                clamp_lower, clamp_upper)
                        return_dict |= _img(frame_seq_next[0],    f"{tag}/t{_step_tag}_sequential",  f"Sequential pred frame {_step_tag}",   clamp_lower, clamp_upper)
                        return_dict |= _img(frame_ss_next[0],     f"{tag}/t{_step_tag}_single_step", f"Single-step pred frame {_step_tag}",  clamp_lower, clamp_upper)
                        return_dict |= _img(frame_cumulative_warp[0], f"{tag}/t{_step_tag}_cumflow_from_init", f"Cumulative-flow-from-init frame {_step_tag}", clamp_lower, clamp_upper)
                    elif C == 2:
                        return_dict |= complex_image_to_wandb(gt_next[0],        f"{tag}/t{_step_tag}_gt_mag",          f"GT mag {_step_tag}",  f"{tag}/t{_step_tag}_gt_phase",          f"GT phase {_step_tag}",  clamp_lower, clamp_upper)
                        return_dict |= complex_image_to_wandb(frame_seq_next[0], f"{tag}/t{_step_tag}_sequential_mag",  f"Seq mag {_step_tag}", f"{tag}/t{_step_tag}_sequential_phase",  f"Seq ph {_step_tag}",    clamp_lower, clamp_upper)
                        return_dict |= complex_image_to_wandb(frame_ss_next[0],  f"{tag}/t{_step_tag}_single_step_mag", f"SS mag {_step_tag}",  f"{tag}/t{_step_tag}_single_step_phase", f"SS ph {_step_tag}",     clamp_lower, clamp_upper)
                        return_dict |= complex_image_to_wandb(frame_cumulative_warp[0], f"{tag}/t{_step_tag}_cumflow_from_init_mag", f"CumFlow mag {_step_tag}", f"{tag}/t{_step_tag}_cumflow_from_init_phase", f"CumFlow phase {_step_tag}", clamp_lower, clamp_upper)
                    elif C == 3:
                        _img = three_channel_image_to_wandb
                        return_dict |= _img(gt_next[0],           f"{tag}/t{_step_tag}_gt",          f"GT frame {_step_tag}",                clamp_lower, clamp_upper)
                        return_dict |= _img(frame_seq_next[0],    f"{tag}/t{_step_tag}_sequential",  f"Sequential pred frame {_step_tag}",   clamp_lower, clamp_upper)
                        return_dict |= _img(frame_ss_next[0],     f"{tag}/t{_step_tag}_single_step", f"Single-step pred frame {_step_tag}",  clamp_lower, clamp_upper)
                        return_dict |= _img(frame_cumulative_warp[0], f"{tag}/t{_step_tag}_cumflow_from_init", f"Cumulative-flow-from-init frame {_step_tag}", clamp_lower, clamp_upper)

            all_seq_mse[_eval_branch].extend(seq_mse_per_frame)
            all_ss_mse[_eval_branch].extend(ss_mse_per_frame)
            all_seq_psnr[_eval_branch].extend(seq_psnr_per_frame)
            all_seq_warp_psnr[_eval_branch].extend(seq_warp_psnr_per_frame)

            if seq_channel_bias_per_frame:
                seq_channel_bias_np = np.stack(seq_channel_bias_per_frame, axis=0)  # (T-1, C)
                ss_channel_bias_np = np.stack(ss_channel_bias_per_frame, axis=0)    # (T-1, C)
                all_seq_channel_bias[_eval_branch].append(seq_channel_bias_np)
                all_ss_channel_bias[_eval_branch].append(ss_channel_bias_np)

            vid_tag = f"video_{file_idx}"
            return_dict[f"full_video_mse/{_eval_branch}/{vid_tag}/sequential_mean"]  = float(np.mean(seq_mse_per_frame))
            return_dict[f"full_video_mse/{_eval_branch}/{vid_tag}/sequential_max"]   = float(np.max(seq_mse_per_frame))
            return_dict[f"full_video_mse/{_eval_branch}/{vid_tag}/single_step_mean"] = float(np.mean(ss_mse_per_frame))
            return_dict[f"full_video_mse/{_eval_branch}/{vid_tag}/single_step_max"]  = float(np.max(ss_mse_per_frame))
            return_dict[f"full_video_warp_only_mse/{_eval_branch}/{vid_tag}/sequential_mean"] = float(np.mean(seq_warp_mse_per_frame))
            return_dict[f"full_video_warp_only_mse/{_eval_branch}/{vid_tag}/single_step_mean"] = float(np.mean(ss_warp_mse_per_frame))
            return_dict[f"full_video_cumulative_warp_mse/{_eval_branch}/{vid_tag}/mean"] = float(np.mean(cumulative_warp_mse_per_frame))
            return_dict[f"full_video_psnr/{_eval_branch}/{vid_tag}/sequential_mean"] = float(np.mean(seq_psnr_per_frame))
            return_dict[f"full_video_psnr_warp_only/{_eval_branch}/{vid_tag}/sequential_mean"] = float(np.mean(seq_warp_psnr_per_frame))
            return_dict[f"full_video_residual_mag/{_eval_branch}/{vid_tag}/sequential_mean"] = float(np.mean(seq_residual_mag_per_frame))
            return_dict[f"full_video_residual_mag/{_eval_branch}/{vid_tag}/single_step_mean"] = float(np.mean(ss_residual_mag_per_frame))
            return_dict[f"full_video_flow_anchor_mismatch/{_eval_branch}/{vid_tag}/mean"] = float(np.mean(flow_anchor_mismatch_per_frame))
            # Per-frame MSE/PSNR arrays logged as summary scalars (frame index encoded in key)
            for t, (smse, ssmse) in enumerate(zip(seq_mse_per_frame, ss_mse_per_frame)):
                return_dict[f"full_video_mse/{_eval_branch}/{vid_tag}/frame_{t:03d}/sequential"]  = float(smse)
                return_dict[f"full_video_mse/{_eval_branch}/{vid_tag}/frame_{t:03d}/single_step"] = float(ssmse)
            for t, (swmse, sswmse) in enumerate(zip(seq_warp_mse_per_frame, ss_warp_mse_per_frame)):
                return_dict[f"full_video_warp_only_mse/{_eval_branch}/{vid_tag}/frame_{t:03d}/sequential"] = float(swmse)
                return_dict[f"full_video_warp_only_mse/{_eval_branch}/{vid_tag}/frame_{t:03d}/single_step"] = float(sswmse)
            for t, cmse in enumerate(cumulative_warp_mse_per_frame):
                return_dict[f"full_video_cumulative_warp_mse/{_eval_branch}/{vid_tag}/frame_{t:03d}"] = float(cmse)
            for t, (psnr_v, wpsnr_v) in enumerate(zip(seq_psnr_per_frame, seq_warp_psnr_per_frame)):
                return_dict[f"full_video_psnr/{_eval_branch}/{vid_tag}/frame_{t:03d}/sequential"] = float(psnr_v)
                return_dict[f"full_video_psnr_warp_only/{_eval_branch}/{vid_tag}/frame_{t:03d}/sequential"] = float(wpsnr_v)
            for t, (srmag, ssrmag) in enumerate(zip(seq_residual_mag_per_frame, ss_residual_mag_per_frame)):
                return_dict[f"full_video_residual_mag/{_eval_branch}/{vid_tag}/frame_{t:03d}/sequential"] = float(srmag)
                return_dict[f"full_video_residual_mag/{_eval_branch}/{vid_tag}/frame_{t:03d}/single_step"] = float(ssrmag)
            for t, fm in enumerate(flow_anchor_mismatch_per_frame):
                return_dict[f"full_video_flow_anchor_mismatch/{_eval_branch}/{vid_tag}/frame_{t:03d}"] = float(fm)

            if seq_channel_bias_per_frame:
                return_dict[f"full_video_color_bias/{_eval_branch}/{vid_tag}/sequential_abs_mean"] = float(np.mean(np.abs(seq_channel_bias_np)))
                return_dict[f"full_video_color_bias/{_eval_branch}/{vid_tag}/single_step_abs_mean"] = float(np.mean(np.abs(ss_channel_bias_np)))

            # --- random-range one-shot prediction from GT frame t0 to t0+/-k (LPIPS + optional FID pool) ---
            k_max = min(random_range_max_k, T - 1)
            if k_max >= 1:
                _rr_rng = np.random.default_rng(20260328 + file_idx * 1000 + (0 if _eval_branch == "r1" else 500))
                rr_lpips_per_sample: list = []
                rr_warp_lpips_per_sample: list = []
                for _ in range(_rr_draws_per_video_branch):
                    k = int(_rr_rng.integers(1, k_max + 1))
                    if _eval_branch == "r1":
                        t0 = int(_rr_rng.integers(0, T - k))
                        anchor = video[t0:t0 + 1]
                        target = video[t0 + k:t0 + k + 1]
                        # additive flow composition over the k-window
                        m_step_rr = motion_bwd[t0:t0 + k].sum(dim=0, keepdim=True)
                        m_step_opp_rr = motion[t0:t0 + k].sum(dim=0, keepdim=True)
                    else:
                        t0 = int(_rr_rng.integers(k, T))
                        anchor = video[t0:t0 + 1]
                        target = video[t0 - k:t0 - k + 1]
                        m_step_rr = motion[t0 - k:t0].sum(dim=0, keepdim=True)
                        m_step_opp_rr = motion_bwd[t0 - k:t0].sum(dim=0, keepdim=True)

                    rr_noise = torch.randn((1, C, H, W), device=device, generator=_eval_gen)
                    pred_rr, warped_rr = coupling_model.predict_adjacent_frame(
                        frame_anchor=anchor,
                        motion_step=m_step_rr,
                        pipe=pipe,
                        residual_mean=_rm,
                        residual_std=_rs,
                        noise=rr_noise,
                        base_grid=base_grid_1,
                        motion_step_opposite=m_step_opp_rr,
                    )
                    rr_lpips = _lpips_distance(pred_rr, target, device)
                    rr_warp_lpips = _lpips_distance(warped_rr, target, device)
                    if rr_lpips is not None:
                        rr_lpips_per_sample.append(rr_lpips)
                        rr_warp_lpips_per_sample.append(rr_warp_lpips)
                    if compute_fid:
                        fid_pred_frames[_eval_branch].append(pred_rr[0].detach().cpu())
                        fid_warp_frames[_eval_branch].append(warped_rr[0].detach().cpu())
                        fid_real_frames[_eval_branch].append(target[0].detach().cpu())

                if rr_lpips_per_sample:
                    all_rr_lpips[_eval_branch].extend(rr_lpips_per_sample)
                    all_rr_warp_lpips[_eval_branch].extend(rr_warp_lpips_per_sample)
                    return_dict[f"full_video_lpips/{_eval_branch}/{vid_tag}/random_range_mean"] = float(np.mean(rr_lpips_per_sample))
                    return_dict[f"full_video_lpips_warp_only/{_eval_branch}/{vid_tag}/random_range_mean"] = float(np.mean(rr_warp_lpips_per_sample))

    for _eval_branch in _eval_branches:
        if all_seq_mse[_eval_branch]:
            return_dict[f"full_video_mse/{_eval_branch}/all/sequential_mean"]  = float(np.mean(all_seq_mse[_eval_branch]))
            return_dict[f"full_video_mse/{_eval_branch}/all/single_step_mean"] = float(np.mean(all_ss_mse[_eval_branch]))
        if all_seq_psnr[_eval_branch]:
            return_dict[f"full_video_psnr/{_eval_branch}/all/sequential_mean"] = float(np.mean(all_seq_psnr[_eval_branch]))
            return_dict[f"full_video_psnr_warp_only/{_eval_branch}/all/sequential_mean"] = float(np.mean(all_seq_warp_psnr[_eval_branch]))
        if all_rr_lpips[_eval_branch]:
            return_dict[f"full_video_lpips/{_eval_branch}/all/random_range_mean"] = float(np.mean(all_rr_lpips[_eval_branch]))
            return_dict[f"full_video_lpips_warp_only/{_eval_branch}/all/random_range_mean"] = float(np.mean(all_rr_warp_lpips[_eval_branch]))

        if all_seq_channel_bias[_eval_branch]:
            seq_concat = np.concatenate(all_seq_channel_bias[_eval_branch], axis=0)  # (N, C)
            ss_concat = np.concatenate(all_ss_channel_bias[_eval_branch], axis=0)    # (N, C)
            return_dict[f"full_video_color_bias/{_eval_branch}/all/sequential_abs_mean"] = float(np.mean(np.abs(seq_concat)))
            return_dict[f"full_video_color_bias/{_eval_branch}/all/single_step_abs_mean"] = float(np.mean(np.abs(ss_concat)))

        if compute_fid:
            fid_recon = _fid_between(fid_pred_frames[_eval_branch], fid_real_frames[_eval_branch], device)
            fid_warp = _fid_between(fid_warp_frames[_eval_branch], fid_real_frames[_eval_branch], device)
            if fid_recon is not None:
                return_dict[f"full_video_fid/{_eval_branch}/random_range"] = fid_recon
            if fid_warp is not None:
                return_dict[f"full_video_fid_warp_only/{_eval_branch}/random_range"] = fid_warp

    return return_dict


def frame_index_transform(frame_index):
    frame_index = sorted(frame_index, key=lambda x: (x[0], x[1]))
    new_frame_index = []
    for (fp1, idx1), (fp2, idx2) in zip(frame_index, frame_index[1:]):
        if fp1 == fp2:
            new_frame_index.append((fp1, (idx1, idx2)))
    return new_frame_index


def frame_index_transform_random_pairs(frame_index, max_frame_distance: int = 5):
    """All pairs (t, t+d) per video with 1 <= d <= max_frame_distance.

    Returns a list of ``(file_path, (t_i, t_j))`` tuples, as :func:`frame_index_transform`.
    """
    from collections import defaultdict

    frame_index = sorted(frame_index, key=lambda x: (x[0], x[1]))
    file_frames: dict[str, list[int]] = defaultdict(list)
    for fp, idx in frame_index:
        file_frames[fp].append(idx)

    new_frame_index = []
    for fp, indices in file_frames.items():
        indices = sorted(set(indices))
        idx_set = set(indices)
        for t_i in indices:
            for d in range(1, max_frame_distance + 1):
                t_j = t_i + d
                if t_j in idx_set:
                    new_frame_index.append((fp, (t_i, t_j)))
    return new_frame_index

@task(cache_policy=INPUTS - "wandb_params_task" - "use_multi_gpu_ddp" - "data_roots", name="Train CPL Task", tags=["train"], version="1.0", retries=0)
def train_probabilistic_coupling_model(
    dataset_train_path : str,
    dataset_val_path : str,
    seed : int,
    coupling_transform_cfg : CacheableDictConfig,
    accelerator_cfg: CacheableDictConfig,
    cache_refresh: bool,
    cache_cfg : CacheableDictConfig,
    dataloader_train_cfg : CacheableDictConfig,
    dataloader_val_cfg : CacheableDictConfig,
    model_skip_mid_block : bool,
    model_cfg : CacheableDictConfig,
    noise_scheduler_cfg: Optional[CacheableDictConfig] = None,
    ema_cfg : Optional[CacheableDictConfig] = None,
    use_random_noise : bool = True,
    optimizer_cfg: Optional[CacheableDictConfig] = None,
    lr_scheduler_cfg: Optional[CacheableDictConfig] = None,
    sample_image_cfg : Optional[CacheableDictConfig] = None,
    store_model_cfg : Optional[CacheableDictConfig] = None,
    storage_settings_cfg : Optional[CacheableDictConfig] = None,
    training_steps : int = 0,
    wandb_params_task: Optional[WandbParamsTask] = None,
    coupling_model_cfg : Optional[CacheableDictConfig] = None,
    data_roots: Optional[DataRoots] = None,
    debug_print_every_steps: int = 100,
    debug_print_max_steps: int = 1000,
    per_channel_residual_scaling: bool = False,
    residual_centering: bool = True,
    residual_scaling: bool = True,
    residual_prior_type: str = "none",
    residual_prior_weight: float = 0.0,
    residual_prior_delta: float = 0.01,
    residual_prior_blur_sigma: float = 0.0,
    residual_prior_blur_kernel_size: int = 0,
    diffusion_process: str = "ddpm",
    bridge_scheduler_cfg: Optional[CacheableDictConfig] = None,
    # Data-parallel training across all GPUs of one node. Excluded from the cache key, since it
    # changes only how training is executed, not what is trained.
    use_multi_gpu_ddp: bool = False,
):
    os.environ["ACCELERATE_DISTRIBUTED_TYPE"] = "NO"
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(get_free_port())
    """
    Trains a UNet2DConditionModel on video frames data  Hydra config.
    """
    # diffusion_process / bridge_scheduler_cfg are part of the cache key; only "ddpm" is implemented.
    if diffusion_process != "ddpm":
        raise ValueError(f"diffusion_process={diffusion_process!r} is not supported (only 'ddpm').")

    # The training itself is in train_probabilistic_coupling_model_inner, which runs either in this
    # process or in one spawned process per GPU (DDP). The wandb kwargs need the Prefect run context,
    # which only exists in this process, so they are computed here and passed down.
    wandb_kwargs = wandb_kwargs_for_prefect_task(wandb_params_task)
    common_train_kwargs = dict(
        dataset_train_path=dataset_train_path,
        dataset_val_path=dataset_val_path,
        seed=seed,
        coupling_transform_cfg=coupling_transform_cfg,
        accelerator_cfg=accelerator_cfg,
        cache_refresh=cache_refresh,
        cache_cfg=cache_cfg,
        dataloader_train_cfg=dataloader_train_cfg,
        dataloader_val_cfg=dataloader_val_cfg,
        model_skip_mid_block=model_skip_mid_block,
        model_cfg=model_cfg,
        noise_scheduler_cfg=noise_scheduler_cfg,
        ema_cfg=ema_cfg,
        use_random_noise=use_random_noise,
        optimizer_cfg=optimizer_cfg,
        lr_scheduler_cfg=lr_scheduler_cfg,
        sample_image_cfg=sample_image_cfg,
        store_model_cfg=store_model_cfg,
        storage_settings_cfg=storage_settings_cfg,
        training_steps=training_steps,
        wandb_params_task=wandb_params_task,
        wandb_kwargs=wandb_kwargs,
        coupling_model_cfg=coupling_model_cfg,
        data_roots=data_roots,
        debug_print_every_steps=debug_print_every_steps,
        debug_print_max_steps=debug_print_max_steps,
        per_channel_residual_scaling=per_channel_residual_scaling,
        residual_centering=residual_centering,
        residual_scaling=residual_scaling,
        residual_prior_type=residual_prior_type,
        residual_prior_weight=residual_prior_weight,
        residual_prior_delta=residual_prior_delta,
        residual_prior_blur_sigma=residual_prior_blur_sigma,
        residual_prior_blur_kernel_size=residual_prior_blur_kernel_size,
    )

    if not use_multi_gpu_ddp:
        return train_probabilistic_coupling_model_inner(**common_train_kwargs, result_holder=None)

    # Multi-GPU path. torch.multiprocessing.spawn instead of accelerate.notebook_launcher, which forks
    # and fails with "Cannot re-initialize CUDA in forked subprocess". Rank 0 returns its StoragePath
    # through a Manager dict (see the end of train_probabilistic_coupling_model_inner).
    from torch.multiprocessing import Manager, spawn as mp_spawn
    manager = Manager()
    result_holder = manager.dict()
    num_processes = torch.cuda.device_count()
    master_port = get_free_port()
    mp_spawn(
        _train_probabilistic_coupling_model_ddp_worker,
        args=(num_processes, master_port, common_train_kwargs, result_holder),
        nprocs=num_processes,
        join=True,
    )
    if "output_dir_in_fs" not in result_holder:
        raise RuntimeError(
            "train_probabilistic_coupling_model_inner did not report an output path via "
            "result_holder - check subprocess logs for a crash on rank 0."
        )
    return StoragePath(
        storage_path=result_holder["output_dir_in_fs"],
        storage_settings=StorageSettings(**storage_settings_cfg.cfg),
    )


def _train_probabilistic_coupling_model_ddp_worker(local_rank, world_size, master_port, common_train_kwargs, result_holder):
    """Per-process entry point for torch.multiprocessing.spawn (must stay top-level to be picklable)."""
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    # NCCL P2P/InfiniBand off by default (collectives can hang otherwise); set them to 0 to re-enable
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    torch.cuda.set_device(local_rank)
    train_probabilistic_coupling_model_inner(**common_train_kwargs, result_holder=result_holder)


def train_probabilistic_coupling_model_inner(
    dataset_train_path : str,
    dataset_val_path : str,
    seed : int,
    coupling_transform_cfg : CacheableDictConfig,
    accelerator_cfg: CacheableDictConfig,
    cache_refresh: bool,
    cache_cfg : CacheableDictConfig,
    dataloader_train_cfg : CacheableDictConfig,
    dataloader_val_cfg : CacheableDictConfig,
    model_skip_mid_block : bool,
    model_cfg : CacheableDictConfig,
    noise_scheduler_cfg: Optional[CacheableDictConfig] = None,
    ema_cfg : Optional[CacheableDictConfig] = None,
    use_random_noise : bool = True,
    optimizer_cfg: Optional[CacheableDictConfig] = None,
    lr_scheduler_cfg: Optional[CacheableDictConfig] = None,
    sample_image_cfg : Optional[CacheableDictConfig] = None,
    store_model_cfg : Optional[CacheableDictConfig] = None,
    storage_settings_cfg : Optional[CacheableDictConfig] = None,
    training_steps : int = 0,
    wandb_params_task: Optional[WandbParamsTask] = None,
    # computed by the caller, which has the Prefect run context (not available in spawned DDP children)
    wandb_kwargs: Optional[dict] = None,
    coupling_model_cfg : Optional[CacheableDictConfig] = None,
    data_roots: Optional[DataRoots] = None,
    debug_print_every_steps: int = 100,
    debug_print_max_steps: int = 1000,
    per_channel_residual_scaling: bool = False,
    residual_centering: bool = True,
    residual_scaling: bool = True,
    residual_prior_type: str = "none",
    residual_prior_weight: float = 0.0,
    residual_prior_delta: float = 0.01,
    residual_prior_blur_sigma: float = 0.0,
    residual_prior_blur_kernel_size: int = 0,
    # multiprocessing.Manager dict, only set on the multi-GPU spawn path
    result_holder=None,
):
    accelerator = Accelerator(**accelerator_cfg.cfg)
    print(f"Distributed type: {accelerator.distributed_type}")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    coupling_model = LearntCouplingModel(**coupling_model_cfg.cfg)

    from omegaconf import OmegaConf
    model, forward_fn = get_diffuser_coupling_model_and_forward_fun(**OmegaConf.to_container(model_cfg.cfg, resolve=True), device=accelerator.device, coupling_model=coupling_model)
    model = model.to(accelerator.device)

    ema = None # will be initialized later

    noise_scheduler = DDPMScheduler(**OmegaConf.to_container(noise_scheduler_cfg.cfg, resolve=True))
    print(f"[CPL Train] Training with prediction_type='{get_prediction_type(noise_scheduler)}'")
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_cfg.cfg)

    if accelerator.is_main_process:
        sample_scheduler = DDIMScheduler.from_config(noise_scheduler.config, **sample_image_cfg.cfg["scheduler"])
        # must use register_to_config, plain assignment has no effect
        sample_scheduler.register_to_config(clip_sample=False)
        sample_scheduler.set_timesteps(sample_image_cfg.cfg["sample_steps"], device=accelerator.device)

        pipe = partial(pipe_image_sampler_coupling, 
            forward_fn=forward_fn,
            sample_scheduler=sample_scheduler
        )

    path_resolver = data_roots.resolver()

    if coupling_transform_cfg.cfg["enabled"]:
        transform_train = CouplingModelPreprocessingTransform(**coupling_transform_cfg.cfg, append_all=False, coupling_model=coupling_model, path_resolver=path_resolver)
        transform_val = CouplingModelPreprocessingTransform(**coupling_transform_cfg.cfg, append_all=False, coupling_model=coupling_model, path_resolver=path_resolver)
        transform_val_with_motion = CouplingModelPreprocessingTransform(**coupling_transform_cfg.cfg, append_all=True, coupling_model=coupling_model, path_resolver=path_resolver)
    else:
        transform_train = None
        transform_val = None
        transform_val_with_motion = None

    dataset_train_path = path_resolver(dataset_train_path)  # upstream tasks return cache-root-relative paths
    dataset_val_path = path_resolver(dataset_val_path)

    if accelerator.is_main_process:
        wandb_run = wandb.init(**wandb_kwargs)

    filter_train = "^(video)|(warp_parameters_motion.*)$"
    train_dataset_cache_cfg = cache_cfg.cfg["train_dataset"]
    cache_ser_suffix = CacheableDict(  # join dicts before hashing to keep filename short
        cfg={
            "cache_cfg": OmegaConf.to_container(cache_cfg.cfg, resolve=True),
            "coupling_transform_cfg": OmegaConf.to_container(coupling_transform_cfg.cfg, resolve=True),
            "coupling_model_cfg": OmegaConf.to_container(coupling_model_cfg.cfg, resolve=True),
        }
    ).ser()
    print(f"[Train CPL][CACHE_KEY] {cache_ser_suffix}")
    if accelerator.is_main_process:
        wandb.log({"train_cpl_cache_key": cache_ser_suffix})

    # max_frame_distance=1 -> consecutive pairs
    _mfd = coupling_model.max_frame_distance
    _frame_index_transform = (
        frame_index_transform
        if _mfd == 1
        else partial(frame_index_transform_random_pairs, max_frame_distance=_mfd)
    )
    # d>1 needs the flow-composing selector
    _time_slice_selector = (
        time_slice_selector
        if _mfd == 1
        else time_slice_selector_with_distance
    )
    print(f"[Train CPL] max_frame_distance={_mfd}, "
          f"frame_index_transform={'consecutive' if _mfd == 1 else 'random_pairs'}, "
          f"time_slice_selector={'standard' if _mfd == 1 else 'with_distance'}")

    if train_dataset_cache_cfg["enabled"]:
        print(f"Caching train on device: {accelerator.device}")
        train_cache_path_base_dir = path_resolver(train_dataset_cache_cfg["cache_path_base_dir"])
        train_cache_path = os.path.join(train_cache_path_base_dir, f"{os.path.basename(dataset_train_path)}_{cache_ser_suffix}_cache.pt")
        os.makedirs(os.path.dirname(train_cache_path), exist_ok=True)
        print(f"[Train CPL][TRAIN_CACHE_PATH] {train_cache_path}")

        # Delete cache file if cache_refresh is True
        if cache_refresh and os.path.exists(train_cache_path):
            print(f"[Train CPL][CACHE_REFRESH][TRAIN] key={cache_ser_suffix} path={train_cache_path}")
            print(f"Cache refresh enabled: removing existing cache file {train_cache_path}")
            os.remove(train_cache_path)

        if (not cache_refresh) and os.path.exists(train_cache_path):
            print(f"[Train CPL][CACHE_HIT][TRAIN] key={cache_ser_suffix} path={train_cache_path}")
            print(f"Loading train dataset directly from cache file (skip H5 indexing): {train_cache_path}")
            train_dataset = CachedH5FileDatasetNew(
                h5_dataset=None,
                cache_path=train_cache_path,
                limit_cache_size=train_dataset_cache_cfg["limit_cache_size"],
                device=train_dataset_cache_cfg["cache_device"] if "cache_device" in train_dataset_cache_cfg else accelerator.device,
                cache_key=cache_ser_suffix,
            )
        else:
            print(f"[Train CPL][CACHE_MISS][TRAIN] key={cache_ser_suffix} path={train_cache_path}")
            train_index_cache_enabled = bool(train_dataset_cache_cfg.get("index_cache_enabled", False))
            train_index_cache_path = None
            if train_index_cache_enabled:
                train_index_cache_base_dir = path_resolver(
                    train_dataset_cache_cfg.get("index_cache_path_base_dir", train_dataset_cache_cfg["cache_path_base_dir"])
                )
                train_index_cache_path = os.path.join(
                    train_index_cache_base_dir,
                    f"{os.path.basename(dataset_train_path)}_{cache_ser_suffix}_frame_index.pt",
                )
                print(f"[Train CPL][TRAIN_INDEX_CACHE_PATH] {train_index_cache_path}")
            train_dataset_raw = ImagesFromH5VideoFileDataset(
                dataset_train_path,
                key_filter=filter_train,
                transform=transform_train,
                device="cpu",
                frame_index_transform=_frame_index_transform,
                time_slice_selector=_time_slice_selector,
                index_cache_path=train_index_cache_path,
                index_cache_refresh=cache_refresh,
            )
            train_dataset = CachedH5FileDatasetNew(
                train_dataset_raw,
                cache_path=train_cache_path,
                limit_cache_size=train_dataset_cache_cfg["limit_cache_size"],
                device=train_dataset_cache_cfg["cache_device"] if "cache_device" in train_dataset_cache_cfg else accelerator.device,
                cache_key=cache_ser_suffix,
            )

    else:
        train_index_cache_enabled = bool(train_dataset_cache_cfg.get("index_cache_enabled", False))
        train_index_cache_path = None
        if train_index_cache_enabled:
            train_index_cache_base_dir = path_resolver(
                train_dataset_cache_cfg.get("index_cache_path_base_dir", train_dataset_cache_cfg["cache_path_base_dir"])
            )
            train_index_cache_path = os.path.join(
                train_index_cache_base_dir,
                f"{os.path.basename(dataset_train_path)}_{cache_ser_suffix}_frame_index.pt",
            )
            print(f"[Train CPL][TRAIN_INDEX_CACHE_PATH] {train_index_cache_path}")
        train_dataset = ImagesFromH5VideoFileDataset(
            dataset_train_path,
            key_filter=filter_train,
            transform=transform_train,
            device="cpu",
            frame_index_transform=_frame_index_transform,
            time_slice_selector=_time_slice_selector,
            index_cache_path=train_index_cache_path,
            index_cache_refresh=cache_refresh,
        )
    val_dataset_cache_cfg = cache_cfg.cfg["val_dataset"]
    filter_val = "^(video)|(warp_parameters_motion.*)$"

    if val_dataset_cache_cfg["enabled"]:
        print(f"Caching val on device: {accelerator.device}")
        val_cache_path_base_dir = path_resolver(val_dataset_cache_cfg["cache_path_base_dir"])
        val_cache_path = os.path.join(val_cache_path_base_dir, f"{os.path.basename(dataset_val_path)}_{cache_ser_suffix}_cache.pt")
        os.makedirs(os.path.dirname(val_cache_path), exist_ok=True)
        print(f"[Train CPL][VAL_CACHE_PATH] {val_cache_path}")

        # Delete cache file if cache_refresh is True
        if cache_refresh and os.path.exists(val_cache_path):
            print(f"[Train CPL][CACHE_REFRESH][VAL] key={cache_ser_suffix} path={val_cache_path}")
            print(f"Cache refresh enabled: removing existing cache file {val_cache_path}")
            os.remove(val_cache_path)

        if (not cache_refresh) and os.path.exists(val_cache_path):
            print(f"[Train CPL][CACHE_HIT][VAL] key={cache_ser_suffix} path={val_cache_path}")
            print(f"Loading val dataset directly from cache file (skip H5 indexing): {val_cache_path}")
            val_dataset = CachedH5FileDatasetNew(
                h5_dataset=None,
                cache_path=val_cache_path,
                limit_cache_size=val_dataset_cache_cfg["limit_cache_size"],
                device=val_dataset_cache_cfg["cache_device"] if "cache_device" in val_dataset_cache_cfg else accelerator.device,
                cache_key=cache_ser_suffix,
            )
        else:
            print(f"[Train CPL][CACHE_MISS][VAL] key={cache_ser_suffix} path={val_cache_path}")
            val_index_cache_enabled = bool(val_dataset_cache_cfg.get("index_cache_enabled", False))
            val_index_cache_path = None
            if val_index_cache_enabled:
                val_index_cache_base_dir = path_resolver(
                    val_dataset_cache_cfg.get("index_cache_path_base_dir", val_dataset_cache_cfg["cache_path_base_dir"])
                )
                val_index_cache_path = os.path.join(
                    val_index_cache_base_dir,
                    f"{os.path.basename(dataset_val_path)}_{cache_ser_suffix}_frame_index.pt",
                )
                print(f"[Train CPL][VAL_INDEX_CACHE_PATH] {val_index_cache_path}")
            val_dataset_raw = ImagesFromH5VideoFileDataset(
                dataset_val_path,
                key_filter=filter_val,
                transform=transform_val,
                device="cpu",
                frame_index_transform=_frame_index_transform,
                time_slice_selector=_time_slice_selector,
                index_cache_path=val_index_cache_path,
                index_cache_refresh=cache_refresh,
            )
            val_dataset = CachedH5FileDatasetNew(
                val_dataset_raw,
                cache_path=val_cache_path,
                limit_cache_size=val_dataset_cache_cfg["limit_cache_size"],
                device=val_dataset_cache_cfg["cache_device"] if "cache_device" in val_dataset_cache_cfg else accelerator.device,
                cache_key=cache_ser_suffix,
            )

    else:
        val_index_cache_enabled = bool(val_dataset_cache_cfg.get("index_cache_enabled", False))
        val_index_cache_path = None
        if val_index_cache_enabled:
            val_index_cache_base_dir = path_resolver(
                val_dataset_cache_cfg.get("index_cache_path_base_dir", val_dataset_cache_cfg["cache_path_base_dir"])
            )
            val_index_cache_path = os.path.join(
                val_index_cache_base_dir,
                f"{os.path.basename(dataset_val_path)}_{cache_ser_suffix}_frame_index.pt",
            )
            print(f"[Train CPL][VAL_INDEX_CACHE_PATH] {val_index_cache_path}")
        val_dataset = ImagesFromH5VideoFileDataset(
            dataset_val_path,
            key_filter=filter_val,
            transform=transform_val,
            device="cpu",
            frame_index_transform=_frame_index_transform,
            time_slice_selector=_time_slice_selector,
            index_cache_path=val_index_cache_path,
            index_cache_refresh=cache_refresh,
        )
    filter_val_with_motion_params = "^(video)|(warp_parameters_motion.*)$"
    val_dataset_with_motion_params = None
    val_dataset_with_motion_cache_cfg = (
        cache_cfg.cfg["val_dataset_with_motion"]
        if "val_dataset_with_motion" in cache_cfg.cfg
        else val_dataset_cache_cfg
    )

    def get_val_dataset_with_motion_params():
        nonlocal val_dataset_with_motion_params
        if val_dataset_with_motion_params is None:
            print("Building val_dataset_with_motion_params (deferred until sampling/evaluation is needed)...")
            if val_dataset_with_motion_cache_cfg["enabled"]:
                val_motion_cache_path_base_dir = path_resolver(val_dataset_with_motion_cache_cfg["cache_path_base_dir"])
                val_motion_cache_path = os.path.join(
                    val_motion_cache_path_base_dir,
                    f"{os.path.basename(dataset_val_path)}_{cache_ser_suffix}_with_motion_cache.pt",
                )
                os.makedirs(os.path.dirname(val_motion_cache_path), exist_ok=True)
                print(f"[Train CPL][VAL_WITH_MOTION_CACHE_PATH] {val_motion_cache_path}")

                if cache_refresh and os.path.exists(val_motion_cache_path):
                    print(f"[Train CPL][CACHE_REFRESH][VAL_WITH_MOTION] key={cache_ser_suffix} path={val_motion_cache_path}")
                    os.remove(val_motion_cache_path)

                if (not cache_refresh) and os.path.exists(val_motion_cache_path):
                    print(f"[Train CPL][CACHE_HIT][VAL_WITH_MOTION] key={cache_ser_suffix} path={val_motion_cache_path}")
                    val_dataset_with_motion_params = CachedH5FileDatasetNew(
                        h5_dataset=None,
                        cache_path=val_motion_cache_path,
                        limit_cache_size=val_dataset_with_motion_cache_cfg["limit_cache_size"],
                        device=val_dataset_with_motion_cache_cfg["cache_device"] if "cache_device" in val_dataset_with_motion_cache_cfg else accelerator.device,
                        cache_key=f"{cache_ser_suffix}_with_motion",
                    )
                else:
                    print(f"[Train CPL][CACHE_MISS][VAL_WITH_MOTION] key={cache_ser_suffix} path={val_motion_cache_path}")
                    val_motion_index_cache_enabled = bool(val_dataset_with_motion_cache_cfg.get("index_cache_enabled", False))
                    val_motion_index_cache_path = None
                    if val_motion_index_cache_enabled:
                        val_motion_index_cache_base_dir = path_resolver(
                            val_dataset_with_motion_cache_cfg.get("index_cache_path_base_dir", val_dataset_with_motion_cache_cfg["cache_path_base_dir"])
                        )
                        val_motion_index_cache_path = os.path.join(
                            val_motion_index_cache_base_dir,
                            f"{os.path.basename(dataset_val_path)}_{cache_ser_suffix}_with_motion_frame_index.pt",
                        )
                        print(f"[Train CPL][VAL_WITH_MOTION_INDEX_CACHE_PATH] {val_motion_index_cache_path}")
                    val_dataset_with_motion_raw = ImagesFromH5VideoFileDataset(
                        dataset_val_path,
                        key_filter=filter_val_with_motion_params,
                        transform=transform_val_with_motion,
                        device="cpu",
                        frame_index_transform=_frame_index_transform,
                        time_slice_selector=_time_slice_selector,
                        index_cache_path=val_motion_index_cache_path,
                        index_cache_refresh=cache_refresh,
                    )
                    val_dataset_with_motion_params = CachedH5FileDatasetNew(
                        val_dataset_with_motion_raw,
                        cache_path=val_motion_cache_path,
                        limit_cache_size=val_dataset_with_motion_cache_cfg["limit_cache_size"],
                        device=val_dataset_with_motion_cache_cfg["cache_device"] if "cache_device" in val_dataset_with_motion_cache_cfg else accelerator.device,
                        cache_key=f"{cache_ser_suffix}_with_motion",
                    )
            else:
                val_motion_index_cache_enabled = bool(val_dataset_with_motion_cache_cfg.get("index_cache_enabled", False))
                val_motion_index_cache_path = None
                if val_motion_index_cache_enabled:
                    val_motion_index_cache_base_dir = path_resolver(
                        val_dataset_with_motion_cache_cfg.get("index_cache_path_base_dir", val_dataset_with_motion_cache_cfg["cache_path_base_dir"])
                    )
                    val_motion_index_cache_path = os.path.join(
                        val_motion_index_cache_base_dir,
                        f"{os.path.basename(dataset_val_path)}_{cache_ser_suffix}_with_motion_frame_index.pt",
                    )
                    print(f"[Train CPL][VAL_WITH_MOTION_INDEX_CACHE_PATH] {val_motion_index_cache_path}")
                val_dataset_with_motion_params = ImagesFromH5VideoFileDataset(
                    dataset_val_path,
                    key_filter=filter_val_with_motion_params,
                    transform=transform_val_with_motion,
                    device="cpu",
                    frame_index_transform=_frame_index_transform,
                    time_slice_selector=_time_slice_selector,
                    index_cache_path=val_motion_index_cache_path,
                    index_cache_refresh=cache_refresh,
                )
        return val_dataset_with_motion_params

    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    from src.utils.batch_sampler_same_shape import BatchSamplerSameShape

    if not "batch_sampler_params" in dataloader_train_cfg.cfg:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, **dataloader_train_cfg.cfg)
    else:
        train_sampler = BatchSamplerSameShape(train_dataset, **dataloader_train_cfg.cfg.batch_sampler_params, target_key_name="image")
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_dict_of_tensors, **dataloader_train_cfg.cfg.dl_params)

    if not "batch_sampler_params" in dataloader_val_cfg.cfg:
        val_dataloader = torch.utils.data.DataLoader(val_dataset, **dataloader_val_cfg.cfg)
    else:
        val_sampler = BatchSamplerSameShape(val_dataset, **dataloader_val_cfg.cfg.batch_sampler_params, target_key_name="image")
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=collate_dict_of_tensors, **dataloader_val_cfg.cfg.dl_params)

    # Compute residual statistics for scaling during training
    residual_stats = compute_residual_statistics(train_dataset, max_samples=500, per_channel=per_channel_residual_scaling)
    residual_std = residual_stats["residual_std"]
    residual_mean = residual_stats["residual_mean"]
    if not residual_centering:
        # std-only mode: divide by empirical std but do not subtract the mean
        residual_mean = [0.0] * len(residual_mean) if isinstance(residual_mean, list) else 0.0
        print(f"[Train CPL] residual_centering=False: zeroing residual_mean (std-only scaling). std={residual_std}")
    if not residual_scaling:
        # no scaling mode: do not scale residuals during training
        residual_mean = [0.0] * len(residual_mean) if isinstance(residual_mean, list) else 0.0
        residual_std = [1.0] * len(residual_std) if isinstance(residual_std, list) else 1.0
        print(f"[Train CPL] residual_scaling=False: zeroing residual_mean and setting residual_std to 1 (no scaling). std={residual_std}, mean={residual_mean}")
    _is_per_ch = isinstance(residual_mean, list)
    print(f"[Train CPL] Will scale residuals during training: per_channel={_is_per_ch}")
    _rm_t = _as_broadcastable(residual_mean, accelerator.device)
    _rs_t = _as_broadcastable(residual_std, accelerator.device)

    from torch.optim.lr_scheduler import OneCycleLR
    lr_scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=optimizer_cfg.cfg["lr"],
        total_steps=training_steps+1,
        pct_start=0.1,
        anneal_strategy='cos',
        cycle_momentum=False,
        base_momentum=0., 
        max_momentum=0.,
        div_factor = 20,
        final_div_factor=100,
        last_epoch=-1
    )

    print(f"Model parameter: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader, lr_scheduler)

    if accelerator.is_main_process:

        output_dir_in_fs = os.path.join(
            wandb_kwargs["config"]["flow_name"] + "_" + wandb_kwargs["config"]["flow_run_name"],
            str(datetime.now().strftime('%Y%m%d_%H%M%S')) + "_" + wandb_kwargs["name"]
        )
        print(f"Output dir in fs: {output_dir_in_fs}")

        storage_settings = StorageSettings(**storage_settings_cfg.cfg)
        filesystem = load_file_system(storage_settings)

        tmp_dir = tempfile.TemporaryDirectory()
        print(f"Temp dir name: {tmp_dir.name}")
        
        file_name = "wandb_kwargs.yaml"
        folder_name = "wandb_params"
        os.makedirs(os.path.join(tmp_dir.name, folder_name), exist_ok=True)
        OmegaConf.save(config=wandb_params_task.config, f=os.path.join(tmp_dir.name, folder_name, file_name))

        file_name = "model_cfg.yaml"
        folder_name = "coupling_config"
        os.makedirs(os.path.join(tmp_dir.name, folder_name), exist_ok=True)
        OmegaConf.save(config=coupling_model_cfg.cfg, f=os.path.join(tmp_dir.name, folder_name, file_name))

        noise_scheduler.save_pretrained(os.path.join(tmp_dir.name, "noise_scheduler"))

        # Save normalization config for inference; per-clip modes only need the mode (stats are refitted).
        from src.utils.video_normalization import VideoNormalization
        _coupling_transform_norm = str(coupling_transform_cfg.cfg.get("normalization", "none"))
        _per_clip_modes = set(VideoNormalization.SUPPORTED_MODES) - {"none", "times2_minus1"}
        # legacy transform modes that are NOT VideoNormalization modes
        _legacy_transform_modes = {"gaussian", "uniform_-1_1", "uniform_0_1"}
        if _coupling_transform_norm in _per_clip_modes and _coupling_transform_norm not in _legacy_transform_modes:
            _norm_to_save = VideoNormalization(mode=_coupling_transform_norm)
            _norm_to_save.save_to_dir(os.path.join(tmp_dir.name, "normalization"))
            print(f"[Train CPL] Saved coupling-transform normalization mode to model dir: {_norm_to_save}")
        else:
            _data_norm = VideoNormalization.read_from_h5_dir(dataset_train_path)
            if _data_norm is not None:
                _data_norm.save_to_dir(os.path.join(tmp_dir.name, "normalization"))
                print(f"[Train CPL] Saved dataset normalization config to model dir: {_data_norm}")
            else:
                print("[Train CPL] WARNING: Could not read normalization from H5 dir; normalization.json not saved.")

        # Save residual scaling statistics for inference
        residual_scaling_cfg = {
            "residual_std": residual_std,
            "residual_mean": residual_mean,
        }
        residual_scaling_path = os.path.join(tmp_dir.name, "residual_scaling.yaml")
        with open(residual_scaling_path, "w") as f:
            yaml.dump(residual_scaling_cfg, f)
        print(f"[Train CPL] Saved residual scaling config: per_channel={isinstance(residual_mean, list)}, std={residual_std}, mean={residual_mean}")

        filesystem.put_directory(tmp_dir.name, output_dir_in_fs, overwrite=True)

        wandb_run.log({
            "dataset_train_path": dataset_train_path,
            "dataset_val_path": dataset_val_path,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "num_params" : sum(p.numel() for p in model.parameters() if p.requires_grad)
        })
    else:
        wandb_run = None

    from tqdm import tqdm
    extra_dict = {}
    fixed_eval_sample_indices = None
    epochs = training_steps // len(train_dataloader) + 1
    # optional fixed number of sample-based evals over the run, else sample_image_every
    _sample_image_times_total = sample_image_cfg.cfg.params.get("sample_image_times_total", None)
    if _sample_image_times_total:
        sample_image_every_effective = max(1, epochs // int(_sample_image_times_total))
    else:
        sample_image_every_effective = sample_image_cfg.cfg.sample_image_every
    epoch_tbar = tqdm(range(epochs), desc="Epoch")
    step = 0
    training_stopped = False
    for epoch in epoch_tbar:

        if training_stopped:
            break

        model.train()

        train_losses = []
        train_mse_losses = []
        train_prior_losses = []
        for batch in train_dataloader:

            if step >= training_steps:
                print(f"Reached max training steps {step}.")
                training_stopped = True
                break

            image = batch["image"].to(accelerator.device, non_blocking=True) # shape (batch, channels, height, width)
            # Scale residuals to approximately unit variance
            image = (image - _rm_t) / _rs_t
            condition = batch["condition"].to(accelerator.device, non_blocking=True) # shape (batch, channels2, height, width)

            assert use_random_noise, "coupling model only trained on random noise for now, warped_noise_sample from dataset not supported for training"
            noise_sample = torch.randn_like(image)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (image.shape[0],), device=image.device)

            noisy_images = noise_scheduler.add_noise(image, noise_sample, timesteps)

            with accelerator.accumulate(model):

                output = forward_fn(noisy_images, timestep=timesteps, condition=condition)
                # target depends on prediction_type (epsilon / v_prediction / sample)
                diffusion_target = get_training_target(
                    noise_scheduler=noise_scheduler,
                    clean_sample=image,
                    noise=noise_sample,
                    timesteps=timesteps,
                )
                mse_loss = torch.nn.functional.mse_loss(output, diffusion_target)
                pred_clean_scaled = predict_clean_from_model_output(
                    noisy_images=noisy_images,
                    model_output=output,
                    timesteps=timesteps,
                    noise_scheduler=noise_scheduler,
                )
                prior_loss = compute_residual_prior_loss(
                    pred_clean_scaled=pred_clean_scaled,
                    residual_mean=_rm_t,
                    residual_std=_rs_t,
                    prior_type=residual_prior_type,
                    prior_delta=residual_prior_delta,
                    prior_blur_sigma=residual_prior_blur_sigma,
                    prior_blur_kernel_size=residual_prior_blur_kernel_size,
                    target_clean_scaled=image,
                    condition=condition,
                    coupling_model=coupling_model,
                )
                loss = mse_loss + float(residual_prior_weight) * prior_loss

                accelerator.backward(loss)
                optimizer.step()

                optimizer.zero_grad()
                lr_scheduler.step()
                train_losses.append(loss.item())
                train_mse_losses.append(mse_loss.item())
                train_prior_losses.append(prior_loss.item())

            if ema_cfg.cfg["enabled"] and step == ema_cfg.cfg["update_step_start"]:
                ema = EMAModel(model.parameters(), decay=ema_cfg.cfg["decay"])
                ema.to(accelerator.device)
                print(f"Initialized EMA model at step {step}.")
            elif step % ema_cfg.cfg["update_ema_every"] == 0 and ema_cfg.cfg["update_step_start"] <=  step and ema is not None:
                ema.step(model.parameters())

            step += 1

        if accelerator.is_main_process:

            model.eval()

            avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0

            train_metrics = {
                "train_loss": avg_train_loss,
                "train_loss_mse": float(np.mean(train_mse_losses)) if train_mse_losses else 0.0,
                "train_loss_residual_prior": float(np.mean(train_prior_losses)) if train_prior_losses else 0.0,
                "epoch": epoch,
                "lr": lr_scheduler.get_last_lr()[0],
            }

            wandb_run.log(train_metrics, step=step)

            if ema is not None:
                ema.store(model.parameters())
                ema.copy_to(model.parameters())

            eval_metrics = evaluate_validation_loss(
                forward_fn=forward_fn,
                val_dataloader=val_dataloader,
                noise_scheduler=noise_scheduler,
                use_random_noise=use_random_noise,
                residual_mean=residual_mean,
                residual_std=residual_std,
                residual_prior_type=residual_prior_type,
                residual_prior_weight=residual_prior_weight,
                residual_prior_delta=residual_prior_delta,
                residual_prior_blur_sigma=residual_prior_blur_sigma,
                residual_prior_blur_kernel_size=residual_prior_blur_kernel_size,
                coupling_model=coupling_model,
                device=accelerator.device,
            )

            if epoch % sample_image_every_effective == 0 and epoch != 0:
                val_dataset_for_sampling = get_val_dataset_with_motion_params()
                if fixed_eval_sample_indices is None:
                    nr_samples_fixed = int(sample_image_cfg.cfg.params.get("nr_of_image_samples", 1))
                    _rng = np.random.default_rng(seed + 20260328)
                    _replace = len(val_dataset_for_sampling) < nr_samples_fixed
                    fixed_eval_sample_indices = _rng.choice(
                        len(val_dataset_for_sampling),
                        size=nr_samples_fixed,
                        replace=_replace,
                    ).tolist()
                    print(f"[Train CPL] Fixed sample indices for WandB preview: {fixed_eval_sample_indices}")
                sample_based_metrics = evaluate_sample_based_losses(
                    val_dataset=val_dataset_for_sampling,
                    pipe=pipe,
                    coupling_model=coupling_model,
                    device=accelerator.device,
                    use_random_noise=use_random_noise,
                    residual_mean=residual_mean,
                    residual_std=residual_std,
                    fixed_sample_indices=fixed_eval_sample_indices,
                    **{k: v for k, v in sample_image_cfg.cfg.params.items() if k not in _PARAMS_NOT_FOR_SAMPLE_LOSSES},
                )
                nr_full_video = sample_image_cfg.cfg.params.get("nr_of_full_video_samples", 0)
                if nr_full_video > 0:
                    _fid_every_n = max(1, int(sample_image_cfg.cfg.params.get("fid_every_n_sample_evals", 5)))
                    _compute_fid = (epoch % (sample_image_every_effective * _fid_every_n) == 0)
                    full_video_metrics = evaluate_full_video_prediction(
                        dataset_val_path=dataset_val_path,
                        pipe=pipe,
                        coupling_model=coupling_model,
                        nr_of_full_video_samples=nr_full_video,
                        device=accelerator.device,
                        clamp_lower=sample_image_cfg.cfg.params.get("clamp_lower", -6),
                        clamp_upper=sample_image_cfg.cfg.params.get("clamp_upper", 6),
                        residual_mean=residual_mean,
                        residual_std=residual_std,
                        nr_random_range_samples_per_video=int(sample_image_cfg.cfg.params.get("nr_random_range_samples_per_video", 2)),
                        random_range_max_k=int(sample_image_cfg.cfg.params.get("random_range_max_k", 6)),
                        compute_fid=_compute_fid,
                        fid_nr_samples=int(sample_image_cfg.cfg.params.get("fid_nr_samples", 64)),
                    )
                    sample_based_metrics = {**sample_based_metrics, **full_video_metrics}
            else:
                sample_based_metrics = {}

            wandb_run.log({**eval_metrics, **sample_based_metrics}, step=step)
            epoch_tbar.set_postfix({**train_metrics, **eval_metrics, **sample_based_metrics})

            if ema is not None:
                if epoch % store_model_cfg.cfg.save_model_every == 0:
                    model_name = f"ema_model_epoch_{epoch}"
                    accelerator.unwrap_model(model).save_pretrained(os.path.join(tmp_dir.name, model_name))
                    filesystem.put_directory(os.path.join(tmp_dir.name, model_name), os.path.join(output_dir_in_fs, model_name), overwrite=True)

                ema.restore(model.parameters())

            if epoch % store_model_cfg.cfg.save_model_every == 0:
                model_name = f"model_epoch_{epoch}"
                accelerator.unwrap_model(model).save_pretrained(os.path.join(tmp_dir.name, model_name))
                filesystem.put_directory(os.path.join(tmp_dir.name, model_name), os.path.join(output_dir_in_fs, model_name), overwrite=True)

        logging.info(f"Epoch {epoch} completed. Extra metrics: {str(extra_dict)}")

    # final evaluation (main process only; `pipe` only exists there)
    if sample_image_cfg.cfg.sample_final and accelerator.is_main_process:
        val_dataset_for_sampling = get_val_dataset_with_motion_params()
        if fixed_eval_sample_indices is None:
            nr_samples_fixed = int(sample_image_cfg.cfg.params.get("nr_of_image_samples", 1))
            _rng = np.random.default_rng(seed + 20260328)
            _replace = len(val_dataset_for_sampling) < nr_samples_fixed
            fixed_eval_sample_indices = _rng.choice(
                len(val_dataset_for_sampling),
                size=nr_samples_fixed,
                replace=_replace,
            ).tolist()
            print(f"[Train CPL] Fixed sample indices for WandB preview: {fixed_eval_sample_indices}")
        sample_based_metrics_final = evaluate_sample_based_losses(
            val_dataset=val_dataset_for_sampling,
            pipe=pipe,
            coupling_model=coupling_model,
            device=accelerator.device,
            use_random_noise=use_random_noise,
            residual_mean=residual_mean,
            residual_std=residual_std,
            fixed_sample_indices=fixed_eval_sample_indices,
            **{k: v for k, v in sample_image_cfg.cfg.params.items() if k not in _PARAMS_NOT_FOR_SAMPLE_LOSSES},
        )
        nr_full_video_final = sample_image_cfg.cfg.params.get("nr_of_full_video_samples", 0)
        if nr_full_video_final > 0:
            full_video_metrics_final = evaluate_full_video_prediction(
                dataset_val_path=dataset_val_path,
                pipe=pipe,
                coupling_model=coupling_model,
                nr_of_full_video_samples=nr_full_video_final,
                device=accelerator.device,
                clamp_lower=sample_image_cfg.cfg.params.get("clamp_lower", -6),
                clamp_upper=sample_image_cfg.cfg.params.get("clamp_upper", 6),
                residual_mean=residual_mean,
                residual_std=residual_std,
                nr_random_range_samples_per_video=int(sample_image_cfg.cfg.params.get("nr_random_range_samples_per_video", 2)),
                random_range_max_k=int(sample_image_cfg.cfg.params.get("random_range_max_k", 6)),
                compute_fid=True,  # one-off final report -- always worth the extra cost here
                fid_nr_samples=int(sample_image_cfg.cfg.params.get("fid_nr_samples", 64)),
            )
            sample_based_metrics_final = {**sample_based_metrics_final, **full_video_metrics_final}
        wandb_run.log(sample_based_metrics_final, step=step)
        logging.info(f"Final sample-based metrics: {sample_based_metrics_final}")

    # Save model
    if accelerator.is_main_process:
        model_name = f"model_final"
        accelerator.unwrap_model(model).save_pretrained(os.path.join(tmp_dir.name, model_name))
        filesystem.put_directory(os.path.join(tmp_dir.name, model_name), os.path.join(output_dir_in_fs, model_name), overwrite=True)
        if result_holder is not None:
            # rank 0 only; passes the result back from the spawned process
            result_holder["output_dir_in_fs"] = output_dir_in_fs
        wandb_run.finish()

    # non-main ranks have no output dir; their return value is discarded anyway
    if not accelerator.is_main_process:
        return None

    return StoragePath(
        storage_path=output_dir_in_fs,
        storage_settings=storage_settings,
    )
