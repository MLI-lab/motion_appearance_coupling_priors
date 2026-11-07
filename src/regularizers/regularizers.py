import torch
from functools import partial
from typing import Callable, Dict, Optional



from typing import List

from src.warp_methods.base_discrete_warp_method import BaseDiscreteWarpMethod

import torch
import torch.nn.functional as F


from src.representations.videovoxel_representation import VideoVoxelRepresentation
def tv_video_loss(
    model : BaseDiscreteWarpMethod | VideoVoxelRepresentation,  # the model itself
    result,  # kspace output of the model
    kspace,  # kspace target
    kcoords,  # corresponding kspace coordinates
    timesteps,
    time_mesh : torch.Tensor,
    spatial_mesh : torch.Tensor,
    kspace_mesh : Optional[torch.Tensor] = None,
    temporal_tv_weight : float = 0.0,
    img_x_tv_weight : Optional[float] = None,
    img_y_tv_weight : Optional[float] = None,
    img_tv_weights_same : bool = True,
    img_tv_take_mag : bool = False,
    batch_size : int = -1,
    norm_order_temporal : int = 1,
    use_same_batch_size : bool = True,
    take_trainer_timesteps : bool = False,
    norm_order_spatial_img : int = 1,
    use_confidence_weights_for_tv : bool = False,
    frame_selection : Optional[List[int]] = None,
    **kwargs
):


    if img_tv_weights_same:
        if img_x_tv_weight is None:
            img_x_tv_weight = img_y_tv_weight
        if img_y_tv_weight is None:
            img_y_tv_weight = img_x_tv_weight

    assert not (img_tv_take_mag and (img_x_tv_weight is None or img_y_tv_weight is None)), "If img_tv_take_mag is True, both img_x_tv_weight and img_y_tv_weight must be provided."

    if img_x_tv_weight is None:
        img_x_tv_weight = 0.0
    if img_y_tv_weight is None:
        img_y_tv_weight = 0.0


    result_spatial = model.full_warp() # this does not include the smap dimension (T, X, Y)

    def norm_mean(x, norm_order):
        if norm_order == 1:
            return x.abs().mean()
        elif norm_order == 2:
            return x.square().mean()
        else:
            raise ValueError(f"Unsupported norm_order: {norm_order}")

    if frame_selection is not None:
        result_spatial = result_spatial[list(frame_selection)]
        print(f"Selected shape is now: {result_spatial.shape}")

    x_tv = norm_mean(result_spatial[:, 1:, :, :] - result_spatial[:, :-1, :, :], norm_order_spatial_img)
    y_tv = norm_mean(result_spatial[:, :, 1:, :] - result_spatial[:, :, :-1, :], norm_order_spatial_img)
    t_tv = norm_mean(result_spatial[1:, :, :, :] - result_spatial[:-1, :, :, :], norm_order_temporal)

    if isinstance(model, VideoVoxelRepresentation):
        if model.add_occlusion_weighting_parameter and use_confidence_weights_for_tv:
            conf = model.get_occlusion_weighting()  # (2, T-1, X, Y)
            conf_fwd_model, conf_bwd_model = conf[0], conf[1]
            occlusion_weights = (1.0-torch.cat([conf_bwd_model, conf_fwd_model[[-1]]], dim=0)).unsqueeze(-1)  # (T-1, X, Y)
            occlusion_weights_x_tv = (occlusion_weights[:, 1:, :, :] * occlusion_weights[:, :-1, :, :])/2
            x_tv = norm_mean(
                occlusion_weights_x_tv * (result_spatial[:, 1:, :, :] - result_spatial[:, :-1, :, :]), norm_order_spatial_img
            )
            occlusion_weights_y_tv = (occlusion_weights[:, :, 1:, :] * occlusion_weights[:, :, :-1, :])/2
            y_tv = norm_mean(
                occlusion_weights_y_tv * (result_spatial[:, :, 1:, :] - result_spatial[:, :, :-1, :]), norm_order_spatial_img
            )
            
    return img_x_tv_weight * x_tv + img_y_tv_weight * y_tv + temporal_tv_weight * t_tv


import torch

import torch


from torchvision.models.optical_flow._utils import make_coords_grid


def _pixel_grid_to_normalized_grid(
    grid_px: torch.Tensor,
    h: int,
    w: int,
    align_corners: bool = True,
) -> torch.Tensor:
    """Convert a pixel-coordinate sampling grid to the normalized grid used by F.grid_sample."""
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
    """Sample ``x`` using a pixel-coordinate grid with explicit padding mode."""
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


def charbonnier(x, eps=1e-3):
    # smooth L1-like: sqrt(x^2 + eps^2)
    return torch.sqrt(x * x + eps * eps)

def huber(x, delta=1.0):
    ax = x.abs()
    return torch.where(ax <= delta, 0.5 * x * x, delta * (ax - 0.5 * delta))
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def phi_from_complex(video, kind="logmag", eps=1e-3):
    """
    video: (B, C, H, W) with C=2 (real, imag) OR (B, 2, H, W)
    returns: (B, 1, H, W) float
    """
    if video.shape[1] != 2:
        raise ValueError(f"Expected complex channels=2, got {video.shape}")
    re, im = video[:, 0], video[:, 1]
    mag = torch.sqrt(re * re + im * im + 1e-12)  # (B,H,W)

    if kind == "mag":
        out = mag
    elif kind == "logmag":
        out = torch.log(mag + eps)
    elif kind == "softlog":  # nice compromise: log(1 + alpha*mag)
        alpha = 10.0
        out = torch.log1p(alpha * mag)
    else:
        raise ValueError(f"Unsupported phi kind: {kind}")

    return out.unsqueeze(1)  # (B,1,H,W)


def lncc_map(I, J, win=9, eps=1e-5):
    """
    Local normalized cross-correlation map between I and J.
    I,J: (B,1,H,W)
    returns: (B,1,H,W) NCC map in [-1,1] (approximately)
    """
    if I.shape != J.shape or I.dim() != 4 or I.shape[1] != 1:
        raise ValueError(f"Expected I,J as (B,1,H,W), got {I.shape} and {J.shape}")

    B, _, H, W = I.shape
    device, dtype = I.device, I.dtype

    # Box filter window
    k = torch.ones((1, 1, win, win), device=device, dtype=dtype)
    win_size = float(win * win)

    # Local sums
    sum_I  = F.conv2d(I, k, padding=win // 2)
    sum_J  = F.conv2d(J, k, padding=win // 2)
    sum_I2 = F.conv2d(I * I, k, padding=win // 2)
    sum_J2 = F.conv2d(J * J, k, padding=win // 2)
    sum_IJ = F.conv2d(I * J, k, padding=win // 2)

    # Means
    mu_I = sum_I / win_size
    mu_J = sum_J / win_size

    # Variances and covariance (numerically stable form)
    var_I = sum_I2 - 2.0 * mu_I * sum_I + mu_I * mu_I * win_size
    var_J = sum_J2 - 2.0 * mu_J * sum_J + mu_J * mu_J * win_size
    cov_IJ = sum_IJ - mu_I * sum_J - mu_J * sum_I + mu_I * mu_J * win_size

    # Clamp variances
    var_I = torch.clamp(var_I, min=0.0)
    var_J = torch.clamp(var_J, min=0.0)

    ncc = cov_IJ / (torch.sqrt(var_I * var_J + eps))
    return ncc


import torch
import torch.nn.functional as F

def _box_filter2d(x: torch.Tensor, win: int) -> torch.Tensor:
    """
    Per-channel 2D box filter via conv2d with groups.
    x: (B, C, X, Y)
    """
    if x.dim() != 4:
        raise ValueError("Expected 2D tensor (B, C, X, Y)")
    B, C, X, Y = x.shape
    pad = win // 2
    filt = torch.ones((C, 1, win, win), device=x.device, dtype=x.dtype)
    return F.conv2d(x, filt, padding=(pad, pad), groups=C)

def clncc_2d(
    a: torch.Tensor,
    b: torch.Tensor,
    win: int = 9,
    eps: float = 1e-6,
    reduction: str = "mean",
    return_map: bool = False,
) -> torch.Tensor:
    """
    Patch-wise complex LNCC (local coherence) for 2D data.

    a, b: complex (B,C,X,Y), or real (B,2C,X,Y) packed as [real..., imag...].
    Returns a scalar similarity (higher is better), or the (B,1,X,Y) map if reduction="none".
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if a.dim() != 4:
        raise ValueError("Expected tensors of shape (B, C, X, Y)")

    # Convert inputs to real/imag per channel
    if torch.is_complex(a):
        ar, ai = a.real, a.imag           # (B,C,X,Y)
        br, bi = b.real, b.imag
        C = a.shape[1]
    else:
        # expect packed real/imag: (B,2C,X,Y)
        if a.shape[1] % 2 != 0:
            raise ValueError("Real input must have even channel count: (B, 2C, X, Y)")
        C = a.shape[1] // 2
        ar, ai = a[:, :C], a[:, C:]
        br, bi = b[:, :C], b[:, C:]

    # Per-channel complex product: a * conj(b)
    # (ar + i ai) * (br - i bi) = (ar*br + ai*bi) + i(ai*br - ar*bi)
    prod_re = ar * br + ai * bi
    prod_im = ai * br - ar * bi

    # Per-channel magnitudes
    mag_a = ar * ar + ai * ai
    mag_b = br * br + bi * bi

    # Local sums per channel (box filter), then sum over channels
    sum_re = _box_filter2d(prod_re, win).sum(dim=1, keepdim=True)  # (B,1,X,Y)
    sum_im = _box_filter2d(prod_im, win).sum(dim=1, keepdim=True)
    Ea = _box_filter2d(mag_a, win).sum(dim=1, keepdim=True)
    Eb = _box_filter2d(mag_b, win).sum(dim=1, keepdim=True)

    # |C| and denominator
    C_abs = torch.sqrt(sum_re * sum_re + sum_im * sum_im + eps)
    denom = torch.sqrt(Ea * Eb + eps) + eps

    clncc_map = C_abs / denom
    clncc_map = torch.clamp(clncc_map, 0.0, 1.0)

    if reduction == "none":
        out = clncc_map
    elif reduction == "mean":
        out = clncc_map.mean()
    elif reduction == "sum":
        out = clncc_map.sum()
    else:
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")

    if return_map:
        return clncc_map, out
    return out


def weighted_mean(x, w, eps=1e-8):
    """
    x: (B,1,H,W)
    w: (B,1,H,W) or (B,H,W) or None
    """
    if w is None:
        return x.mean()
    if w.dim() == 3:
        w = w.unsqueeze(1)
    if w.shape != x.shape:
        raise ValueError(f"Weight shape {w.shape} must match x shape {x.shape}")
    return (w * x).sum() / (w.sum() + eps)



from typing import Dict
def fwd_bwd_motion_consistency_image_domain(
    model: VideoVoxelRepresentation,
    result,  # kspace output of the model
    kspace,  # kspace target
    kcoords,  # corresponding kspace coordinates
    timesteps,
    time_mesh : torch.Tensor,
    spatial_mesh : torch.Tensor,
    iteration : Optional[int] = 0,
    kspace_mesh : Optional[torch.Tensor] = None,
    weight : float = 1.0,
    weight_fwd_relative : float = 1.0,
    weight_bwd_relative : float = 1.0,
    occlusion_weight : float = 1.0,
    occlusion_fwd_relative : float = 1.0,
    occlusion_bwd_relative : float = 1.0,
    loss_norm : str = 'l1',
    allow_flow_antisymmetry : bool = False,
    enable_occlusion_masking : bool = True,
    tau_pix : float = 1.0,
    sharpness : float = 5.0,
    detach_mask : bool = True,
    cycle_consistency_weight : float = 0.0,
    charbonnier_eps : float = 1e-3,
    huber_delta_img : float = 1.0,
    huber_delta_flow : float = 1.0,
    padding_mode: str = "zeros",
    conf_floor = 1e-2,
    additive_gain_l1_weight : float = 0.0,
    domain_transform: Optional[Callable] = None,
    scale_factor = None,
    apply_scaling_factor_from_setup : bool = True,
    **kwargs
):
    """
        Symmetric motion/video coupling:
         lambda sum_{t=1}^{T-1} (|| v_{t+1} - GridResample(v_{t}, grid + m_{t}) ||_1 + backward_weight_relative || v_{t} - GridResample(v_{t+1}, grid - m_{t}) ||_1)

        domain_transform(video, flow_fwd, flow_bwd) optionally evaluates the loss in another domain (e.g. a VAE latent).
    """
    video = model.video_param.squeeze(dim=1).movedim(-1,1) # (T, X, Y, C) -> (T, C, X)
    assert len(video.shape) == 4, "Video should be of shape (T, C, X, Y)"

    T, _, H, W = video.shape

    flow_fwd = model.get_velocity_field(scale=False)[0].permute(3,1,2,0)
    # sure about sign here?

    if model.add_motion_bwd_parameter:
        flow_bwd = model.get_velocity_field(scale=False, ret_bwd=True)[0].permute(3,1,2,0)
    elif allow_flow_antisymmetry or weight_bwd_relative == 0:
        flow_bwd = -flow_fwd
    else:
        raise ValueError("Backward flow parameter is not enabled in the model, but weight_bwd_relative is nonzero and allow_flow_antisymmetry is False. Please enable backward flow parameter or set weight_bwd_relative to 0 or allow_flow_antisymmetry to True.")

    if domain_transform is not None:
        # the confidence maps below stay at the occlusion parameter's (pixel) resolution
        assert not enable_occlusion_masking, "enable_occlusion_masking is not supported together with domain_transform"
        if apply_scaling_factor_from_setup and scale_factor is not None:
            video = video * scale_factor  # undo scale_measurement: e.g. the VAE encoder needs [0, 1] pixels
        # the transform takes (N,2,H,W) motion tensors, the flows here are channels-last
        video, flow_fwd, flow_bwd = domain_transform(video, flow_fwd.movedim(-1, 1), flow_bwd.movedim(-1, 1))
        flow_fwd, flow_bwd = flow_fwd.movedim(1, -1), flow_bwd.movedim(1, -1)
        T, _, H, W = video.shape

    # this is implemented using the raft convention
    base_grid = make_coords_grid(batch_size=T-1, h=H,w=W, device=str(video.device)).movedim(1,-1)
    grid_fwd = base_grid + flow_fwd  # (T-1,H,W,2)
    grid_bwd = base_grid + flow_bwd  # (T-1,H,W,2)
    
    vid_fwd = _grid_sample_pixels(
        video[1:],
        grid_fwd,
        align_corners=True,
        mode="bilinear",
        padding_mode=padding_mode,
    )
    vid_bwd = _grid_sample_pixels(
        video[:-1],
        grid_bwd,
        align_corners=True,
        mode="bilinear",
        padding_mode=padding_mode,
    )

    # occlusion confidence via fwd-bwd consistency: bwd flow sampled at u + fwd(u)
    flow_fwd_warped = _grid_sample_pixels(
        flow_fwd.movedim(-1,1),
        grid_bwd,
        align_corners=True,
        mode="bilinear",
        padding_mode=padding_mode,
    )
    flow_bwd_warped = _grid_sample_pixels(
        flow_bwd.movedim(-1,1),
        grid_fwd,
        align_corners=True,
        mode="bilinear",
        padding_mode=padding_mode,
    )
    fb_err = torch.linalg.norm(flow_fwd + flow_bwd_warped.movedim(1,-1), dim=-1)  # (T-1,H,W) pixels
    fb_bwd_error = torch.linalg.norm(flow_bwd + flow_fwd_warped.movedim(1,-1), dim=-1)  # (T-1,H,W) pixels
    conf_fwd = torch.exp( -fb_err.square() / (2 * tau_pix * tau_pix) )*(1-conf_floor) + conf_floor  # (T-1,H,W)
    conf_bwd = torch.exp( -fb_bwd_error.square() / (2 * tau_pix * tau_pix) )*(1-conf_floor) + conf_floor  # (T-1,H,W)
    conf = model.get_occlusion_weighting()  # (2, T-1, X, Y)
    conf_fwd_model, conf_bwd_model = conf[0], conf[1]
    if conf_fwd.shape[-2:] != conf_fwd_model.shape[-2:]:
        # flows were domain-transformed to another resolution (e.g. the latent grid)
        conf_fwd = F.interpolate(conf_fwd.unsqueeze(1), size=conf_fwd_model.shape[-2:], mode="nearest").squeeze(1)
        conf_bwd = F.interpolate(conf_bwd.unsqueeze(1), size=conf_bwd_model.shape[-2:], mode="nearest").squeeze(1)
    _occ_total = occlusion_fwd_relative + occlusion_bwd_relative
    occlusion_fwd_relative = occlusion_fwd_relative / _occ_total
    occlusion_bwd_relative = occlusion_bwd_relative / _occ_total
    conf_loss = occlusion_weight * (
        occlusion_fwd_relative * F.mse_loss(conf_fwd_model,conf_fwd) +
        occlusion_bwd_relative * F.mse_loss(conf_bwd_model,conf_bwd)
    )
    # for the confidence maps we need (T-1, C, H, W)

    # -- Now apply the confidence masking to the cycle consistency loss --
    if detach_mask:
        conf_fwd = conf_fwd_model.detach() # here we use the model now
        conf_bwd = conf_bwd_model.detach() # here we use the model now
    else:
        conf_fwd = conf_fwd_model
        conf_bwd = conf_bwd_model

    diff_fwd = (vid_fwd - video[:-1]) #.abs()
    diff_bwd = (vid_bwd - video[1:]) #.abs()
    if enable_occlusion_masking:
        diff_fwd = diff_fwd * conf_fwd[:, None]  # (T-1,C,H,W) * (T-1,H,W) -> (T-1,C,H,W)
        diff_bwd = diff_bwd * conf_bwd[:, None]  # (T-1,C,H,W) * (T-1,H,W) -> (T-1,C,H,W)

    # weight fwd and backward such that it sums up to 1
    total_weight = weight_fwd_relative + weight_bwd_relative
    weight_fwd_relative = weight_fwd_relative / total_weight
    weight_bwd_relative = weight_bwd_relative / total_weight
    if loss_norm == 'l2':
        return 0.5 * weight * weight_fwd_relative * diff_fwd.pow(2).mean() + 0.5 * weight * weight_bwd_relative * diff_bwd.pow(2).mean() + conf_loss
    elif loss_norm == 'l1':  # l1 loss
        return 0.5 * weight * weight_fwd_relative * diff_fwd.abs().mean() + 0.5 * weight * weight_bwd_relative * diff_bwd.abs().mean() + conf_loss
    elif loss_norm == 'charbonnier':
        return (
            0.5 * weight * weight_fwd_relative * charbonnier(diff_fwd, eps=charbonnier_eps).mean()
            + 0.5 * weight * weight_bwd_relative * charbonnier(diff_bwd, eps=charbonnier_eps).mean()
            + conf_loss
        )

    elif loss_norm == 'huber':
        return (
            0.5 * weight * weight_fwd_relative * huber(diff_fwd, delta=huber_delta_img).mean()
            + 0.5 * weight * weight_bwd_relative * huber(diff_bwd, delta=huber_delta_img).mean()
            + conf_loss
        )
    elif loss_norm == 'clncc':
        # complex local normalized cross correlation
        # we have: (video[:-1], vid_fwd) and (video[1:], vid_bwd) need to be compared respectively, shapes are : (T-1,C,H,W) and (T-1,C,H,W)
        clncc_map1 = clncc_2d(video[:-1], vid_fwd, win=9, eps=1e-5, reduction='none')  # (T-1,1,H,W) 
        clncc_map2 = clncc_2d(video[1:], vid_bwd, win=9, eps=1e-5, reduction='none')  # (T-1,1,H,W)
        # apply confidence as weights
        w_fwd = conf_fwd.unsqueeze(1) if enable_occlusion_masking else None  # (T-1,1,H,W)
        w_bwd = conf_bwd.unsqueeze(1) if enable_occlusion_masking else None
        loss_fwd = 1.0 - weighted_mean(clncc_map1, w_fwd)  # convert similarity to loss
        loss_bwd = 1.0 - weighted_mean(clncc_map2, w_bwd)
        return (
            0.5 * weight * weight_fwd_relative * loss_fwd
            + 0.5 * weight * weight_bwd_relative * loss_bwd
            + conf_loss
        )
    elif loss_norm == 'lncc':
        # Choose feature for similarity
        # Recommended defaults for cine: logmag (or softlog)
        phi_kind = "logmag"
        phi_eps = 1e-3
        win = 9
        ncc_eps = 1e-5

        # Target vs warped (note: video is (T,C,H,W))
        I_fwd = phi_from_complex(video[:-1], kind=phi_kind, eps=phi_eps)   # (T-1,1,H,W)
        J_fwd = phi_from_complex(vid_fwd,    kind=phi_kind, eps=phi_eps)   # (T-1,1,H,W)

        I_bwd = phi_from_complex(video[1:],  kind=phi_kind, eps=phi_eps)
        J_bwd = phi_from_complex(vid_bwd,    kind=phi_kind, eps=phi_eps)

        ncc_fwd = lncc_map(I_fwd, J_fwd, win=win, eps=ncc_eps)  # (T-1,1,H,W)
        ncc_bwd = lncc_map(I_bwd, J_bwd, win=win, eps=ncc_eps)

        # Apply confidence as weights (soft occlusion)
        w_fwd = conf_fwd.unsqueeze(1) if enable_occlusion_masking else None  # (T-1,1,H,W)
        w_bwd = conf_bwd.unsqueeze(1) if enable_occlusion_masking else None

        # Convert similarity to loss: (1 - mean NCC)
        loss_fwd = 1.0 - weighted_mean(ncc_fwd, w_fwd)
        loss_bwd = 1.0 - weighted_mean(ncc_bwd, w_bwd)

        return (
            0.5 * weight * weight_fwd_relative * loss_fwd
            + 0.5 * weight * weight_bwd_relative * loss_bwd
            + conf_loss
        )
    else:
        raise ValueError(f"Unsupported loss_norm: {loss_norm}")

from src.regularizers.flow_estimators import load_flow_estimator, estimate_flow_for_video_with
from src.regularizers.model_initializers.video_frame_init import process_video_for_raft

_CACHE={}

import hashlib
def make_key(t: torch.Tensor):
    t = t.detach().cpu().contiguous()
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()

def motion_regularizer_flow_distance(
    model: VideoVoxelRepresentation,
    result,  # kspace output of the model
    kspace,  # kspace target
    kcoords,  # corresponding kspace coordinates
    timesteps,
    raft_model,
    pseudoinverse : torch.Tensor,
    time_mesh : torch.Tensor,
    spatial_mesh : torch.Tensor,
    iteration : Optional[int] = 0,
    kspace_mesh : Optional[torch.Tensor] = None,
    weight_relative_fwd : float = 1.0,
    weight_relative_bwd : float = 1.0,
    weight : float = 1.0,
    loss_norm : str = 'l1',
    charbonnier_eps : float = 1e-3,
    huber_delta_flow : float = 1.0,
    clamp_upper_lower_flow : float = 4.0,
    raft_enable_upsample_flow : bool = True,
    flow_estimate_params : Dict = {},
    flow_noise_std : Optional[float] = None,
    flow_estimator : str = 'raft',
    flow_estimator_kwargs : Dict = {},
    # spatial smoothness (Horn-Schunck / TV) on the optimised velocity field
    spatial_horn_schunck_weight: float = 0.0,   # set > 0 to enable
    spatial_horn_schunck_type: str = "l1",       # "l1" = Charbonnier TV, "l2" = HS quadratic
    spatial_horn_schunk_charb_const: float = 1e-3,  # Charbonnier epsilon (l1 mode only)
    **kwargs
):
    # pseudoinverse has shape (T, X, Y, C)

    # _CACHE can outlive a run (reused Ray workers), so noise/estimator settings must be in the key
    key = (
        make_key(pseudoinverse),
        flow_noise_std,
        flow_estimator,
        tuple(sorted(flow_estimator_kwargs.items())),
    )
    if key in _CACHE:
        velo_field_fwd, velo_field_bwd = _CACHE[key]
    else:
        raft_dataset = process_video_for_raft(pseudoinverse)  # (T, C, X, Y)
        velo_field_fwd = estimate_flow_for_video_with(flow_estimator, raft_model, raft_dataset.movedim(-1,1), enable_upsample_flow=raft_enable_upsample_flow, forward_flow=True, **flow_estimate_params).movedim(0,-1)[None]  # (T-1, 2, X, Y)
        velo_field_bwd = estimate_flow_for_video_with(flow_estimator, raft_model, raft_dataset.movedim(-1,1), enable_upsample_flow=raft_enable_upsample_flow, forward_flow=False, **flow_estimate_params).movedim(0,-1)[None]  # (T-1, 2, X, Y)
        if flow_noise_std is not None and flow_noise_std > 0:
            # one-time corruption of the cached target (persistent noise, not per-step)
            velo_field_fwd = velo_field_fwd + torch.randn_like(velo_field_fwd) * flow_noise_std
            velo_field_bwd = velo_field_bwd + torch.randn_like(velo_field_bwd) * flow_noise_std
        _CACHE[key] = (velo_field_fwd.detach(), velo_field_bwd.detach())
    # this so far has been the convention...

    model_velocity_field_fwd = model.get_velocity_field(scale=False, ret_bwd=False)  # (1, 2, H, W, T-1)
    model_velocity_field_bwd = model.get_velocity_field(scale=False, ret_bwd=True)  # (1, 2, H, W, T-1)

    # implement a cycle consistency loss here..


    total_weight = weight_relative_fwd + weight_relative_bwd
    weight_relative_fwd = weight_relative_fwd / total_weight
    weight_relative_bwd = weight_relative_bwd / total_weight
    if loss_norm == 'l2':
        pi_loss = 0.5 * weight * weight_relative_fwd * (velo_field_fwd - model_velocity_field_fwd).pow(2).mean() + 0.5 * weight * weight_relative_bwd * (velo_field_bwd - model_velocity_field_bwd).pow(2).mean()
    elif loss_norm == 'l1':
        pi_loss = 0.5 * weight * weight_relative_fwd * (velo_field_fwd - model_velocity_field_fwd).abs().mean() + 0.5 * weight * weight_relative_bwd * (velo_field_bwd - model_velocity_field_bwd).abs().mean()
    elif loss_norm == 'charbonnier':
        pi_loss = (
            0.5 * weight * weight_relative_fwd * charbonnier(velo_field_fwd - model_velocity_field_fwd, eps=charbonnier_eps).mean()
            + 0.5 * weight * weight_relative_bwd * charbonnier(velo_field_bwd - model_velocity_field_bwd, eps=charbonnier_eps).mean()
        )
    elif loss_norm == 'huber':
        pi_loss = (
            0.5 * weight * weight_relative_fwd * huber(velo_field_fwd - model_velocity_field_fwd, delta=huber_delta_flow).mean()
            + 0.5 * weight * weight_relative_bwd * huber(velo_field_bwd - model_velocity_field_bwd, delta=huber_delta_flow).mean()
        )
    else:
        raise ValueError(f"Unsupported loss_norm: {loss_norm}")

    # Horn-Schunck / TV smoothness on the optimised velocity field
    hs_loss = model_velocity_field_fwd.new_tensor(0.0)
    if spatial_horn_schunck_weight > 0:
        # V_fwd: (1, 2, H, W, T-1) -> (H, W, T-1, 2)
        Vf = model_velocity_field_fwd.squeeze(0).permute(1, 2, 3, 0)   # (H, W, T-1, 2)
        Vb = model_velocity_field_bwd.squeeze(0).permute(1, 2, 3, 0)   # (H, W, T-1, 2)

        def _spatial_tv(V):
            gx = V[1:, :, :, :] - V[:-1, :, :, :]  # (H-1, W, T-1, 2)
            gy = V[:, 1:, :, :] - V[:, :-1, :, :]  # (H, W-1, T-1, 2)
            if spatial_horn_schunck_type == "l1":
                # Charbonnier-smoothed isotropic TV (well-behaved at 0)
                tv = (
                    torch.sqrt(gx.square().sum(dim=-1) + spatial_horn_schunk_charb_const).mean()
                    + torch.sqrt(gy.square().sum(dim=-1) + spatial_horn_schunk_charb_const).mean()
                )
            elif spatial_horn_schunck_type == "l2":
                tv = gx.square().mean() + gy.square().mean()
            else:
                raise ValueError(f"Unsupported spatial_horn_schunck_type: {spatial_horn_schunck_type!r}")
            return tv

        hs_loss = spatial_horn_schunck_weight * (
            0.5 * weight_relative_fwd * _spatial_tv(Vf)
            + 0.5 * weight_relative_bwd * _spatial_tv(Vb)
        )

    return pi_loss + hs_loss

def composite_motion_apperance_coupling_regularizer(
    motion_regularizer,
    appearance_regularizer,
    coupling_regularizer,
    **params
):
    def _merge(sub, outer):
        # keys in the sub-regularizer dict take priority over outer params
        filtered = {k: v for k, v in outer.items() if k not in sub}
        return {**sub, **filtered}

    motion_regularizer_fn = regularizer_fn_resolver(**_merge(motion_regularizer, params)) if motion_regularizer is not None else None
    appearance_regularizer_fn = regularizer_fn_resolver(**_merge(appearance_regularizer, params)) if appearance_regularizer is not None else None
    coupling_regularizer_fn = regularizer_fn_resolver(**_merge(coupling_regularizer, params)) if coupling_regularizer is not None else None

    def composite_regularizer_fn(
        model : BaseDiscreteWarpMethod,  # the model itself
        result,  # kspace output of the model
        **kwargs
    ):
        total_loss = 0.0
        if motion_regularizer_fn is not None:
            total_loss = total_loss + motion_regularizer_fn(model=model, result=result, **kwargs)
        if appearance_regularizer_fn is not None:
            total_loss = total_loss + appearance_regularizer_fn(model=model, result=result, **kwargs)
        if coupling_regularizer_fn is not None:
            total_loss = total_loss + coupling_regularizer_fn(model=model, result=result, **kwargs)
        return total_loss
    return composite_regularizer_fn



def regularizer_fn_resolver(name, **params):
    if "diffuser" in name and "cpl" in name and "latent" in name:
        # Before the generic "diffuser"+"cpl" branch below, whose substring test would also match.
        from src.regularizers.diffusers_cpl_latent_regularizers import (
            diffusers_cpl_latent_regularizer_resolver,
        )
        return diffusers_cpl_latent_regularizer_resolver(name, **params)
    elif "diffuser" in name and "cpl" in name:
        from src.regularizers.diffusers_cpl_regularizers import (
            diffusers_cpl_regularizer_resolver,
        )
        return diffusers_cpl_regularizer_resolver(name, **params)
    elif "diffusion_idm" in name:
        from src.regularizers.diffusion_idm_regularizers import (
            diffusion_idm_regularizer_resolver,
        )
        return diffusion_idm_regularizer_resolver(name, **params)
    elif name == "tv_video":
        return partial(tv_video_loss, **params)
    elif name == "composite_motion_apperance_coupling_regularizer":
        return composite_motion_apperance_coupling_regularizer(**params)
    elif name == "fwd_bwd_motion_consistency_image_domain":
        return partial(fwd_bwd_motion_consistency_image_domain, **params)
    elif name == "fwd_bwd_motion_consistency_latent_domain":
        # the same fwd/bwd photometric consistency, evaluated on the frozen SD1.5 VAE's latent
        # grid: video encoded, flows resized and rescaled to latent resolution
        from src.regularizers.diffusers_cpl_latent_regularizers import FrozenLatentEncoder
        latent_encoder = FrozenLatentEncoder(
            pretrained_model_dir=params.pop("pretrained_model_dir", "model_cache/sd15"),
            device=params.get("device", "cuda:0"),
            vae_subfolder=params.pop("vae_subfolder", "vae"),
            assume_pixel_range=params.pop("assume_pixel_range", "0_1"),
            path_resolver=params.get("path_resolver"),
        )
        return partial(fwd_bwd_motion_consistency_image_domain, **params, domain_transform=latent_encoder.encode_video_and_motion)
    elif name == "motion_regularizer_flow_distance":
        # flow_estimator selects the optical-flow backend for the flowdist target (see flow_estimators.py)
        flow_estimator = params.pop("flow_estimator", "raft")
        flow_estimator_kwargs = params.pop("flow_estimator_kwargs", {})
        raft_model = load_flow_estimator(flow_estimator, device="cuda:0", **flow_estimator_kwargs)
        return partial(motion_regularizer_flow_distance, **params, raft_model=raft_model, flow_estimator=flow_estimator, flow_estimator_kwargs=flow_estimator_kwargs)
    elif name == "none" or name == "None":
        return None
    else:
        raise ValueError(f"Unknown regularizer function name: {name}")
