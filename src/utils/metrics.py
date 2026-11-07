from typing import Optional, Union, Any, Tuple

from torch import Tensor
import torch

from skimage.metrics import structural_similarity

import numpy as np
import scipy
import logging
import math

def PSNR(
        rec: Tensor,
        gt: Tensor,
    ) -> float:
    return PSNR_pt(
        reconstruction=rec,
        ground_truth=gt,
        data_range=torch.max(gt)
    )

def normalize(img):
    img = img - torch.min(img)
    img = img / torch.max(img)
    return img


def PSNR_pt(
    reconstruction: Tensor,
    ground_truth  : Tensor,
    data_range: Optional[float] = None
    ) -> np.number:

    mse = (reconstruction - ground_truth).square().mean()
    if data_range is None:
        data_range = torch.max(ground_truth) - np.min(ground_truth)
    return 20*torch.log10(data_range) - 10*torch.log10(mse)

def normalize(
    x: Union[Tensor, np.ndarray], 
    inplace: bool = False
    ) -> Union[Tensor, np.ndarray]:

    """
    Normalize the input as ``(x - x.min()) / (x.max() - x.min())``, optionally inplace.
    """
    if inplace:
        x -= x.min()
        x /= x.max()
    else:
        x = x - x.min()
        x = x / x.max()
    return x




# ------------------------
# CMR helper metrics
# ------------------------

def nmse(pred: Tensor, gt: Tensor) -> float:
    """Normalized MSE = ||pred-gt||^2 / ||gt||^2 (works with real or complex tensors)."""
    diff = pred - gt
    num = (diff.abs() ** 2).mean()
    denom = (gt.abs() ** 2).mean() + 1e-10
    return float((num / denom).item())


def ssim_slicewise_mag(gt_mag: Union[Tensor, np.ndarray], pred_mag: Union[Tensor, np.ndarray], data_range: Optional[float] = None) -> float:
    """SSIM averaged over time axis for magnitude videos.

    Args:
        gt_mag: (T, X, Y) magnitude ground truth
        pred_mag: (T, X, Y) magnitude prediction
        data_range: optional range; if None, inferred from gt
    """
    if torch.is_tensor(gt_mag):
        gt_mag_np = gt_mag.detach().cpu().numpy()
    else:
        gt_mag_np = gt_mag
    if torch.is_tensor(pred_mag):
        pred_mag_np = pred_mag.detach().cpu().numpy()
    else:
        pred_mag_np = pred_mag

    if gt_mag_np.shape != pred_mag_np.shape:
        raise ValueError(f"Shape mismatch for SSIM slicewise: {pred_mag_np.shape} vs {gt_mag_np.shape}")

    if data_range is None:
        data_range = float(gt_mag_np.max() - gt_mag_np.min())

    ssim_list = []
    for t in range(gt_mag_np.shape[0]):
        ssim_list.append(
            structural_similarity(
                gt_mag_np[t],
                pred_mag_np[t],
                data_range=data_range,
            )
        )
    return float(np.mean(ssim_list)) if ssim_list else 0.0


def tssim_temporal_mag(
    gt_mag: Union[Tensor, np.ndarray],
    pred_mag: Union[Tensor, np.ndarray],
    sample_stride: Optional[int] = None,
    win_size: Optional[int] = None,
) -> float:
    """Temporal SSIM averaged over sampled spatial locations.

    Args:
        gt_mag: (T, X, Y) magnitude ground truth
        pred_mag: (T, X, Y) magnitude prediction
        sample_stride: spatial stride for sampling (default picks ~150 voxels)
        win_size: SSIM window size along time (odd, <= T)

    Returns:
        Mean temporal SSIM across sampled voxels.
    """
    if torch.is_tensor(gt_mag):
        gt_mag_np = gt_mag.detach().cpu().numpy()
    else:
        gt_mag_np = gt_mag
    if torch.is_tensor(pred_mag):
        pred_mag_np = pred_mag.detach().cpu().numpy()
    else:
        pred_mag_np = pred_mag

    if gt_mag_np.shape != pred_mag_np.shape:
        raise ValueError(f"Shape mismatch for temporal SSIM: {pred_mag_np.shape} vs {gt_mag_np.shape}")

    T, X, Y = gt_mag_np.shape
    if T < 3:
        return 1.0 if np.allclose(gt_mag_np, pred_mag_np, rtol=1e-4) else 0.0

    if win_size is None:
        win_size = min(5, T if T % 2 == 1 else T - 1)
        win_size = max(3, win_size)

    num_voxels = X * Y
    if sample_stride is None:
        sample_stride = max(1, int(math.ceil(math.sqrt(num_voxels / 150))))

    ssim_vals = []
    for x in range(0, X, sample_stride):
        for y in range(0, Y, sample_stride):
            gt_temporal = gt_mag_np[:, x, y]
            pred_temporal = pred_mag_np[:, x, y]

            temporal_range = float(gt_temporal.max() - gt_temporal.min())
            if temporal_range > 1e-10:
                try:
                    ssim_vals.append(
                        structural_similarity(
                            gt_temporal,
                            pred_temporal,
                            data_range=temporal_range,
                            win_size=win_size,
                        )
                    )
                except (ValueError, RuntimeError):
                    ssim_vals.append(0.0)
            else:
                pred_range = float(pred_temporal.max() - pred_temporal.min())
                ssim_vals.append(1.0 if pred_range < 1e-10 else 0.0)

    return float(np.mean(ssim_vals)) if ssim_vals else 0.0


def psnr_compl(pred: Tensor, gt: Tensor, data_range: Optional[float] = None) -> float:
    """PSNR on the complex-valued error (unlike magnitude PSNR, this also penalizes phase error).

    Args:
        pred, gt: complex tensors, or real tensors with a trailing real/imag dim of size 2.
    """
    pred_c = torch.view_as_complex(pred.contiguous()) if not pred.is_complex() else pred
    gt_c = torch.view_as_complex(gt.contiguous()) if not gt.is_complex() else gt

    mse = (pred_c - gt_c).abs().pow(2).mean()
    if data_range is None:
        data_range = float(gt_c.abs().max())
    if data_range <= 0:
        data_range = 1e-8
    mse_val = float(mse.item())
    if mse_val == 0.0:
        return float("inf")
    return 20 * math.log10(data_range) - 10 * math.log10(mse_val)


def ssim3d(
    gt_mag: Union[Tensor, np.ndarray],
    pred_mag: Union[Tensor, np.ndarray],
    data_range: Optional[float] = None,
    win_size: Optional[int] = None,
) -> float:
    """Volumetric SSIM over the whole (T, X, Y) magnitude volume with a single 3D window (penalizes flicker).

    win_size: if None, the largest odd size <= 7 that fits every axis.
    """
    if torch.is_tensor(gt_mag):
        gt_mag_np = gt_mag.detach().cpu().numpy()
    else:
        gt_mag_np = gt_mag
    if torch.is_tensor(pred_mag):
        pred_mag_np = pred_mag.detach().cpu().numpy()
    else:
        pred_mag_np = pred_mag

    if gt_mag_np.shape != pred_mag_np.shape:
        raise ValueError(f"Shape mismatch for SSIM 3D: {pred_mag_np.shape} vs {gt_mag_np.shape}")

    if data_range is None:
        data_range = float(gt_mag_np.max() - gt_mag_np.min())

    smallest_dim = min(gt_mag_np.shape)
    if win_size is None:
        win_size = min(7, smallest_dim)
        if win_size % 2 == 0:
            win_size -= 1
        win_size = max(3, win_size)

    if win_size > smallest_dim:
        # Clip too short for a 3D window: fall back to per-frame SSIM instead of crashing mid-training.
        return ssim_slicewise_mag(gt_mag_np, pred_mag_np, data_range=data_range)

    return float(structural_similarity(gt_mag_np, pred_mag_np, data_range=data_range, win_size=win_size))