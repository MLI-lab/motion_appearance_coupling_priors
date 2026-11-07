"""Transport of pixel-resolution motion fields to the latent grid of a Stable-Diffusion-style
VAE (downsampling factor f): m_lat = Resize_{HxW -> H/f x W/f}(m) / f, as in MGLD (arXiv:2312.00853).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def motion_to_latent(motion: torch.Tensor, downsample_factor: int, mode: str = "area") -> torch.Tensor:
    """motion: (N, 2, H, W) displacement in pixel units, or (1, 2, H, W, T). Returns the same layout at (H/f, W/f).

    ``area`` mode: a latent cell gets the mean motion of its f*f pixels (and all pixels keep a gradient).
    """
    f = int(downsample_factor)
    if f == 1:
        return motion
    if motion.ndim == 5:
        # (1, 2, H, W, T) -> (T, 2, H, W) -> resize -> back
        m = motion[0].permute(3, 0, 1, 2)
        return motion_to_latent(m, f, mode=mode).permute(1, 2, 3, 0)[None]
    if motion.ndim != 4 or motion.shape[1] != 2:
        raise ValueError(f"motion_to_latent expects (N,2,H,W) or (1,2,H,W,T), got {tuple(motion.shape)}")
    _, _, h_in, w_in = motion.shape
    if h_in % f != 0 or w_in % f != 0:
        raise ValueError(f"motion_to_latent: spatial size {(h_in, w_in)} is not divisible by f={f}")
    h_out, w_out = h_in // f, w_in // f
    resized = F.interpolate(motion, size=(h_out, w_out), mode=mode)
    scale = torch.tensor([w_out / w_in, h_out / h_in], dtype=resized.dtype, device=resized.device).view(1, 2, 1, 1)
    return resized * scale
