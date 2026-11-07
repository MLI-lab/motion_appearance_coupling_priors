"""Pluggable optical-flow estimator backends ("raft", "classical" Farneback via OpenCV).

All return a detached (T-1, 2, H, W) pixel-displacement tensor, (dx, dy), frame i -> i+1.
"""
from typing import Optional, Dict
import logging

import torch

logger = logging.getLogger(__name__)


# shared full-frame scale+pad helpers, so all backends run at the same working resolution
def _round_up_to_multiple_of_8(size: int) -> int:
    return ((int(size) + 7) // 8) * 8


def _scale_and_pad(frames: torch.Tensor, min_short_side: int):
    """See raft_based_regularizers._scale_and_pad_for_raft. Returns contiguous frames."""
    from src.regularizers.raft_based_regularizers import _scale_and_pad_for_raft
    frames, scaled_size, original_size = _scale_and_pad_for_raft(frames, min_short_side)
    return frames.contiguous(), scaled_size, original_size


def _unpad_and_rescale_flow(flow: torch.Tensor, scaled_size, original_size):
    """See raft_based_regularizers._unpad_and_rescale_flow."""
    from src.regularizers.raft_based_regularizers import _unpad_and_rescale_flow as _impl
    return _impl(flow, scaled_size, original_size)


def _video_to_3ch_01(video: torch.Tensor) -> torch.Tensor:
    """video: (T, C, H, W) raw-domain -> (T, 3, H, W) float32 in [0, 1], same channel handling as the raft backend."""
    from src.utils.utils import scale_to_unit_interval

    T, C, H, W = video.shape
    if C == 1:
        video_3c = video.repeat(1, 3, 1, 1)
    elif C == 2:
        video_3c = torch.linalg.norm(video, dim=1, keepdim=True).repeat(1, 3, 1, 1)
    elif C >= 3:
        video_3c = video[:, :3]
    else:
        raise ValueError(f"Unsupported channel count C={C}. Expected 1, 2, or >=3.")

    video_3c = video_3c.to(dtype=torch.float32)
    return scale_to_unit_interval(video_3c)


# "raft" backend
def _load_raft(device: str, checkpoint_path: Optional[str] = None, weights: Optional[str] = None, **kwargs):
    from src.regularizers.raft_based_regularizers import load_raft_model
    return load_raft_model(checkpoint_path=checkpoint_path, device=device, weights=weights)


def _raft_estimate_flow_for_video(model, video, enable_upsample_flow=True, forward_flow=True,
                                   interpolate_to_target_size=None, **kwargs) -> torch.Tensor:
    from src.regularizers.raft_based_regularizers import estimate_flow_for_video
    return estimate_flow_for_video(
        model, video, enable_upsample_flow=enable_upsample_flow, forward_flow=forward_flow,
        interpolate_to_target_size=interpolate_to_target_size,
    )


# "classical" backend: Farneback via OpenCV, on CPU per frame pair
def _load_classical(device: str, checkpoint_path: Optional[str] = None, **kwargs):
    import cv2  # noqa: F401  (import check only -- raises a clear error if opencv is missing)
    return None  # no weights/model object to hold onto


def _classical_estimate_flow_for_video(model, video, enable_upsample_flow=True, forward_flow=True,
                                        interpolate_to_target_size=None,
                                        farneback_params: Optional[Dict] = None, **kwargs) -> torch.Tensor:
    import cv2
    import numpy as np

    T, C, H, W = video.shape
    assert T >= 2, "Need at least two frames to compute optical flow."

    video_01 = _video_to_3ch_01(video)  # (T, 3, H, W) in [0, 1]
    original_size = (H, W)

    target_size = _round_up_to_multiple_of_8(int(interpolate_to_target_size) if interpolate_to_target_size else 256)
    video_01, scaled_size, original_size = _scale_and_pad(video_01, target_size)

    # grayscale: plain channel mean
    gray = (video_01.mean(dim=1, keepdim=False).clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()  # (T, Ht, Wt)

    img1 = gray[:-1]
    img2 = gray[1:]
    if not forward_flow:
        img1, img2 = img2, img1

    params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    if farneback_params:
        params.update(farneback_params)

    flows = []
    for i in range(img1.shape[0]):
        flow_i = cv2.calcOpticalFlowFarneback(img1[i], img2[i], None, **params)  # (Ht, Wt, 2) as (dx, dy)
        flows.append(flow_i)
    flow = torch.from_numpy(np.stack(flows, axis=0)).movedim(-1, 1).to(device=video.device, dtype=torch.float32)  # (N, 2, Hs, Ws)

    flow = _unpad_and_rescale_flow(flow, scaled_size, original_size)
    return flow


_BACKENDS = {
    "raft": (_load_raft, _raft_estimate_flow_for_video),
    "classical": (_load_classical, _classical_estimate_flow_for_video),
}


def load_flow_estimator(name: str, device: str = "cuda:0", checkpoint_path: Optional[str] = None, **kwargs):
    """Load the flow-estimator model for backend `name` (None for backends without weights, e.g. "classical")."""
    if name not in _BACKENDS:
        raise ValueError(f"Unknown flow_estimator {name!r}. Available: {sorted(_BACKENDS)}")
    load_fn, _ = _BACKENDS[name]
    return load_fn(device=device, checkpoint_path=checkpoint_path, **kwargs)


def estimate_flow_for_video_with(name: str, model, video: torch.Tensor, **kwargs) -> torch.Tensor:
    """video: (T, C, H, W) raw-domain tensor, same input convention as
    raft_based_regularizers.estimate_flow_for_video. Returns (T-1, 2, H, W)."""
    if name not in _BACKENDS:
        raise ValueError(f"Unknown flow_estimator {name!r}. Available: {sorted(_BACKENDS)}")
    _, estimate_fn = _BACKENDS[name]
    return estimate_fn(model, video, **kwargs)


def estimate_displacement_field_with(name: str, model, video: torch.Tensor, enable_upsample_flow: bool = True,
                                      forward_flow: bool = True, **kwargs) -> torch.Tensor:
    """Cumulative-sum per-pair flow into a (1, 2, H, W, T) displacement field, zero at t=0."""
    if name == "raft":
        from src.regularizers.raft_based_regularizers import estimate_displacement_field_with_raft
        return estimate_displacement_field_with_raft(model, video, enable_upsample_flow=enable_upsample_flow, forward_flow=forward_flow)

    flow_field = estimate_flow_for_video_with(name, model, video, enable_upsample_flow=enable_upsample_flow,
                                               forward_flow=forward_flow, **kwargs)  # (T-1, 2, H, W)
    _, C, Hf, Wf = flow_field.shape
    velo_field = torch.cat(
        [
            torch.zeros((1, 2, Hf, Wf, 1), device=flow_field.device),
            flow_field.movedim(0, -1)[None],
        ],
        dim=-1,
    )
    return torch.cumsum(velo_field, dim=-1)  # (1, 2, H, W, T)
