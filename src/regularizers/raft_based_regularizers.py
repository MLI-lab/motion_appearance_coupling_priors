import torch
import torch.nn.functional as F
from typing import Optional
from torch import Tensor
import logging

logger = logging.getLogger(__name__)


def load_raft_model(checkpoint_path: Optional[str] = None, device: str = "cuda:0", weights: Optional[str] = None):
    """
    Load a RAFT optical flow model (torchvision weights, falling back to checkpoint_path).

    weights: name of a torchvision Raft_Large_Weights member (e.g. "C_T_V2"); None = DEFAULT (C_T_SKHT_V2).
    """
    try:
        # Try to load from torchvision (recommended as it includes official pretrained weights)
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        logger.info("Loading RAFT model from torchvision...")
        weights_enum = getattr(Raft_Large_Weights, weights) if weights is not None else Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights_enum, progress=True)
        model = model.to(device)
        model.eval()
        # torchvision leaves requires_grad=True on the weights
        model.requires_grad_(False)
        return model
    except Exception as e:
        logger.warning(f"Failed to load RAFT from torchvision: {e}")
        
    try:
        # Fallback: try to load from custom checkpoint
        if checkpoint_path is not None:
            logger.info(f"Loading RAFT model from checkpoint: {checkpoint_path}")
            from torchvision.models.optical_flow import raft_large
            model = raft_large(weights=None, progress=True)
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            model.requires_grad_(False)
            return model
        else:
            raise ValueError("No checkpoint path provided and torchvision RAFT loading failed")
    except Exception as e:
        logger.error(f"Failed to load RAFT model: {e}")
        raise



def _round_up_to_multiple_of_8(size: int) -> int:
    return ((int(size) + 7) // 8) * 8


def _scale_and_pad_for_raft(frames: Tensor, min_short_side: int):
    """Upscale (nearest) the whole frame (N, C, H, W) so its short side is >= min_short_side
    (never downscale or crop), then reflect-pad H and W to multiples of 8.

    Returns (frames_padded, scaled_size, original_size).
    """
    N, C, H, W = frames.shape
    scale = max(1.0, min_short_side / min(H, W))
    if scale != 1.0:
        scaled_size = (int(round(H * scale)), int(round(W * scale)))
        # nearest matters for RAFT input quality here
        frames = F.interpolate(frames, size=scaled_size, mode='nearest')
    else:
        scaled_size = (H, W)
        frames = frames.contiguous()
    pad_h = (-scaled_size[0]) % 8
    pad_w = (-scaled_size[1]) % 8
    if pad_h or pad_w:
        frames = F.pad(frames, (0, pad_w, 0, pad_h), mode='reflect')
    return frames, scaled_size, (H, W)


def _unpad_and_rescale_flow(flow: Tensor, scaled_size, original_size):
    """Inverse of _scale_and_pad_for_raft: unpad, resize the (N, 2, Hs, Ws) flow to (H, W) and
    rescale each channel to original-resolution pixel units (channel 0 = x, channel 1 = y)."""
    Hs, Ws = scaled_size
    H, W = original_size
    flow = flow[:, :, :Hs, :Ws]
    if (Hs, Ws) != (H, W):
        flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=True)
        axis_scale = torch.tensor([W / Ws, H / Hs], device=flow.device, dtype=flow.dtype).view(1, 2, 1, 1)
        flow = flow * axis_scale
    return flow


import torchvision
# no_grad: the flow is always used detached; otherwise the RAFT graph stays alive until backward (~8-32 GiB)
@torch.no_grad()
def estimate_flow_for_video(
    raft_model, video, enable_upsample_flow: bool = True, forward_flow: bool = True,
    interpolate_to_target_size: Optional[int] = None, gaussian_blur_before_flow: bool = False, gaussian_blur_kernel_size: int = 5, gaussian_blur_sigma: float = 1.0
):

    # video: (T, C, H, W)
    T, C, H, W = video.shape
    assert T >= 2, "Need at least two frames to compute optical flow."

    # Make sure we have 3 channels for RAFT
    if C == 1:
        video_3c = video.repeat(1, 3, 1, 1)
    elif C == 2:
        # real/imaginary channels of complex-valued MRI data -> magnitude image
        video_3c = torch.linalg.norm(video, dim=1, keepdim=True).repeat(1, 3, 1, 1)
    elif C >= 3:
        video_3c = video[:, :3]  # use first 3 channels if more are present
    else:
        raise ValueError(f"Unsupported channel count C={C}. Expected 1, 2, or >=3.")

    video_3c = video_3c.to(device=video.device, dtype=torch.float32)
    original_size = (H,W)

    from src.utils.utils import scale_to_unit_interval
    video_01 = scale_to_unit_interval(video_3c)  # (T, 3, H, W) in [0,1]

    if interpolate_to_target_size is not None:
        interpolate_to_target_size = int(interpolate_to_target_size)
    else:
        interpolate_to_target_size = 256
    interpolate_to_target_size = _round_up_to_multiple_of_8(interpolate_to_target_size)

    # RAFT needs H, W divisible by 8 and works best at >= ~256 px
    video_01, scaled_size, original_size = _scale_and_pad_for_raft(video_01, interpolate_to_target_size)

    # Build consecutive frame pairs in a single batch: N = T-1
    img1 = video_01[:-1]  # (T-1, 3, Hs, Ws)
    img2 = video_01[1:]   # (T-1, 3, Hs, Ws)

    if not forward_flow:
        img1, img2 = img2, img1  # swap order for backward flow

    def preprocess(batch):
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            ]
        )
        batch = transforms(batch)
        return batch

    img1_p = preprocess(img1)
    img2_p = preprocess(img2)

    raft_out = raft_model(img1_p, img2_p)

    # torchvision RAFT returns a list/tuple of flow predictions; last is highest quality
    flow_p = raft_out[-1] if isinstance(raft_out, (list, tuple)) else raft_out

    # undo scale+pad, flow back in original-resolution pixel units
    flow = _unpad_and_rescale_flow(flow_p, scaled_size, original_size)

    return flow  # (T-1, 2, H, W)



def estimate_displacement_field_with_raft(raft_model, video, enable_upsample_flow : bool = True, forward_flow: bool = True):
    flow_field = estimate_flow_for_video(raft_model, video, enable_upsample_flow=enable_upsample_flow, forward_flow=forward_flow)  # (T-1, 2, H, W)
    _, C, Hf, Wf = flow_field.shape
    velo_field = torch.cat(
        [
            torch.zeros( (1, 2, Hf, Wf, 1), device=video.device),
            flow_field.movedim(0, -1)[None]
        ],
        dim=-1
    ).to(video.device)
    return torch.cumsum(velo_field, dim=-1)  # shape (1, 2, H, W, T)

