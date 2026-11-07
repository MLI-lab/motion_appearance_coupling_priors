import torch
from typing import Optional


def process_video_for_raft(video: torch.Tensor) -> torch.Tensor:
    # video shape: (T, H, W, C)
    C = video.shape[-1]
    if C == 1:
        video_raft = video.repeat(1,1,1,3)  # shape (T, H, W, 3)
    elif C == 2:
        video_raft = video.norm(dim=-1, keepdim=True).repeat(1,1,1,3)  # shape (T, H, W, 3)
    elif C == 3:
        video_raft = video
    else:
        raise ValueError(f"Unsupported number of channels: {C}")
    return video_raft

def video_and_motion_init_full(
    dataset,
    motion_init="random",
    device="cpu",
    raft_enable_upsample_flow=False,
    clamp_lower: float = -6.0,
    clamp_upper: float = 6.0,
    flow_noise_std: Optional[float] = None,
    flow_estimator: str = "raft",
    flow_estimator_kwargs: dict = {},
    motion_source_dataset: Optional[torch.Tensor] = None,
    video_init: str = "pseudoinv",
    video_init_noise_std: Optional[float] = None,
    video_init_alpha: float = 0.1,
    reference_video: Optional[torch.Tensor] = None,
    **kwargs
) -> dict:
    # video_init: "pseudoinv" (default) | "random" | "mc_warp" (frame 0 chained through the backward flow)
    #   | "target_plus_alias" / "target_plus_noise" (oracle starts, need reference_video); motion always from the pseudoinverse
    from src.regularizers.flow_estimators import load_flow_estimator, estimate_displacement_field_with

    # motion_source_dataset: estimate motion from a different video than the one used for initialization
    motion_source = dataset if motion_source_dataset is None else motion_source_dataset
    dataset_raft = process_video_for_raft(motion_source).to(device)  # shape (T, H, W, 3)

    flow_model = load_flow_estimator(flow_estimator, device=device, **flow_estimator_kwargs)
    motion = estimate_displacement_field_with(flow_estimator, flow_model, dataset_raft.movedim(-1,1), enable_upsample_flow=raft_enable_upsample_flow, forward_flow=True).to(device)
    motion_bwd = estimate_displacement_field_with(flow_estimator, flow_model, dataset_raft.movedim(-1,1), enable_upsample_flow=raft_enable_upsample_flow, forward_flow=False).to(device)
    if flow_noise_std is not None and flow_noise_std > 0:
        # one-time corruption of the motion estimate (only used by motion_init="raft")
        motion = motion + torch.randn_like(motion) * flow_noise_std
        motion_bwd = motion_bwd + torch.randn_like(motion_bwd) * flow_noise_std
    if motion_init == "random":
        motion = torch.randn_like(motion) * motion.std()
        motion_bwd = torch.randn_like(motion_bwd) * motion_bwd.std()
    elif motion_init == "zero":
        motion = torch.zeros_like(motion)
        motion_bwd = torch.zeros_like(motion_bwd)
    elif motion_init == "raft":
        pass
    else:
        raise ValueError(f"Unknown motion initialization type: {motion_init}")

    if dataset.ndim == 5:
        video_dataset = dataset # has already the expected shape
    elif dataset.ndim == 4:
        video_dataset = dataset[:,None]  # add singleton dimension for compatibility with expected shape (T, C, H, W, C)
    else:
        raise ValueError(f"Unsupported dataset shape: {dataset.shape}")
    video_dataset = video_dataset.to(device)

    if video_init == "pseudoinv":
        pass
    elif video_init == "random":
        std = video_dataset.std() if video_init_noise_std is None else float(video_init_noise_std)
        video_dataset = torch.randn_like(video_dataset) * std + video_dataset.mean()
    elif video_init == "mc_warp":
        video_dataset = motion_compensated_video_init(video_dataset, motion_bwd)
    elif video_init in ("target_plus_alias", "target_plus_noise"):
        if reference_video is None:
            raise ValueError(f"video_init={video_init} needs reference_video (the clean clip in the pseudoinverse's domain)")
        ref = reference_video.to(device)
        ref = ref if ref.ndim == 5 else ref[:, None]
        alias = video_dataset - ref
        if video_init == "target_plus_noise":
            noise = torch.randn_like(alias)
            alias = noise * (alias.norm() / noise.norm().clamp_min(1e-12))
        video_dataset = ref + float(video_init_alpha) * alias
    else:
        raise ValueError(f"Unknown video initialization type: {video_init}")

    return {
        "video" : video_dataset,  # shape (T, H, W, C) -> (T, 1, H, W, C)
        "motion" : motion,
        "motion_bwd" : motion_bwd,
        "template" : video_dataset[[0]]  # shape (1, H, W, C)
    }


@torch.no_grad()
def motion_compensated_video_init(video: torch.Tensor, motion_bwd: torch.Tensor) -> torch.Tensor:
    """Frame 0 of ``video`` chained through the backward flow: x_{t+1} = x_t warped forward to t+1.

    video: (T, 1, H, W, C); motion_bwd: (1, 2, H, W, T) displacement field (zero at t=0).
    """
    from torchvision.models.optical_flow._utils import make_coords_grid, grid_sample
    from src.warp_methods.base_warp_method import displacement_field_to_velocity_field

    T, _, H, W, _ = video.shape
    m_bwd = displacement_field_to_velocity_field(motion_bwd.to(video.device))  # (1, 2, H, W, T-1) per-pair flow
    base_grid = make_coords_grid(batch_size=1, h=H, w=W, device=str(video.device)).movedim(1, -1)  # (1, H, W, 2)
    frames = [video[0]]  # (1, H, W, C)
    for t in range(T - 1):
        grid = base_grid + m_bwd[..., t].movedim(1, -1).to(video.dtype)
        prev = frames[-1].movedim(-1, 1)  # (1, C, H, W)
        frames.append(grid_sample(prev, grid, align_corners=True).movedim(1, -1))
    return torch.stack(frames, dim=0)  # (T, 1, H, W, C)