import wandb
from torch import Tensor
import torch
from typing import Dict, Optional, Tuple
import numpy as np
import logging

from omegaconf import DictConfig, OmegaConf

def flatten_hydra_config(cfg : DictConfig) -> dict:
    out_dict = {}
    _flatten_hydra_config(cfg, out_dict)
    return out_dict

def _flatten_hydra_config(cfg : DictConfig, out_dict : dict, prefix : str = ""):
    new_prefix = prefix + "." if len(prefix) > 0 else ""
    for key, value in cfg.items():
        if isinstance(value, DictConfig):
            _flatten_hydra_config(value, out_dict, new_prefix + key)
        else:
            out_dict[new_prefix + key] = value


from pydantic import BaseModel
from prefect.context import get_run_context
from src.prefect.api_helper import get_flow_name_from_api
import uuid

class WandbParamsTask(BaseModel):
    flow_name : str
    flow_run_name : str
    flow_run_id : uuid.UUID
    name_aux : str = ""
    log : bool = False
    log_artifact: bool = False
    project: str = ""
    entity : str = ""
    code_dir : str = ""
    config : dict = {}

def gather_flow_infos_for_wandb():
    """
        Returns the wandb kwargs from the prefect flow context.
        This is used to log the wandb run in the flow.
    """
    run_context = get_run_context()
    assert run_context is not None, "No run context found. This function should only be called within a Prefect flow."
    
    flow_run = run_context.flow_run
    return WandbParamsTask(
        flow_name = get_flow_name_from_api(),
        flow_run_name = flow_run.name,
        flow_run_id = flow_run.id,
        name_aux = ""
    )

def wandb_kwargs_for_prefect_task(params : WandbParamsTask):
    """
        Returns the wandb kwargs for the prefect task.
        This is used to log the wandb run in the task.
    """

    run_context = get_run_context()
    assert run_context is not None, "No run context found. This function should only be called within a Prefect task."

    # task run has: name (Recon Task-5ba), task_key (recon_task-fa6af2dc), tags (['recon']), version (1.0), retries (0), flow run id, 
    task_run_name = run_context.task_run.name if run_context.task_run else None
    task_id = run_context.task_run.id if run_context.task_run else None
    task_tags = run_context.task_run.tags if run_context.task_run else None
    task_name = "-".join(task_run_name.split("-")[:-2])

    config = params.config
    config['flow_name'] = params.flow_name
    config['flow_run_name'] = params.flow_run_name
    config['flow_run_id'] = str(params.flow_run_id)
    config['name_aux'] = params.name_aux
    config['task_name'] = task_name
    config['task_run_name'] = task_run_name
    config['task_id'] = str(task_id)
    config['task_tags'] = task_tags

    return {
        'project': params.project,
        'entity': params.entity,
        'mode': 'online' if params.log else 'disabled',
        'settings': wandb.Settings(code_dir=params.code_dir),
        'group' : f"{params.flow_run_name}",
        'tags' : task_tags,
        'name': f"{task_run_name}_{params.name_aux}",
        'job_type' : task_name,
        'config' : params.config
    }


import torch
from typing import Optional
import math
import numpy as np
import matplotlib.pyplot as plt

from src.representations.base_representation import BaseRepresentation

def wandb_video_animation(
    mod : BaseRepresentation,
    spatial_mesh: torch.Tensor,
    time_mesh : torch.Tensor,
    frame_processor,
    nr_frames: int = 50,
    time_start: float = 0.0,
    time_end: Optional[float] = None,
    spatial_factor: float = 1.0,
    chunks=1,
    fps=1,
    skip_temp: bool = False,
    device: str = "cuda:0",
    save : bool = False,
    save_path : str = "",
    normalize_frames = True, # whether to normalize the frames,
    **kwargs
):
    
    print(f"Unused kwargs: {kwargs}")    

    spatial_mesh_flat = spatial_mesh.reshape(-1, 2)
    matrix_size = spatial_mesh.shape[:-1]

    if nr_frames > 0:
        time_start = float(max(time_start, time_mesh[0]))
        time_end = float(min(time_end if time_end is not None else time_mesh[-1], time_mesh[-1]))
        times = torch.linspace(time_start, time_end, nr_frames).to(device)
    else:
        times = time_mesh.to(device)

    x = mod.full_warp(timesteps=times).detach().cpu()
    return wandb.Video(frame_processor(x), fps=fps, format="gif")

import matplotlib.pyplot as plt
from typing import Optional
def wandb_quiver_animation(field, duration_in_s : float =4, scale=1, figsize=(5,5), use_pairwise_deltas=False, dpi=200):
    
    """
        Parameters:
        - deformation_field: Tensor of shape (T, H, W, 2) representing the deformation field.
        - duration_in_s: Duration of the animation in seconds.
        - scale: Scale factor for the arrows in the quiver plot.
    """

    if use_pairwise_deltas:
        flow_video_shifted = torch.zeros_like(field)
        flow_video_shifted[1:, ...] = field[:-1, ...]
        field = field - flow_video_shifted

    assert field.ndim == 4, "Video tensor must have shape (batch_size, num_frames, channels, height, width)"
    # input shape is (C, H, W, T), want (T, H, W, C)
    frames = []
    for i in range(len(field)):
        # Adjust figsize to match field dimensions (aspect ratio)
        h, w = field.shape[1:3]
        aspect_ratio = w / h
        base_height = figsize[1]
        fig, ax = plt.subplots(figsize=(base_height * aspect_ratio, base_height))
        q = ax.quiver(
            field[i, :, :, 0],  # x-coordinates (u)
            field[i, :, :, 1],  # y-coordinates (v)
            color='blue',  # color of the arrows
            scale=scale,  # scaling factor for the arrows
            angles='xy',  # specify that the angles are in xy coordinates
            scale_units='xy'  # scale the arrows by the data units
        )
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.set_title(f"time = {i * float(duration_in_s) / field.shape[0]:.3f} s")
        fig.set_dpi(dpi)
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frame = frame[..., :3]  # Drop alpha channel to keep (H, W, 3)
        frames.append(frame)
        plt.close(fig)
    frames_np = np.stack(frames, axis=0)  # (T, H, W, C)
    frames_np = np.transpose(frames_np, (0, 3, 1,2))  # (T, C, H, W)

    return wandb.Video(
        frames_np,
        fps=field.shape[0] // duration_in_s,
        format="gif",
        caption="Quiver Animation of Deformation Field")


import numpy as np
import torch
import wandb

def _hsv_to_rgb_torch(hsv: torch.Tensor) -> torch.Tensor:
    """
    Convert HSV to RGB (all in [0,1]) using torch ops.
    hsv: (..., 3) tensor with channels (h, s, v)
    returns: (..., 3) RGB in [0,1]
    """
    h, s, v = hsv.unbind(dim=-1)  # each (...,)

    h6 = h * 6.0
    i = torch.floor(h6).to(torch.int64)  # (...,)
    f = h6 - i

    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    i_mod = i % 6

    # Build rgb with piecewise selection
    r = torch.empty_like(v)
    g = torch.empty_like(v)
    b = torch.empty_like(v)

    mask = i_mod == 0
    r[mask], g[mask], b[mask] = v[mask], t[mask], p[mask]

    mask = i_mod == 1
    r[mask], g[mask], b[mask] = q[mask], v[mask], p[mask]

    mask = i_mod == 2
    r[mask], g[mask], b[mask] = p[mask], v[mask], t[mask]

    mask = i_mod == 3
    r[mask], g[mask], b[mask] = p[mask], q[mask], v[mask]

    mask = i_mod == 4
    r[mask], g[mask], b[mask] = t[mask], p[mask], v[mask]

    mask = i_mod == 5
    r[mask], g[mask], b[mask] = v[mask], p[mask], q[mask]

    rgb = torch.stack([r, g, b], dim=-1)
    return rgb.clamp(0.0, 1.0)


@torch.no_grad()
def flow_video_to_wandb_torch(
    flow_video: torch.Tensor,
    fps: float | None = None,
    total_duration: float | None = None,
    max_magnitude: float | None = None,
    percentile: float = 99.5,
    l_percentile: float = 0.5,
    base_reference: bool = False,
    use_pairwise_deltas: bool = False,
    numerical_stability_eps: float = 1e-8,
) -> wandb.Video:
    """
    Convert a flow video (F, 2, H, W) tensor into a wandb.Video using torch
    (no OpenCV). Produces an RGB optical-flow visualization.

    Args:
        flow_video: torch.Tensor, shape (F, 2, H, W), float.
                   Can be on GPU or CPU.
        fps: frames per second. Provide either fps or total_duration.
        total_duration: total duration in seconds.
        max_magnitude: optional fixed magnitude for normalization (recommended
                       if you want consistent color scaling across runs).
        percentile: robust scaling percentile if max_magnitude is None.
        base_reference: we use the first frame as a reference if true and calculate the deltas of the next frames to it.
        use_pairwise_deltas: if true, compute deltas between consecutive frames instead of to the base reference frame.

    Returns:
        wandb.Video (uint8 RGB)
    """
    assert flow_video.ndim == 4 and flow_video.shape[-1] == 2, \
        "Expected flow_video of shape (F, 2, H, W)."

    # input shape is now (F, H, W, 2)
    F = flow_video.shape[0]

    # the flow video has shape (F, H, W, 2)

    if base_reference:
        flow_video = flow_video - flow_video[0:1, ...]

    if use_pairwise_deltas:
        flow_video_shifted = torch.zeros_like(flow_video)
        flow_video_shifted[1:, ...] = flow_video[:-1, ...]
        flow_video = flow_video - flow_video_shifted

    # Determine fps
    if fps is None:
        assert total_duration is not None, "Provide fps or total_duration."
        fps = int(round(F / float(total_duration)))
    fps = int(fps)

    # Compute magnitude (global)
    u = flow_video[..., 0]  # (F, H, W)
    v = flow_video[..., 1]
    mag = torch.sqrt(u * u + v * v )  # (F, H, W)

    if max_magnitude is None:
        # torch.quantile works on GPU in recent PyTorch; fall back to CPU if needed
        try:
            max_magnitude_t = torch.quantile(mag.reshape(-1), percentile / 100.0)
            min_magnitude = torch.quantile(mag.reshape(-1), l_percentile / 100.0)
        except Exception:
            max_magnitude_t = torch.quantile(mag.detach().cpu().reshape(-1), percentile / 100.0).to(mag.device)
            min_magnitude = torch.quantile(mag.detach().cpu().reshape(-1), l_percentile / 100.0).to(mag.device)
        max_magnitude = float(max_magnitude_t.item()) + numerical_stability_eps
        min_magnitude = float(min_magnitude.item())
    else:
        max_magnitude = float(max_magnitude) + numerical_stability_eps
        min_magnitude = 0.0

    # Angle in [-pi, pi]
    ang = torch.atan2(v, u)  # (F, H, W)

    # HSV in [0,1]
    hue = (ang + torch.pi) / (2.0 * torch.pi)              # (F, H, W)
    mag[mag <= min_magnitude] = 0.0
    sat = torch.clamp(mag / max_magnitude, 0.0, 1.0)       # (F, H, W)
    val = torch.ones_like(sat)                             # (F, H, W)

    hsv = torch.stack([hue, sat, val], dim=-1)             # (F, H, W, 3)
    rgb = _hsv_to_rgb_torch(hsv)                           # (F, H, W, 3), float [0,1]

    # Convert to uint8 video on CPU for wandb
    video_uint8 = (rgb.movedim(-1,1) * 255.0).round().to(torch.uint8).cpu().numpy()  # (F, H, W, 3)
    return wandb.Video(video_uint8, fps=fps, format="gif")


@torch.no_grad()
def flow_overlay_video_to_wandb(
    flow_video: torch.Tensor,
    video: torch.Tensor,
    fps: int | None = None,
    total_duration: float | None = None,
    arrow_stride: int = 8,
    arrow_scale: float | None = None,
    arrow_alpha: float = 0.8,
    use_pairwise_deltas: bool = True,
    max_frames: int | None = None,
) -> wandb.Video:
    """
    Overlay a yellow velocity field on top of video frames. The flow input is a
    displacement field; if use_pairwise_deltas is True we visualize frame-to-frame
    velocity (flow[t] - flow[t-1]) to show where content will move next.

    Inputs:
        flow_video: (F, 2, H, W) or (F, H, W, 2) displacement field.
        video: (F, H, W), (F, 1, H, W), (F, 3, H, W), or (F, H, W, 3) frames.
        arrow_stride: sample every N pixels for quiver density.
        arrow_scale: matplotlib quiver scale (None lets matplotlib auto-scale).
        arrow_alpha: transparency for arrows.
        use_pairwise_deltas: visualize velocity as successive-frame deltas.
        max_frames: optional cap on frames to render (starting from first).
    """
    assert flow_video.ndim == 4, "flow_video must be 4D"
    assert video.ndim in (3, 4), "video must be (F,H,W) or (F,C,H,W)"

    # normalize flow to channel-first
    if flow_video.shape[1] == 2 and flow_video.shape[-1] != 2:
        flow_cf = flow_video
    elif flow_video.shape[-1] == 2 and flow_video.shape[1] != 2:
        flow_cf = flow_video.permute(0, 3, 1, 2)
    elif flow_video.shape[1] == 2 and flow_video.shape[-1] == 2:
        flow_cf = flow_video
    else:
        raise ValueError(f"Unexpected flow shape {tuple(flow_video.shape)}")

    F = flow_cf.shape[0]
    if max_frames is not None:
        F = min(F, max_frames)
        flow_cf = flow_cf[:F]
        video = video[:F]

    if use_pairwise_deltas:
        flow_shifted = torch.zeros_like(flow_cf)
        flow_shifted[1:] = flow_cf[:-1]
        flow_cf = flow_cf - flow_shifted

    if fps is None:
        assert total_duration is not None, "Provide fps or total_duration."
        fps = int(round(F / float(total_duration)))
    fps = int(fps)

    # video to (F, H, W, 3) in [0,1]
    if video.ndim == 3:  # (F, H, W)
        video_cf = video[:, None, ...]
    else:
        video_cf = video
    if video_cf.shape[1] == 1:
        video_cf = video_cf.repeat(1, 3, 1, 1)
    elif video_cf.shape[1] == 2:
        video_cf = video_cf.norm(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    elif video_cf.shape[1] == 3:
        pass
    elif video_cf.shape[-1] == 3 and video_cf.shape[1] != 3:
        video_cf = video_cf.permute(0, 3, 1, 2)
    elif video_cf.shape[-1] == 2 and video_cf.shape[1] != 2:
        video_cf = video_cf.permute(0, 3,1,2).norm(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    else:
        raise ValueError(f"Unsupported video shape {tuple(video.shape)}")

    vid_np = video_cf.detach().float()
    vid_np = vid_np - vid_np.min()
    vid_np = vid_np / (vid_np.max() + 1e-8)
    vid_np = vid_np.clamp(0, 1).cpu().numpy()  # (F,3,H,W)

    frames = []
    for t in range(F):
        u = flow_cf[t, 0].detach().cpu().numpy()
        v = flow_cf[t, 1].detach().cpu().numpy()

        h, w = u.shape
        y = np.arange(0, h, arrow_stride)
        x = np.arange(0, w, arrow_stride)
        xx, yy = np.meshgrid(x, y)
        uu = u[yy, xx]
        vv = v[yy, xx]

        fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
        ax.imshow(np.transpose(vid_np[t], (1, 2, 0)))
        ax.quiver(
            xx, yy, uu, vv,
            color=(1.0, 1.0, 0.0, arrow_alpha),
            angles='xy', scale_units='xy', scale=arrow_scale,
        )
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, w); ax.set_ylim(h, 0)

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frame = frame[..., :3]
        frames.append(frame)
        plt.close(fig)

    frames_np = np.stack(frames, axis=0)  # (F, H, W, 3)
    frames_np = np.transpose(frames_np, (0, 3, 1, 2))  # (F, 3, H, W)
    return wandb.Video(frames_np, fps=fps, format="gif")



    