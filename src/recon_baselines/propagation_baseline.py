from __future__ import annotations

import os
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
import numpy as np
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from omegaconf import OmegaConf
from torchvision.models.optical_flow._utils import grid_sample, make_coords_grid

from src.domain_specific_setups.natural_video_forward_models import LinearForwardModel
from src.recon_baselines.recon_baseline_resolver import (
    BaseReconBaseline,
    EvaluatorFunction,
    resolve_recon_baseline,
)
from src.regularizers.raft_based_regularizers import (
    estimate_flow_for_video,
    load_raft_model,
)
from src.tasks.train_probabilistic_coupling_model import LearntCouplingModel


def _as_broadcastable(val, device):
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


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    try:
        return cfg[key] if key in cfg else default
    except Exception:
        return getattr(cfg, key, default)


def _to_dict(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    return OmegaConf.to_container(cfg, resolve=True)


def _measurement_to_flow_video(measurement: torch.Tensor, upsample_factor: int = 4) -> torch.Tensor:
    # input: (T, H, W, C); upsampled since RAFT needs spatial dims >= 128
    video = measurement.movedim(-1, 1)  # (T, C, H, W)
    if video.shape[1] == 2:
        video = torch.linalg.norm(video, dim=1, keepdim=True)
    if upsample_factor != 1:
        video = torch.nn.functional.interpolate(video, scale_factor=upsample_factor, mode="nearest")
    return video


def _fb_consistency_confidence(
    motion_step: torch.Tensor,
    motion_step_opposite: torch.Tensor,
    grid: torch.Tensor,
    tau_pix: float,
    conf_floor: float,
) -> torch.Tensor:
    """Forward-backward flow consistency confidence, in [conf_floor, 1].

    motion_step / motion_step_opposite : (B, 2, H, W)
    grid                               : (B, H, W, 2) = base_grid + motion_step
    Returns (B, H, W).
    """
    flow_opp_warped = grid_sample(motion_step_opposite, grid, align_corners=True, mode="bilinear")
    fb_error = torch.linalg.norm(motion_step + flow_opp_warped, dim=1)  # (B, H, W)
    return torch.exp(-fb_error.square() / (2.0 * tau_pix * tau_pix)) * (1.0 - conf_floor) + conf_floor


def _residual_norm(diff: torch.Tensor, loss_norm: str, charbonnier_eps: float = 1e-3) -> torch.Tensor:
    if loss_norm == "l1":
        return diff.abs().mean()
    if loss_norm == "l2":
        return diff.square().mean()
    if loss_norm == "charbonnier":
        return torch.sqrt(diff.square() + charbonnier_eps * charbonnier_eps).mean()
    raise ValueError(f"Unsupported guidance_loss_norm: {loss_norm}")


def _resize_flow(flow: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    # flow: (T-1, 2, H, W)
    h_old, w_old = flow.shape[-2:]
    if h_old == target_h and w_old == target_w:
        return flow
    scale_y = float(target_h) / float(h_old)
    scale_x = float(target_w) / float(w_old)
    flow_rs = F.interpolate(flow, size=(target_h, target_w), mode="bilinear", align_corners=True)
    flow_rs[:, 0] = flow_rs[:, 0] * scale_x
    flow_rs[:, 1] = flow_rs[:, 1] * scale_y
    return flow_rs


def _estimate_pairwise_flows(measurement: torch.Tensor, h: int, w: int, device: str):
    """RAFT forward (t -> t+1) and backward (t+1 -> t) flows of every consecutive frame pair of
    the measurement, resized to (h, w). Same estimation as CouplingPropagationMethod.propagate().
    Returns (flow_fwd, flow_bwd), each (T-1, 2, h, w)."""
    flow_video = _measurement_to_flow_video(measurement)
    raft_model = load_raft_model(device=device)
    # run RAFT on the full (upsampled) frame
    _raft_target_size = max(flow_video.shape[-2], flow_video.shape[-1])
    flow_fwd = estimate_flow_for_video(
        raft_model=raft_model,
        video=flow_video,
        enable_upsample_flow=False,
        forward_flow=True,
        interpolate_to_target_size=_raft_target_size,
    )
    flow_bwd = estimate_flow_for_video(
        raft_model=raft_model,
        video=flow_video,
        enable_upsample_flow=False,
        forward_flow=False,
        interpolate_to_target_size=_raft_target_size,
    )
    return _resize_flow(flow_fwd, h, w), _resize_flow(flow_bwd, h, w)


# Single-step next-frame prediction targets: "previous" = x_t from x_{t+1} (r0), "next" = x_{t+1} from x_t (r1).
# Must match NaturalVideoSetup(next_frame_prediction_target=...) so only predicted frames are scored.
SINGLE_STEP_TARGETS = ("previous", "next")


def _check_single_step_target(target: str) -> None:
    if target not in SINGLE_STEP_TARGETS:
        raise ValueError(f"single_step_target must be one of {SINGLE_STEP_TARGETS}, got {target!r}")


class BasePropagationMethod(ABC):
    @abstractmethod
    def propagate(
        self,
        first_frame_chw: torch.Tensor,
        measurement: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        # returns (T, C, H, W)
        raise NotImplementedError()

    def predict_single_step(
        self,
        anchor_video_chw: torch.Tensor,
        measurement: torch.Tensor,
        device: str,
        target: str = "previous",
    ) -> torch.Tensor:
        # anchor_video_chw: (T, C, H, W) true frames; returns (T, C, H, W), see SINGLE_STEP_TARGETS
        raise NotImplementedError(f"{type(self).__name__} does not support single-step prediction")


class WarpPropagationMethod(BasePropagationMethod):
    def __init__(
        self,
        interpolation_mode: str = "bilinear",
        align_corners: bool = True,
    ):
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners

    @torch.no_grad()
    def propagate(
        self,
        first_frame_chw: torch.Tensor,
        measurement: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        flow_video = _measurement_to_flow_video(measurement)
        raft_model = load_raft_model(device=device)

        # run RAFT on the full (upsampled) frame
        _raft_target_size = max(flow_video.shape[-2], flow_video.shape[-1])

        # backward pairwise flow (t+1 -> t), consistent with existing noise warping code paths
        flow_bwd = estimate_flow_for_video(
            raft_model=raft_model,
            video=flow_video,
            enable_upsample_flow=False,
            forward_flow=False,
            interpolate_to_target_size=_raft_target_size,
        )

        h, w = first_frame_chw.shape[-2:]
        flow_bwd = _resize_flow(flow_bwd, h, w)
        t = measurement.shape[0]

        flow_full = torch.cat(
            [torch.zeros((1, 2, h, w), device=first_frame_chw.device), flow_bwd],
            dim=0,
        )
        displacement = torch.cumsum(flow_full, dim=0)

        base_grid = make_coords_grid(batch_size=t, h=h, w=w, device=device).movedim(1, -1)
        grid = base_grid + displacement.movedim(1, -1)

        x0 = first_frame_chw[None]
        return grid_sample(
            x0.expand(t, -1, -1, -1),
            grid,
            align_corners=self.align_corners,
            mode=self.interpolation_mode,
        )

    @torch.no_grad()
    def predict_single_step(
        self,
        anchor_video_chw: torch.Tensor,
        measurement: torch.Tensor,
        device: str,
        target: str = "previous",
    ) -> torch.Tensor:
        _check_single_step_target(target)
        t, _, h, w = anchor_video_chw.shape
        flow_fwd, flow_bwd = _estimate_pairwise_flows(measurement, h, w, device)
        base_grid = make_coords_grid(batch_size=t - 1, h=h, w=w, device=device).movedim(1, -1)
        if target == "previous":
            # x_t ~ W(x_{t+1}, m^f_t), t = 0..T-2; the last frame is the anchor itself
            warped = grid_sample(anchor_video_chw[1:], base_grid + flow_fwd.movedim(1, -1),
                                 align_corners=self.align_corners, mode=self.interpolation_mode)
            return torch.cat([warped, anchor_video_chw[-1:]], dim=0)
        # x_{t+1} ~ W(x_t, m^b_t), t = 0..T-2; the first frame is the anchor itself
        warped = grid_sample(anchor_video_chw[:-1], base_grid + flow_bwd.movedim(1, -1),
                             align_corners=self.align_corners, mode=self.interpolation_mode)
        return torch.cat([anchor_video_chw[:1], warped], dim=0)


class CouplingPropagationMethod(BasePropagationMethod):
    def __init__(
        self,
        path_resolver,
        device: str,
        pretrained_model_dir=None,
        pretrained_model_path=None,
        input_normalization: str = "auto",
        residual_branch: str = "auto",
        use_final_model: bool = True,
        use_ema: bool = True,
        use_epoch: Optional[int] = None,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        scheduler: Optional[Dict[str, Any]] = None,
        frame0_channel_std_floor: float = 1e-3,
        # --- motion-consistency posterior sampling (guidance) ---
        guidance_weight: float = 0.0,
        guidance_mode: str = "dps",
        guidance_loss_norm: str = "l1",
        guidance_weight_fwd_relative: float = 1.0,
        guidance_weight_bwd_relative: float = 0.0,
        guidance_enable_occlusion_masking: bool = False,
        guidance_tau_pix: Optional[float] = None,
        guidance_conf_floor: Optional[float] = None,
        guidance_normalize_gradient: bool = True,
        # number of sampled residuals averaged per step (>1 approximates the posterior mean)
        num_samples: int = 1,
    ):
        if int(num_samples) < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        self.num_samples = int(num_samples)
        self.path_resolver = path_resolver
        self.device = device
        self.use_final_model = use_final_model
        self.use_ema = use_ema
        self.use_epoch = use_epoch
        self.input_normalization = input_normalization
        self.residual_branch = residual_branch
        self.num_inference_steps = int(num_inference_steps)
        self.eta = float(eta)
        self.scheduler_cfg = scheduler or {}
        self.frame0_channel_std_floor = float(frame0_channel_std_floor)
        self.available_residual_branches: tuple[str, ...] = ("r1",)
        self.selected_residual_branch: str = "r1"
        self.requires_last_frame: bool = False

        # guidance_weight > 0: approximate posterior sampling from
        #     p(r_t | cond) * exp(-w * || conf ⊙ (x_t - warp(x_{t-1})) ||)  (+ optional backward term)
        if guidance_mode not in ("dps", "x0"):
            raise ValueError(f"guidance_mode must be 'dps' or 'x0', got {guidance_mode!r}")
        self.guidance_weight = float(guidance_weight)
        self.guidance_mode = guidance_mode
        self.guidance_loss_norm = guidance_loss_norm
        self.guidance_weight_fwd_relative = float(guidance_weight_fwd_relative)
        self.guidance_weight_bwd_relative = float(guidance_weight_bwd_relative)
        self.guidance_enable_occlusion_masking = bool(guidance_enable_occlusion_masking)
        self.guidance_tau_pix = guidance_tau_pix
        self.guidance_conf_floor = guidance_conf_floor
        self.guidance_normalize_gradient = bool(guidance_normalize_gradient)

        if self.use_final_model:
            subfolder = "model_final"
        elif self.use_ema:
            if self.use_epoch is None:
                raise ValueError("use_epoch must be provided when use_final_model=False and use_ema=True")
            subfolder = f"ema_model_epoch_{int(self.use_epoch)}"
        else:
            if self.use_epoch is None:
                raise ValueError("use_epoch must be provided when use_final_model=False and use_ema=False")
            subfolder = f"model_epoch_{int(self.use_epoch)}"

        if pretrained_model_path is not None:
            self._load_from_storage_path(pretrained_model_path, subfolder)
        elif pretrained_model_dir is not None:
            self._load_from_local_dir(path_resolver(pretrained_model_dir), subfolder)
        else:
            raise ValueError(
                "CouplingPropagationMethod requires either pretrained_model_path "
                "(StoragePath from a training task) or pretrained_model_dir (local path)."
            )

    def _load_from_local_dir(self, model_base_dir: str, subfolder: str) -> None:
        """Load model, scheduler and coupling config from a local directory."""
        from src.utils.video_normalization import VideoNormalization
        import yaml

        # Probabilistic coupling trainer saves everything flat at the root.
        model_path = os.path.join(model_base_dir, subfolder)
        self.model = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path=model_path).to(self.device)
        self.model.eval()
        noise_scheduler = DDPMScheduler.from_pretrained(os.path.join(model_base_dir, "noise_scheduler"))
        self.sample_scheduler = DDIMScheduler.from_config(noise_scheduler.config, **self.scheduler_cfg)
        coupling_cfg = OmegaConf.load(
            os.path.join(model_base_dir, "coupling_config", "model_cfg.yaml")
        )
        self.coupling_model = LearntCouplingModel(**_to_dict(coupling_cfg))
        self._resolve_residual_branch_from_model()
        self.zero_condition = torch.zeros((1, 1, 1), device=self.device)

        # Load normalization from model dir, honouring any explicit override.
        _norm_dir = os.path.join(model_base_dir, "normalization")
        if self.input_normalization != "auto":
            self.video_normalization: VideoNormalization = VideoNormalization(mode=self.input_normalization)
        elif os.path.isdir(_norm_dir):
            self.video_normalization = VideoNormalization.from_dir(_norm_dir)
        else:
            self.video_normalization = VideoNormalization(mode="none")

        # Load residual scaling from model dir (always at root).
        _residual_scaling_path = os.path.join(model_base_dir, "residual_scaling.yaml")
        self.residual_std = 1.0
        self.residual_mean = 0.0
        if os.path.exists(_residual_scaling_path):
            try:
                with open(_residual_scaling_path, "r") as f:
                    residual_cfg = yaml.safe_load(f)
                    _rs = residual_cfg.get("residual_std", 1.0)
                    _rm = residual_cfg.get("residual_mean", 0.0)
                    self.residual_std = _rs if isinstance(_rs, list) else float(_rs)
                    self.residual_mean = _rm if isinstance(_rm, list) else float(_rm)
                    print(f"[Propagation] Loaded residual scaling: per_channel={isinstance(self.residual_std, list)}, std={self.residual_std}, mean={self.residual_mean}")
            except Exception as e:
                print(f"[Propagation] WARNING: Failed to load residual scaling: {e}")
        else:
            print(f"[Propagation] WARNING: No residual_scaling.yaml found at {_residual_scaling_path}")

    def _load_from_storage_path(self, pretrained_model_path: Any, subfolder: str) -> None:
        """Download from remote StoragePath (training task result) and load."""
        import tempfile
        import yaml
        from functools import partial as functools_partial
        from src.prefect.setup import load_file_system, download_directory_to_temp_on_enter
        from src.regularizers.diffusers_cpl_regularizers import get_coupling_config
        from src.utils.video_normalization import VideoNormalization

        with tempfile.TemporaryDirectory() as tmp_dir:
            filesystem = load_file_system(pretrained_model_path.storage_settings)
            download_mgr = functools_partial(
                download_directory_to_temp_on_enter,
                filesystem=filesystem,
                temp_folder=tmp_dir,
                fs_folder=pretrained_model_path.storage_path,
            )

            # Probabilistic coupling trainer saves everything flat at root.
            with download_mgr(ex_subfolder=subfolder):
                self.model = UNet2DConditionModel.from_pretrained(
                    pretrained_model_name_or_path="./"
                ).to(self.device)
            self.model.eval()

            with download_mgr(ex_subfolder="noise_scheduler"):
                noise_scheduler = DDPMScheduler.from_pretrained("./")
            self.sample_scheduler = DDIMScheduler.from_config(noise_scheduler.config, **self.scheduler_cfg)

            coupling_cfg = get_coupling_config(download_mgr)

            if self.input_normalization != "auto":
                self.video_normalization: VideoNormalization = VideoNormalization(mode=self.input_normalization)
            else:
                try:
                    with download_mgr(ex_subfolder="normalization"):
                        self.video_normalization = VideoNormalization.from_dir("./")
                except Exception:
                    self.video_normalization = VideoNormalization(mode="none")

            # read directly: ex_subfolder="." gives a key ending in "/." which MinIO rejects
            self.residual_std = 1.0
            self.residual_mean = 0.0
            residual_scaling_key = os.path.join(pretrained_model_path.storage_path, "residual_scaling.yaml")
            try:
                residual_cfg = yaml.safe_load(filesystem.read_path(residual_scaling_key))
                _rs = residual_cfg.get("residual_std", 1.0)
                _rm = residual_cfg.get("residual_mean", 0.0)
                self.residual_std = _rs if isinstance(_rs, list) else float(_rs)
                self.residual_mean = _rm if isinstance(_rm, list) else float(_rm)
                print(f"[Propagation] Loaded residual scaling: per_channel={isinstance(self.residual_std, list)}, std={self.residual_std}, mean={self.residual_mean}")
            except Exception as e:
                print(f"[Propagation] WARNING: Failed to load residual scaling from storage: {e}")

        self.coupling_model = LearntCouplingModel(**_to_dict(coupling_cfg))
        self._resolve_residual_branch_from_model()
        self.zero_condition = torch.zeros((1, 1, 1), device=self.device)

    def _resolve_residual_branch_from_model(self) -> None:
        """Determine available residual branches from the model config and select one (auto prefers r1)."""
        # Support new training_branches string attr and legacy learn_fwd_bwd bool
        _tb = getattr(self.coupling_model, "training_branches", None)
        if _tb is not None:
            if _tb == "both":
                available = ("r0", "r1")
            elif _tb in ("r0", "r1"):
                available = (_tb,)
            else:
                available = ("r0", "r1")  # unknown value → treat as both
        else:
            # Legacy checkpoint: fall back to learn_fwd_bwd bool
            _lfb = getattr(self.coupling_model, "learn_fwd_bwd", True)
            available = ("r0", "r1") if bool(_lfb) else ("r0",)

        requested = str(self.residual_branch)
        if requested == "auto":
            selected = "r1" if "r1" in available else available[0]
        elif requested in ("r0", "r1"):
            if requested not in available:
                raise ValueError(
                    f"Requested residual_branch={requested!r}, but available branches in checkpoint are {available}."
                )
            selected = requested
        else:
            raise ValueError("residual_branch must be one of {'auto', 'r0', 'r1'}")

        self.available_residual_branches = available
        self.selected_residual_branch = selected
        self.requires_last_frame = selected == "r0"
        print(
            "[Propagation] Residual branch selection "
            f"requested={requested!r} available={available} selected={selected!r}"
        )

    def _build_guidance_context(
        self,
        anchor_model: torch.Tensor,
        warped: torch.Tensor,
        motion_step: torch.Tensor,
        motion_step_opposite: torch.Tensor,
        base_grid: torch.Tensor,
        residual_mean,
        residual_std,
    ) -> Optional[Dict[str, Any]]:
        """Everything the motion-consistency likelihood needs for one propagation step.

        anchor_model : (1, C, H, W) previous frame x_{t-1}, in model (normalized) space.
        warped       : (1, C, H, W) anchor warped by motion_step — the frame the propagated
                       frame is expected to stay close to.
        """
        if self.guidance_weight == 0.0:
            return None

        tau_pix = self.guidance_tau_pix
        if tau_pix is None:
            tau_pix = float(getattr(self.coupling_model, "confidence_tau_pix", 1.0))
        conf_floor = self.guidance_conf_floor
        if conf_floor is None:
            conf_floor = float(getattr(self.coupling_model, "confidence_floor", 0.0))

        grid_primary = base_grid + motion_step.movedim(1, -1)
        # maps the propagated frame back onto the anchor's timestamp
        grid_opposite = base_grid + motion_step_opposite.movedim(1, -1)

        conf_fwd = conf_bwd = None
        if self.guidance_enable_occlusion_masking:
            conf_fwd = _fb_consistency_confidence(
                motion_step, motion_step_opposite, grid_primary, tau_pix, conf_floor
            )[:, None]  # (1, 1, H, W)
            conf_bwd = _fb_consistency_confidence(
                motion_step_opposite, motion_step, grid_opposite, tau_pix, conf_floor
            )[:, None]

        return {
            "anchor": anchor_model,
            "warped": warped,
            "grid_opposite": grid_opposite,
            "conf_fwd": conf_fwd,
            "conf_bwd": conf_bwd,
            "residual_mean": residual_mean,
            "residual_std": residual_std,
        }

    def _frame_from_scaled_residual(self, residual_scaled: torch.Tensor, guidance: Dict[str, Any]) -> torch.Tensor:
        residual = residual_scaled * guidance["residual_std"] + guidance["residual_mean"]
        if getattr(self.coupling_model, "prediction_target", "residual") == "full_frame":
            return residual
        return guidance["warped"] + residual

    def _motion_consistency_loss(self, residual_scaled: torch.Tensor, guidance: Dict[str, Any]) -> torch.Tensor:
        """|| conf ⊙ (x_t - warp(x_{t-1})) || (+ optional backward round-trip term)."""
        frame = self._frame_from_scaled_residual(residual_scaled, guidance)

        w_fwd = self.guidance_weight_fwd_relative
        w_bwd = self.guidance_weight_bwd_relative
        total_w = w_fwd + w_bwd
        if total_w <= 0.0:
            raise ValueError("guidance_weight_fwd_relative + guidance_weight_bwd_relative must be > 0")
        w_fwd, w_bwd = w_fwd / total_w, w_bwd / total_w

        loss = frame.new_zeros(())
        if w_fwd > 0.0:
            diff = frame - guidance["warped"]
            if guidance["conf_fwd"] is not None:
                diff = diff * guidance["conf_fwd"]
            loss = loss + w_fwd * _residual_norm(diff, self.guidance_loss_norm)
        if w_bwd > 0.0:
            back_warped = grid_sample(frame, guidance["grid_opposite"], align_corners=True)
            diff = back_warped - guidance["anchor"]
            if guidance["conf_bwd"] is not None:
                diff = diff * guidance["conf_bwd"]
            loss = loss + w_bwd * _residual_norm(diff, self.guidance_loss_norm)
        return loss

    def _predict_residual_x0(self, sample: torch.Tensor, model_out: torch.Tensor, t) -> torch.Tensor:
        """Tweedie estimate of the clean residual from the current noisy iterate."""
        alpha_prod_t = self.sample_scheduler.alphas_cumprod.to(sample.device)[int(t)]
        sqrt_ap = alpha_prod_t.sqrt()
        sqrt_1map = (1.0 - alpha_prod_t).sqrt()
        prediction_type = getattr(self.sample_scheduler.config, "prediction_type", "epsilon")
        if prediction_type == "epsilon":
            return (sample - sqrt_1map * model_out) / sqrt_ap
        if prediction_type == "sample":
            return model_out
        if prediction_type == "v_prediction":
            return sqrt_ap * sample - sqrt_1map * model_out
        raise ValueError(f"Unsupported scheduler prediction_type: {prediction_type}")

    def _sample_residual(
        self,
        condition: torch.Tensor,
        channels: int,
        h: int,
        w: int,
        guidance: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        sample = torch.randn((1, channels, h, w), device=self.device)
        self.sample_scheduler.set_timesteps(self.num_inference_steps, device=self.device)

        # disable clipping of residuals; item assignment on the FrozenDict config has no effect
        self.sample_scheduler.register_to_config(clip_sample=False)

        guided = guidance is not None and self.guidance_weight != 0.0

        for t in self.sample_scheduler.timesteps:
            if not guided:
                pred = self.model(
                    sample=torch.cat([sample, condition], dim=1),
                    timestep=t,
                    encoder_hidden_states=self.zero_condition,
                ).sample
                sample = self.sample_scheduler.step(pred, t, sample, eta=self.eta).prev_sample
                continue

            if self.guidance_mode == "dps":
                # full DPS: differentiate the likelihood through the denoiser
                with torch.enable_grad():
                    sample_in = sample.detach().requires_grad_(True)
                    pred = self.model(
                        sample=torch.cat([sample_in, condition], dim=1),
                        timestep=t,
                        encoder_hidden_states=self.zero_condition,
                    ).sample
                    loss = self._motion_consistency_loss(
                        self._predict_residual_x0(sample_in, pred, t), guidance
                    )
                    grad = torch.autograd.grad(loss, sample_in)[0]
                pred = pred.detach()
            else:
                # "x0": denoiser output treated as constant, gradient only through the Tweedie map
                pred = self.model(
                    sample=torch.cat([sample, condition], dim=1),
                    timestep=t,
                    encoder_hidden_states=self.zero_condition,
                ).sample
                with torch.enable_grad():
                    sample_in = sample.detach().requires_grad_(True)
                    loss = self._motion_consistency_loss(
                        self._predict_residual_x0(sample_in, pred.detach(), t), guidance
                    )
                    grad = torch.autograd.grad(loss, sample_in)[0]

            sample = self.sample_scheduler.step(pred, t, sample.detach(), eta=self.eta).prev_sample
            if self.guidance_normalize_gradient:
                # unit-RMS gradient so guidance_weight is comparable across configs
                grad = grad / (grad.square().mean().sqrt() + 1e-12)
            sample = (sample - self.guidance_weight * grad).detach()

        return sample.detach()

    def _sample_residual_mean(
        self,
        condition: torch.Tensor,
        channels: int,
        h: int,
        w: int,
        guidance: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Mean of self.num_samples independent _sample_residual() draws (sequential, so
        num_samples=1 consumes the RNG exactly like a single call)."""
        total = self._sample_residual(condition=condition, channels=channels, h=h, w=w, guidance=guidance)
        for _ in range(self.num_samples - 1):
            total = total + self._sample_residual(condition=condition, channels=channels, h=h, w=w, guidance=guidance)
        return total / self.num_samples

    def _predict_from_anchor(
        self,
        anchor: torch.Tensor,
        motion_step: torch.Tensor,
        motion_step_opposite: torch.Tensor,
        clip_norm,
        base_grid: torch.Tensor,
        residual_mean,
        residual_std,
    ) -> torch.Tensor:
        """One propagation step: the frame W(anchor, motion_step) + sampled residual (or the
        sampled full frame), with anchor (1, C, H, W) in the original domain."""
        _, channels, h, w = anchor.shape
        anchor_model = clip_norm.normalize_fchw(anchor)
        condition, warped = self.coupling_model.build_condition_from_anchor_and_motion(
            frame_anchor=anchor_model,
            motion_step=motion_step,
            motion_step_opposite=motion_step_opposite,
        )
        guidance = self._build_guidance_context(
            anchor_model=anchor_model,
            warped=warped,
            motion_step=motion_step,
            motion_step_opposite=motion_step_opposite,
            base_grid=base_grid,
            residual_mean=residual_mean,
            residual_std=residual_std,
        )
        residual_scaled = self._sample_residual_mean(
            condition=condition, channels=channels, h=h, w=w, guidance=guidance
        )
        residual = residual_scaled * residual_std + residual_mean
        if getattr(self.coupling_model, 'prediction_target', 'residual') == 'full_frame':
            return clip_norm.denormalize_fchw(residual)
        return clip_norm.denormalize_fchw(residual + warped)

    @torch.no_grad()
    def predict_single_step(
        self,
        anchor_video_chw: torch.Tensor,
        measurement: torch.Tensor,
        device: str,
        target: str = "previous",
    ) -> torch.Tensor:
        # branch is determined by the prediction direction, not by residual_branch
        _check_single_step_target(target)
        branch = "r0" if target == "previous" else "r1"
        if branch not in self.available_residual_branches:
            raise ValueError(
                f"single_step_target={target!r} needs residual branch {branch!r}, but the checkpoint "
                f"only provides {self.available_residual_branches}."
            )
        t, _, h, w = anchor_video_chw.shape
        flow_fwd, flow_bwd = _estimate_pairwise_flows(measurement, h, w, device)
        _rm = _as_broadcastable(self.residual_mean, self.device)
        _rs = _as_broadcastable(self.residual_std, self.device)
        base_grid = make_coords_grid(batch_size=1, h=h, w=w, device=str(anchor_video_chw.device)).movedim(1, -1)

        from src.utils.video_normalization import VideoNormalization
        predicted = []
        for i in range(t - 1):
            anchor = anchor_video_chw[i + 1 : i + 2] if target == "previous" else anchor_video_chw[i : i + 1]
            # per-anchor normalization stats, as propagate() fits them on its (single) anchor
            clip_norm = VideoNormalization.fit_fchw(
                mode=self.video_normalization.mode,
                video_fchw=anchor,
                std_floor=self.frame0_channel_std_floor,
            )
            motion_step, motion_step_opposite = (
                (flow_fwd[i : i + 1], flow_bwd[i : i + 1]) if target == "previous"
                else (flow_bwd[i : i + 1], flow_fwd[i : i + 1])
            )
            predicted.append(self._predict_from_anchor(
                anchor, motion_step, motion_step_opposite, clip_norm, base_grid, _rm, _rs
            ))
        if target == "previous":
            return torch.cat(predicted + [anchor_video_chw[-1:]], dim=0)
        return torch.cat([anchor_video_chw[:1]] + predicted, dim=0)

    @torch.no_grad()
    def propagate(
        self,
        first_frame_chw: torch.Tensor,
        measurement: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        flow_video = _measurement_to_flow_video(measurement)
        raft_model = load_raft_model(device=device)

        # run RAFT on the full (upsampled) frame
        _raft_target_size = max(flow_video.shape[-2], flow_video.shape[-1])

        flow_fwd = estimate_flow_for_video(
            raft_model=raft_model,
            video=flow_video,
            enable_upsample_flow=False,
            forward_flow=True,
            interpolate_to_target_size=_raft_target_size,
        )
        flow_bwd = estimate_flow_for_video(
            raft_model=raft_model,
            video=flow_video,
            enable_upsample_flow=False,
            forward_flow=False,
            interpolate_to_target_size=_raft_target_size,
        )

        h, w = first_frame_chw.shape[-2:]
        flow_fwd = _resize_flow(flow_fwd, h, w)
        flow_bwd = _resize_flow(flow_bwd, h, w)

        anchor = first_frame_chw[None]
        channels = int(anchor.shape[1])
        t = measurement.shape[0]

        # Fit per-clip normalization stats from the first frame.
        from src.utils.video_normalization import VideoNormalization
        clip_norm = VideoNormalization.fit_fchw(
            mode=self.video_normalization.mode,
            video_fchw=anchor,
            std_floor=self.frame0_channel_std_floor,
        )

        _rm = _as_broadcastable(self.residual_mean, self.device)
        _rs = _as_broadcastable(self.residual_std, self.device)

        # For r1: motion_step = flow_bwd (warp x_t → x_{t+1}), opposite = flow_fwd
        # For r0: motion_step = flow_fwd (warp x_{t+1} → x_t), opposite = flow_bwd
        base_grid = make_coords_grid(batch_size=1, h=h, w=w, device=str(anchor.device)).movedim(1, -1)

        if self.selected_residual_branch == "r1":
            frames = [anchor]
            x_t = anchor
            for i in range(t - 1):
                x_t_model = clip_norm.normalize_fchw(x_t)
                condition, warped_xt = self.coupling_model.build_condition_from_anchor_and_motion(
                    frame_anchor=x_t_model,
                    motion_step=flow_bwd[i : i + 1],
                    motion_step_opposite=flow_fwd[i : i + 1],
                )
                guidance = self._build_guidance_context(
                    anchor_model=x_t_model,
                    warped=warped_xt,
                    motion_step=flow_bwd[i : i + 1],
                    motion_step_opposite=flow_fwd[i : i + 1],
                    base_grid=base_grid,
                    residual_mean=_rm,
                    residual_std=_rs,
                )
                residual_scaled = self._sample_residual_mean(
                    condition=condition, channels=channels, h=h, w=w, guidance=guidance
                )
                residual = residual_scaled * _rs + _rm
                if getattr(self.coupling_model, 'prediction_target', 'residual') == 'full_frame':
                    # Model predicts the full frame directly.
                    x_t_model = residual
                else:
                    x_t_model = residual + warped_xt
                x_t = clip_norm.denormalize_fchw(x_t_model)
                frames.append(x_t)
            return torch.cat(frames, dim=0)

        if self.selected_residual_branch == "r0":
            # For r0 we propagate backward in time from the anchor (which must be the last frame).
            frames_reverse = [anchor]
            x_t1 = anchor
            for i in reversed(range(t - 1)):
                x_t1_model = clip_norm.normalize_fchw(x_t1)
                condition, warped_xt1 = self.coupling_model.build_condition_from_anchor_and_motion(
                    frame_anchor=x_t1_model,
                    motion_step=flow_fwd[i : i + 1],
                    motion_step_opposite=flow_bwd[i : i + 1],
                )
                guidance = self._build_guidance_context(
                    anchor_model=x_t1_model,
                    warped=warped_xt1,
                    motion_step=flow_fwd[i : i + 1],
                    motion_step_opposite=flow_bwd[i : i + 1],
                    base_grid=base_grid,
                    residual_mean=_rm,
                    residual_std=_rs,
                )
                residual_scaled = self._sample_residual_mean(
                    condition=condition, channels=channels, h=h, w=w, guidance=guidance
                )
                residual = residual_scaled * _rs + _rm
                if getattr(self.coupling_model, 'prediction_target', 'residual') == 'full_frame':
                    # Model predicts the full frame directly.
                    x_t_model = residual
                else:
                    x_t_model = residual + warped_xt1
                x_t = clip_norm.denormalize_fchw(x_t_model)
                frames_reverse.append(x_t)
                x_t1 = x_t
            frames_forward = list(reversed(frames_reverse))
            return torch.cat(frames_forward, dim=0)

        raise ValueError(f"Unsupported selected residual branch: {self.selected_residual_branch}")


class PropagationBaseline(BaseReconBaseline):
    def __init__(
        self,
        path_resolver,
        device: str,
        initializer,
        propagation,
        nfe: Optional[float] = None,
        pretrained_model_path=None,
        # "rollout": autoregressive propagation from one anchor frame
        # "single_step": every frame predicted from its true neighbour (see SINGLE_STEP_TARGETS)
        mode: str = "rollout",
        single_step_target: str = "previous",
    ):
        super().__init__()
        self.path_resolver = path_resolver
        self.device = device
        if mode not in ("rollout", "single_step"):
            raise ValueError(f"mode must be 'rollout' or 'single_step', got {mode!r}")
        _check_single_step_target(single_step_target)
        if mode == "single_step" and _cfg_get(initializer, "name") not in {None, "", "none", "null"}:
            raise ValueError("mode='single_step' anchors on the measurement's own frames; initializer must be null")
        self.mode = mode
        self.single_step_target = single_step_target

        self.initializer: Optional[BaseReconBaseline] = None
        init_name = _cfg_get(initializer, "name") if initializer is not None else None
        init_name_normalized = str(init_name).lower() if init_name is not None else None
        if init_name_normalized not in {None, "", "none", "null"}:
            init_params = _to_dict(_cfg_get(initializer, "params", {}))

            # Re-wrap nested dicts as DictConfig so sub-initializers can use attribute access
            # (e.g. I2SBBaseline does self.opt.device = device)
            init_params = {
                k: OmegaConf.create(v) if isinstance(v, dict) else v
                for k, v in init_params.items()
            }

            # Pass nfe to initializer if provided and not already in params
            if nfe is not None and "nfe" not in init_params:
                init_params["nfe"] = nfe

            self.initializer = resolve_recon_baseline(
                name=str(init_name),
                path_resolver=path_resolver,
                device=device,
                **init_params,
            )

        prop_name = _cfg_get(propagation, "name")
        prop_params = _to_dict(_cfg_get(propagation, "params", {}))
        if prop_name == "warp":
            self.propagator: BasePropagationMethod = WarpPropagationMethod(**prop_params)
        elif prop_name == "coupling":
            self.propagator = CouplingPropagationMethod(
                path_resolver=path_resolver,
                device=device,
                pretrained_model_path=pretrained_model_path,
                **prop_params,
            )
        else:
            raise ValueError(f"Unknown propagation method: {prop_name}")

    def run(
        self,
        measurement: torch.Tensor,
        forward_model: LinearForwardModel,
        evaluator: EvaluatorFunction | partial[Dict[Any, Any]],
        **kwargs,
    ) -> torch.Tensor:
        del evaluator
        if self.mode == "single_step":
            anchor_video = forward_model.pseudoinv(measurement, cdim=-1) if forward_model is not None else measurement
            video_chw = self.propagator.predict_single_step(
                anchor_video_chw=anchor_video.movedim(-1, 1),
                measurement=measurement,
                device=self.device,
                target=self.single_step_target,
            )
            return video_chw.movedim(1, -1).clamp(0.0, 1.0)

        anchor_measurement = measurement[-1:] if getattr(self.propagator, "requires_last_frame", False) else measurement[:1]

        if self.initializer is not None:
            anchor_frame_video = self.initializer.run(
                measurement=anchor_measurement,
                forward_model=forward_model,
                evaluator=EvaluatorFunction(),
                regularizer_wrapper=None,
            )
        elif forward_model is not None:
            anchor_frame_video = forward_model.pseudoinv(anchor_measurement, cdim=-1)
        else:
            anchor_frame_video = anchor_measurement
        anchor_frame_chw = anchor_frame_video[0].movedim(-1, 0)

        video_chw = self.propagator.propagate(
            first_frame_chw=anchor_frame_chw,
            measurement=measurement,
            device=self.device,
        )
        return video_chw.movedim(1, -1).clamp(0.0, 1.0)
