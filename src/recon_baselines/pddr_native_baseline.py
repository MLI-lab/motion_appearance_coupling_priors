"""
PDDR baseline (Fuernrohr & Heckel 2026, https://github.com/MLI-lab/pddr) 
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import tqdm
import wandb
import yaml

from src.domain_specific_setups.natural_video_forward_models import LinearForwardModel
from src.recon_baselines.pddr_native.unet import SpatioTemporalUNetModel
from src.recon_baselines.recon_baseline_resolver import BaseReconBaseline, EvaluatorFunction
from src.regularizers.baseline_regularizer_wrapper import BaselineRegularizerWrapper


def _make_loss_fn(name: str):
    if name == "l2":
        return F.mse_loss
    elif name == "l1":
        return F.l1_loss
    else:
        raise ValueError(f"Unknown loss type: {name}")


def _linear_beta_schedule(timesteps: int) -> torch.Tensor:
    """Matches ``cardiac_diffusion.diffusion.Diffusion.beta_schedule(schedule='linear')``."""
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


class PDDRNativeBaseline(BaseReconBaseline):
    """Piecewise diffusion-regularized reconstruction using the PDDR authors' own pretrained
    checkpoint and native model architecture."""

    def __init__(
        self,
        recon_args: dict,
        device: str,
        model_dir,  # data-root-relative ('manual/...') or absolute path, resolved via path_resolver
        path_resolver=None,
        checkpoint_name: str = "model.pt",
        use_ema: bool = True,
        **_kwargs,
    ):
        super().__init__()

        self.recon_args = recon_args
        self.device = device

        # relative paths are relative to a data root, not the working directory
        resolved_model_dir = path_resolver(model_dir) if path_resolver is not None else model_dir
        self._load_from_checkpoint(Path(resolved_model_dir), checkpoint_name=checkpoint_name, use_ema=use_ema)

    def _load_from_checkpoint(self, model_dir: Path, *, checkpoint_name: str, use_ema: bool) -> None:
        with open(model_dir / "configs" / "diffusion.yaml", "r") as f:
            diffusion_cfg = yaml.safe_load(f)

        unet_cfg = diffusion_cfg["SpatioTemporalUNetModel"]
        model = SpatioTemporalUNetModel(**unet_cfg).to(self.device)

        checkpoint_path = model_dir / "models" / checkpoint_name
        model_data = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        # keys look like "module.restoration_fn.<unet key>" (DDP-wrapped Diffusion); Diffusion's own buffers are dropped
        raw_state_dict = model_data["ema"] if use_ema else model_data["model"]
        state_dict = {}
        for k, v in raw_state_dict.items():
            if k.startswith("module."):
                k = k[len("module."):]
            if k.startswith("restoration_fn."):
                state_dict[k[len("restoration_fn."):]] = v
        model.load_state_dict(state_dict)
        model.eval()
        model.requires_grad_(False)

        self.model = model

        diff_cfg = diffusion_cfg["diffusion"]
        if diff_cfg.get("schedule", "linear") != "linear":
            raise NotImplementedError(
                f"Only the 'linear' beta schedule is currently vendored (checkpoint requests "
                f"'{diff_cfg.get('schedule')}')."
            )
        self.noise_based = bool(diff_cfg.get("noise_based", True))
        self.num_train_timesteps = int(diff_cfg["timesteps"])

        betas = _linear_beta_schedule(self.num_train_timesteps).to(self.device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - self.alphas_cumprod).sqrt()

        print(
            f"[PDDRNativeBaseline] Loaded checkpoint from {checkpoint_path} - "
            f"{self.num_train_timesteps} timesteps, use_ema={use_ema}"
        )

    def _schedule_timestep(self, it: int, optim_steps: int, t_max: int, time_sampling: str) -> int:
        # matches Variation.regularization's exact timestep schedule (cardiac_diffusion/variational.py)
        if time_sampling == "descending":
            t = t_max * (optim_steps - it) // optim_steps
        elif time_sampling == "fixed":
            t = t_max
        else:  # "random", matching Variation's else-branch (torch.randint(0, max_timestep, ...))
            t = int(torch.randint(0, t_max, (1,)).item())
        return max(0, min(t, self.num_train_timesteps - 1))

    def _reg_factor(self, t_val: int, reg_param: float) -> torch.Tensor:
        """reg_param * sqrt(1 - alpha_bar_t) / sqrt(alpha_bar_t), as in Variation.regularization."""
        alpha_bar_t = self.alphas_cumprod[t_val]
        return reg_param * (1 - alpha_bar_t).sqrt() / alpha_bar_t.sqrt()

    def _choose_slab(
        self, num_frames: int, slab_size: int, slab_sampling: str, iteration: int
    ) -> Tuple[int, int]:
        """As Variation.choose_slab: one active slab per iteration, random or rotating."""
        if slab_sampling == "random":
            q = int(torch.randint(0, num_frames, (1,)).item())
            start, stop = q - slab_size // 2, q + slab_size // 2 + slab_size % 2
            if start < 0:
                start, stop = 0, slab_size
            if stop > num_frames:
                start, stop = num_frames - slab_size, num_frames
        elif slab_sampling == "sliding_window":
            if slab_size >= num_frames:
                start, stop = 0, num_frames
            else:
                max_slabs = int(math.ceil(num_frames / slab_size))
                s = iteration % max_slabs
                start, stop = s * slab_size, (s + 1) * slab_size
            if stop > num_frames:
                start, stop = num_frames - slab_size, num_frames
        else:
            raise ValueError(f"Unknown slab_sampling: {slab_sampling}")
        return start, stop

    def _regularization_loss(
        self,
        x: torch.Tensor,  # (T, X, Y, C) model estimate, real-valued
        t_val: int,
        start: int,
        stop: int,
        regularization_loss_fn,
        prior_scale: torch.Tensor,
    ) -> torch.Tensor:
        slab = x[start:stop] * prior_scale  # (f, X, Y, C)
        slab_model = slab.movedim(-1, 0).unsqueeze(0)  # (1, C, f, X, Y)

        noise = torch.randn_like(slab_model)
        t_batch = torch.full((slab_model.shape[0],), t_val, device=self.device, dtype=torch.long)
        sqrt_ac = self.sqrt_alphas_cumprod[t_batch].view(-1, 1, 1, 1, 1)
        sqrt_omac = self.sqrt_one_minus_alphas_cumprod[t_batch].view(-1, 1, 1, 1, 1)
        noisy_slab = sqrt_ac * slab_model + sqrt_omac * noise

        estimation = self.model(noisy_slab, t_batch)
        ground = noise if self.noise_based else slab_model
        return regularization_loss_fn(ground, estimation)

    @staticmethod
    def _tv_loss(x: torch.Tensor) -> torch.Tensor:
        """Temporal (frame-axis) total variation of a (T, X, Y, C) tensor."""
        return (x[:-1] - x[1:]).abs().mean() + (x[1:] - x[:-1]).abs().mean()

    def run(
        self,
        measurement: torch.Tensor,
        forward_model: LinearForwardModel,
        evaluator: EvaluatorFunction,
        regularizer_wrapper: Optional[BaselineRegularizerWrapper] = None,
        flow_reference_video: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs

        recon_args = self.recon_args
        optim_steps = int(recon_args.get("optim_steps", 100))
        reg_param = float(recon_args.get("reg_param", 0.03))
        slab_size = int(recon_args.get("slab_size", 36))
        slab_sampling = recon_args.get("slab_sampling", "sliding_window")
        timestep_factor = float(recon_args.get("timestep_factor", 0.4))
        time_sampling = recon_args.get("time_sampling", "descending")
        consistency_loss_fn = _make_loss_fn(recon_args.get("consistency_loss", "l2"))
        regularization_loss_fn = _make_loss_fn(recon_args.get("regularization_loss", "l2"))
        lr = float(recon_args.get("lr", 1.0))
        use_lr_scheduler = bool(recon_args.get("lr_scheduler", True))
        lr_scheduler_factor = float(recon_args.get("lr_scheduler_factor", 0.25))
        lr_scheduler_patience = int(recon_args.get("lr_scheduler_patience", 25))
        tv_reg_param = float(recon_args.get("tv_reg_param", 0.0))
        eval_every_iters = int(recon_args.get("eval_every_iters", 5))
        # dataset normalization already matches PDDR's training scale; prior_scale=1.0 reproduces upstream
        prior_scale = torch.tensor(float(recon_args.get("prior_scale", 1.0)), device=self.device)

        # zero-filled initialization (measurement: (T, Ncoils, X, Y, 2) masked k-space)
        x = forward_model.adjoint(measurement, cdim=-1).detach().clone()  # (T, X, Y, C)

        num_frames = x.shape[0]
        slab_size = min(slab_size, num_frames)
        t_max = int(timestep_factor * (self.num_train_timesteps - 1))

        x.requires_grad_(True)

        # betas=(0.9, 0.99), matching Variation.fit exactly (not PyTorch's Adam default (0.9, 0.999))
        optimizer = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.99))
        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=lr_scheduler_factor, patience=lr_scheduler_patience
            )
            if use_lr_scheduler
            else None
        )

        if regularizer_wrapper is not None:
            pseudoinv_seed_chw = x.detach().movedim(-1, 1)
            if flow_reference_video is not None and flow_reference_video.shape != pseudoinv_seed_chw.shape:
                raise ValueError(
                    f"flow_reference_video shape {tuple(flow_reference_video.shape)} != x shape {tuple(pseudoinv_seed_chw.shape)}"
                )
            regularizer_wrapper.set_pseudoinv(
                flow_reference_video if flow_reference_video is not None else pseudoinv_seed_chw
            )

        eval_results: Dict[str, Any] = {}
        pbar = tqdm.tqdm(range(optim_steps), desc="PDDR-native")
        for it in pbar:
            optimizer.zero_grad()

            dc_loss = consistency_loss_fn(forward_model(x, cdim=-1), measurement)

            t_val = self._schedule_timestep(it, optim_steps, t_max, time_sampling)
            start, stop = self._choose_slab(num_frames, slab_size, slab_sampling, it)
            reg_loss = self._regularization_loss(x, t_val, start, stop, regularization_loss_fn, prior_scale)
            reg_factor = self._reg_factor(t_val, reg_param)

            loss = dc_loss + reg_factor * reg_loss
            if tv_reg_param > 0.0:
                loss = loss + tv_reg_param * self._tv_loss(x)

            loss.backward()
            optimizer.step()
            # as in Variation.fit: scheduler only after the first 5% of iterations
            if scheduler is not None and it > optim_steps * 0.05:
                scheduler.step(loss.item())

            if regularizer_wrapper is not None:
                alpha_bar_t = self.alphas_cumprod[t_val]
                snr_t = (alpha_bar_t.sqrt() / (1 - alpha_bar_t).sqrt()).item()
                with torch.no_grad():
                    x_updated = regularizer_wrapper(x.detach().movedim(-1, 1), step=t_val, snr_t=snr_t).movedim(1, -1)
                    # in-place copy keeps x tied to the Adam state
                    x.data.copy_(x_updated)

            if it % eval_every_iters == 0:
                with torch.no_grad():
                    eval_results = evaluator(x.detach())
                    pbar.set_description(f"(@it {it}, t={t_val}) {eval_results}", refresh=True)
                    if wandb.run is not None:
                        wandb.log({
                            **eval_results,
                            "dc_loss": dc_loss.item(),
                            "reg_loss": reg_loss.item(),
                            "reg_factor": reg_factor.item(),
                            "global_step": it,
                        })

        return x.detach()
