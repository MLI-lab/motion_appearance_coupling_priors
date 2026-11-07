"""
Latent-diffusion inverse-problem baselines on a frozen Stable Diffusion 1.5 (VAE + UNet).

dc_mode="psld": PSLD (Rout et al., NeurIPS 2023, arXiv:2307.00619, Algorithm 2).
dc_mode="dds":  latent DDS / TReg-style CG projection (Chung et al. ICLR 2024; Kim et al. ICLR 2024).
"""

from __future__ import annotations

import os
import warnings
from typing import List, Optional

import torch
import torch as th
import tqdm
import wandb

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from src.domain_specific_setups.natural_video_forward_models import LinearForwardModel
from src.recon_baselines.recon_baseline_resolver import BaseReconBaseline, EvaluatorFunction
from src.regularizers.baseline_regularizer_wrapper import BaselineRegularizerWrapper
from src.utils.latent_motion import motion_to_latent


def CG(A, b, x, n_inner=5, eps=1e-5):
    """Conjugate gradients on an unbatched tensor - solve A(x) = b."""
    r = b - A(x)
    p = r.clone()
    rsold = th.sum(r * r)

    for _ in range(n_inner):
        Ap = A(p)
        pAp = th.sum(p * Ap)
        a = rsold / (pAp + 1e-12)
        x = x + a * p
        r = r - a * Ap
        rsnew = th.sum(r * r)
        if th.sqrt(rsnew).abs() < eps:
            break
        p = r + (rsnew / (rsold + 1e-12)) * p
        rsold = rsnew
    return x


_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def _forward_model_is_frame_separable(forward_model: LinearForwardModel) -> bool:
    """True when A acts on every frame independently, so the PSLD loss can be differentiated chunk-wise."""
    from src.domain_specific_setups.natural_video_forward_models import IdentityForwardModel
    if isinstance(forward_model, IdentityForwardModel):
        return True
    return int(getattr(forward_model, "kernel_size_t", 0)) == 1


class LDMBaseline(BaseReconBaseline):

    def __init__(
        self,
        path_resolver,
        recon_args: dict,
        nfe: int,
        noise_mode: str,
        noise_params: dict,
        device: str,
        pretrained_model_dir: str = "model_cache/sd15",
        vae_subfolder: str = "vae",
        unet_subfolder: str = "unet",
        scheduler_subfolder: str = "scheduler",
        text_encoder_subfolder: str = "text_encoder",
        tokenizer_subfolder: str = "tokenizer",
        assume_pixel_range: str = "0_1",
        dtype: str = "float16",
        gradient_checkpointing: bool = True,
        **_kwargs,  # pretrained_model_path{,2}, pretrained_vdm_model_path: not used by this baseline
    ):
        super().__init__()

        self.recon_args = dict(recon_args)
        self.noise_mode = noise_mode
        self.noise_params = dict(noise_params)
        self.nfe = int(nfe)
        self.device = device
        if assume_pixel_range not in ("0_1", "minus1_1"):
            raise ValueError(f"Unknown assume_pixel_range: {assume_pixel_range!r}")
        self.assume_pixel_range = assume_pixel_range
        self.dtype = _DTYPES[dtype]

        model_dir = path_resolver(pretrained_model_dir)
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"SD checkpoint directory {model_dir} not found; add `sd15` to the flow's required_datasets "
                f"(hydra/task/download_dataset/sd15.yaml) so it is staged before the baseline runs."
            )
        print(f"[LDMBaseline] loading SD components from {model_dir} ({dtype})")
        fp_kwargs = {"local_files_only": True}

        # SD runs in fp16; forward model / evaluator side stays fp32 (bridged in _encode/_decode)
        self.vae = AutoencoderKL.from_pretrained(model_dir, subfolder=vae_subfolder, torch_dtype=self.dtype, **fp_kwargs).to(device)
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae.enable_slicing()

        self.unet = UNet2DConditionModel.from_pretrained(model_dir, subfolder=unet_subfolder, torch_dtype=self.dtype, **fp_kwargs).to(device)
        self.unet.requires_grad_(False)
        self.unet.eval()
        if gradient_checkpointing:
            # diffusers >= 0.30: also effective in eval mode
            self.unet.enable_gradient_checkpointing()
            self.vae.enable_gradient_checkpointing()

        self.noise_scheduler = DDPMScheduler.from_pretrained(model_dir, subfolder=scheduler_subfolder, **fp_kwargs)
        self.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)
        self.num_train_timesteps = int(self.noise_scheduler.config.num_train_timesteps)

        self.scaling_factor = float(getattr(self.vae.config, "scaling_factor", 0.18215))
        # Number of 2x-downsampling encoder blocks (standard SD VAE: 4 blocks -> f=8).
        self.downsample_factor = 2 ** (len(self.vae.config.down_block_types) - 1)
        self.latent_channels = int(self.vae.config.latent_channels)

        tokenizer = CLIPTokenizer.from_pretrained(model_dir, subfolder=tokenizer_subfolder, **fp_kwargs)
        text_encoder = CLIPTextModel.from_pretrained(model_dir, subfolder=text_encoder_subfolder, torch_dtype=self.dtype, **fp_kwargs).to(device)
        text_encoder.requires_grad_(False)
        text_encoder.eval()
        with torch.no_grad():
            empty_tokens = tokenizer(
                [""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
            ).input_ids.to(device)
            self.empty_text_embedding = text_encoder(empty_tokens)[0]  # (1, 77, 768)
        del text_encoder, tokenizer  # only needed once, for the unconditional embedding

    # ------------------------------------------------------------------ VAE / UNet bridges

    def _encode(self, x_pixel: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) fp32 pixels -> (B,4,H/f,W/f) scaled latents. Differentiable."""
        x = x_pixel.to(self.dtype)
        if self.assume_pixel_range == "0_1":
            x = x * 2.0 - 1.0
        z = self.vae.encode(x).latent_dist.mode()
        return z * self.scaling_factor

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B,4,H/f,W/f) scaled latents -> (B,3,H,W) fp32 pixels. Differentiable."""
        x = self.vae.decode(z.to(self.dtype) / self.scaling_factor).sample
        if self.assume_pixel_range == "0_1":
            x = (x + 1.0) / 2.0
        return x.float()

    def _unet_forward(self, zt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        encoder_hidden_states = self.empty_text_embedding.expand(zt.shape[0], -1, -1)
        return self.unet(zt.to(self.dtype), t, encoder_hidden_states=encoder_hidden_states).sample

    def _alpha_bar(self, t: int) -> torch.Tensor:
        """alpha_bar_t as a fp32 scalar tensor; t = -1 (the final DDIM target) yields 1."""
        if t < 0:
            return torch.ones((), device=self.device)
        return self.alphas_cumprod[int(t)]

    # ------------------------------------------------------------------ noise sampler

    def _build_latent_noise_sampler(self, flow_seed: torch.Tensor, latent_shape: torch.Size):
        """Noise sampler on the latent grid. Temporally-warped modes get the RAFT flow of the
        pixel-space ``flow_seed`` (T,C,H,W) transported to latent resolution."""
        from src.warp_methods.noise_warp_util import get_noise_warping_method
        noise_warp_method = get_noise_warping_method(
            name=self.noise_mode, **self.noise_params, target_shape=latent_shape, device=self.device,
        )
        if hasattr(noise_warp_method, "initialize_flow_from_pairwise_flow"):
            from src.regularizers.raft_based_regularizers import estimate_flow_for_video, load_raft_model
            raft_model = load_raft_model(device=str(self.device))
            with torch.no_grad():
                flow_pixel = estimate_flow_for_video(raft_model, flow_seed, enable_upsample_flow=False, forward_flow=False)  # (T-1,2,H,W)
            flow_latent = motion_to_latent(flow_pixel, self.downsample_factor)
            noise_warp_method.initialize_flow_from_pairwise_flow(flow_latent)
            del raft_model
        else:
            noise_warp_method.initialize_flow(flow_seed)  # no-op for fully_random / consistent / fixed_random

        def sample() -> torch.Tensor:
            return noise_warp_method.sample_noise().to(self.dtype)

        return sample

    # ------------------------------------------------------------------ PSLD

    def _psld_step(
        self,
        zt: torch.Tensor,
        t_scalar: int,
        alpha_bar: torch.Tensor,
        measurement_img: torch.Tensor,   # (T, X, Y, C) fp32, the domain A acts in
        at_y_img: torch.Tensor,          # (T, X, Y, C) = A^T y, precomputed
        forward_model: LinearForwardModel,
        chunks: List[List[int]],
        omega: float,
        gamma: float,
    ):
        """Returns (eps, z0_hat, grad) for the whole video, each (T,4,h,w) in self.dtype, with
        grad = d/dz_t [omega * sum_frames ||y - A D(z0)||_2 + gamma * sum_frames ||z0 - E(x_star)||_2]."""
        eps_all = torch.empty_like(zt)
        z0_all = torch.empty_like(zt)
        grad_all = torch.zeros_like(zt)
        need_grad = (omega != 0.0) or (gamma != 0.0)

        for idx in chunks:
            zt_c = zt[idx].detach().requires_grad_(need_grad)
            t_c = torch.full((len(idx),), int(t_scalar), device=self.device, dtype=torch.long)
            with torch.set_grad_enabled(need_grad):
                eps_c = self._unet_forward(zt_c, t_c)
                z0_c = (zt_c - (1.0 - alpha_bar).sqrt().to(zt_c.dtype) * eps_c) / alpha_bar.sqrt().to(zt_c.dtype)
                if need_grad:
                    x0_img = self._decode(z0_c).movedim(1, -1)  # (n, X, Y, C) fp32
                    y_c = measurement_img[idx]
                    a_x0 = forward_model(x0_img, cdim=-1)
                    loss = torch.zeros((), device=self.device)
                    if omega != 0.0:
                        meas_err = torch.linalg.norm((y_c - a_x0).flatten(1), dim=1).sum()
                        loss = loss + omega * meas_err
                    if gamma != 0.0:
                        x_star = at_y_img[idx] + x0_img - forward_model.adjoint(a_x0, cdim=-1)
                        z_star = self._encode(x_star.movedim(-1, 1))
                        glue_err = torch.linalg.norm((z0_c.float() - z_star.float()).flatten(1), dim=1).sum()
                        loss = loss + gamma * glue_err
                    grad_c = torch.autograd.grad(loss, zt_c)[0]
                    grad_all[idx] = grad_c.detach().to(zt.dtype)
            eps_all[idx] = eps_c.detach()
            z0_all[idx] = z0_c.detach()
        return eps_all, z0_all, grad_all

    @torch.no_grad()
    def _dds_step(
        self,
        zt: torch.Tensor,
        t_scalar: int,
        alpha_bar: torch.Tensor,
        measurement_img: torch.Tensor,   # (T, X, Y, C) fp32
        at_y_img: torch.Tensor,          # (T, X, Y, C) = A^T y, precomputed
        forward_model: LinearForwardModel,
        cg_n_inner: int,
        cg_eps: float,
        cg_prox_weight: float,
        pixel_wrapper: Optional[BaselineRegularizerWrapper] = None,
        snr_t: Optional[torch.Tensor] = None,
    ):
        """Returns (eps, z0_hat_dc): Tweedie latent decoded, made data-consistent by CG in pixel space, re-encoded."""
        del measurement_img  # DC only needs A^T y and A^T A
        n = zt.shape[0]
        t = torch.full((n,), int(t_scalar), device=self.device, dtype=torch.long)
        eps = self._unet_forward(zt, t)
        z0 = (zt - (1.0 - alpha_bar).sqrt().to(zt.dtype) * eps) / alpha_bar.sqrt().to(zt.dtype)
        x0_img = self._decode(z0).movedim(1, -1)  # (T, X, Y, C) fp32

        lam = float(cg_prox_weight)
        if lam > 0.0:
            def AtA(x):
                return forward_model.adjoint(forward_model(x, cdim=-1), cdim=-1) + lam * x
            b = at_y_img + lam * x0_img
        else:
            def AtA(x):
                return forward_model.adjoint(forward_model(x, cdim=-1), cdim=-1)
            b = at_y_img
        x_hat = CG(AtA, b, x0_img, n_inner=int(cg_n_inner), eps=float(cg_eps))
        if pixel_wrapper is not None:
            with torch.enable_grad():  # the wrapper runs its own inner optimizer
                x_hat = pixel_wrapper(x_hat.movedim(-1, 1), step=t, snr_t=snr_t).detach().movedim(1, -1)
        z0_dc = self._encode(x_hat.movedim(-1, 1)).to(zt.dtype)
        return eps, z0_dc

    @torch.no_grad()
    def _ddim_invert(self, z0: torch.Tensor, times_asc: List[int]) -> torch.Tensor:
        """Deterministic DDIM inversion (eta = 0) of clean latents z0 along the ascending grid ``times_asc``."""
        n = z0.shape[0]
        z = z0.to(self.dtype)
        for t_cur, t_next in zip([-1] + list(times_asc[:-1]), times_asc):
            a_cur = self._alpha_bar(t_cur)
            a_next = self._alpha_bar(t_next)
            t_eval = max(int(t_cur), 0)
            eps = self._unet_forward(z, torch.full((n,), t_eval, device=self.device, dtype=torch.long))
            z0_hat = (z - (1.0 - a_cur).sqrt().to(z.dtype) * eps) / a_cur.sqrt().to(z.dtype)
            z = a_next.sqrt().to(z.dtype) * z0_hat + (1.0 - a_next).sqrt().to(z.dtype) * eps
        return z

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

        # latent-domain wrapper acts on the DDIM anchor latent; pixel-domain one only in dds mode
        wrapper_domain = getattr(regularizer_wrapper, "domain", "pixel") if regularizer_wrapper is not None else None
        if regularizer_wrapper is not None and wrapper_domain == "latent":
            if int(getattr(regularizer_wrapper, "latent_downsample_factor", 0)) != int(self.downsample_factor):
                raise ValueError(
                    f"LDMBaseline: regularizer_wrapper.latent_downsample_factor="
                    f"{getattr(regularizer_wrapper, 'latent_downsample_factor', None)} != VAE factor {self.downsample_factor}"
                )

        # channel-last (T, X, Y, C) on the forward-model side; always pass cdim explicitly
        measurement = measurement.detach().float()
        at_y_img = forward_model.adjoint(measurement, cdim=-1)  # (T, X, Y, C) = A^T y
        pseudoinv = at_y_img.movedim(-1, 1)  # (T, C, X, Y)

        if flow_reference_video is not None and flow_reference_video.shape != pseudoinv.shape:
            raise ValueError(
                f"flow_reference_video shape {tuple(flow_reference_video.shape)} != pseudoinv shape {tuple(pseudoinv.shape)}"
            )
        flow_seed = flow_reference_video if flow_reference_video is not None else pseudoinv

        T, C, H, W = pseudoinv.shape
        f = self.downsample_factor
        if H % f != 0 or W % f != 0:
            raise ValueError(f"LDMBaseline: spatial size {(H, W)} must be divisible by the VAE factor f={f}")
        latent_shape = torch.Size((T, self.latent_channels, H // f, W // f))
        noise_sampler = self._build_latent_noise_sampler(flow_seed, latent_shape)
        if regularizer_wrapper is not None:
            regularizer_wrapper.set_pseudoinv(flow_seed)  # RAFT at pixel resolution (-> latent grid inside, in latent domain)

        T_sampling = int(self.nfe + 1)
        eta = float(self.recon_args.get("eta", 1.0))
        dc_mode = str(self.recon_args.get("dc_mode", "psld"))
        if dc_mode not in ("psld", "dds"):
            raise ValueError(f"LDMBaseline: unknown dc_mode {dc_mode!r} (want 'psld' or 'dds')")
        omega = float(self.recon_args.get("omega", 1.0))
        gamma = float(self.recon_args.get("gamma", 0.1))
        cg_n_inner = int(self.recon_args.get("cg_n_inner", 5))
        cg_eps = float(self.recon_args.get("cg_eps", 1e-5))
        cg_prox_weight = float(self.recon_args.get("cg_prox_weight", 0.0))
        eval_every_iters = int(self.recon_args.get("eval_every_iters", 5))
        if regularizer_wrapper is not None and wrapper_domain == "pixel" and dc_mode != "dds":
            raise ValueError("LDMBaseline: a pixel-domain regularizer_wrapper needs dc_mode=dds (it acts on the CG-corrected frames)")
        init_mode = str(self.recon_args.get("init_mode", "noise"))
        init_frac = float(self.recon_args.get("init_frac", 1.0))
        if init_mode not in ("noise", "sdedit", "ddim_inversion"):
            raise ValueError(f"LDMBaseline: unknown init_mode {init_mode!r} (want 'noise', 'sdedit' or 'ddim_inversion')")
        if init_mode == "noise":
            init_frac = 1.0
        grad_chunk_frames = int(self.recon_args.get("grad_chunk_frames", 0) or 0)

        if grad_chunk_frames > 0 and grad_chunk_frames < T and not _forward_model_is_frame_separable(forward_model):
            warnings.warn(
                f"LDMBaseline: forward model {type(forward_model).__name__} couples frames; ignoring "
                f"grad_chunk_frames={grad_chunk_frames} and differentiating the whole video at once.",
                RuntimeWarning,
            )
            grad_chunk_frames = 0
        step = grad_chunk_frames if grad_chunk_frames > 0 else T
        chunks = [list(range(s, min(s + step, T))) for s in range(0, T, step)]

        num_ts = self.num_train_timesteps
        skip = num_ts // T_sampling
        times = range(0, num_ts, skip)
        times_next = [-1] + list(times[:-1])
        times_pair = list(zip(reversed(times), reversed(times_next)))

        if init_frac < 1.0:
            # warm start from the encoded pseudoinverse at t = init_frac * num_ts
            tau = int(init_frac * num_ts)
            times_pair = [(i, j) for (i, j) in times_pair if i <= tau]
            tau_actual = times_pair[0][0] if times_pair else 0
            with torch.no_grad():
                z_meas = self._encode(pseudoinv)
            if init_mode == "sdedit":
                a_tau = self._alpha_bar(tau_actual).to(self.dtype)
                zt = a_tau.sqrt() * z_meas + (1.0 - a_tau).sqrt() * noise_sampler()
            else:  # ddim_inversion (VISION-XL's pseudo-batch inversion): deterministic, no noise
                times_asc = [i for (i, _) in reversed(times_pair)]  # ascending, ends at tau_actual
                zt = self._ddim_invert(z_meas, times_asc)
            print(f"[LDMBaseline] init_mode={init_mode}: starting at t={tau_actual}, {len(times_pair)} sampling steps")
        else:
            zt = noise_sampler()  # (T, 4, H/f, W/f)

        step_k = 0
        eval_results = {}
        pbar = tqdm.tqdm(times_pair, total=len(times_pair), desc=f"LDM/{dc_mode}")
        for i, j in pbar:
            at = self._alpha_bar(i)
            at_next = self._alpha_bar(j)

            if dc_mode == "psld":
                eps, z0_hat, grad = self._psld_step(
                    zt, i, at, measurement, at_y_img, forward_model, chunks, omega=omega, gamma=gamma,
                )
            else:
                eps, z0_hat = self._dds_step(
                    zt, i, at, measurement, at_y_img, forward_model,
                    cg_n_inner=cg_n_inner, cg_eps=cg_eps, cg_prox_weight=cg_prox_weight,
                    pixel_wrapper=regularizer_wrapper if wrapper_domain == "pixel" else None,
                    snr_t=(at.sqrt() / (1.0 - at).sqrt()),
                )
                grad = torch.zeros_like(zt)

            if regularizer_wrapper is not None and wrapper_domain == "latent":
                # latent motion regularization on the anchor latent (wrapper runs its own optimizer)
                t_vec = torch.full((zt.shape[0],), int(i), device=self.device, dtype=torch.long)
                snr_t = at.sqrt() / (1.0 - at).sqrt()
                with torch.enable_grad():
                    z0_hat = regularizer_wrapper(z0_hat.detach().float(), step=t_vec, snr_t=snr_t).detach().to(zt.dtype)

            with torch.no_grad():
                c1 = ((1.0 - at_next).sqrt() * eta).to(self.dtype)
                c2 = ((1.0 - at_next).sqrt() * ((1.0 - eta ** 2) ** 0.5)).to(self.dtype)
                noise = noise_sampler()
                zt = at_next.sqrt().to(self.dtype) * z0_hat + c1 * noise + c2 * eps - grad

            if step_k % eval_every_iters == 0 or j < 0:
                with torch.no_grad():
                    eval_img = self._decode(zt).movedim(1, -1)  # (T, X, Y, C)
                    eval_results = evaluator(eval_img)
                    eval_results = {
                        **eval_results,
                        "dc_grad_norm": grad.float().norm().item(),
                        "z0_std": z0_hat.float().std().item(),
                    }
                pbar.set_description(f"(@it {step_k}) {eval_results}", refresh=True)
                if wandb.run is not None:
                    wandb.log({**eval_results, "global_step": step_k})
            step_k += 1

        with torch.no_grad():
            return self._decode(zt).movedim(1, -1)  # (T, X, Y, C), pixel domain
