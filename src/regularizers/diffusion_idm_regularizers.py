import torch
import math
from typing import Optional
from torch import Tensor
from src.diffmodels.sde import DDPM
from src.diffmodels.networks.archs.unet import UNetModel
import wandb
from functools import partial

from src.representations.base_representation import BaseRepresentation
from src.regularizers.timestep_schedules import compute_scheduled_timestep

def diffusion_variational_idm_regularizer(
    model : BaseRepresentation,  # representation model, e.g. BaseImageWithINNDeformationField
    result: Tensor,  # kspace line result or part of rasterized image
    kspace: Tensor,  # kspace line target or part of rasterized target image
    kcoords: Tensor,
    timesteps: Tensor,  # timesteps for the kspace line
    spatial_mesh: Tensor,  # spatial mesh for interpolation
    kspace_mesh: Tensor,  # kspace mesh for interpolation
    time_mesh: Tensor,  # time mesh for interpolation
    score: UNetModel,
    sde: DDPM,
    scale_factor : float = 1.0,
    reg_strength: float = 1.0,
    data_multiplier: float = 1.0,
    num_train_steps_factor_lower: float = 0.0,
    num_train_steps_factor_upper: float = 0.5,
    adapt_reg_strength: Optional[bool] = None,  # deprecated — use denoise_term_weight
    denoise_term_weight: Optional[str] = None,
    timestep_nr_factor : float = 1.0,
    take_trainer_timesteps : bool = False,
    timestep_mode: str = "uniform_cont",
    noise_mode : str = "fully_random",
    fixed_noise_sample : Optional[Tensor] = None,
    consistent_timestep : bool = False,
    normalization : str = "none",
    mixture_param : float = 0.0,
    timestep_schedule: str = "random",
    timestep_schedule_ddim_steps: Optional[int] = None,
    timestep_jitter_fraction: float = 0.0,
    optimization_progress: Optional[float] = None,
    compensate_passthrough_alpha: bool = False,
    apply_scaling_factor_from_setup : bool = True,
    **kwargs
) -> Tensor:

    # sample a random time point  between min and max
    if take_trainer_timesteps:
        tau = timesteps
    elif timestep_mode == "uniform_cont":
        D =  min(max(math.floor(result.shape[0]*timestep_nr_factor), 1), result.shape[0])
        tmin, tmax = time_mesh.min(), time_mesh.max()
        tau = torch.rand((D,), device=result.device) * (tmax - tmin) + tmin
    elif timestep_mode == "uniform_discrete":
        D =  min(max(math.floor(result.shape[0]*timestep_nr_factor), 1), result.shape[0])
        # now take a random subset of D timesteps from the time_mesh
        tau = time_mesh[torch.randperm(time_mesh.shape[0], device=result.device)][:D]
    else:
        raise ValueError(f"Unknown timestep sampling mode: {timestep_mode}")

    output = model.full_warp(timesteps=tau).movedim(-1,1) # (T', C, X, Y)

    output = (output * scale_factor if apply_scaling_factor_from_setup else output) * data_multiplier

    if normalization == "2times_minus1":
        output = output * 2 - 1 # scale to [-1, 1], assuming the output is in [0, 1] 
    elif normalization == "to_01":
        output = (output - output.min()) / (output.max() - output.min() + 1e-8) # scale to [0, 1]
    elif normalization == "to_11":
        output = (output - output.min()) / (output.max() - output.min() + 1e-8) * 2 - 1 # scale to [-1, 1]
    elif normalization == "standardize":
        output = (output - output.mean()) / (output.std() + 1e-8) # standardize to zero mean and unit variance
    elif normalization == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

    if noise_mode == "fully_random":
        z = torch.randn_like(output)
    elif noise_mode == "consistent":
        z = torch.randn((1, *output.shape[1:]), device=output.device).expand(output.shape)
    elif noise_mode == "fixed" and fixed_noise_sample is not None:
        z = fixed_noise_sample.expand(output.shape)
    elif noise_mode == "fixed_consistent_mixture" and fixed_noise_sample is not None and mixture_param is not None:
        random_noise = torch.randn((1, *output.shape[1:]), device=output.device).expand(output.shape)
        fixed_noise = fixed_noise_sample.expand(output.shape)
        z = (1-mixture_param) * fixed_noise + mixture_param * random_noise
    elif noise_mode == "fixed_nonconsistent" and fixed_noise_sample is not None and mixture_param is not None:
        random_noise = torch.randn_like(output)
        fixed_noise = fixed_noise_sample.expand(output.shape)
        z = (1-mixture_param) * fixed_noise + mixture_param * random_noise
    else:
        raise ValueError(f"Unknown noise mode: {noise_mode}")

    # sample t in the sampling chain
    t_lo = max(0, math.floor(num_train_steps_factor_lower * sde.num_steps))
    t_hi = max(t_lo + 1, math.floor(num_train_steps_factor_upper * sde.num_steps))
    t = compute_scheduled_timestep(
        t_lo=t_lo,
        t_hi=t_hi,
        batch_size=output.shape[0],
        device=output.device,
        schedule=timestep_schedule,
        optimization_progress=optimization_progress,
        jitter_fraction=timestep_jitter_fraction,
        consistent_timestep=consistent_timestep,
        ddim_num_steps=timestep_schedule_ddim_steps,
    )

    mean, std = sde.marginal_prob(output, t)
    perturbed_x = mean + z * std[:, None, None, None]

    zhat = score(perturbed_x, t)

    if zhat.size(1) == 2 * perturbed_x.size(1):
        # this occurs when learn_sigma is enabled for the trained network
        zhat = zhat[:, :perturbed_x.size(1)]

    res = (z - zhat).pow(2).flatten(start_dim=1).mean(dim=1)

    # Denoiser term weighting (Mardani et al., 2023, RED-diff)
    # Resolve denoise_term_weight from the (deprecated) adapt_reg_strength flag
    _weight_mode = denoise_term_weight
    if _weight_mode is None:
        _weight_mode = "linear" if adapt_reg_strength else "const"

    bar_a = sde.marginal_prob_mean(t)  # sqrt(alpha_bar_t)
    snr_inv = std / bar_a              # sqrt(1-alpha_bar) / sqrt(alpha_bar) = 1/SNR

    if _weight_mode == "const":
        w_t = torch.ones_like(snr_inv)
    elif _weight_mode == "linear":
        w_t = snr_inv
    elif _weight_mode == "sqrt":
        w_t = snr_inv.sqrt()
    elif _weight_mode == "square":
        w_t = snr_inv.square()
    elif _weight_mode == "log":
        w_t = (snr_inv + 1.0).log()
    elif _weight_mode == "trunc_linear":
        w_t = snr_inv.clamp(max=1.0)
    elif _weight_mode == "power2over3":
        w_t = snr_inv.pow(2.0 / 3.0)
    else:
        raise ValueError(
            f"Unknown denoise_term_weight: {_weight_mode!r}. "
            f"Choose from: const, linear, sqrt, square, log, trunc_linear, power2over3."
        )

    # pass-through gradient picks up an implicit sqrt(a_bar_t) factor; divide it out to match RED-diff
    if compensate_passthrough_alpha:
        w_t = w_t / bar_a.clamp(min=1e-6)

    reg_strength_t = w_t * reg_strength

    return (reg_strength_t * res).mean()

def diffusion_idm_regularizer_resolver(
    name,
    spatial_mesh,
    kspace_mesh,
    time_mesh,
    device,
    path_resolver,
    model_path,
    cache_dir,
    use_score_pass_through,
    arch,
    sde,
    reg_params,
    **kwargs,
):
    if name == "diffusion_idm_regularizer":
        # first goal is to have the svi approach implemented, but in variational form
        from src.diffmodels.sde import load_sde_model
        from src.utils.pass_through import ScoreWithIdentityGradWrapper
        from src.diffmodels.utils_load import create_dense_model

        cache_dir = path_resolver(cache_dir)
        print(f"Cache_dir resolved to: {cache_dir}")
        from src.utils.model_cache import ensure_cached_download
        model_path_resolved = ensure_cached_download(model_path, cache_dir)

        print(f"Loading model from {model_path_resolved}...")
        score = create_dense_model(**arch)
        score.load_state_dict(torch.load(model_path_resolved, map_location="cpu"))
        score.to(device)
        score.eval()

        sde = load_sde_model(sde)
        wandb.run.summary["num_params_score"] = sum(
            p.numel() for p in score.parameters() if p.requires_grad
        )
        if use_score_pass_through:
            score = ScoreWithIdentityGradWrapper(score=score)

        X, Y, _ = spatial_mesh.shape
        nr_channels = arch["in_channels"]
        fixed_noise_sample = torch.randn((1, nr_channels, X, Y), device=device)

        return partial(
            diffusion_variational_idm_regularizer,
            spatial_mesh=spatial_mesh,
            kspace_mesh=kspace_mesh,
            time_mesh=time_mesh,
            score=score,
            sde=sde,
            device=device,
            fixed_noise_sample=fixed_noise_sample,
            **reg_params
        )

    else:
        raise ValueError(f"Unknown diffusion regularizer function name: {name}")
