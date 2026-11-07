import tempfile
import torch
from torch import Tensor
from src.prefect.setup import load_file_system, StoragePath
from src.representations.base_representation import BaseRepresentation

from functools import partial
import torch
from omegaconf import OmegaConf
import math
from typing import Optional
from torch import Tensor
from diffusers import DDPMScheduler
from src.tasks.train_probabilistic_coupling_model import LearntCouplingModel

class GradientPassThrough(torch.autograd.Function):
    """RED-diff style score evaluation: network in the forward pass, identity surrogate for its
    Jacobian in the backward pass. The condition channels of the concatenated input get zero gradient.
    """
    @staticmethod
    def forward(ctx, result, t, condition_variable, model):
        ctx.input_shape = result.shape
        # Call the transformer as usual
        return model(
            result, t, condition_variable
        ).sample

    @staticmethod
    def backward(ctx, grad_output):
        # Skip Jacobian → identity on the sample channels, no gradient into the condition
        n_sample_channels = grad_output.shape[1]
        assert n_sample_channels <= ctx.input_shape[1], (
            f"GradientPassThrough: model output has {n_sample_channels} channels but the "
            f"concatenated input only has {ctx.input_shape[1]}"
        )
        if grad_output.shape == ctx.input_shape:
            grad_input = grad_output
        else:
            grad_input = grad_output.new_zeros(ctx.input_shape)
            grad_input[:, :n_sample_channels] = grad_output
        return (grad_input, None, None, None) # first one gets gradient

def get_forward_fn(
    download_directory_to_temp_on_enter_mgr,
    use_final_model: bool,
    use_epoch: int,
    use_pass_through: bool,
    device,
    model_type: str,
    use_ema : bool = True
):

    if use_final_model:
        model_subfolder = "model_final"
    elif use_ema:
        model_subfolder = f"ema_model_epoch_{use_epoch}"
    else:
        model_subfolder = f"model_epoch_{use_epoch}"
    print(f"Using model from subfolder: {model_subfolder}")

    model_channel_nr = None
    with download_directory_to_temp_on_enter_mgr(ex_subfolder=model_subfolder):
        # assumed shape for the sample: (B, C, T, H, W), some models need (B, T, C, H, W)
        if model_type == "UNet2DCondition":
            from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
            model = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path="./").to(device)
            model_channel_nr = int(model._internal_dict.in_channels)
            zero_condition = torch.zeros((1, 1, 1), device=device)
            if use_pass_through:
                forward_fn = (
                    lambda sample, timestep, condition: GradientPassThrough.apply(
                        torch.cat([sample, condition], dim=1),  # shape (B, 2, C, H, W)
                        timestep,
                        zero_condition.repeat_interleave(sample.shape[0], dim=0),
                        model,
                    )
                )
            else:
                forward_fn = lambda sample, timestep, condition: model(
                    sample=torch.cat([sample, condition], dim=1),  # shape (B, 2, C, H, W)
                    timestep=timestep,
                    encoder_hidden_states=zero_condition.repeat_interleave(
                        sample.shape[0], dim=0
                    ),
                ).sample
        else:
            raise ValueError(f"Unknown model type {model_type}")

    return forward_fn, model_channel_nr

def get_train_wandb_kwargs(download_directory_to_temp_on_enter):
    # used to find out how the model is pretrained (e.g. on motion or video)
    with download_directory_to_temp_on_enter(ex_subfolder="wandb_params"):
        model_train_cfg = OmegaConf.load("wandb_kwargs.yaml")
    return model_train_cfg

def get_coupling_config(download_directory_to_temp_on_enter):
    with download_directory_to_temp_on_enter(ex_subfolder="coupling_config"):
        coupling_config = OmegaConf.load("model_cfg.yaml")
    return coupling_config

def get_residual_scaling(download_directory_to_temp_on_enter):
    """Load residual_mean / residual_std saved during training (residual_scaling.yaml).
    Returns (residual_mean, residual_std); falls back to (0.0, 1.0) if not found."""
    try:
        with download_directory_to_temp_on_enter(ex_subfolder=""):
            import yaml as _yaml
            with open("residual_scaling.yaml", "r") as _f:
                _cfg = _yaml.safe_load(_f)
        residual_mean = _cfg.get("residual_mean", 0.0)
        residual_std  = _cfg.get("residual_std",  1.0)
        print(f"[CPL Reg] Loaded residual scaling: mean={residual_mean}, std={residual_std}")
        return residual_mean, residual_std
    except Exception as e:
        print(f"[CPL Reg] Could not load residual_scaling.yaml ({e}); using mean=0, std=1")
        return 0.0, 1.0

def get_forward_fn_and_noise_scheduler(download_directory_to_temp_on_enter, use_final_model : bool, use_epoch : int, use_pass_through: bool, device, model_type : str, use_ema : bool = False):

    forward_fn, model_channel_nr = get_forward_fn(download_directory_to_temp_on_enter, use_final_model, use_epoch, use_pass_through, device, model_type, use_ema=use_ema)

    with download_directory_to_temp_on_enter(ex_subfolder="noise_scheduler"):
        noise_scheduler = DDPMScheduler.from_pretrained("./")

    return forward_fn, noise_scheduler, model_channel_nr

def get_snr_inv(noise_scheduler : DDPMScheduler, timesteps, original_samples):
    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=original_samples.device)
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(dtype=original_samples.dtype)
    timesteps = timesteps.to(original_samples.device)

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    return (sqrt_one_minus_alpha_prod / sqrt_alpha_prod)

from src.utils.diffusion_parameterization import (
    _alpha_terms,
    get_prediction_type,
    get_training_target,
    model_output_to_epsilon,
)
def video_normalization(rasterized_video, normalization):
    if normalization == "gaussian":
        # normalize to zero mean and unit variance
        rasterized_video = (rasterized_video - rasterized_video.mean()) / (rasterized_video.std() + 1e-8)
    elif normalization == "uniform_-1_1":
        # normalize to [-1, 1]
        rasterized_video = 2 * (rasterized_video - rasterized_video.min()) / (rasterized_video.max() - rasterized_video.min() + 1e-8) - 1
    elif normalization == "uniform_0_1":
        # normalize to [0, 1]
        rasterized_video = (rasterized_video - rasterized_video.min()) / (rasterized_video.max() - rasterized_video.min() + 1e-8)
    elif normalization == "video_div_by_std":
        # normalize video by its own std
        rasterized_video = rasterized_video / (rasterized_video.std())
    elif normalization == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization type: {normalization}")
    return rasterized_video

from src.representations.base_representation import BaseRepresentation
from src.representations.videovoxel_representation import VideoVoxelRepresentation

def variational_coupling_regularizer(
    # ---
    model: BaseRepresentation,  # representation model, e.g. BaseImageWithINNDeformationField
    result: Tensor,  # kspace line result or part of rasterized image
    kspace: Tensor,  # kspace line target or part of rasterized target image
    kcoords: Tensor,
    timesteps: Tensor,  # timesteps for the kspace line
    spatial_mesh: Tensor,  # spatial mesh for interpolation
    kspace_mesh: Tensor,  # kspace mesh for interpolation
    time_mesh: Tensor,  # time mesh for interpolation
    # ---
    forward_fn, # this can be a diffusion pipeline as before
    noise_scheduler,
    coupling_model : LearntCouplingModel,
    weight : float,
    num_train_steps_factor_lower : float,
    num_train_steps_factor_upper : float,
    prior_scaling: float = 1.0,
    adapt_reg_strength: bool = False,
    reg_strength_mode: str = "snr_inv", # only used when adapt_reg_strength=True, see below

    include_backward : bool = True,
    occlusion_weight : float = 1.0,
    occlusion_fwd_relative : float = 1.0,
    occlusion_bwd_relative : float = 1.0,
    tau_pix : float = 1.0,
    conf_floor : float = 0.0,
    enable_occlusion_masking : bool = False,
    timestep_provided : Optional[int] = None, # fixed timestep for all samples instead of random sampling
    take_provided_timestep : bool = False,
    channel_timestep_sampling_independent : bool = False,
    timestep_sampling_random : bool = True,
    scale_factor : float = 1.0,
    apply_scaling_factor_from_setup : bool = True,
    normalization : str = "none",
    loss_norm : str = "l2",
    target_l1_weight : Optional[float] = None, # if set, scale of the classical l1 warp-residual term
    optimization_progress : Optional[float] = None, # fraction of the fit loop completed, in [0, 1]
    use_sin_cos_annealing: bool = False,
    use_two_phase_schedule: bool = False, # hard switch from L1-only to diffusion-only at phase_split
    phase_split: float = 0.5, # fraction of optimization_progress at which phase 1 (classical only) switches to phase 2 (diffusion only)
    residual_mean: float = 0.0,
    residual_std: float = 1.0,
    use_target_l1_same_as_diffusion_loss_if_none : bool = True, # if target_l1 is None, use the same as diffusion loss
    recon_max_frame_distance : Optional[int] = 1, # d ~ U{1..K}; 1 = consecutive pairs, None = K from the checkpoint
    variational_loss_space : str = "epsilon", # space in which the score-matching residual is formed, see below
    domain_transform=None, # optional (video, motion, motion_bwd) transform, e.g. to VAE latents
    **kwargs,
):

    assert isinstance(model, VideoVoxelRepresentation)

    _d = coupling_model.sample_recon_frame_distance(recon_max_frame_distance)

    video = (model.video_param * scale_factor if apply_scaling_factor_from_setup else model.video_param) * prior_scaling
    if normalization == "none":
        pass
    elif normalization == "times2_minus1":
        video = video * 2 - 1
    elif normalization == "video_div_by_std":
        video = video / (video.std().detach() + 1e-8)
    else:
        raise ValueError(f"Unknown normalization type: {normalization}")

    target, condition, aux_params = coupling_model.video_voxel_repr_to_training_data(
        video_param=video,
        motion_param=model.get_velocity_field(scale=False),
        motion_bwd_param=model.get_velocity_field(scale=False, ret_bwd=True),
        return_aux_params=True,
        frame_distance=_d,
        domain_transform=domain_transform,
    )

    conf = model.get_occlusion_weighting()  # (2, T-1, X, Y)
    if _d == 1:
        conf_fwd_model = conf[0]   # (T-1, H, W)
        conf_bwd_model = conf[1]
    else:
        # Average single-step occlusion weights over the d steps of each pair.
        T_minus1 = conf.shape[1]
        n_pairs = T_minus1 - _d + 1
        conf_fwd_model = torch.stack([conf[0, t:t+_d].mean(0) for t in range(n_pairs)], dim=0)
        conf_bwd_model = torch.stack([conf[1, t:t+_d].mean(0) for t in range(n_pairs)], dim=0)
    _occ_total = occlusion_fwd_relative + occlusion_bwd_relative
    occlusion_fwd_relative = occlusion_fwd_relative / _occ_total
    occlusion_bwd_relative = occlusion_bwd_relative / _occ_total
    # aux conf maps live on domain_transform's grid (e.g. latent); resize to the model's grid
    _aux_conf_fwd, _aux_conf_bwd = aux_params["conf_fwd"], aux_params["conf_bwd"]
    if _aux_conf_fwd.shape[-2:] != conf_fwd_model.shape[-2:]:
        _aux_conf_fwd = torch.nn.functional.interpolate(_aux_conf_fwd.unsqueeze(1), size=conf_fwd_model.shape[-2:], mode="nearest").squeeze(1)
        _aux_conf_bwd = torch.nn.functional.interpolate(_aux_conf_bwd.unsqueeze(1), size=conf_bwd_model.shape[-2:], mode="nearest").squeeze(1)
    conf_loss = occlusion_weight * (
        occlusion_fwd_relative * (conf_fwd_model - _aux_conf_fwd).square().mean() +
        occlusion_bwd_relative * (conf_bwd_model - _aux_conf_bwd).square().mean()
    )

    # warp residual for the classical L1 term, laid out like `target`
    mc_residual = aux_params["mc_residual"]

    # Normalize target to match training distribution: during training image = (image_raw - rm) / rs
    from src.tasks.train_probabilistic_coupling_model import _as_broadcastable
    _rm = _as_broadcastable(residual_mean, model.device)
    _rs = _as_broadcastable(residual_std,  model.device)
    target = (target - _rm) / _rs

    target_l1_loss = mc_residual.abs().mean()

    assert not (use_sin_cos_annealing and use_two_phase_schedule), (
        "use_sin_cos_annealing and use_two_phase_schedule are mutually exclusive"
    )

    diff_weight = weight
    if target_l1_weight is None and use_target_l1_same_as_diffusion_loss_if_none:
        l1_weight = weight
    l1_weight = target_l1_weight if target_l1_weight is not None else 0.0

    if use_sin_cos_annealing:
        assert optimization_progress is not None, "For sin-cos annealing, optimization_progress must be provided"
        sin_cos_factor = 0.5 * (1 + math.cos(math.pi * optimization_progress)) # starts with 1 and ends with 0
        diff_weight = (1 - sin_cos_factor) * diff_weight
        l1_weight = sin_cos_factor * l1_weight
    elif use_two_phase_schedule:
        assert optimization_progress is not None, "For the two-phase schedule, optimization_progress must be provided"
        if phase_split >= 1.0 or optimization_progress < phase_split:
            diff_weight = 0.0  # phase 1: classical (L1) prior only
        else:
            l1_weight = 0.0  # phase 2: diffusion model only

    # skip the expensive diffusion forward pass when it is weighted by zero
    if diff_weight == 0.0:
        if loss_norm in ("l1", "l2"):
            return l1_weight * target_l1_loss + conf_loss
        else:
            raise ValueError(f"Unknown loss norm: {loss_norm}")

    noise_sample = torch.randn_like(target)

    # here we perform independent slicing
    if timestep_provided is not None and take_provided_timestep:
        diff_timesteps = torch.full(
            size=(noise_sample.shape[0],),
            fill_value=timestep_provided.float().mean(),
            device=model.device,
        ).int()
    elif channel_timestep_sampling_independent:
        if timestep_sampling_random:
            diff_timesteps = torch.randint(
                math.floor(num_train_steps_factor_lower * noise_scheduler.num_train_timesteps),
                math.ceil(num_train_steps_factor_upper * noise_scheduler.num_train_timesteps),
                (noise_sample.shape[0],),
                device=model.device,
            )
        else:
            # deterministic descending schedule: t_hi at progress 0 -> t_lo at progress 1
            assert optimization_progress is not None, "For non-random timestep sampling with independent channel-timestep sampling, optimization_progress must be provided"
            _t_lo = math.floor(num_train_steps_factor_lower * noise_scheduler.num_train_timesteps)
            _t_hi = math.ceil(num_train_steps_factor_upper * noise_scheduler.num_train_timesteps) - 1
            _t = _t_lo + int(round((1.0 - float(optimization_progress)) * (_t_hi - _t_lo)))
            diff_timesteps = torch.full(
                (noise_sample.shape[0],),
                int(min(max(_t, _t_lo), _t_hi)),
                device=model.device,
            )
    else:
        diff_timesteps = torch.randint(
            math.floor(num_train_steps_factor_lower * noise_scheduler.num_train_timesteps),
            math.ceil(num_train_steps_factor_upper * noise_scheduler.num_train_timesteps),
            (1,),
            device=model.device,
        ).repeat_interleave(noise_sample.shape[0], dim=0)

    # proceed with variational loss
    noisy_video = noise_scheduler.add_noise(target, noise_sample, diff_timesteps) # (B*(T-1), C, H, W)

    # fwd pass and condition
    model_out = forward_fn(noisy_video, diff_timesteps, condition=condition) # (B*(T-1), C, H, W)

    if adapt_reg_strength:
        reg_strength_t = get_snr_inv(noise_scheduler, diff_timesteps, noisy_video).flatten()
        # "snr_inv" weights the squared residual (effective sqrt(1-a_bar) in the update);
        # "red_diff_gradient" divides out sqrt(a_bar) to match the RED-diff paper's lambda_t
        if reg_strength_mode == "red_diff_gradient":
            sqrt_alpha_bar_t, _ = _alpha_terms(noise_scheduler, diff_timesteps, reg_strength_t)
            reg_strength_t = reg_strength_t / sqrt_alpha_bar_t.flatten()
        elif reg_strength_mode != "snr_inv":
            raise ValueError(
                f"Unknown reg_strength_mode: {reg_strength_mode!r} "
                "(expected 'snr_inv' or 'red_diff_gradient')"
            )
    else:
        reg_strength_t = torch.ones( (model_out.shape[0],), device=model.device)

    # By default the residual is formed in epsilon space (weights were tuned against it);
    # "native" matches the training loss, which for v_prediction differs by a factor a_bar.
    if variational_loss_space == "native":
        diff_target = get_training_target(
            noise_scheduler=noise_scheduler,
            clean_sample=target,
            noise=noise_sample,
            timesteps=diff_timesteps,
        )
        diff_pred = model_out
    elif variational_loss_space in ("epsilon", "epsilon_detached"):
        diff_target = noise_sample
        diff_pred = model_output_to_epsilon(
            noise_scheduler=noise_scheduler,
            model_output=model_out,
            noisy_sample=noisy_video.detach() if variational_loss_space == "epsilon_detached" else noisy_video,
            timesteps=diff_timesteps,
        )
    else:
        raise ValueError(
            f"Unknown variational_loss_space: {variational_loss_space!r} "
            "(expected 'epsilon', 'epsilon_detached' or 'native')"
        )

    diff = (diff_pred - diff_target) # shape (B*(T-1), C, H, W) || do we need abs here?

    # occlusion mask has shape: (T-1, H, W)
    if enable_occlusion_masking:
        if include_backward:
            conf = torch.cat([aux_params["conf_fwd"], aux_params["conf_bwd"]], dim=0)  # (2*(T-1), H, W)
        else:
            conf = aux_params["conf_fwd"]  # (T-1, H, W)
        diff = diff * conf[:, None]  # (T-1,C,H,W) * (T-1,H,W) -> (T-1,C,H,W)


    # reshape, not view: diff is not always contiguous
    if loss_norm == "l2":
        return diff_weight * (reg_strength_t[:,None] * diff.square().reshape(diff.shape[0], -1)).mean() + conf_loss + l1_weight * target_l1_loss
    elif loss_norm == "l1":
        return diff_weight * (reg_strength_t[:,None] * diff.abs().reshape(diff.shape[0], -1)).mean() + conf_loss + l1_weight * target_l1_loss
    else:
        raise ValueError(f"Unknown loss norm: {loss_norm}")

def load_cpl_checkpoint_kwargs(pretrained_model_path: StoragePath, device, params: dict):
    """Load a trained LearntCouplingModel checkpoint and resolve the kwargs shared by all diffusers_cpl
    regularizer variants. Returns (common_kwargs, extra_dict)."""
    with tempfile.TemporaryDirectory() as tmp_dir_path:
        from src.prefect.setup import download_directory_to_temp_on_enter
        download_directory_to_temp_on_enter_mgr = partial(
            download_directory_to_temp_on_enter,
            filesystem=load_file_system(pretrained_model_path.storage_settings),
            temp_folder=tmp_dir_path,
            fs_folder=pretrained_model_path.storage_path)
        forward_fn, noise_scheduler, model_channel_nr = get_forward_fn_and_noise_scheduler(download_directory_to_temp_on_enter_mgr, device=device, **params["forward_fn_params"])
        print(f"[CPL Reg] Checkpoint prediction_type: '{get_prediction_type(noise_scheduler)}'")

        wandb_train_cfg = get_train_wandb_kwargs(download_directory_to_temp_on_enter_mgr)
        coupling_config = get_coupling_config(download_directory_to_temp_on_enter_mgr)
        residual_mean, residual_std = get_residual_scaling(download_directory_to_temp_on_enter_mgr)

    extra_dict = {}
    if "prior_scaling" in params["reg_params_var_spec"]:
        print(f"Using prior scaling from reg_params: {params['reg_params_var_spec']['prior_scaling']}")
        if "coupling_transform.constant_scaling_factor" in wandb_train_cfg:
            scaling_factor_by_wandb_cfg = wandb_train_cfg['coupling_transform.constant_scaling_factor']
            if float(scaling_factor_by_wandb_cfg) != float(params["reg_params_var_spec"]['prior_scaling']):
                print(f"Conflict between prior scaling in reg_params: {params['reg_params_var_spec']['prior_scaling']} and wandb config: {scaling_factor_by_wandb_cfg}. Using reg_params value.")
    elif "coupling_transform.constant_scaling_factor" in wandb_train_cfg:
        extra_dict['prior_scaling'] = wandb_train_cfg['coupling_transform.constant_scaling_factor']
        print(f"Using prior scaling from wandb config")

    if "coupling_transform.constant_scaling_factor" in wandb_train_cfg:
        print(f"Using prior scaling from wandb config: {wandb_train_cfg['coupling_transform.constant_scaling_factor']}")
    else:
        print(f"No prior scaling found in wandb config or reg_params, using default of 1.0")

    coupling_model = LearntCouplingModel(**dict(coupling_config))

    common_kwargs = dict(
        forward_fn=forward_fn,
        noise_scheduler=noise_scheduler,
        model_channel_nr=model_channel_nr,
        wandb_train_cfg=wandb_train_cfg,
        coupling_model=coupling_model,
        residual_mean=residual_mean,
        residual_std=residual_std,
    )
    return common_kwargs, extra_dict


def diffusers_cpl_regularizer_resolver(
    name,
    spatial_mesh,
    kspace_mesh,
    time_mesh,
    frame_processor_for_visualization,
    device,
    pretrained_model_name_or_path,
    image_conditioning_use_target,
    target_image,
    reg_params,
    accelerator,
    pretrained_model_path : StoragePath,
    pretrained_model_path2 : Optional[StoragePath] = None, # for coupling model pretrained separately from the video model
    **params,
):

    if name == "diffuser_pretrained_cpl_regularizer":
        common_kwargs, extra_dict = load_cpl_checkpoint_kwargs(pretrained_model_path, device, params)

        return partial(
            variational_coupling_regularizer,
            spatial_mesh=spatial_mesh,
            kspace_mesh=kspace_mesh,
            time_mesh=time_mesh,
            **common_kwargs,
            **reg_params,
            **params["reg_params_var_spec"],
            **extra_dict
        )
    else:
        raise ValueError(f"Unknown diffusers_idm regularizer name: {name}")
