"""Latent-space (SD-VAE) variant of the coupling ("cpl") regularizer and the frozen VAE encoder
shared with the coupling model's latent training transform. Only the coupling loss is evaluated in latent space."""
from __future__ import annotations

import os
from functools import partial
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from diffusers import AutoencoderKL

from src.prefect.setup import StoragePath
from src.regularizers.diffusers_cpl_regularizers import (
    load_cpl_checkpoint_kwargs,
    variational_coupling_regularizer,
)
from src.utils.latent_motion import motion_to_latent

# diffusers default for SD 1.x/2.x VAEs; overridden by the checkpoint's config.scaling_factor
SD_VAE_SCALING_FACTOR_DEFAULT = 0.18215
_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


class FrozenLatentEncoder:
    """Frozen SD-style VAE mapping pixel video frames and motion fields to its latent grid.

    encode is NOT under no_grad: gradients must flow through the encoder back to the pixel-space video.
    """

    def __init__(
        self,
        pretrained_model_dir: str = "model_cache/sd15",
        device: str = "cpu",
        vae_subfolder: Optional[str] = "vae",
        assume_pixel_range: str = "0_1",  # "0_1" or "minus1_1"
        scaling_factor: Optional[float] = None,
        path_resolver: Optional[Callable[[str], str]] = None,
        autocast_dtype: Optional[str] = None,  # e.g. "bfloat16": run the encoder under autocast (halves activation memory)
    ):
        self.device = device
        if assume_pixel_range not in ("0_1", "minus1_1"):
            raise ValueError(f"Unknown assume_pixel_range: {assume_pixel_range!r}")
        self.assume_pixel_range = assume_pixel_range
        self.autocast_dtype = _DTYPES[autocast_dtype] if autocast_dtype else None

        model_dir = path_resolver(pretrained_model_dir) if path_resolver is not None else pretrained_model_dir
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"FrozenLatentEncoder: VAE directory {model_dir} not found; add `sd15` to the flow's "
                f"required_datasets (hydra/task/download_dataset/sd15.yaml) so it is staged first."
            )
        kwargs = {"subfolder": vae_subfolder} if vae_subfolder else {}
        self.vae = AutoencoderKL.from_pretrained(model_dir, local_files_only=True, **kwargs).to(device)
        self.vae.requires_grad_(False)
        self.vae.eval()

        self.scaling_factor = (
            float(scaling_factor)
            if scaling_factor is not None
            else float(getattr(self.vae.config, "scaling_factor", SD_VAE_SCALING_FACTOR_DEFAULT))
        )
        # first down_block never downsamples; SD VAE has 4 blocks -> f=8
        self.downsample_factor = 2 ** (len(self.vae.config.down_block_types) - 1)
        self.latent_channels = int(self.vae.config.latent_channels)

    def _to_vae_input(self, x: Tensor) -> Tensor:
        return x * 2.0 - 1.0 if self.assume_pixel_range == "0_1" else x

    def encode(self, x: Tensor) -> Tensor:
        """x: (B, C, H, W) pixel frames in `assume_pixel_range` -> (B, latent_channels, H/f, W/f)
        scaled latents (posterior mean, not a sample)."""
        x = self._to_vae_input(x.to(self.device))
        if self.autocast_dtype is not None and str(self.device).startswith("cuda"):
            with torch.autocast("cuda", dtype=self.autocast_dtype):
                z = self.vae.encode(x).latent_dist.mode()
            z = z.float()
        else:
            z = self.vae.encode(x.to(self.vae.dtype)).latent_dist.mode()
        return z * self.scaling_factor

    def decode(self, z: Tensor) -> Tensor:
        """Inverse of encode (for visualisation): scaled latents -> pixel frames in `assume_pixel_range`."""
        x = self.vae.decode(z.to(self.device, self.vae.dtype) / self.scaling_factor).sample
        return (x + 1.0) / 2.0 if self.assume_pixel_range == "0_1" else x

    def motion_to_latent(self, v: Tensor) -> Tensor:
        return motion_to_latent(v.to(self.device), self.downsample_factor)

    def encode_video_and_motion(self, video: Tensor, motion: Tensor, motion_bwd: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """video: (F,C,H,W) pixels; motion/motion_bwd: (N,2,H,W) pixel-unit flows -> latent-space
        equivalents (the `domain_transform` signature)."""
        return self.encode(video), self.motion_to_latent(motion), self.motion_to_latent(motion_bwd)


def autoencoder_gluing_loss(
    *,
    latent_encoder: FrozenLatentEncoder,
    model,
    scale_factor=None,
    apply_scaling_factor_from_setup: bool = True,
    detach_target: bool = True,
) -> Tensor:
    """PSLD-style gluing penalty mean((x - D(E(x)))^2) on the pixel video.

    Penalises the part of x the frozen autoencoder cannot represent. detach_target=True treats
    D(E(x)) as constant (no backward through the VAE). Returned in the scaled optimisation domain.
    """
    sf = 1.0
    if apply_scaling_factor_from_setup and scale_factor is not None:
        sf = scale_factor

    video = model.video_param * sf if apply_scaling_factor_from_setup and scale_factor is not None else model.video_param
    x_pix = video.squeeze(dim=1).movedim(-1, 1)  # (T, C, H, W), [0, 1] pixel domain

    if detach_target:
        with torch.no_grad():
            x_roundtrip = latent_encoder.decode(latent_encoder.encode(x_pix.detach()))
        glue = (x_pix - x_roundtrip.to(x_pix.dtype)).square().mean()
    else:
        glue = (x_pix - latent_encoder.decode(latent_encoder.encode(x_pix)).to(x_pix.dtype)).square().mean()

    if apply_scaling_factor_from_setup and scale_factor is not None:
        # back into the domain the data-consistency loss is computed in
        glue = glue / (sf.square().reshape(()) if torch.is_tensor(sf) else sf ** 2)
    return glue


def variational_coupling_regularizer_latent(
    *,
    latent_encoder: FrozenLatentEncoder,
    glue_weight: float = 0.0,
    glue_detach_target: bool = True,
    **kwargs,
):
    """variational_coupling_regularizer with video + motion transported to the frozen VAE's latent grid.

    glue_weight > 0 adds glue_weight * autoencoder_gluing_loss(...)."""
    loss = variational_coupling_regularizer(domain_transform=latent_encoder.encode_video_and_motion, **kwargs)

    model = kwargs.get("model", None)
    if not glue_weight or model is None:
        return loss

    glue = autoencoder_gluing_loss(
        latent_encoder=latent_encoder,
        model=model,
        scale_factor=kwargs.get("scale_factor", None),
        apply_scaling_factor_from_setup=kwargs.get("apply_scaling_factor_from_setup", True),
        detach_target=glue_detach_target,
    )
    try:  # piggy-back on the fit loop's own wandb step (commit=False), never a step of its own
        import wandb
        if wandb.run is not None:
            wandb.log({"reg/glue_mse": glue.item()}, commit=False)
    except Exception:  # noqa: BLE001 - logging must never break the fit
        pass
    return loss + glue_weight * glue


def diffusers_cpl_latent_regularizer_resolver(
    name,
    spatial_mesh,
    kspace_mesh,
    time_mesh,
    frame_processor_for_visualization,
    device,
    pretrained_model_name_or_path,   # unused (kept for signature parity with the pixel resolver)
    image_conditioning_use_target,
    target_image,
    reg_params,
    accelerator,
    pretrained_model_path: StoragePath,  # the *latent* coupling-model checkpoint (trained with reds_rgb_model_latent)
    pretrained_model_path2: Optional[StoragePath] = None,
    pretrained_model_dir: str = "model_cache/sd15",
    vae_subfolder: Optional[str] = "vae",
    assume_pixel_range: str = "0_1",
    autocast_dtype: Optional[str] = None,
    path_resolver: Optional[Callable[[str], str]] = None,
    **params,
):
    if name != "diffuser_pretrained_cpl_regularizer_latent":
        raise ValueError(f"Unknown diffusers_cpl_latent regularizer name: {name}")

    latent_encoder = FrozenLatentEncoder(
        pretrained_model_dir=pretrained_model_dir,
        device=device,
        vae_subfolder=vae_subfolder,
        assume_pixel_range=assume_pixel_range,
        path_resolver=path_resolver,
        autocast_dtype=autocast_dtype,
    )
    print(f"[CPL Reg/latent] VAE f={latent_encoder.downsample_factor} c={latent_encoder.latent_channels} scaling={latent_encoder.scaling_factor}")

    common_kwargs, extra_dict = load_cpl_checkpoint_kwargs(pretrained_model_path, device, params)
    # model_channel_nr = latent_channels + condition channels
    try:
        expected = latent_encoder.latent_channels + int(common_kwargs["coupling_model"].get_extra_input_channels(latent_encoder.latent_channels))
    except Exception:  # noqa: BLE001 - advisory check only
        expected = None
    if expected is not None and int(common_kwargs["model_channel_nr"]) != expected:
        raise ValueError(
            f"[CPL Reg/latent] checkpoint UNet has {common_kwargs['model_channel_nr']} input channels, a latent "
            f"coupling model on this VAE needs {expected}: this checkpoint was not trained in latent space "
            f"(train with +exps=reds_rgb_model_latent)."
        )

    return partial(
        variational_coupling_regularizer_latent,
        latent_encoder=latent_encoder,
        spatial_mesh=spatial_mesh,
        kspace_mesh=kspace_mesh,
        time_mesh=time_mesh,
        **common_kwargs,
        **reg_params,
        **params["reg_params_var_spec"],
        **extra_dict,
    )
