"""Helpers for working with the different DDPM output parameterizations.

The models in this repo are trained with a diffusers ``DDPMScheduler`` whose
``prediction_type`` decides what the network output means.  With

    x_t = sqrt(a_bar) * x_0 + sqrt(1 - a_bar) * eps

the three supported parameterizations are

    prediction_type   training target                        eps from output                    x_0 from output
    ---------------   ------------------------------------   -------------------------------   -------------------------------
    epsilon           eps                                    out                               (x_t - sqrt(1-a_bar)*out)/sqrt(a_bar)
    v_prediction      sqrt(a_bar)*eps - sqrt(1-a_bar)*x_0     sqrt(a_bar)*out + sqrt(1-a_bar)*x_t   sqrt(a_bar)*x_t - sqrt(1-a_bar)*out
    sample            x_0                                    (x_t - sqrt(a_bar)*out)/sqrt(1-a_bar)  out
"""

from typing import Tuple

import torch


SUPPORTED_PREDICTION_TYPES = ("epsilon", "v_prediction", "sample")


def get_prediction_type(noise_scheduler) -> str:
    """Return the scheduler's prediction type, defaulting to ``epsilon``."""
    return str(getattr(noise_scheduler.config, "prediction_type", "epsilon"))


def _alpha_terms(
    noise_scheduler,
    timesteps: torch.Tensor,
    reference: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (sqrt(a_bar_t), sqrt(1 - a_bar_t)) broadcastable against ``reference``."""
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(
        device=reference.device,
        dtype=reference.dtype,
    )
    alpha_bar_t = alphas_cumprod[timesteps.to(reference.device)]
    shape = (-1,) + (1,) * (reference.dim() - 1)
    alpha_bar_t = alpha_bar_t.view(shape)
    sqrt_alpha_bar_t = alpha_bar_t.clamp_min(0.0).sqrt().clamp_min(1e-8)
    sqrt_one_minus_alpha_bar_t = (1.0 - alpha_bar_t).clamp_min(0.0).sqrt().clamp_min(1e-8)
    return sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t


def get_training_target(
    noise_scheduler,
    clean_sample: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """The regression target the network is trained against for this scheduler."""
    prediction_type = get_prediction_type(noise_scheduler)
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "v_prediction":
        # v = sqrt(a_bar) * eps - sqrt(1 - a_bar) * x_0
        return noise_scheduler.get_velocity(clean_sample, noise, timesteps)
    if prediction_type == "sample":
        return clean_sample
    raise ValueError(
        f"Unsupported scheduler prediction_type: {prediction_type!r} "
        f"(supported: {SUPPORTED_PREDICTION_TYPES})"
    )


def model_output_to_epsilon(
    noise_scheduler,
    model_output: torch.Tensor,
    noisy_sample: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Convert a network output into the equivalent epsilon estimate.

    This is the inverse of :func:`get_training_target`: feeding the exact training
    target back in returns the noise that generated ``noisy_sample``.
    """
    prediction_type = get_prediction_type(noise_scheduler)
    if prediction_type == "epsilon":
        return model_output
    sqrt_ab, sqrt_1mab = _alpha_terms(noise_scheduler, timesteps, noisy_sample)
    if prediction_type == "v_prediction":
        return sqrt_ab * model_output + sqrt_1mab * noisy_sample
    if prediction_type == "sample":
        return (noisy_sample - sqrt_ab * model_output) / sqrt_1mab
    raise ValueError(
        f"Unsupported scheduler prediction_type: {prediction_type!r} "
        f"(supported: {SUPPORTED_PREDICTION_TYPES})"
    )


def predict_clean_from_model_output(
    noise_scheduler,
    model_output: torch.Tensor,
    noisy_sample: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Recover the predicted clean sample x_0 from a network output (Tweedie estimate)."""
    prediction_type = get_prediction_type(noise_scheduler)
    if prediction_type == "sample":
        return model_output
    sqrt_ab, sqrt_1mab = _alpha_terms(noise_scheduler, timesteps, noisy_sample)
    if prediction_type == "epsilon":
        return (noisy_sample - sqrt_1mab * model_output) / sqrt_ab
    if prediction_type == "v_prediction":
        return sqrt_ab * noisy_sample - sqrt_1mab * model_output
    raise ValueError(
        f"Unsupported scheduler prediction_type: {prediction_type!r} "
        f"(supported: {SUPPORTED_PREDICTION_TYPES})"
    )
