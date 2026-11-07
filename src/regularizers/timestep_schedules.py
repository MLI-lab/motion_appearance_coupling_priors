"""Timestep annealing schedules for variational diffusion regularizers.

Anneal the DSM timestep from t_hi (coarse structure) to t_lo (fine detail) as optimization_progress
goes 0 -> 1, with optional jitter. Not from Mardani et al. (2023); that is the SNR loss weighting.
"""

import math
from typing import Optional, Union

import torch
from torch import Tensor


def compute_scheduled_timestep(
    t_lo: int,
    t_hi: int,
    batch_size: int,
    device: Union[torch.device, str],
    schedule: str = "random",
    optimization_progress: Optional[float] = None,
    jitter_fraction: float = 0.2,
    consistent_timestep: bool = False,
    ddim_num_steps: Optional[int] = None,
) -> Tensor:
    """Return a (batch_size,) long tensor of diffusion timesteps in [t_lo, t_hi) according to *schedule*.

    schedule: "random", "ddim", "linear", "sqrt", "cosine", "quadratic", "log" or "inverse_sqrt".
    optimization_progress in [0, 1] is required for all but "random".
    jitter_fraction: fraction of (t_hi - t_lo) used as jitter around the centre (0 = deterministic, 1 = random).
    """
    assert t_hi > t_lo, f"t_hi ({t_hi}) must be > t_lo ({t_lo})"
    span = t_hi - t_lo  # > 0

    if schedule == "random":
        n = 1 if consistent_timestep else batch_size
        t = torch.randint(t_lo, t_hi, (n,), device=device)
        if consistent_timestep:
            t = t.expand(batch_size)
        return t

    # For all annealing schedules we need optimization_progress
    if optimization_progress is None:
        # Fallback to random if progress is unavailable
        n = 1 if consistent_timestep else batch_size
        t = torch.randint(t_lo, t_hi, (n,), device=device)
        if consistent_timestep:
            t = t.expand(batch_size)
        return t

    p = max(0.0, min(1.0, optimization_progress))  # clamp

    # Compute fraction-of-span already descended (0 → at t_hi, 1 → at t_lo)
    if schedule == "ddim":
        # DDIM-like respacing: choose a deterministic point on a coarse
        # grid in timestep space and optionally add local jitter.
        num_ddim_steps = int(span if ddim_num_steps is None else ddim_num_steps)
        num_ddim_steps = max(2, min(num_ddim_steps, span))
        if num_ddim_steps < 2:
            num_ddim_steps = 2
        grid = torch.linspace(
            float(t_hi - 1),
            float(t_lo),
            steps=int(num_ddim_steps),
            device=device,
            dtype=torch.float32,
        )
        idx = int(round(p * (num_ddim_steps - 1)))
        idx = max(0, min(int(num_ddim_steps - 1), idx))
        centre = float(grid[idx].item())
        # Jitter is local to one DDIM interval.
        local_span = max(1.0, span / float(num_ddim_steps - 1))
        half_jitter = 0.5 * jitter_fraction * local_span
        lo_j = max(t_lo, centre - half_jitter)
        hi_j = min(t_hi, centre + half_jitter + 1.0)

        lo_j_int = max(t_lo, int(math.floor(lo_j)))
        hi_j_int = max(lo_j_int + 1, min(t_hi, int(math.ceil(hi_j))))

        n = 1 if consistent_timestep else batch_size
        t = torch.randint(lo_j_int, hi_j_int, (n,), device=device)
        if consistent_timestep:
            t = t.expand(batch_size)
        return t
    elif schedule == "linear":
        frac = p
    elif schedule == "sqrt":
        frac = math.sqrt(p)
    elif schedule == "cosine":
        frac = 0.5 * (1.0 - math.cos(math.pi * p))
    elif schedule == "quadratic":
        frac = p * p
    elif schedule == "log":
        # log(1 + p*(e-1)) / log(e) = log(1 + p*(e-1))
        frac = math.log(1.0 + p * (math.e - 1.0))
    elif schedule == "inverse_sqrt":
        # 1 - 1/sqrt(1+p)  normalised so that frac(1)=1
        raw_max = 1.0 - 1.0 / math.sqrt(2.0)  # value at p=1
        frac = (1.0 - 1.0 / math.sqrt(1.0 + p)) / raw_max if raw_max > 0 else p
    else:
        raise ValueError(
            f"Unknown timestep schedule: {schedule!r}. "
            f"Choose from: random, ddim, linear, sqrt, cosine, quadratic, log, inverse_sqrt."
        )

    # Deterministic centre (high → low as frac goes 0 → 1)
    centre = t_hi - frac * span  # float in [t_lo, t_hi]

    # Add jitter
    half_jitter = 0.5 * jitter_fraction * span
    lo_j = max(t_lo, centre - half_jitter)
    hi_j = min(t_hi, centre + half_jitter)

    # Ensure valid range
    lo_j_int = max(t_lo, int(math.floor(lo_j)))
    hi_j_int = max(lo_j_int + 1, min(t_hi, int(math.ceil(hi_j))))

    n = 1 if consistent_timestep else batch_size
    t = torch.randint(lo_j_int, hi_j_int, (n,), device=device)
    if consistent_timestep:
        t = t.expand(batch_size)
    return t
