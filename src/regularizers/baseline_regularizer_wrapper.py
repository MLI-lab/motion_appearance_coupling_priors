# Wrapping the variational regularization step to be a x <- regularizer(x) step, with potentially multiple steps of a loss function gradient iterate

import math
from src.regularizers.model_initializers.video_frame_init import video_and_motion_init_full
class BaselineRegularizerWrapper:

    def __init__(self, regularizer_fn, spatial_mesh, time_mesh, device, iterations : int,
                 motion_refresh_every: int = 0, motion_lr: float = 0.0,
                 use_sin_cos_annealing: bool = False, num_timesteps: int = 1000,
                 flow_noise_std: float = None,
                 domain: str = "pixel",
                 latent_downsample_factor: int = 8,
                 **kwargs):
        """
        Args:
            iterations: gradient-descent steps per call.
            motion_refresh_every: if > 0, re-estimate motion from the current reconstruction every N outer steps.
            motion_lr: learning rate for the motion parameters (0.0 keeps motion frozen).
            use_sin_cos_annealing: scale the regularizer loss by a cosine factor from 1 (first/highest t) to 0.
            flow_noise_std: one-time Gaussian noise on the RAFT motion estimate.
            domain: "pixel": __call__ gets pixel frames (T, C, X, Y); "latent": latents (T, c, X/f, Y/f),
                with RAFT motion estimated on pixels and transported to the latent grid (MGLD-style).
            latent_downsample_factor: f of the VAE (8 for SD1.5); latent mode only.
        """
        self.regularizer_fn = regularizer_fn
        self.spatial_mesh = spatial_mesh
        self.time_mesh = time_mesh
        self.device = device
        self.iterations = iterations
        self.motion_refresh_every = int(motion_refresh_every)
        self.motion_lr = float(motion_lr)
        self.use_sin_cos_annealing = bool(use_sin_cos_annealing)
        self.num_timesteps = int(num_timesteps)
        self.flow_noise_std = flow_noise_std
        if domain not in ("pixel", "latent"):
            raise ValueError(f"BaselineRegularizerWrapper: unknown domain {domain!r} (want 'pixel' or 'latent')")
        self.domain = domain
        self.latent_downsample_factor = int(latent_downsample_factor)
        if self.domain == "latent" and self.motion_refresh_every > 0:
            raise ValueError("BaselineRegularizerWrapper: motion_refresh_every is not supported in latent domain")
        self._call_count = 0

    def set_pseudoinv(self, pseudoinv):
        self.init_dict = video_and_motion_init_full(
            dataset=pseudoinv.movedim(1,-1), # from (T, C, X, Y) to (T, X, Y, C)
            motion_init="raft",
            device=self.device,
            flow_noise_std=self.flow_noise_std,
        )
        if self.domain == "latent":
            # pixel-resolution motion -> latent grid, plus a matching spatial mesh
            from src.utils.latent_motion import motion_to_latent
            from src.dataset.cinemri_dataset import get_spatial_mesh_for_matrix_size
            f = self.latent_downsample_factor
            self.init_dict["motion"] = motion_to_latent(self.init_dict["motion"], f)
            self.init_dict["motion_bwd"] = motion_to_latent(self.init_dict["motion_bwd"], f)
            _, _, X, Y, _ = self.init_dict["motion"].shape
            self.spatial_mesh = get_spatial_mesh_for_matrix_size((X, Y), self.device)
            self.init_dict.pop("template", None)
    
    def _maybe_refresh_motion(self, x):
        """Re-estimate motion from the current reconstruction using RAFT."""
        if self.motion_refresh_every <= 0:
            return
        if self._call_count % self.motion_refresh_every != 0:
            return
        if self._call_count == 0:
            return  # skip on first call — pseudoinverse motion is fine

        try:
            fresh = video_and_motion_init_full(
                dataset=x.movedim(1, -1),  # (T, C, X, Y) -> (T, X, Y, C)
                motion_init="raft",
                device=self.device,
                flow_noise_std=self.flow_noise_std,
            )
            self.init_dict["motion"] = fresh["motion"]
            self.init_dict["motion_bwd"] = fresh["motion_bwd"]
        except Exception as e:
            import warnings
            warnings.warn(
                f"Motion refresh failed at call {self._call_count}: {e}; "
                "keeping previous motion estimate.",
                RuntimeWarning,
            )

    def __call__(self, x, step : int, snr_t : float):
        # x: (T, C, X, Y) pixel frames, or latents (T, c, X/f, Y/f) in latent domain
        from src.representations.videovoxel_representation import VideoVoxelRepresentation

        self._maybe_refresh_motion(x)
        self._call_count += 1

        self.init_dict["video"] = x.movedim(1,-1)[:,None]  # shape (T, X, Y, C)
        representation = VideoVoxelRepresentation(
            spatial_mesh=self.spatial_mesh,
            time_mesh=self.time_mesh,
            device=self.device,
            learnable=True,
            add_motion_parameter=True,
            add_motion_bwd_parameter=True,
            add_occlusion_weighting_parameter=True,
            **self.init_dict
        )
        optimizer = representation.setup_optimizer(
            lr=1.0, motion_lr=self.motion_lr, template_lr=0.0, occlusion_lr=0.0
        )
        # progress 0 at the first (largest) t -> 1 at t=0; cosine factor goes 1 -> 0
        current_t = float(step.float().mean()) if hasattr(step, 'float') else float(step)
        progress = 1.0 - current_t / max(self.num_timesteps, 1)
        if self.use_sin_cos_annealing:
            annealing_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            annealing_factor = 1.0

        for i in range(self.iterations):
            optimizer.zero_grad()
            loss = self.regularizer_fn(representation, result=None, kspace=None, kcoords=None, timesteps=self.time_mesh, timestep_provided=step, optimization_progress=progress)
            (loss * annealing_factor).backward()
            optimizer.step()
        x = representation.video_param.squeeze(1).movedim(-1,1)  # shape (T, C, X, Y)
        return x