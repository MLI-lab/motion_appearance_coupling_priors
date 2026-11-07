from src.warp_methods.base_warp_method import BaseWarpMethod, velocity_field_to_displacement_field, displacement_field_to_velocity_field
from typing import Optional, Tuple
import torch
from src.utils.wandb_utils import wandb_quiver_animation, flow_video_to_wandb_torch, flow_overlay_video_to_wandb
from src.regularizers.deformation_models.DiscreteDeformationModel import initialize_param
from abc import abstractmethod
import wandb
from src.utils.metrics import PSNR, nmse, ssim_slicewise_mag

class BaseDiscreteWarpMethod(BaseWarpMethod):
    """
        Implementation of the BaseWarpMethod, where the parameters are given as:
        - spatial_param: the parameters defining the spatial deformation field. Shape: (C1, H, W, C2)
        - temporal_param: the parameters defining the temporal deformation field. Shape: (1, 2, X, Y, T) 

        and a default implementation for fit_to_video, setup optimizer and wandb_animation.
    """

    def __init__(
        self,
        base_tensor: torch.Tensor,
        spatial_mesh: torch.Tensor,
        time_mesh: torch.Tensor,
        device: str = 'cuda:0',
        temp_parameter_init: Optional[torch.Tensor] = None,
        temporal_param_is_velocity : bool = False,
        deformation_model: dict = {}):

        super().__init__()

        self.base_tensor = base_tensor # (C1, X, Y, C2)
        self.spatial_mesh = spatial_mesh # (X, Y, 2)
        self.time_mesh = time_mesh # (T, 1)
        self.device = device

        self.spatial_param = torch.nn.Parameter(
            self.base_tensor.contiguous(),
            requires_grad=True
        ) # (C, X, Y)
        temporal_model_param_init, self.temporal_param_ref, self.temporal_model_cfg = initialize_param(**deformation_model, spatial_mesh=spatial_mesh, time_mesh=time_mesh) # (1, 2, X, Y, T)
        if temp_parameter_init is not None:
            assert temp_parameter_init.shape == temporal_model_param_init.shape, f"Provided temp_parameter_init has shape {temp_parameter_init.shape}, but expected {temporal_model_param_init.shape}."
            temporal_model_param_init = temp_parameter_init
        self.temporal_param_is_velocity = temporal_param_is_velocity
        if self.temporal_param_is_velocity:
            temporal_model_param_init = displacement_field_to_velocity_field(temporal_model_param_init, assert_first_frame_zero=True)
        self.temporal_param = torch.nn.Parameter(temporal_model_param_init.to(self.device), requires_grad=True)



    def get_temp_param_scaling(self) -> torch.Tensor:
        return torch.tensor([self.spatial_mesh.shape[0], self.spatial_mesh.shape[1]], device=self.device)[None, :, None, None, None]

    def get_displacement_field(self, scale : bool = False) -> torch.Tensor:
        param = (
            self.temporal_param
            if not self.temporal_param_is_velocity
            else velocity_field_to_displacement_field(self.temporal_param)
        )
        return param if not scale else param / self.get_temp_param_scaling()

    def get_velocity_field(self, scale : bool = False) -> torch.Tensor:
        param = (
            self.temporal_param
            if self.temporal_param_is_velocity
            else displacement_field_to_velocity_field(self.temporal_param)
        )
        return param if not scale else param / self.get_temp_param_scaling()


    def setup_optimizer(
        self, base_lr: float, temp_lr : float
    ) -> torch.optim.Optimizer:

        params = [
            {"params": [self.spatial_param], "lr": base_lr, "name": "base"},
            {"params": [self.temporal_param], "lr": temp_lr, "name": "temporal"},
        ]

        self.optimizer = torch.optim.Adam(params, lr=0.000, eps=1e-15)

        return self.optimizer

    def fit_to_video(
        self,
        video: torch.Tensor,
        base_lr: float,
        temp_lr: float,
        num_iterations: int,
        target_psnr: float,
        print_loss_every: int,
        use_tqdm: bool,
        lr_factor,
        lr_patience,
        regularizer_cfg,
        data_con_weight : float = 1.0,
    ) -> None:

        optimizer = self.setup_optimizer(base_lr=base_lr, temp_lr=temp_lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=lr_patience, min_lr=1e-6)
        from src.regularizers.regularizers import regularizer_fn_resolver
        reg_fn = regularizer_fn_resolver(**regularizer_cfg)
        from tqdm import trange
        pbar = trange(num_iterations, desc="Fitting to video") if use_tqdm else range(num_iterations)
        for it in pbar:
            optimizer.zero_grad()

            video_warped = self.full_warp()
            loss = data_con_weight * (video_warped - video).square().mean() + reg_fn(model=self)

            loss.backward()
            optimizer.step()

            lr_scheduler.step(loss)

            psnr = PSNR(video_warped, video)
            ssim_slicewise = ssim_slicewise_mag(video.norm(dim=-1), video_warped.norm(dim=-1))
            nmse_val = nmse(video_warped.norm(dim=-1), video.norm(dim=-1))
            if video_warped.shape[-1] > 1:
                psnr_mag = PSNR(video_warped.norm(dim=-1), video.norm(dim=-1))
            else:
                psnr_mag = None
            if print_loss_every is not None and it % print_loss_every == 0:
                print(f"Initialization iteration {it}, loss: {loss.item()}, PSNR: {psnr}: PSNR_magnitude: {psnr_mag}, SSIM_slicewise: {ssim_slicewise}, NMSE: {nmse_val}")
            if psnr >= target_psnr:
                print(f"Stopping early at iteration {it} due to reaching target PSNR: {psnr}, target was {target_psnr}.")
                return

            if use_tqdm: 
                pbar.set_postfix({"loss": loss.item(), "PSNR": psnr, "PSNR_magnitude": psnr_mag, "SSIM_slicewise": ssim_slicewise, "NMSE": nmse_val})

    def wandb_animation(self, quiver_enable, quiver_cfg, color_enable, color_cfg, overlay_enable, overlay_cfg, reference_video=None):

        # temporal param has shape. (1, 2, X, Y, T) -> (T, X, Y, 2)
        diff = self.get_displacement_field(scale=True)[0].detach().cpu().permute(3, 1, 2, 0)
        duration_in_s = self.time_mesh.max().item() - self.time_mesh.min().item()
        fps = len(self.time_mesh) / duration_in_s
        video = self.full_warp().norm(dim=-1).unsqueeze(1) # (T, X, Y, C) -> (T, 1, X, Y)
        ret = {}
        if quiver_enable:
            ret.update({
                "temporal_param_quiver" : wandb_quiver_animation(diff, duration_in_s=duration_in_s, **quiver_cfg),
                "temporal_param_quiver_velocity" : wandb_quiver_animation(diff, duration_in_s=duration_in_s, use_pairwise_deltas=True,**quiver_cfg),
            })
        if color_enable:
            ret.update({
                "temporal_param_color" : flow_video_to_wandb_torch(diff, fps=fps, **color_cfg),
                "temporal_param_color_velocity" : flow_video_to_wandb_torch(diff, fps=fps, use_pairwise_deltas=True, **color_cfg),
            })
        if overlay_enable:
            ret.update({
                "overlay" : flow_overlay_video_to_wandb(diff, video, fps=fps, **overlay_cfg),
            })
            if reference_video is not None:
                if reference_video.shape[1] == 2:
                    reference_video = reference_video.norm(dim=1, keepdim=True)
                ret["overlay_ref"] = flow_overlay_video_to_wandb(diff, reference_video, fps=fps, **overlay_cfg)
        return ret
