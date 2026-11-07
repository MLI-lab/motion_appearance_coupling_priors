
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch

from src.representations.base_representation import BaseRepresentation

def displacement_field_to_velocity_field(displ_field : torch.Tensor, keep_first_frame : bool = False, raft_conv : bool = False, assert_first_frame_zero : bool = False) -> torch.Tensor:
    # inverse of velocity_field_to_displacement_field
    diff = displ_field[:, :, :, :, 1:] - displ_field[:, :, :, :, :-1]
    if raft_conv:
        diff = -diff[:, [1,0]]
    if keep_first_frame:
        return torch.cat(
            [
                displ_field[...,[0]],
                diff
            ],
            dim=-1
        ).to(displ_field.device)
    else:
        if assert_first_frame_zero:
            torch.testing.assert_close(displ_field[...,0], torch.zeros_like(displ_field[...,0]))
        return diff
        
def velocity_field_to_displacement_field(velo_field : torch.Tensor, add_first_frame : bool = True, raft_conv : bool = False) -> torch.Tensor:
    _, _, H, W, T = velo_field.shape
    if add_first_frame:
        velo_field = torch.cat(
            [
                torch.zeros( (1, 2, H, W, 1), device=velo_field.device),
                velo_field
            ],
            dim=-1
        ).to(velo_field.device)
        if raft_conv:
            velo_field = -velo_field[:, [1,0]]
    elif raft_conv:
        raise ValueError("Raft convention only supported when adding first frame.")
    return torch.cumsum(velo_field, dim=-1)  # shape (1, 2, H, W, T)

class BaseWarpMethod(BaseRepresentation):
    """
        The base class for all warp methods.

        Usually based on two torch Parameters:
            - spatial_param: the parameters defining the spatial deformation field. Shape: (C1, H, W, C2)
            - temporal_param: the parameters defining the temporal deformation field. Shape: (1, 2, X, Y, T)

        Usually saved meshes via initialization:
            - spatial_mesh: the spatial mesh to sample from. Shape: (H, W, 2)
            - time_mesh: the time mesh to sample from. Shape: (T,)
    """

    @abstractmethod
    def forward(self, x : torch.Tensor, t : torch.Tensor, base_tensor_replacement : Optional[torch.Tensor], spatial_mesh_replacement : Optional[torch.Tensor], skip_temp : bool) -> torch.Tensor:
        """
        Obtain the warped elements at positions x and time t.

        Args:
            - x (torch.Tensor): The spatial positions to sample from. Shape: (T, B, 2), where B might be X*Y
            - t (torch.Tensor): The timesteps to sample from. Shape: (T,)
            - base_tensor_replacement (torch.Tensor, optional): If provided, use this tensor instead of self.spatial_param as the base tensor. Shape: (C1, H, W, C2)
            - spatial_mesh_replacement (torch.Tensor, optional): If provided, use this tensor instead of self.spatial_mesh as the spatial mesh. Shape: (H, W, 2)

        Returns:
            - warped_tensor (torch.Tensor): The warped tensor. Shape: (T, B, C1, C2)
        """
        pass

    @abstractmethod
    def full_warp(self, timesteps : Optional[torch.Tensor] = None, base_tensor_replacement : Optional[torch.Tensor] = None, spatial_mesh_replacement : Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Warp the base_tensor to a certain point in time and using the full spatial_mesh at that time by default.

        Args:
         - timesteps (torch.Tensor, optional): If provided, use these timesteps instead of self.time_mesh. Shape: (T,)
         - base_tensor_replacement (torch.Tensor, optional): If provided, use this tensor instead of self.spatial_param as the base tensor. Shape: (C1, H, W, C2)
         - spatial_mesh_replacement (torch.Tensor, optional): If provided, use this tensor instead of self.spatial_mesh as the spatial mesh. Shape: (H, W, 2)

        Returns:
            - warped_tensor (torch.Tensor): The warped tensor. Shape: (T, H, W, C1*C2)        
        """ 
        pass

    def kspace_is_cft(self) -> bool:
        return False

    @abstractmethod
    def setup_optimizer(self, **params) -> torch.optim.Optimizer:
        """
        """
        pass

    @abstractmethod
    def fit_to_video(self, video: torch.Tensor, **fit_params) -> None:
        """
        Fit the model to the given video.

        Args:
         - video (torch.Tensor): The video to fit to. Shape: (F, C, H, W)
         - fit_params (Dict[str, Any]): Additional parameters for fitting.
        """ 
        pass

    def wandb_animation(self, **kwargs) -> dict:
        return {}