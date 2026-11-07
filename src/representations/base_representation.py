
import torch
from abc import ABC, abstractmethod
from typing import Optional


"""
    Abstract base class for representations.
    Used as interface during reconstruction.

    WarpMethods inherit from BaseRepresentation and offer the additional functionality of warping a replaceable base tensor.
    That is, they have a base tensor (e.g. an image) that can be deformed via spatial and temporal deformation fields.

    Another example of BaseRepresentation could be INNs, untrained methods, or simple volume based representations which do not offer this possibility. 
"""

class BaseRepresentation(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self,
            x: torch.Tensor, t: torch.Tensor, **kwargs
        ) -> torch.Tensor:
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


    @abstractmethod
    def setup_optimizer(self, **params) -> torch.optim.Optimizer:
        """
        """

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

    def wandb_animation(self, **kwargs) -> dict:
        return {}