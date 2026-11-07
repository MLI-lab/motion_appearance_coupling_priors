"""Wraps a warp method / representation so it can be used for MRI reconstruction (k-space forward, smaps)."""

from src.warp_methods.base_warp_method import BaseRepresentation
import torch
from typing import Optional
import fastmri
from src.warp_methods.utils import grid_interpolation_batched

from src.representations.base_representation import BaseRepresentation

import math

class RepresentationMRIWrapper:

    def __init__(self, 
            spatial_mesh : torch.Tensor,
            kspace_mesh : torch.Tensor,
            time_mesh : torch.Tensor,
            smaps : torch.Tensor,
            upsampling_factor: int = 1,
            use_upsampled_meshes: bool = False,
            sensmaps_in_base : bool = False,
            use_smaps : bool = False,
            interpolation_mode : str = "bilinear",
            padding_mode : str = "border",
            align_corners : bool = True
        ):
        
        self.spatial_mesh = spatial_mesh  # (H, W, 2)
        self.spatial_mesh_flat = spatial_mesh.reshape(-1, 2)  # (H*W, 2)
        self.kspace_mesh = kspace_mesh  # (H, W, 2)
        self.time_mesh = time_mesh  # (T,)
        self.smaps = smaps.to(spatial_mesh.device)  # (C, H, W, 2)
        self.use_smaps = use_smaps

        self.interpolation_mode = interpolation_mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

        self.sensmaps_in_base = sensmaps_in_base

        self.unset_model()

    def set_model(self, model: BaseRepresentation):
        self.model = model

    def unset_model(self):
        self.model = None

    def kspace_is_cft(self) -> bool:
        assert self.model is not None, "Model is not set in RepresentationMRIWrapper."
        return self.model.kspace_is_cft()

    def forward_spatial(
        self,
        x: torch.Tensor, t: torch.Tensor,
        skip_temp: bool = False, skip_smaps: bool = False, 
        spatial_mesh_replacement : Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:

        assert self.model is not None, "Model is not set in RepresentationMRIWrapper."
        
        """
            x has shape (T, B, 2)
            t has shape (T,)
        """
        
        T = t.shape[0]

        if self.sensmaps_in_base:
            
            if self.smaps is not None and self.use_smaps and not skip_smaps:
                return self.model.forward(
                    x, t, skip_temp=skip_temp, spatial_mesh_replacement=self.spatial_mesh,
                    base_tensor_replacement=torch.view_as_real(torch.view_as_complex(self.smaps) * torch.view_as_complex(self.model.spatial_param)),
                    **kwargs
                )  # (T, B, C1, C2)
            else:
                return self.model.forward(
                    x, t, skip_temp=skip_temp, spatial_mesh_replacement=spatial_mesh_replacement,
                    **kwargs
                )  # (T, B, C1, C2)
        else:
            warped_result = self.model.forward(
                x, t, skip_temp=skip_temp, spatial_mesh_replacement=spatial_mesh_replacement,
                **kwargs
            ) # (T, B, C1, C2)

            assert warped_result.shape[-2] == 1, f"Expected C1 to be 1, got {warped_result.shape[-2]}"

            if self.smaps is not None and self.use_smaps and not skip_smaps:
                smaps_sampled = grid_interpolation_batched(
                    grid_values=x, 
                    base_value=self.smaps[None].expand(T, -1, -1, -1, -1), # (T, C, H, W, 2)
                    mesh=self.spatial_mesh if spatial_mesh_replacement is None else spatial_mesh_replacement,  # (H, W, 2)
                    interpolation_mode=self.interpolation_mode,
                    padding_mode=self.padding_mode,
                    align_corners=self.align_corners,
                )
                return torch.view_as_real(
                    torch.view_as_complex(warped_result) * torch.view_as_complex(smaps_sampled)
                )  # (T, B, C1, C2) * (T, NC, 1, 2) -> (T, B, NC, 2)
            else:
                return warped_result

    def forward_kspace(
        self,
        k: torch.Tensor, t: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        
        """
            k has shape  (T, M, 2)
            t has shape (T)
        """

        assert self.model is not None, "Model is not set in RepresentationMRIWrapper."

        T = t.shape[0]
        X, Y, _ = self.spatial_mesh.shape  # (X, Y, 2)
        
        spatial_mesh_flat_rep = self.spatial_mesh_flat[None, :, :].expand(T, -1, -1)  # (T, X*Y, 2)
        full_image_at_t = self.forward_spatial(
           spatial_mesh_flat_rep, t, **kwargs
        ).reshape(T, X, Y, -1, 2)  # (T, B, C, 2) -> (T, X, Y, C, 2)

        kspace_at_t = fastmri.fft2c(full_image_at_t.movedim(-2,1)) # T, C, X, Y, 2

        return grid_interpolation_batched(
            grid_values=k,  # (T, B, 2)
            base_value=kspace_at_t,  # (T, C, X, Y, 2)
            mesh=self.kspace_mesh,
            interpolation_mode=self.interpolation_mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )

    def full_warp_kspace(self,
            kspace_mesh: torch.Tensor, 
            timesteps : torch.Tensor, 
            skip_temp=True, 
            device: str = "cuda:0", 
            chunks=5, 
            kspace_mesh_replacement : Optional[torch.Tensor] = None,
            **kwargs
        ):

        assert self.model is not None, "Model is not set in RepresentationMRIWrapper."
        
        T = timesteps.shape[0]
        matrix_size = kspace_mesh.shape[:-1]
        kspace_mesh_flat = kspace_mesh.reshape(-1, 2)
        result_chunks = []

        C = self.smaps.shape[-1]

        chunk_size = math.ceil(len(kspace_mesh_flat) / chunks)
        for j in range(0, len(kspace_mesh_flat), chunk_size):
            chunk = kspace_mesh_flat[j : j + chunk_size][None].to(device)
            result_chunk = self.forward_kspace(
                chunk.expand(T, -1, -1), timesteps, skip_temp=skip_temp, **kwargs
            )
            result_chunks.append(result_chunk)
        return torch.cat(result_chunks, dim=1).reshape(T, matrix_size[0], matrix_size[1], -1, C)

    def full_warp_spatial(
            self,
            spatial_mesh: torch.Tensor, 
            timesteps : torch.Tensor, 
            skip_temp=True, 
            skip_smaps=False, 
            device: str = "cuda:0", 
            chunks=5, 
            spatial_factor = 1.0, 
            spatial_mesh_replacement : Optional[torch.Tensor] = None,
            **kwargs):

        
        assert self.model is not None, "Model is not set in RepresentationMRIWrapper."
        
        T = timesteps.shape[0]
        matrix_size = spatial_mesh.shape[:-1]
        spatial_mesh_flat = spatial_mesh.reshape(-1, 2)
        result_chunks = []

        C = self.smaps.shape[-1]

        chunk_size = math.ceil(len(spatial_mesh_flat) / chunks)
        for j in range(0, len(spatial_mesh_flat), chunk_size):
            chunk = spatial_mesh_flat[j : j + chunk_size][None].to(device) * spatial_factor
            _, S, _ = chunk.shape
            result_chunk = self.forward_spatial(
                chunk.expand(T, -1, -1), timesteps, skip_temp=skip_temp, skip_smaps=skip_smaps, spatial_mesh_replacement=spatial_mesh_replacement, **kwargs
            ).reshape(T, S, -1, C)
            result_chunks.append(result_chunk)
        return torch.cat(result_chunks, dim=1).reshape(T, matrix_size[0], matrix_size[1], -1, C)


