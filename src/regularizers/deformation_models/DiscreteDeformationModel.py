
import logging
import torch
from typing import Optional


"""
    Discrete model, based on a 3D representation.

    Suppose the spatial_mesh has shape (X,Y,2) and the time_mesh (T,).
    The deformation field relealizes a learnt mapping spatial_mesh x time_mesh -> spatial_mesh.
    Here, we implement this as a tensor (X, Y, 2, T, 2), e.g. (168, 256, 2, 6000, 2).
    With our input (B, 3), we can simply use trilinear interpolation to obtain (B, 2).
"""


def initialize_param(
    spatial_mesh: torch.Tensor,
    time_mesh: torch.Tensor,
    grid_interpolation_mode: str = "bilinear",
    grid_padding_mode: str = "border", # zeros or reflective can lead to wrong results here
    grid_align_corners: bool = False,
    init_interpolation_mode: str = "trilinear",
    interpolation_factor_x : Optional[float] = None,
    interpolation_factor_y : Optional[float] = None,
    interpolation_factor_t : Optional[float] = None):

    spatial_mesh = spatial_mesh
    time_mesh = time_mesh
    interpolation_mode = grid_interpolation_mode
    padding_mode = grid_padding_mode
    align_corners = grid_align_corners

    T = time_mesh.shape[0]
    X, Y, _ = spatial_mesh.shape  # (X, Y, 2)
    xt_mesh = torch.cat(
        [
            spatial_mesh[:,:,None,:].expand(-1,-1,T,-1),
            time_mesh[None,None,:,None].expand(X, Y, -1, 1),
        ], dim=-1
    )

    mesh_min = xt_mesh.view(-1, 3).min(dim=0)[0]
    mesh_max = xt_mesh.view(-1, 3).max(dim=0)[0]
    mesh_center = (mesh_max + mesh_min) / 2
    mesh_half_range = (mesh_max - mesh_min) / 2

    # X, Y, 2 -> 1, X, Y, 1, 2 -> 1, X, Y, T, 2 -> 1, 2, X, Y, T
    param_init = spatial_mesh[None, :, :, None, :].repeat(1, 1, 1, T, 1).movedim(-1, 1) # shape (1, 2, X, Y, T)
    if interpolation_factor_y is None and not interpolation_factor_x is None:
        logging.info("interpolation_factor_y is None, using interpolation_factor_x for both dimensions.")
        interpolation_factor_y = interpolation_factor_x
    if interpolation_factor_x is None and not interpolation_factor_y is None:
        logging.info("interpolation_factor_x is None, using interpolation_factor_y for both dimensions.")
        interpolation_factor_x = interpolation_factor_y

    if interpolation_factor_x is not None and  interpolation_factor_y is not None and interpolation_factor_t is not None:
        factors = (interpolation_factor_x, interpolation_factor_y, interpolation_factor_t)
        param_init = torch.nn.functional.interpolate(
            param_init, scale_factor=factors, mode=init_interpolation_mode, align_corners=align_corners
        )

    param_ref = param_init.clone()
    param_init = torch.zeros_like(param_init)

    return param_init.contiguous(), param_ref.contiguous(), {
        "mesh_center": mesh_center,
        "mesh_half_range": mesh_half_range,
        "interpolation_mode": interpolation_mode,
        "padding_mode": padding_mode,
        "align_corners": align_corners
    }
