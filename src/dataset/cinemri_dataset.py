# %%
import os, argparse
from types import SimpleNamespace
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt

def get_meshes_and_corrections_for_matrix_size(matrix_size, device):
    spatial_grid_l, spatial_grid_u = 0.0, 1.0

    # spatial mesh
    spatial_mesh = torch.stack(
        torch.meshgrid(
            *[
                torch.linspace(spatial_grid_l, spatial_grid_u, matrix_size_i)
                for matrix_size_i in matrix_size
            ]
        ),
        dim=-1,
    ).to(device)

    deltas = [
        (spatial_grid_u - spatial_grid_l) / (matrix_size_i - 1)
        for matrix_size_i in matrix_size
    ]
    dft_normalization_factor = torch.from_numpy(
        np.array(np.sqrt(np.prod(matrix_size)) * np.prod(deltas))
    ).to(
        device
    )

    # kspace mesh
    kspace_mesh = torch.stack(
        torch.meshgrid(
            *[
                (-(matrix_size_i // 2) + torch.arange(matrix_size_i))
                / matrix_size_i
                / delta_i
                for delta_i, matrix_size_i in zip(deltas, matrix_size)
            ]
        ),
        dim=-1,
    ).to(device)

    from torch.fft import ifftshift

    mean_spatial_grid = ifftshift(spatial_mesh)[0, 0]
    dft_phase_correction = torch.exp(
        2j * torch.pi * torch.sum(mean_spatial_grid * kspace_mesh, dim=-1)
    )

    return (
        spatial_mesh,
        kspace_mesh,
        dft_normalization_factor,
        dft_phase_correction,
    )

def get_spatial_mesh_for_matrix_size(matrix_size, device, spatial_grid_l=0.0, spatial_grid_u=1.0):

    # spatial mesh
    spatial_mesh = torch.stack(
        torch.meshgrid(
            *[
                torch.linspace(spatial_grid_l, spatial_grid_u, matrix_size_i)
                for matrix_size_i in matrix_size
            ]
        ),
        dim=-1,
    ).to(device)

    return spatial_mesh


