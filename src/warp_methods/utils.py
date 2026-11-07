import torch


def grid_interpolation_batched(
    grid_values: torch.Tensor,  # (T, B, 2)
    base_value: torch.Tensor,   # (T, C, X, Y, 2)
    mesh: torch.Tensor,         # (X, Y, 2)
    interpolation_mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
):

    # base_value has shape: (C, X, Y, 2)
    # grid_values has shape: (B, 2)
    # grid_resample expects:
    # - input of shape (N, C, Hin, Win)
    # - grid of shape (N, Hout, Wout, 2)
    # - output is of shape (N, C, Hout, Wout)
    T, B, _ = grid_values.shape
    T, C, X, Y, C2 = base_value.shape
    param_reformed = base_value.movedim(1, -2).reshape(T, X, Y, C*C2).movedim(-1,1)  # Reshape to (T, C*2, X, Y)
    # mesh is of type [x1,x2]x[y1,y2], but need [-1,1]x[-1,1]
    # mesh is (X, Y, 2), so get min/max for each axis
    mesh_min = mesh.view(-1, 2).min(dim=0)[0]
    mesh_max = mesh.view(-1, 2).max(dim=0)[0]
    mesh_center = (mesh_max + mesh_min) / 2
    mesh_half_range = (mesh_max - mesh_min) / 2
    grid = (grid_values - mesh_center[None, None, :]) / mesh_half_range[None, None, :]

    grid = grid[:, :, None, :]  # Reshape to (T, B, 1, 2)
    grid_reformed = torch.stack([grid[:,:,:,1], grid[:,:,:,0]], dim=-1)

    return (
        torch.nn.functional.grid_sample(
            param_reformed,
            grid=grid_reformed.to(dtype=param_reformed.dtype),
            mode=interpolation_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )  # output is (T, C*2, B, 1)
        .moveaxis(1, -1)
        .view(T,B, C, C2)
        .contiguous()
    )