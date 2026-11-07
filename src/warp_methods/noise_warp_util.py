import torch
import torch.nn.functional as F
from typing import Optional, Any

from src.ray_setup_hooks import install_nvdiffrast
install_nvdiffrast()
import nvdiffrast.torch as dr

from src.regularizers.raft_based_regularizers import estimate_flow_for_video, load_raft_model
from src.dataset.cinemri_dataset import get_spatial_mesh_for_matrix_size

def cond_noise_sampling(src_noise, level=3, device="cpu"):
    """
        Performs conditional upsampling of a noise sample.

        Parameters:
        - src_noise (torch.Tensor): Input noise tensor of shape (B, C, H, W).
        - level (int): Upsampling level k.
        - device (str): Device to perform the computation on.
        Returns:
        - up_noise (torch.Tensor): Upsampled noise tensor of shape (B, C, H * 2^k, W * 2^k).
    """

    B, C, H, W = src_noise.shape
    up_factor = 2 ** level
    upscaled_means = torch.nn.functional.interpolate(src_noise, scale_factor=(up_factor, up_factor), mode='nearest')

    up_H = up_factor * H
    up_W = up_factor * W

    raw_rand = torch.randn(B, C, up_H, up_W, device=device)

    Z_mean = raw_rand.unfold(2, up_factor, up_factor).unfold(3, up_factor, up_factor).mean((4, 5))

    Z_mean = torch.nn.functional.interpolate(Z_mean, scale_factor=up_factor, mode='nearest')
    mean_removed_rand = raw_rand - Z_mean

    up_noise = upscaled_means / up_factor + mean_removed_rand

    return up_noise

def get_triangle_vertex_map(tr_H : int, tr_W : int):
    """
        Returns the triangle_verex map for the pixel traingulation.

        More specifically, it returns a tensor of shape (num_triangles, 3), where num_triangles = W * H * 8. The values in the tensor are the indices of the vertices in the 2x+1-upscaled map (flattened out) that correspond to the vertices of the triangles.

        Parameters:
        - tr_H (int): Height of the triangulated image.
        - trW (int): Width of the triangulated image.

        Returns:
        - vertices (torch.Tensor): Tensor of shape (num_triangles, 3)
    """

    # per_tri_verts defining 8 polygonal partitions for a single original pixel (top left pixel)
    front_tri_verts = torch.tensor([[0, 1, 1+tr_W], [0, tr_W, 1+tr_W], [tr_W, 1+tr_W, 1+2*tr_W], [tr_W, 2*tr_W, 1+2*tr_W]])
    per_tri_verts = torch.cat((front_tri_verts, front_tri_verts + 1),dim=0)

    # Defining 'starting vertex indices' representing original pixels at 2x upscaled pixel map
    width = torch.arange(0, tr_W - 1, 2)
    height = torch.arange(0, tr_H-1, 2) * (tr_W)

    # In the end we have W*H*8 triangles, each triangle is defined by 3 vertices

    # start_idxs is a list of the top left vertex of each original pixel with indexes in the 2x upscaled map (leaving out every second row and column in particular)
    start_idxs = (width[None,...] + height[...,None]).reshape(-1,1) # shape: (tr_W-1)*(tr_H-1), 1
    # vertices is a list of (num_triangles, 3) which contains the indexes into the vertex position list referring to which vertices belong to which triangle
    vertices = (start_idxs.repeat(1,8)[...,None] + per_tri_verts[None,...]).reshape(-1,3) # ((W*H, 1)>(W*H,8,1) + (1, 8, 3)) -> (W*H,8,3) = (num_triangles, 3)

    return vertices

def warp_base_tensor_batch(H: int, W: int, glctx, up_base_tensor: torch.Tensor, defo_map: torch.Tensor, vertices : torch.Tensor, normalize_for_noise : bool = True) -> torch.Tensor:
    """
    Efficient batched version of warp_base_tensor without explicit for-loop over time dimension.
    """
    device = up_base_tensor.device
    defo_map = defo_map.to(device)

    assert vertices.numel() == 3*8*H*W, "Vertices shape is not correct for the given H and W"

    tr_H, tr_W = 2*H + 1, 2*W + 1
    B, C, H_up, W_up = up_base_tensor.shape
    T = defo_map.shape[-1]

    # --- Step 1: Upsample all deformation maps ---
    defo_map_scaled = 2*defo_map - 1  # scale to [-1, 1]
    defo_map_scaled = defo_map_scaled.permute(3,0,1,2)  # (2, H,W, T) -> (T, 2, H, W)
    defo_up = torch.nn.functional.interpolate(defo_map_scaled, size=(tr_H, tr_W), mode='bicubic', align_corners=True) # HIWYN uses bicubic here
    defo_up = defo_up.permute(0, 2, 3, 1)  # (T, H, W, 2)

    # --- Step 2: Flatten vertices for rasterization ---
    coords_len = tr_H * tr_W
    defo_flat = defo_up.reshape(T, -1, 2).flip(-1)  # (T, coords_len, 2)
    warped_vtx_pos = torch.cat([
        defo_flat, 
        torch.zeros(T, coords_len, 1, device=device),
        torch.ones(T, coords_len, 1, device=device)
    ], dim=-1)  # (T, coords_len, 4)

    # --- Step 3: Rasterize all frames at once ---
    rast_result = dr.rasterize(glctx, warped_vtx_pos, vertices, resolution=[H_up, W_up])
    rast_out: Any = rast_result[0] if isinstance(rast_result, tuple) else rast_result

    rast = rast_out[:, :, :, 3:].permute(0, 3, 1, 2).to(torch.int64)  # (T, 1, H_up, W_up)

    indices = (rast - 1) // 8 + 1  # map triangles to original pixels
    indices_flat = indices.reshape(T, -1)  # (T, H_up*W_up)

    # --- Step 4: Flatten the upsampled base tensor ---
    up_base_flat = up_base_tensor.reshape(B*C, -1)  # (B*C, H_up*W_up)

    # Repeat for all T frames
    indices_flat_repeat = indices_flat.unsqueeze(1).repeat(1, B*C, 1)  # (T, B*C, H_up*W_up)
    up_base_flat_repeat = up_base_flat.unsqueeze(0).repeat(T, 1, 1)  # (T, B*C, H_up*W_up)

    # --- Step 5: Scatter-add aggregation ---
    fin_v_val = torch.zeros(T, B*C, H*W+1, device=device, dtype=up_base_flat.dtype)
    fin_v_val.scatter_add_(2, indices_flat_repeat, up_base_flat_repeat)

    ones_flat = torch.ones_like(up_base_flat[:1])
    ones_repeat = ones_flat.unsqueeze(0).repeat(T, 1, 1)  # (T, 1, H_up*W_up)

    fin_v_num = torch.zeros(T, 1, H*W+1, device=device, dtype=up_base_flat.dtype)
    fin_v_num.scatter_add_(2, indices_flat_repeat[:, :1, :], ones_repeat)

    # Pixels with no contributing source samples
    invalid_mask = (fin_v_num == 0)

    # Safe denominator for normalization
    safe_fin_v_num = fin_v_num.clone()
    safe_fin_v_num[invalid_mask] = 1

    if normalize_for_noise:
        final_values = fin_v_val / torch.sqrt(safe_fin_v_num)
    else:
        final_values = fin_v_val / safe_fin_v_num

    # Remove dummy 0 index
    final_values = final_values[..., 1:]      # (T, B*C, H*W)
    invalid_mask = invalid_mask[..., 1:]      # (T, 1, H*W)

    # Fill uncovered regions with fresh Gaussian noise
    invalid_mask_exp = invalid_mask.expand(-1, B*C, -1)
    fresh_noise = torch.randn_like(final_values)
    final_values = torch.where(invalid_mask_exp, fresh_noise, final_values)

    # --- Step 6: Reshape to original tensor shape ---
    warped_tensor = final_values.reshape(T, B, C, H, W).permute(0, 3, 4, 2, 1)  # (T, H, W, C, B)
    warped_tensor = warped_tensor.squeeze(-1)  # remove batch dim if B=1

    return warped_tensor

from abc import ABC, abstractmethod
class NoiseWarpingUtil(ABC):
    
    def __init__(self):
        # initialize the noise warping using the video
        pass

    @abstractmethod
    def initialize_flow(self, video : torch.Tensor):
        # initialize the flow field using the video
        pass

    @abstractmethod
    def sample_noise(self, noise_frame : Optional[torch.Tensor] = None) -> torch.Tensor:
        # sample noise according to the noise mode and the initialized video
        pass

class FullyRandomNoiseUtil(NoiseWarpingUtil):

    def __init__(self, target_shape : torch.Size, device : torch.device):
        self.target_shape = target_shape
        self.device = device

    def initialize_flow(self, video : torch.Tensor):
        # no flow to initialize for fully random noise
        pass

    def sample_noise(self, noise_frame : Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.randn(self.target_shape, device=self.device)

class FixedNoiseUtil(NoiseWarpingUtil):

    def __init__(self, target_shape : torch.Size, device : torch.device, consistent : bool = True):
        self.target_shape = target_shape
        self.device = device
        if consistent:
            self.fixed_noise = torch.randn((1, target_shape[1], target_shape[2], target_shape[3]), device=device).repeat(target_shape[0], 1, 1, 1)
        else:
            self.fixed_noise = torch.randn(self.target_shape, device=self.device)

    def initialize_flow(self, video : torch.Tensor):
        # no flow to initialize for fixed noise
        pass

    def sample_noise(self, noise_frame : Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.fixed_noise

class ConsistentNoiseUtil(NoiseWarpingUtil):
    
    def __init__(self, target_shape : torch.Size, device : torch.device):
        self.target_shape = target_shape
        self.device = device

    def initialize_flow(self, video : torch.Tensor):
        # no flow to initialize for consistent noise
        pass

    def sample_noise(self, noise_frame : Optional[torch.Tensor] = None) -> torch.Tensor:
        T = self.target_shape[0]
        spatial_noise = torch.randn(self.target_shape[1:], device=self.device)
        return spatial_noise.unsqueeze(0).expand(T, *([-1]* (len(self.target_shape)-1)))

from torchvision.models.optical_flow._utils import make_coords_grid, grid_sample
class GridResamplingNoiseUtil(NoiseWarpingUtil):
    # extract the flow (not displacement fields) and use make_grid and other utils to warp the noise
    def __init__(self, target_shape : torch.Size, device : torch.device, interpolation_mode : str, consecutive_warp     : bool = False):
        self.target_shape = target_shape
        self.device = device
        self.interpolation_mode = interpolation_mode
        self.consecutive_warp = consecutive_warp

        T, C, H, W = target_shape
        # torchvision optical-flow grid_sample expects absolute coordinates with shape (N, H, W, 2)
        # make_coords_grid returns (N, 2, H, W), so we move channels to the last dimension.
        self.base_grid = make_coords_grid(
            h=H,
            w=W,
            batch_size=T,
            device=str(device)
        ).movedim(1, -1)  # (T, H, W, 2)

        self.H = H
        self.W = W
        self.T = T

    def initialize_flow(self, video : torch.Tensor, raft_model : Optional[torch.nn.Module] = None):

        if raft_model is None:
            raft_model = load_raft_model(device=str(video.device))

        flow_field = estimate_flow_for_video(raft_model, video, enable_upsample_flow=False, forward_flow=False)  # (T-1, 2, H, W) # Note the False here
        self.initialize_flow_from_pairwise_flow(flow_field)

    def initialize_flow_from_pairwise_flow(self, flow_field : torch.Tensor):
        """Set up the warp grids from a precomputed flow field (T-1, 2, H, W): backward flow,
        pixel units at this util's H x W (same as initialize_flow's RAFT output)."""
        assert flow_field.shape[-2:] == (self.H, self.W), f"flow field {tuple(flow_field.shape)} does not match target H,W=({self.H},{self.W})"
        self.flow_field = flow_field.to(self.device)
        self.pairwise_flow = self.flow_field

        flow_field = torch.cat(
            [torch.zeros(1, 2, self.H, self.W, device=self.device), self.flow_field],
            dim=0,
        )  # (T, 2, H, W)

        self.displacement_field = torch.cumsum(flow_field, dim=0)  # (T, 2, H, W)

        flow_field_xy = flow_field.movedim(1, -1)                 # (T, H, W, 2)
        displacement_field_xy = self.displacement_field.movedim(1, -1)  # (T, H, W, 2)

        self.grid_flow_bwd = self.base_grid + flow_field_xy
        self.grid_displ_bwd = self.base_grid + displacement_field_xy

    def sample_noise(self, noise_frame : Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise_frame is None:
            noise_frame = torch.randn((1, self.target_shape[1], self.target_shape[2], self.target_shape[3]), device=self.device)
        if noise_frame.ndim == 3:
            noise_frame = noise_frame[None]
        elif noise_frame.ndim == 4 and noise_frame.shape[0] != 1:
            noise_frame = noise_frame[:1]

        if self.consecutive_warp:
            frames = [noise_frame]
            for t in range(1, self.T):
                frames.append(
                    grid_sample(
                        frames[-1],
                        self.grid_flow_bwd[t:t+1],
                        align_corners=True,
                        mode=self.interpolation_mode
                    )
                )
            return torch.cat(frames, dim=0)
        else:
            return grid_sample(
                noise_frame.expand(self.T, -1, -1, -1),
                self.grid_displ_bwd,
                align_corners=True,
                mode=self.interpolation_mode) # (1, C, H, W) -> (T, C, H, W)
        
class IntegralNoiseUtil(NoiseWarpingUtil):

    def __init__(self, target_shape : torch.Size, device : torch.device, upsampling_level : int = 3):
        # initialize the noise warping using the video
        self.target_shape = target_shape
        self.device = device
        self.upsampling_level = upsampling_level

        use_opengl = False
        self.glctx = dr.RasterizeGLContext(device=device) if use_opengl else dr.RasterizeCudaContext(device=device)

        _, C, H, W = target_shape
        self.spatial_mesh = get_spatial_mesh_for_matrix_size( (H,W), device)
        self.H = H
        self.tr_H = 2*H + 1
        self.W = W
        self.tr_W = 2*W + 1

        self.vertices_spatial_mesh = get_triangle_vertex_map(self.tr_H, self.tr_W).int().to(self.device) # shape (num_triangles, 3)

    def initialize_flow(self, video : torch.Tensor, raft_model : Optional[torch.nn.Module] = None):
        if raft_model is None:
            raft_model = load_raft_model(device=str(video.device))

        flow_field = estimate_flow_for_video(raft_model, video, enable_upsample_flow=False, forward_flow=False)  # (T-1, 2, H, W) # Note the False here
        self.initialize_flow_from_pairwise_flow(flow_field)

    def initialize_flow_from_pairwise_flow(self, flow_field : torch.Tensor):
        """Build the deformation map from a precomputed flow field (T-1, 2, H, W): backward flow,
        pixel units at this util's H x W."""
        assert flow_field.shape[-2:] == (self.H, self.W), f"flow field {tuple(flow_field.shape)} does not match target H,W=({self.H},{self.W})"
        flow_field = flow_field.to(self.device)
        self.pairwise_flow = flow_field
        _, C, Hf, Wf = flow_field.shape
        velo_field = torch.cat(
            [
                torch.zeros( (1, 2, Hf, Wf, 1), device=self.device),
                (flow_field).movedim(0, -1)[None, [1,0]] # note we don't take minus but flip xy
            ],
            dim=-1
        ).to(self.device)
        displacement_field = torch.cumsum(velo_field, dim=-1)  # shape (1, 2, H, W, T)    

        scaling = torch.tensor([self.H, self.W], device=self.device)[None, :, None, None, None]
        self.defo_map = (
            self.spatial_mesh.movedim(-1, 0)[None, :, :, :, None].expand(displacement_field.shape)
            + displacement_field / scaling
        ) # shape (1, 2, H, W, T)

    def sample_noise(self, noise_frame : Optional[torch.Tensor] = None) -> torch.Tensor:

        if noise_frame is None:
            noise_frame = torch.randn((1, self.target_shape[1], self.target_shape[2], self.target_shape[3]), device=self.device)
        if noise_frame.ndim == 3:
            noise_frame = noise_frame[None]
        elif noise_frame.ndim == 4 and noise_frame.shape[0] != 1:
            noise_frame = noise_frame[:1]
         
        upsampled_noise = cond_noise_sampling(noise_frame, level=self.upsampling_level, device=str(self.device))

        return warp_base_tensor_batch(
            H=self.H,
            W=self.W,
            glctx=self.glctx,
            up_base_tensor=upsampled_noise,
            defo_map=self.defo_map[0],
            vertices=self.vertices_spatial_mesh,
            normalize_for_noise=True
        ).movedim(-1,1)  # (T, H, W, C) -> (T, C, H, W)


def get_noise_warping_method(name : str, **kwargs):
    if name == "integral_noise":
        return IntegralNoiseUtil(**kwargs)
    elif name == "bicubic":
        return GridResamplingNoiseUtil(**kwargs, interpolation_mode="bicubic")
    elif name == "bilinear":
        return GridResamplingNoiseUtil(**kwargs, interpolation_mode="bilinear")
    elif name == "nearest":
        return GridResamplingNoiseUtil(**kwargs, interpolation_mode="nearest")
    elif name == "fully_random":
        return FullyRandomNoiseUtil(**kwargs)
    elif name == "fixed_random":
        return FixedNoiseUtil(**kwargs)
    elif name == "consistent":
        return ConsistentNoiseUtil(**kwargs)
    else:
        raise ValueError(f"Invalid noise warping method name: {name}")