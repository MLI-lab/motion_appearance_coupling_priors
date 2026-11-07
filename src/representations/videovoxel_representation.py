from typing import Optional, Dict
import torch
import numpy as np

from src.representations.base_representation import BaseRepresentation
from src.warp_methods.base_warp_method import velocity_field_to_displacement_field, displacement_field_to_velocity_field
from src.utils.wandb_utils import wandb_quiver_animation, flow_video_to_wandb_torch, flow_overlay_video_to_wandb


def time_slice_selector(param : torch.Tensor, name : str, slice) -> torch.Tensor:
	assert len(slice) == 2 and slice[1] - slice[0] == 1, f"Expected slice of length 1 for time slicing, got {slice}"
	if "video_param" in name:
		# video param has shape (T, C, X, Y, C2)
		return param[slice]
	elif "motion_param" in name:
		# motion param has shape (1, 2, X, Y, T) or (1,2,X,Y,T-1)
		return param[..., slice[0]]
	elif "motion_bwd_param" in name:
		# backward motion param has shape (1, 2, X, Y, T) or (1,2,X,Y,T-1)
		return param[..., slice[0]]
	elif "occlusion_weighting_param" in name:
		# occlusion weighting param has shape (2,T-1, X, Y)
		return param[:, slice[0]]
	elif "template_param" in name:
		# template param has shape (1, X, Y, C2)
		return param
	elif "additive_gains_param" in name:
		# additive gains param has shape (T, C, X, Y, C2)
		return param[slice]
	else:
		return param[slice] # we just assume its in the first dimension for simplicity


def time_slice_selector_with_distance(param, name: str, slice) -> "np.ndarray":
	"""Like time_slice_selector but for an arbitrary frame distance, slice=[t_i, t_j].

	Motion fields are composed additively over the intermediate steps, giving (1, 2, H, W).
	param may be a numpy array or an h5py dataset.
	"""
	t_i, t_j = int(slice[0]), int(slice[1])
	distance = t_j - t_i
	assert distance >= 1, f"Expected t_j > t_i, got slice={slice}"

	if "video_param" in name:
		# Load the two endpoint frames only.
		return np.asarray(param[[t_i, t_j]])

	elif "motion_param" in name or "motion_bwd_param" in name:
		# motion shape: (1, 2, H, W, T-1)
		# Compose flows additively: sum of single-step flows between t_i and t_j.
		if distance == 1:
			return np.asarray(param[..., t_i])
		flows = np.asarray(param[..., t_i:t_j])  # (1, 2, H, W, distance)
		composed = flows.sum(axis=-1)             # (1, 2, H, W)
		return composed

	elif "occlusion_weighting_param" in name:
		# (2, T-1, X, Y) -> average across the distance
		if distance == 1:
			return np.asarray(param[:, t_i])
		weights = np.asarray(param[:, t_i:t_j])  # (2, distance, X, Y)
		return weights.mean(axis=1)               # (2, X, Y)

	elif "template_param" in name:
		return np.asarray(param)

	elif "additive_gains_param" in name:
		return np.asarray(param[[t_i, t_j]])

	else:
		return np.asarray(param[[t_i, t_j]])

class VideoVoxelRepresentation(BaseRepresentation):
	"""
	Representation that directly samples a fixed video volume via trilinear interpolation over (t, y, x).

	video: (T, C, X, Y, C2), spatial_mesh: (X, Y, 2), time_mesh: (T,)
	forward: x (Tb, B, 2), t (Tb,) -> (Tb, B, C, C2)
	"""

	def __init__(
		self,
		video: torch.Tensor,  # (T, C, X, Y, C2)
		spatial_mesh: torch.Tensor,  # (X, Y, 2)
		time_mesh: torch.Tensor,  # (T,)
		motion: Optional[torch.Tensor] = None,
		motion_bwd : Optional[torch.Tensor] = None,
		occlusion_weighting: Optional[torch.Tensor] = None,
		template : Optional[torch.Tensor] = None,
		additive_gains : Optional[torch.Tensor] = None,
		motion_is_velocity: bool = True,
		add_motion_parameter : bool = False,
		add_motion_bwd_parameter : bool = False,
		add_template_parameter : bool = False,
		add_occlusion_weighting_parameter : bool = False,
		add_additive_gains_parameter : bool = False,
		grid_interpolation_mode: str = "bilinear",
		padding_mode: str = "zeros",
		align_corners: bool = True,
		device: str = "cuda:0",
		learnable: bool = True,
		**kwargs
	):
		super().__init__()

		assert video.ndim == 5, f"video must be (T, C, X, Y, C2); got {video.shape}"
		self.device = device
		self.video_param = torch.nn.Parameter(
			video.clone().contiguous().to(device), requires_grad=learnable
		)  # (T, C, X, Y, C2)

		self.spatial_mesh = spatial_mesh.to(device)  # (X, Y, 2)
		self.time_mesh = time_mesh.to(device)  # (T,)

		# Cache mesh normalization stats for grid_sample
		mesh_min = self.spatial_mesh.view(-1, 2).min(dim=0)[0]
		mesh_max = self.spatial_mesh.view(-1, 2).max(dim=0)[0]
		self.sp_center = (mesh_max + mesh_min) / 2.0
		self.sp_half_range = (mesh_max - mesh_min) / 2.0

		t_min = self.time_mesh.min()
		t_max = self.time_mesh.max()
		self.t_center = (t_max + t_min) / 2.0
		self.t_half_range = (t_max - t_min) / 2.0

		self.grid_interpolation_mode = grid_interpolation_mode
		self.padding_mode = padding_mode
		self.align_corners = align_corners

		self.motion_is_velocity = motion_is_velocity
		self.add_template_parameter = add_template_parameter
		self.add_motion_parameter = add_motion_parameter
		if self.add_motion_parameter:
			if motion is None:
				# Initialize zero motion field
				motion = torch.zeros(
					(1, 2, spatial_mesh.shape[0], spatial_mesh.shape[1], time_mesh.shape[0]),
					device=device,
				)
			if motion_is_velocity:
				from src.warp_methods.base_warp_method import displacement_field_to_velocity_field
				motion = displacement_field_to_velocity_field(motion)
			self.motion_param = torch.nn.Parameter(
				motion.contiguous().to(device), requires_grad=learnable
			)  # (1, 2, X, Y, T) or (1,2,X,Y,T-1)

		self.add_motion_bwd_parameter = add_motion_bwd_parameter
		if self.add_motion_bwd_parameter:
			if motion_bwd is None:
				# Initialize zero motion field
				motion_bwd = torch.zeros(
					(1, 2, spatial_mesh.shape[0], spatial_mesh.shape[1], time_mesh.shape[0]),
					device=device,
				)
			if motion_is_velocity:
				from src.warp_methods.base_warp_method import displacement_field_to_velocity_field
				motion_bwd = displacement_field_to_velocity_field(motion_bwd)
			self.motion_bwd_param = torch.nn.Parameter(
				motion_bwd.contiguous().to(device), requires_grad=learnable
			)  # (1, 2, X, Y, T) or (1,2,X,Y,T-1)
		if self.add_template_parameter:
			if template is None:
				# Initialize zero template
				template = torch.zeros(
					(1, spatial_mesh.shape[0], spatial_mesh.shape[1], video.shape[4]),
					device=device,
				)
			self.template_param = torch.nn.Parameter(
				template.contiguous().to(device), requires_grad=learnable
			)  # (1, X, Y, C2)
		self.add_occlusion_weighting_parameter = add_occlusion_weighting_parameter
		if self.add_occlusion_weighting_parameter:
			if occlusion_weighting is None:
				# Initialize occlusion weighting to all ones
				time_dim = self.motion_param.shape[-1] if self.add_motion_parameter else time_mesh.shape[0]
				occlusion_weighting = torch.ones(
					(2, time_dim, spatial_mesh.shape[0], spatial_mesh.shape[1]),
					device=device,
				)
			self.occlusion_weighting_param = torch.nn.Parameter(
				occlusion_weighting.contiguous().to(device), requires_grad=learnable
			)  # (2,T-1, X, Y)

		self.add_additive_gains_parameter = add_additive_gains_parameter

	def kspace_is_cft(self) -> bool:
		return False


	def _build_grid_3d(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
		"""
		Build a 3D grid for torch.grid_sample over (t, y, x).

		x: (Tb, B, 2) in physical coords
		t: (Tb,) in physical coords

		Returns grid of shape (1, Tb*B, 1, 1, 3) with order (t, y, x) normalized to [-1, 1].
		"""
		Tb, B, _ = x.shape
		# Normalize spatial coords to [-1, 1]
		x_norm = (x - self.sp_center[None, None, :]) / (self.sp_half_range[None, None, :] + 1e-12)
		# Normalize time coords to [-1, 1]
		t_expanded = t[:, None, None].expand(-1, B, 1)
		t_norm = (t_expanded - self.t_center) / (self.t_half_range + 1e-12)

		# Compose (x, y, t) then reorder to (x_grid, y_grid, z_grid) = (y_norm, x_norm, t_norm)
		# because grid[...,0] indexes W (our Y), grid[...,1] indexes H (our X), grid[...,2] indexes D (our T)
		xt = torch.cat([x_norm, t_norm], dim=-1).reshape(-1, 3)  # (Tb*B, 3) as (x, y, t)
		grid = xt[None, :, None, None, :]  # (1, Tb*B, 1, 1, 3)
		grid_reordered = torch.stack(
			[grid[..., 1], grid[..., 0], grid[..., 2]], dim=-1
		)  # (1, Tb*B, 1, 1, 3) as (y, x, t) -> (x_grid, y_grid, z_grid)
		return grid_reordered.to(dtype=self.video_param.dtype, device=self.video_param.device)

	def forward(
		self,
		x: torch.Tensor,
		t: torch.Tensor,
		base_tensor_replacement: Optional[torch.Tensor] = None,  # unused, kept for API compatibility
		spatial_mesh_replacement: Optional[torch.Tensor] = None,  # unused, kept for API compatibility
		skip_temp: bool = False,  # unused, no warping in this representation
		noise_correction: bool = False,  # unused, kept for API compatibility
		**kwargs,
	) -> torch.Tensor:
		"""
		Sample the video at positions (x, t) using 3D grid_sample.
		Returns (Tb, B, C, C2)
		"""
		Tb, B, _ = x.shape

		# Prepare input volume for grid_sample: (N=1, C'=C*C2, D=T, H=X, W=Y)
		T, C, X, Y, C2 = self.video_param.shape
		vol = self.video_param.permute(1, 4, 0, 2, 3).reshape(1, C * C2, T, X, Y)

		grid = self._build_grid_3d(x.to(self.device), t.to(self.device))  # (1, Tb*B, 1, 1, 3)

		sampled = torch.nn.functional.grid_sample(
			vol,
			grid,
			mode=self.grid_interpolation_mode,
			padding_mode=self.padding_mode,
			align_corners=self.align_corners,
		)  # (1, C*C2, Tb*B, 1, 1)

		sampled = (
			sampled.view(1, C, C2, Tb * B, 1, 1)
			.permute(3, 0, 1, 2, 4, 5)
			.reshape(Tb * B, C, C2)
		)  # (Tb*B, C, C2)

		return sampled.view(Tb, B, C, C2).contiguous()

	def full_warp(
		self,
		timesteps: Optional[torch.Tensor] = None,
		base_tensor_replacement: Optional[torch.Tensor] = None,  # unused
		spatial_mesh_replacement: Optional[torch.Tensor] = None,  # allow alt mesh
		noise_correction: bool = False,  # unused
	) -> torch.Tensor:
		"""
		Return the full frames at the requested timesteps sampled on the given spatial mesh.
		Output: (Treq, X, Y, C*C2)
		"""
		if timesteps is None:
			timesteps = self.time_mesh

		smesh = self.spatial_mesh if spatial_mesh_replacement is None else spatial_mesh_replacement
		X, Y, _ = smesh.shape

		# Build flat query grid
		spatial_mesh_flat_rep = smesh.view(-1, 2)[None, :, :].expand(timesteps.numel(), -1, -1)  # (Treq, X*Y, 2)
		result = self.forward(spatial_mesh_flat_rep, timesteps, skip_temp=True)
		return result.view(timesteps.numel(), X, Y, -1)
	
	def get_fps(self) -> float:
		duration_in_s = self.time_mesh.max().item() - self.time_mesh.min().item()
		return len(self.time_mesh) / duration_in_s
	
	def setup_optimizer(
        self, lr: float, motion_lr: Optional[float] = None, template_lr: Optional[float] = None, occlusion_lr : Optional[float] = None
    ) -> torch.optim.Optimizer:
		params = [
            {"params": [self.video_param], "lr": lr, "name": "video"},
        ]
		if self.add_motion_parameter:
			assert motion_lr is not None, "motion_lr must be provided if motion parameter is added."
			params.append({"params": [self.motion_param], "lr": motion_lr, "name": "motion"})
		if self.add_motion_bwd_parameter:
			assert motion_lr is not None, "motion_lr must be provided if backward motion parameter is added."
			params.append({"params": [self.motion_bwd_param], "lr": motion_lr, "name": "motion_bwd"})
		if self.add_template_parameter:
			assert template_lr is not None, "template_lr must be provided if template parameter is added."
			params.append({"params": [self.template_param], "lr": template_lr, "name": "template"})
		if self.add_occlusion_weighting_parameter:
			assert occlusion_lr is not None, "occlusion_lr must be provided if occlusion weighting parameter is added."
			params.append({"params": [self.occlusion_weighting_param], "lr": occlusion_lr, "name": "occlusion_weighting"})

		if self.add_additive_gains_parameter:
			assert template_lr is not None, "template_lr must be provided if additive gains parameter is added."
			params.append({"params": [self.additive_gains_param], "lr": template_lr, "name": "additive_gains"})

		self.optimizer = torch.optim.Adam(params, lr=0.000, eps=1e-15)
		return self.optimizer
	
	def get_temp_param_scaling(self) -> torch.Tensor:
		return torch.tensor([self.spatial_mesh.shape[0], self.spatial_mesh.shape[1]], device=self.device)[None, :, None, None, None]
	
	def get_displacement_field(self, scale : bool = False, ret_bwd : bool = False) -> torch.Tensor:
		if ret_bwd and not self.add_motion_bwd_parameter:
			raise ValueError("Cannot return backward motion parameter if add_motion_bwd_parameter is False.")
		param = (
            self.motion_param if not ret_bwd else self.motion_bwd_param
            if not self.motion_is_velocity
            else velocity_field_to_displacement_field(self.motion_param if not ret_bwd else self.motion_bwd_param)
        )
		return param if not scale else param / self.get_temp_param_scaling()
	
	def get_velocity_field(self, scale : bool = False, ret_bwd : bool = False) -> torch.Tensor:
		if ret_bwd and not self.add_motion_bwd_parameter:
			raise ValueError("Cannot return backward motion parameter if add_motion_bwd_parameter is False.")
		param = (
            self.motion_param if not ret_bwd else self.motion_bwd_param
            if self.motion_is_velocity
            else displacement_field_to_velocity_field(self.motion_param if not ret_bwd else self.motion_bwd_param)
        )
		return param if not scale else param / self.get_temp_param_scaling()

	def get_occlusion_weighting(self) -> torch.Tensor:
		assert self.add_occlusion_weighting_parameter, "Cannot return occlusion weighting if add_occlusion_weighting_parameter is False."
		return torch.sigmoid(self.occlusion_weighting_param)
	
	def wandb_animation(self, quiver_enable, quiver_cfg, color_enable, color_cfg, overlay_enable, overlay_cfg, reference_video=None) -> Dict:
		# video param has shape (T, C, X, Y, C2)
		# video visualization
		video = self.full_warp().movedim(-1,1) # (T, X, Y, C) -> (T, C, X, Y)
		from src.tasks.visualize_vdm_training_dataset import single_channel_video_to_wandb, complex_video_to_wandb, three_channel_video_to_wandb
		fps = self.get_fps()
		ret = {}
		if video.shape[1] == 1:
			ret |= single_channel_video_to_wandb(video, "video_param", "Video Parameter", clamp_lower=video.min().item(), clamp_upper=video.max().item(), fps=fps)
		elif video.shape[1] == 2:
			ret |= complex_video_to_wandb(video, "video_param_mag", "Video Parameter Magnitude", "video_param_angle", "Video Parameter Phase", clamp_lower=video.min().item(), clamp_upper=video.max().item(), fps=fps)
		elif video.shape[1] == 3:
			ret |= three_channel_video_to_wandb(video, "video_param_rgb", "Video Parameter RGB", clamp_lower=video.min().item(), clamp_upper=video.max().item(), fps=fps)
		else:
			raise ValueError("Cannot visualize videos with more than 3 channels.")

		# next we upload a video where we can see the reference video and our video side-by-side as well as a video of the differences
		if reference_video is not None:
			C = reference_video.shape[-1]
			reference_video_l = reference_video.movedim(-1,1) # (T, X, Y, C) -> (T, C, X, Y)
			diff_video = video.norm(dim=1, keepdim=True) - reference_video_l.norm(dim=1, keepdim=True) # (T, C, X, Y)
			diff_video_pos = diff_video.clamp(min=0).repeat(1,C,1,1) # showing where the video is brighter than the reference
			diff_video_neg = -diff_video.clamp(min=0).repeat(1,C,1,1) # showing where the video is darker than the reference
			video_concat = torch.cat([video, reference_video_l, diff_video_pos, diff_video_neg], dim=3) # (T, C, Y, 3X)
			if video_concat.shape[1] == 1:
				ret |= single_channel_video_to_wandb(video_concat, "video_comparison", "Video Comparison (Param | Reference | Abs Diff)", clamp_lower=video_concat.min().item(), clamp_upper=video_concat.max().item(), fps=fps)
			elif video_concat.shape[1] == 2:
				ret |= complex_video_to_wandb(video_concat, "video_comparison_mag", "Video Comparison Magnitude (Param | Reference | Abs Diff)", "video_comparison_angle", "Video Comparison Phase (Param | Reference | Abs Diff)", clamp_lower=video_concat.min().item(), clamp_upper=video_concat.max().item(), fps=fps)
			elif video_concat.shape[1] == 3:
				ret |= three_channel_video_to_wandb(video_concat, "video_comparison_rgb", "Video Comparison RGB (Param | Reference | Abs Diff)", clamp_lower=video_concat.min().item(), clamp_upper=video_concat.max().item(), fps=fps)
			else:
				raise ValueError("Cannot visualize videos with more than 3 channels.")
	
		if self.add_motion_parameter:
			diff = self.get_displacement_field(scale=True)[0].detach().cpu().permute(3, 1, 2, 0)
			duration_in_s = float(self.time_mesh.max().item() - self.time_mesh.min().item())
			fps = len(self.time_mesh) / duration_in_s
			video = self.full_warp().norm(dim=-1).unsqueeze(1) # (T, X, Y, C) -> (T, 1, X, Y)
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

		if self.add_motion_bwd_parameter:
			diff_bwd = self.get_displacement_field(scale=True, ret_bwd=True)[0].detach().cpu().permute(3, 1, 2, 0)
			duration_in_s = float(self.time_mesh.max().item() - self.time_mesh.min().item())
			fps = len(self.time_mesh) / duration_in_s
			video = self.full_warp().norm(dim=-1).unsqueeze(1) # (T, X, Y, C) -> (T, 1, X, Y)
			if quiver_enable:
				ret.update({
					"temporal_param_quiver_bwd" : wandb_quiver_animation(diff_bwd, duration_in_s=duration_in_s, **quiver_cfg),
					"temporal_param_quiver_velocity_bwd" : wandb_quiver_animation(diff_bwd, duration_in_s=duration_in_s, use_pairwise_deltas=True,**quiver_cfg),
				})
			if color_enable:
				ret.update({
					"temporal_param_color_bwd" : flow_video_to_wandb_torch(diff_bwd, fps=fps, **color_cfg),
					"temporal_param_color_velocity_bwd" : flow_video_to_wandb_torch(diff_bwd, fps=fps, use_pairwise_deltas=True, **color_cfg),
				})
			if overlay_enable:
				ret.update({
					"overlay_bwd" : flow_overlay_video_to_wandb(diff_bwd, video, fps=fps, **overlay_cfg),
				})
				if reference_video is not None:
					if reference_video.shape[1] == 2:
						reference_video = reference_video.norm(dim=1, keepdim=True)
					ret["overlay_ref_bwd"] = flow_overlay_video_to_wandb(diff_bwd, reference_video, fps=fps, **overlay_cfg)

		if self.add_occlusion_weighting_parameter:
			occ_weighting_video_fwd = self.get_occlusion_weighting()[0,:,None] # (2,Tp, X, Y) -> (T, 1, X, Y)
			occ_weighting_video_bwd = self.get_occlusion_weighting()[1,:,None] # (2,Tp, X, Y) -> (T, 1, X, Y)
			ret |= single_channel_video_to_wandb(occ_weighting_video_fwd, "occlusion_weighting_fwd", "Occlusion Weighting Forward", clamp_lower=occ_weighting_video_fwd.min().item(), clamp_upper=occ_weighting_video_fwd.max().item(), fps=fps)
			ret |= single_channel_video_to_wandb(occ_weighting_video_bwd, "occlusion_weighting_bwd", "Occlusion Weighting Backward", clamp_lower=occ_weighting_video_bwd.min().item(), clamp_upper=occ_weighting_video_bwd.max().item(), fps=fps)
					

		return ret
	
	def fit_to_video(
        self,
		video,
        **kwargs
    ) -> None:
		# No fitting needed since this representation directly samples the video
		pass