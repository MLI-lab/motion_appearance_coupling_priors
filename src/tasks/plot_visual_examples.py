import hashlib
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from prefect import task
from omegaconf import OmegaConf

from src.prefect.caching import CacheableDict, CacheableDictConfig, CacheableListConfig
from src.prefect.setup import StoragePath, StorageSettings, load_file_system
from src.utils.wandb_utils import WandbParamsTask, wandb_kwargs_for_prefect_task


def _normalize_param_name(param_name: str) -> str:
	return str(param_name).strip().strip('"').strip("'").lstrip("+")


def _resolve_column_name(requested_name: Optional[str], available_columns: List[str]) -> Optional[str]:
	if requested_name is None:
		return None
	if requested_name in available_columns:
		return requested_name

	normalized_requested = _normalize_param_name(requested_name)
	normalized_to_actual = {
		_normalize_param_name(col): col for col in available_columns
	}
	return normalized_to_actual.get(normalized_requested)


def _result_to_dict(result: Any) -> Dict[str, Any]:
	if isinstance(result, tuple) and len(result) == 2:
		metrics, storage = result
		result_dict = _result_to_dict(metrics)
		if isinstance(storage, StoragePath):
			result_dict["storage_path"] = storage.model_dump(mode="python")
		elif isinstance(storage, dict):
			result_dict["storage_path"] = storage
		return result_dict
	if isinstance(result, CacheableDict):
		return dict(result.cfg)
	if isinstance(result, CacheableDictConfig):
		container = OmegaConf.to_container(result.cfg, resolve=True)
		return cast(Dict[str, Any], container) if isinstance(container, dict) else {"value": container}
	if isinstance(result, StoragePath):
		return {
			"storage_path": result.model_dump(mode="python"),
		}
	if isinstance(result, dict):
		return dict(result)
	cfg_value = getattr(result, "cfg", None)
	if cfg_value is not None and not isinstance(result, tuple):
		return dict(cfg_value)
	raise TypeError(f"Unsupported result type for visual plotting: {type(result)}")


def _ordered_unique(values: Sequence[Any]) -> List[Any]:
	ordered = []
	for value in values:
		if not any((pd.isna(value) and pd.isna(existing)) or value == existing for existing in ordered):
			ordered.append(value)
	return ordered


def _make_safe_name(value: Any) -> str:
	text = str(value)
	safe_chars = []
	for char in text:
		if char.isalnum() or char in ["-", "_", "."]:
			safe_chars.append(char)
		else:
			safe_chars.append("_")
	safe = "".join(safe_chars).strip("_")
	return safe[:120] if safe else "value"


def _format_caption_value(value: Any) -> str:
	if pd.isna(value):
		return "nan"
	if isinstance(value, (bool, np.bool_)):
		return "True" if bool(value) else "False"
	if isinstance(value, (int, np.integer)):
		return str(int(value))
	if isinstance(value, (float, np.floating)):
		return f"{float(value):.4g}"
	return str(value)


def _is_bool_like_param_values(values: Sequence[Any]) -> bool:
	allowed_text = {"0", "1", "true", "false"}
	for value in values:
		if pd.isna(value):
			continue
		if isinstance(value, (bool, np.bool_)):
			continue
		if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
			continue
		if isinstance(value, (float, np.floating)) and float(value) in (0.0, 1.0):
			continue
		if str(value).strip().lower() in allowed_text:
			continue
		return False
	return True


def _format_param_label(param_name: Optional[str], value: Any, is_boolean_param: bool = False) -> str:
	if is_boolean_param:
		value_text = str(value).strip().lower()
		if value_text in ["1", "true"]:
			formatted_value = "True"
		elif value_text in ["0", "false"]:
			formatted_value = "False"
		else:
			formatted_value = _format_caption_value(value)
	else:
		formatted_value = _format_caption_value(value)
	if param_name is None:
		return formatted_value
	return f"{param_name}={formatted_value}"


def _flow_to_rgb_video(flow: torch.Tensor) -> torch.Tensor:
	"""Convert pairwise optical flow (T, 2, H, W) → (T, H, W, 3) float32 RGB in [0,1].
	Uses the official Middlebury color wheel via torchvision.utils.flow_to_image.
	"""
	from torchvision.utils import flow_to_image as _tv_flow_to_image
	# flow_to_image expects (N, 2, H, W) float and returns (N, 3, H, W) uint8
	img = _tv_flow_to_image(flow.float())  # (T, 3, H, W) uint8
	return img.permute(0, 2, 3, 1).float() / 255.0  # (T, H, W, 3) in [0, 1]


def _compute_error_map(recon: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
	"""Compute per-pixel absolute error map (T, H, W, 3) colorised with viridis."""
	import matplotlib.cm as _cm

	def _to_thwc(x: torch.Tensor) -> torch.Tensor:
		x = torch.as_tensor(x).float().cpu()
		if x.ndim == 4:
			return x if x.shape[-1] in (1, 2, 3) else x.movedim(1, -1)
		if x.ndim == 3:
			return x.unsqueeze(-1)
		return x.unsqueeze(0).unsqueeze(-1)

	# both use the reference's display window so recon outliers don't rescale the error
	ref_t = _to_thwc(reference)
	window = _reference_display_window(ref_t)
	r_n = _apply_display_window(_to_thwc(recon), window)
	ref_n = _apply_display_window(ref_t, window)
	if r_n.shape[-1] > ref_n.shape[-1]:
		ref_n = ref_n.expand_as(r_n)
	elif ref_n.shape[-1] > r_n.shape[-1]:
		r_n = r_n.expand_as(ref_n)
	error = (r_n - ref_n).norm(dim=-1)  # (T, H, W)
	err_max = float(error.max())
	if err_max > 1e-8:
		error = error / err_max
	cmap = _cm.get_cmap("viridis")
	rgb = cmap(error.numpy())[..., :3].astype(np.float32)  # (T, H, W, 3)
	return torch.from_numpy(rgb)


def _compute_warp_residual_video(video_cf: torch.Tensor, flow_full: torch.Tensor) -> torch.Tensor:
	"""Compute warping residuals colorised with viridis.

	|frame[t] - warp(frame[t+1], flow_fwd[t])|, i.e. the CPL r0 training target.

	video_cf  : (T, C, H, W) float, channel-first
	flow_full : (T, 2, H, W) float (last frame duplicated so length == T)
	Returns   : (T, H, W, 3) float32 RGB in [0, 1] using viridis colormap
	"""
	import matplotlib.cm as _cm
	from torchvision.models.optical_flow._utils import make_coords_grid, grid_sample as _ov_gs

	T, C, H, W = video_cf.shape
	residuals = []
	for t in range(T - 1):
		base_grid = make_coords_grid(batch_size=1, h=H, w=W, device="cpu").movedim(1, -1)  # (1, H, W, 2)
		grid = base_grid + flow_full[t:t + 1].movedim(1, -1)  # (1, H, W, 2)
		warped_back = _ov_gs(video_cf[t + 1:t + 2].cpu(), grid, align_corners=True)  # warp frame[t+1] → t
		residual = (video_cf[t:t + 1].cpu() - warped_back).abs().mean(dim=1)  # (1, H, W)
		residuals.append(residual)
	residuals.append(torch.zeros(1, H, W))  # last frame: no next frame → zero residual
	residual_map = torch.cat(residuals, dim=0)  # (T, H, W)
	r_max = float(residual_map.max())
	if r_max > 1e-8:
		residual_map = residual_map / r_max
	cmap = _cm.get_cmap("viridis")
	rgb = cmap(residual_map.numpy())[..., :3].astype(np.float32)
	return torch.from_numpy(rgb)  # (T, H, W, 3)


def _enrich_payload_in_place(
	payload: Dict[str, Any],
	raft_model: Any,
	flow_source: str,
	compute_error: bool,
	compute_confidence: bool = False,
	compute_pseudorec_flow: bool = False,
	compute_warp_residual: bool = False,
) -> None:
	"""Lazily add 'flow_rgb', 'confidence_mask' and 'error_map' keys to an already-loaded payload dict.

	Flow and confidence come from the reference when available, else the recon.
	"""
	if raft_model is not None and (
		"flow_rgb" not in payload
		or (compute_confidence and "confidence_mask" not in payload)
		or (compute_pseudorec_flow and "flow_rgb_pseudorec" not in payload)
		or (compute_warp_residual and "warp_residual" not in payload)
	):
		# Prefer 'reference' for a stable, method-independent flow; fall back to flow_source then 'recon'
		_ref = payload.get("reference")
		_fs = payload.get(flow_source)
		_rc = payload.get("recon")
		source_video = _ref if _ref is not None else (_fs if _fs is not None else _rc)
		if source_video is not None:
			try:
				from src.regularizers.raft_based_regularizers import estimate_flow_for_video
				dev = str(next(raft_model.parameters()).device)
				vid = torch.as_tensor(source_video).float()
				if vid.ndim == 4:
					vid = vid if vid.shape[1] in (1, 2, 3) else vid.movedim(-1, 1)
				elif vid.ndim == 3:
					vid = vid.unsqueeze(1)
				with torch.no_grad():
					flow = estimate_flow_for_video(
						raft_model, vid.to(dev), enable_upsample_flow=False, forward_flow=True
					)  # (T-1, 2, H, W)
				flow_full = torch.cat([flow, flow[-1:]], dim=0).cpu()  # (T, 2, H, W)
				if compute_warp_residual and "warp_residual" not in payload:
					try:
						payload["warp_residual"] = _compute_warp_residual_video(vid.cpu(), flow_full)
					except Exception as _wr_err:
						print(f"[WARN] Warp residual computation failed: {_wr_err}")
				if "flow_rgb" not in payload:
					payload["flow_rgb"] = _flow_to_rgb_video(flow_full)
				if compute_confidence and "confidence_mask" not in payload:
					try:
						with torch.no_grad():
							bwd_flow = estimate_flow_for_video(
								raft_model, vid.to(dev), enable_upsample_flow=False, forward_flow=False
							)  # (T-1, 2, H, W)
						bwd_full = torch.cat([bwd_flow, bwd_flow[-1:]], dim=0).cpu()  # (T, 2, H, W)
						# Forward-backward consistency: ideally fwd + warp(bwd) ≈ 0; approximate with fwd + bwd
						fb_error = (flow_full + bwd_full).norm(dim=1)  # (T, H, W)
						tau = 2.0  # pixels
						floor = 0.05
						confidence = torch.exp(-fb_error.pow(2) / (2.0 * tau ** 2))  # (T, H, W) in [0,1]
						confidence_floored = confidence * (1.0 - floor) + floor  # (T, H, W)
						conf_gray = confidence_floored.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()  # (T, H, W, 3)
						payload["confidence_mask"] = conf_gray
					except Exception as _conf_err:
						print(f"[WARN] Confidence mask computation failed: {_conf_err}")
				if compute_pseudorec_flow and "flow_rgb_pseudorec" not in payload:
					_pseudo = payload.get("pseudoinv")
					if _pseudo is not None:
						try:
							pseudo_vid = torch.as_tensor(_pseudo).float()
							if pseudo_vid.ndim == 4:
								pseudo_vid = pseudo_vid if pseudo_vid.shape[1] in (1, 2, 3) else pseudo_vid.movedim(-1, 1)
							elif pseudo_vid.ndim == 3:
								pseudo_vid = pseudo_vid.unsqueeze(1)
							with torch.no_grad():
								pseudo_flow = estimate_flow_for_video(
									raft_model, pseudo_vid.to(dev), enable_upsample_flow=False, forward_flow=True
								)  # (T-1, 2, H, W)
							pseudo_flow_full = torch.cat([pseudo_flow, pseudo_flow[-1:]], dim=0).cpu()  # (T, 2, H, W)
							payload["flow_rgb_pseudorec"] = _flow_to_rgb_video(pseudo_flow_full)
						except Exception as _pseudo_flow_err:
							print(f"[WARN] Pseudorec flow estimation failed: {_pseudo_flow_err}")
			except Exception as _flow_err:
				print(f"[WARN] Flow estimation failed: {_flow_err}")
	if compute_error and "error_map" not in payload:
		recon_t = payload.get("recon")
		ref_t = payload.get("reference")
		if recon_t is not None and ref_t is not None:
			try:
				payload["error_map"] = _compute_error_map(recon_t, ref_t)
			except Exception as _err_err:
				print(f"[WARN] Error map computation failed: {_err_err}")


def _save_tensor_video_to_file(frames: torch.Tensor, path: str, fps: int = 10) -> bool:
	"""Save a (T, H, W, 3) float32 [0,1] tensor as an MP4 video file. Returns True on success."""
	try:
		import imageio
		arr = (frames.numpy() * 255).clip(0, 255).astype(np.uint8)
		imageio.mimwrite(path, [arr[t] for t in range(arr.shape[0])], fps=fps, format="mp4")
		return True
	except Exception as _vid_err:
		print(f"[WARN] Could not save video to {path}: {_vid_err}")
		return False


def _sanitize_for_filename(value: Any) -> str:
	"""Return a filesystem-safe slug derived from a parameter value."""
	import re
	s = str(value).strip()
	s = re.sub(r"[^\w\-]+", "_", s)
	s = re.sub(r"_+", "_", s).strip("_")
	return s or "unknown"


def _filter_metrics_by_cfg(
	metrics: List[str],
	metrics_filter_cfg: Optional[CacheableListConfig],
	warning_context: str,
) -> List[str]:
	if metrics_filter_cfg is None:
		return metrics

	metric_filters = [str(v).strip().lower() for v in metrics_filter_cfg.cfg if str(v).strip()]
	if len(metric_filters) == 0:
		return metrics

	filtered_metrics = [
		metric for metric in metrics
		if any(metric_filter in metric.lower() for metric_filter in metric_filters)
	]
	if len(filtered_metrics) == 0:
		print(
			f"[WARN] No metrics matched filters {metric_filters} for {warning_context}. Falling back to all numeric metrics."
		)
		return metrics
	return filtered_metrics


def _resolve_overlay_metrics(
	df_plot: pd.DataFrame,
	sweep_params: List[str],
	metrics_filter_cfg: Optional[CacheableListConfig],
	max_metrics: int = 2,
) -> List[str]:
	numeric_metrics: List[str] = []
	seen = set()
	for col_idx, col_name in enumerate(df_plot.columns):
		if col_name in seen:
			continue
		seen.add(col_name)
		if col_name in sweep_params or str(col_name) == "save_results":
			continue
		if pd.api.types.is_numeric_dtype(df_plot.iloc[:, col_idx]):
			numeric_metrics.append(str(col_name))

	filtered = _filter_metrics_by_cfg(
		metrics=numeric_metrics,
		metrics_filter_cfg=metrics_filter_cfg,
		warning_context="visual_examples_metric_overlay",
	)
	return filtered[:max(0, int(max_metrics))]


def _metric_overlay_lines_with_precision(
	row_dict: Mapping[Any, Any],
	metric_names: Sequence[str],
	decimals: int,
) -> List[str]:
	lines: List[str] = []
	for metric_name in metric_names:
		if metric_name not in row_dict:
			continue
		value = row_dict.get(metric_name)
		if value is None or pd.isna(value):
			continue
		if isinstance(value, (float, int, np.floating, np.integer)):
			value_text = f"{float(value):.{decimals}f}"
		else:
			value_text = str(value)
		lines.append(f"{metric_name}: {value_text}")
	return lines


def _draw_metric_overlay(
	ax,
	metric_lines: Sequence[str],
	text_color: str,
	fontsize: float,
	anchor_x: float = 0.02,
	anchor_y: float = 0.02,
) -> None:
	if len(metric_lines) == 0:
		return
	ax.text(
		anchor_x,
		anchor_y,
		"\n".join(metric_lines),
		transform=ax.transAxes,
		ha="right" if anchor_x > 0.5 else "left",
		va="bottom",
		fontsize=fontsize,
		color=text_color,
		bbox={"facecolor": "black", "alpha": 0.75, "pad": 2.5, "edgecolor": "none"},
	)


def _value_matches(series_value: Any, expected_value: Any) -> bool:
	if isinstance(expected_value, (list, tuple, set)):
		return any(_value_matches(series_value, candidate) for candidate in expected_value)
	if pd.isna(series_value) and pd.isna(expected_value):
		return True
	if series_value == expected_value:
		return True
	return str(series_value) == str(expected_value)


def _collapse_stochastic_params(
	df: pd.DataFrame,
	sweep_params: List[str],
	requested_stochastic_params: Sequence[Any],
) -> Tuple[pd.DataFrame, List[str]]:
	resolved_stochastic_params: List[str] = []
	for param in requested_stochastic_params:
		resolved_param = _resolve_column_name(str(param), df.columns.tolist())
		if resolved_param is None:
			print(f"[WARN] Visual examples stochastic parameter '{param}' not found; skipping it.")
			continue
		if resolved_param not in sweep_params:
			print(f"[WARN] Visual examples stochastic parameter '{param}' resolved to '{resolved_param}', but it is not a sweep parameter; skipping it.")
			continue
		if resolved_param not in resolved_stochastic_params:
			resolved_stochastic_params.append(resolved_param)

	if len(resolved_stochastic_params) == 0:
		return df, resolved_stochastic_params

	non_stochastic_sweep_params = [
		param for param in sweep_params if param in df.columns and param not in resolved_stochastic_params
	]
	print(
		f"[INFO] Visual examples: stochastic params={resolved_stochastic_params}; "
		f"collapsing to first sample per non-stochastic setting {non_stochastic_sweep_params}."
	)

	before_count = len(df)
	df_sorted = df.sort_values(resolved_stochastic_params)
	if len(non_stochastic_sweep_params) > 0:
		df_collapsed = df_sorted.groupby(non_stochastic_sweep_params, dropna=False, as_index=False).first()
	else:
		df_collapsed = df_sorted.head(1)
	print(f"[INFO] Visual examples: stochastic collapse kept {len(df_collapsed)}/{before_count} rows.")
	return df_collapsed, resolved_stochastic_params


def _brightness_match_video(video: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
	"""Linearly rescale *video* to the mean/std of *reference*, clamped to the reference's range."""
	v = torch.as_tensor(video).float()
	r = torch.as_tensor(reference).float()
	v_mean, v_std = float(v.mean()), float(v.std())
	r_mean, r_std = float(r.mean()), float(r.std())
	if v_std < 1e-8:
		return v
	scale = r_std / v_std if r_std > 1e-8 else 1.0
	matched = (v - v_mean) * scale + r_mean
	r_min, r_max = float(r.min()), float(r.max())
	return matched.clamp(r_min, r_max)


def _to_thwc(video: torch.Tensor) -> torch.Tensor:
	"""Normalize a raw (possibly channel-first, possibly 2-channel flow-like) tensor into (T, H, W, C<=3)."""
	x = torch.as_tensor(video).detach().cpu()

	if x.ndim == 4:
		if x.shape[-1] in (1, 3):
			frames = x
		elif x.shape[1] in (1, 3):
			frames = x.movedim(1, -1)
		elif x.shape[-1] == 2:
			frames = x.norm(dim=-1, keepdim=True)
		elif x.shape[1] == 2:
			frames = x.movedim(1, -1).norm(dim=-1, keepdim=True)
		else:
			raise ValueError(f"Unsupported 4D video shape for visualization: {tuple(x.shape)}")
	elif x.ndim == 3:
		if x.shape[-1] in (1, 3):
			frames = x.unsqueeze(0)
		elif x.shape[0] in (1, 3):
			frames = x.movedim(0, -1).unsqueeze(0)
		else:
			frames = x.unsqueeze(-1)
	elif x.ndim == 2:
		frames = x.unsqueeze(0).unsqueeze(-1)
	else:
		raise ValueError(f"Unsupported tensor shape for visualization: {tuple(x.shape)}")

	if frames.ndim != 4:
		raise ValueError(f"Expected THWC-like frames, got shape {tuple(frames.shape)}")
	if frames.shape[-1] == 2:
		frames = frames.norm(dim=-1, keepdim=True)
	if frames.shape[-1] == 1:
		frames = frames.repeat(1, 1, 1, 3)
	elif frames.shape[-1] > 3:
		frames = frames[..., :3]
	return frames


# intensity panels share the reference's display window; derived maps are already in [0, 1]
_INTENSITY_PANEL_KEYS = ("recon", "pseudoinv", "reference", "anchor_frame")


def _reference_display_window(reference: Optional[torch.Tensor]) -> Optional[Tuple[float, float]]:
	"""Shared display window (lo, hi) for all intensity panels: the reference's value range.

	Keeps all methods on one intensity scale and clamps their outliers."""
	if reference is None:
		return None
	ref = torch.nan_to_num(_to_thwc(reference).float())
	lo = float(ref.min())
	hi = float(ref.max())
	if not (hi > lo):
		return None
	return lo, hi


def _apply_display_window(frames: torch.Tensor, window: Optional[Tuple[float, float]]) -> torch.Tensor:
	"""Map *frames* linearly so that window (lo, hi) -> [0, 1] and clamp. With no window,
	fall back to the panel's own min/max stretch when it leaves [0, 1] (derived panels and
	entries without a stored reference)."""
	frames = torch.nan_to_num(frames.float())
	if window is None:
		f_min = float(frames.min())
		f_max = float(frames.max())
		if f_min < 0.0 or f_max > 1.0:
			if f_max > f_min:
				frames = (frames - f_min) / (f_max - f_min)
			else:
				frames = torch.zeros_like(frames)
		return frames.clamp(0.0, 1.0)
	lo, hi = window
	return ((frames - lo) / (hi - lo)).clamp(0.0, 1.0)


def _select_frame(video: torch.Tensor, frame_index: int, window: Optional[Tuple[float, float]] = None) -> np.ndarray:
	frames = _to_thwc(video)
	idx = min(max(int(frame_index), 0), frames.shape[0] - 1)
	return _apply_display_window(frames[idx], window).numpy()


def _prepare_video_thwc_normalized(video: torch.Tensor, window: Optional[Tuple[float, float]] = None) -> torch.Tensor:
	"""Whole clip through one display window (no per-frame flicker). Returns (T, H, W, 3) float32 in [0, 1]."""
	return _apply_display_window(_to_thwc(video), window)


def _select_time_slice(
	video: torch.Tensor,
	mode: str,
	center_row: int = -1,
	center_col: int = -1,
	window: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
	frames = _to_thwc(video)
	height = frames.shape[1]
	width = frames.shape[2]
	if mode == "x-t":
		if center_row < 0:
			row_idx = height // 2
		else:
			row_idx = min(max(int(center_row), 0), height - 1)
		slice_img = torch.nan_to_num(frames[:, row_idx, :, :].float())  # (T, X, C)
	elif mode == "y-t":
		if center_col < 0:
			col_idx = width // 2
		else:
			col_idx = min(max(int(center_col), 0), width - 1)
		slice_img = torch.nan_to_num(frames[:, :, col_idx, :].float()).movedim(0, 1)  # (Y, T, C)
	else:
		raise ValueError(f"Unsupported time slice mode: {mode}")

	return _apply_display_window(slice_img, window).numpy()


def _resize_hwc_nearest(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
	src_h, src_w = image.shape[0], image.shape[1]
	if src_h == target_h and src_w == target_w:
		return image
	y_idx = np.linspace(0, max(0, src_h - 1), num=target_h).round().astype(int)
	x_idx = np.linspace(0, max(0, src_w - 1), num=target_w).round().astype(int)
	return image[y_idx][:, x_idx]


def _compose_frame_with_time_slice(
	main_frame: np.ndarray,
	time_slice: np.ndarray,
	mode: str,
	size_ratio: float,
) -> np.ndarray:
	if time_slice is None or mode == "none":
		return main_frame
	height = main_frame.shape[0]
	width = main_frame.shape[1]
	if mode == "x-t":
		target_h = max(1, int(round(height * max(0.05, float(size_ratio)))))
		slice_resized = _resize_hwc_nearest(time_slice, target_h=target_h, target_w=width)
		return np.concatenate([main_frame, slice_resized], axis=0)
	if mode == "y-t":
		target_w = max(1, int(round(width * max(0.05, float(size_ratio)))))
		slice_resized = _resize_hwc_nearest(time_slice, target_h=height, target_w=target_w)
		return np.concatenate([main_frame, slice_resized], axis=1)
	return main_frame


def _overlay_anchor_for_time_slice(mode: str, size_ratio: float) -> Tuple[float, float]:
	if mode == "x-t":
		boundary = float(size_ratio) / (1.0 + float(size_ratio))
		return 0.02, min(0.98, boundary + 0.02)
	return 0.02, 0.02


def _perceptual_error_map(recon: torch.Tensor, reference: torch.Tensor, lpips_fn: Any) -> torch.Tensor:
	"""Per-pixel LPIPS distance (T, H, W) between *recon* and *reference*, both seen through the
	reference's display window, i.e. the difference as it shows up in the plotted panels. Falls
	back to the squared intensity error when no LPIPS model is available."""
	window = _reference_display_window(reference)
	rec = _apply_display_window(_to_thwc(recon), window)
	ref = _apply_display_window(_to_thwc(reference), window)
	num_frames = min(rec.shape[0], ref.shape[0])
	rec, ref = rec[:num_frames], ref[:num_frames]
	if lpips_fn is None:
		return (rec - ref).pow(2).mean(dim=-1)
	device = next(lpips_fn.parameters()).device
	maps = []
	with torch.no_grad():
		for t in range(num_frames):
			# LPIPS expects (N, 3, H, W) in [-1, 1]
			rec_t = rec[t:t + 1].movedim(-1, 1).to(device) * 2.0 - 1.0
			ref_t = ref[t:t + 1].movedim(-1, 1).to(device) * 2.0 - 1.0
			maps.append(lpips_fn(rec_t, ref_t)[0, 0].cpu())
	return torch.stack(maps)


def _select_zoom_boxes(
	error_map: torch.Tensor,
	frame_indices: Sequence[int],
	box_size: float,
	zoom_factor: float,
	border_margin: float = 0.0,
	shared_across_frames: bool = False,
) -> Dict[int, Tuple[Tuple[float, float, float, float], float]]:
	"""Per candidate frame, the *box_size* window with the largest mean of *error_map* (T, H, W),
	skipping windows under the lower-left inset or within *border_margin* of the edges.
	Returns {frame_idx: ((y0, x0, h, w) as frame fractions, mean error)}."""
	num_frames, height, width = error_map.shape
	box_h = min(height, max(1, int(round(box_size * height))))
	box_w = min(width, max(1, int(round(box_size * width))))
	window_means = torch.nn.functional.avg_pool2d(error_map[:, None].float(), (box_h, box_w), stride=1)[:, 0]
	# exclude windows overlapping the lower-left inset
	inset_h = min(height, int(round(box_h * zoom_factor)))
	inset_w = min(width, int(round(box_w * zoom_factor)))
	ys = torch.arange(window_means.shape[1])[:, None]
	xs = torch.arange(window_means.shape[2])[None, :]
	excluded = (ys + box_h > height - inset_h) & (xs < inset_w)
	margin_h = int(round(border_margin * height))
	margin_w = int(round(border_margin * width))
	if margin_h > 0 or margin_w > 0:
		excluded = excluded | (ys < margin_h) | (ys + box_h > height - margin_h) | (xs < margin_w) | (xs + box_w > width - margin_w)
	if not bool(excluded.all()):
		window_means[:, excluded] = float("-inf")
	frame_ts = {int(frame_idx): min(max(int(frame_idx), 0), num_frames - 1) for frame_idx in frame_indices}
	if shared_across_frames:
		shared_means = window_means[sorted(set(frame_ts.values()))].mean(dim=0)
		window_means = shared_means[None].expand(num_frames, -1, -1)
	boxes = {}
	for frame_idx, t in frame_ts.items():
		score, flat_idx = window_means[t].flatten().max(dim=0)
		y0, x0 = divmod(int(flat_idx), window_means.shape[2])
		boxes[frame_idx] = ((y0 / height, x0 / width, box_h / height, box_w / width), float(score))
	return boxes


def _box_pixels(frame_shape: Sequence[int], box: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
	"""(y0, x0, h, w) fractions of a frame -> integer pixel box clipped to the frame."""
	height, width = frame_shape[0], frame_shape[1]
	y0 = min(int(round(box[0] * height)), height - 1)
	x0 = min(int(round(box[1] * width)), width - 1)
	box_h = max(1, min(int(round(box[2] * height)), height - y0))
	box_w = max(1, min(int(round(box[3] * width)), width - x0))
	return y0, x0, box_h, box_w


def _draw_box_outline(frame: np.ndarray, box: Tuple[float, float, float, float]) -> np.ndarray:
	"""Copy of *frame* (H, W, 3) with *box* outlined in red (same line width as _draw_zoom_inset)."""
	height, width = frame.shape[0], frame.shape[1]
	top, left, h, w = _box_pixels(frame.shape, box)
	line_width = max(1, int(round(min(height, width) / 128)))
	out = frame.copy()
	red = np.array([1.0, 0.0, 0.0], dtype=out.dtype)
	out[top:top + line_width, left:left + w] = red
	out[top + h - line_width:top + h, left:left + w] = red
	out[top:top + h, left:left + line_width] = red
	out[top:top + h, left + w - line_width:left + w] = red
	return out


def _zoom_crop(frame: np.ndarray, box: Tuple[float, float, float, float], upscale: int) -> np.ndarray:
	"""The *box* region of *frame*, enlarged *upscale* times by pixel repetition (no detail invented)."""
	top, left, h, w = _box_pixels(frame.shape, box)
	crop = frame[top:top + h, left:left + w]
	return np.repeat(np.repeat(crop, upscale, axis=0), upscale, axis=1)


def _save_panel_with_zoom_variants(path: str, frame: np.ndarray, zoom_box: Optional[Tuple[float, float, float, float]], zoom_factor: float, upscale: int) -> None:
	"""Save *frame* (H, W, 3) in [0, 1]; with a zoom box, draw the inset and also write
	<name>__plain / __box / __crop variants."""
	if zoom_box is None:
		plt.imsave(path, frame)
		return
	stem, ext = os.path.splitext(path)
	plt.imsave(path, _draw_zoom_inset(frame, zoom_box, zoom_factor))
	plt.imsave(f"{stem}__plain{ext}", frame)
	plt.imsave(f"{stem}__box{ext}", _draw_box_outline(frame, zoom_box))
	plt.imsave(f"{stem}__crop{ext}", _zoom_crop(frame, zoom_box, upscale))


def _reference_flows(payload: Dict[str, Any], raft_model: Any) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
	"""RAFT forward (t -> t+1) and backward (t+1 -> t) flows (T-1, 2, H, W) of the payload's
	reference, estimated exactly like NaturalVideoSetup's ground-truth flows; cached in the payload."""
	if "_reference_flows" in payload:
		return payload["_reference_flows"]
	reference = payload.get("reference")
	if raft_model is None or reference is None:
		return None
	from src.regularizers.raft_based_regularizers import estimate_flow_for_video
	device = str(next(raft_model.parameters()).device)
	video = _to_thwc(reference).movedim(-1, 1).to(device)  # (T, 3, H, W)
	flow_fwd = estimate_flow_for_video(raft_model, video, forward_flow=True).cpu()
	flow_bwd = estimate_flow_for_video(raft_model, video, forward_flow=False).cpu()
	payload["_reference_flows"] = (flow_fwd, flow_bwd)
	return payload["_reference_flows"]


def _next_frame_flows_to_anchor(flow_fwd: torch.Tensor, flow_bwd: torch.Tensor, target: str) -> Tuple[torch.Tensor, torch.Tensor]:
	"""(flow from each predicted frame to its anchor, opposite flow) for single-step next-frame
	prediction, see NaturalVideoSetup._compute_next_frame_prediction_metrics: "next" predicts frame
	t+1 from frame t (backward flow), "previous" frame t from frame t+1 (forward flow)."""
	return (flow_fwd, flow_bwd) if target == "previous" else (flow_bwd, flow_fwd)


def _pad_to_video(per_pair: torch.Tensor, target: str, fill: float) -> torch.Tensor:
	"""Per-pair maps (T-1, ...) indexed by predicted frame -> (T, ...); the anchor frame (the first
	frame for "next", the last for "previous") gets *fill*."""
	anchor = torch.full_like(per_pair[:1], fill)
	return torch.cat([per_pair, anchor]) if target == "previous" else torch.cat([anchor, per_pair])


def _next_frame_confidence_video(flows: Tuple[torch.Tensor, torch.Tensor], target: str, tau_pix: float) -> torch.Tensor:
	"""Soft forward-backward confidence exp(-|m_to + W(m_from)|^2 / (2 tau^2)) of the flow that
	drives each predicted frame's warp, in that frame's coordinates (the nfp_*_conf split of
	NaturalVideoSetup); white = reliable, the anchor frame is all white. Returns (T, H, W, 3)."""
	from torchvision.models.optical_flow._utils import make_coords_grid, grid_sample as _ov_gs
	flow_to_anchor, flow_from_anchor = _next_frame_flows_to_anchor(*flows, target)
	num_pairs, _, height, width = flow_to_anchor.shape
	base_grid = make_coords_grid(batch_size=num_pairs, h=height, w=width, device="cpu").movedim(1, -1)
	flow_from_anchor_warped = _ov_gs(flow_from_anchor, base_grid + flow_to_anchor.movedim(1, -1), align_corners=True, mode="bilinear")
	fb_error_sq = (flow_to_anchor + flow_from_anchor_warped).square().sum(dim=1)  # (T-1, H, W)
	confidence = _pad_to_video(torch.exp(-fb_error_sq / (2.0 * float(tau_pix) ** 2)), target, 1.0)
	return confidence.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()


def _anchor_frame_video(reference: torch.Tensor, target: str) -> torch.Tensor:
	"""For every frame of a single-step next-frame prediction, the ground-truth frame it was
	predicted from: frame t-1 for "next", frame t+1 for "previous" (the anchor frame shows itself)."""
	frames = _to_thwc(reference)
	if target == "previous":
		return torch.cat([frames[1:], frames[-1:]])
	return torch.cat([frames[:1], frames[:-1]])


def _frame_metric_lines(
	recon: torch.Tensor,
	reference: torch.Tensor,
	frame_idx: int,
	lpips_fn: Any,
	decimals: int,
) -> List[str]:
	"""PSNR (data range 1 after the reference display window, as nfp_rgb_peak_snr) and LPIPS of a
	single frame, for metric overlays on figures whose rows are frames."""
	window = _reference_display_window(reference)
	rec = torch.from_numpy(_select_frame(recon, frame_idx, window)).float()
	ref = torch.from_numpy(_select_frame(reference, frame_idx, window)).float()
	mse = float((rec - ref).square().mean().clamp_min(1e-12))
	lines = [f"PSNR {-10.0 * np.log10(mse):.{decimals}f} dB"]
	if lpips_fn is not None:
		device = next(lpips_fn.parameters()).device
		with torch.no_grad():
			value = lpips_fn(rec.movedim(-1, 0)[None].to(device) * 2.0 - 1.0, ref.movedim(-1, 0)[None].to(device) * 2.0 - 1.0).mean()
		lines.append(f"LPIPS {float(value):.{max(decimals, 3)}f}")
	return lines


def _draw_zoom_inset(frame: np.ndarray, box: Tuple[float, float, float, float], zoom_factor: float) -> np.ndarray:
	"""Outline *box* (y0, x0, h, w as fractions of the frame) in red and paste the region, enlarged
	by *zoom_factor* (nearest neighbour, so no detail is invented) and outlined in red as well, into
	the lower-left corner of a copy of *frame* (H, W, 3) in [0, 1]."""
	height, width = frame.shape[0], frame.shape[1]
	y0 = min(int(round(box[0] * height)), height - 1)
	x0 = min(int(round(box[1] * width)), width - 1)
	box_h = max(1, min(int(round(box[2] * height)), height - y0))
	box_w = max(1, min(int(round(box[3] * width)), width - x0))
	inset_h = min(height, int(round(box_h * zoom_factor)))
	inset_w = min(width, int(round(box_w * zoom_factor)))
	ys = y0 + np.minimum((np.arange(inset_h) / zoom_factor).astype(int), box_h - 1)
	xs = x0 + np.minimum((np.arange(inset_w) / zoom_factor).astype(int), box_w - 1)
	out = frame.copy()
	out[height - inset_h:, :inset_w] = frame[ys][:, xs]
	line_width = max(1, int(round(min(height, width) / 128)))
	red = np.array([1.0, 0.0, 0.0], dtype=out.dtype)
	for top, left, h, w in ((y0, x0, box_h, box_w), (height - inset_h, 0, inset_h, inset_w)):
		out[top:top + line_width, left:left + w] = red
		out[top + h - line_width:top + h, left:left + w] = red
		out[top:top + h, left:left + line_width] = red
		out[top:top + h, left + w - line_width:left + w] = red
	return out


def _load_visual_payload(
	result_cfg: Mapping[Any, Any],
	cache: Dict[str, Dict[str, Any]],
	temp_root: str,
) -> Optional[Dict[str, Any]]:
	storage_path_info = result_cfg.get("storage_path")
	output_dir_in_fs = None
	storage_settings_dict = None
	if storage_path_info is not None:
		if isinstance(storage_path_info, StoragePath):
			output_dir_in_fs = storage_path_info.storage_path
			storage_settings = storage_path_info.storage_settings
		else:
			output_dir_in_fs = storage_path_info.get("storage_path")
			storage_settings_dict = storage_path_info.get("storage_settings")
			storage_settings = StorageSettings(**storage_settings_dict) if storage_settings_dict is not None else None
	else:
		output_dir_in_fs = result_cfg.get("output_dir_in_fs")
		storage_settings_dict = result_cfg.get("storage_settings")
		storage_settings = StorageSettings(**storage_settings_dict) if storage_settings_dict is not None else None

	if not output_dir_in_fs:
		print("[WARN] Skipping visual example entry without storage_path metadata.")
		return None
	if output_dir_in_fs in cache:
		return cache[output_dir_in_fs]

	if storage_settings is None:
		print(
			f"Result '{output_dir_in_fs}' has no storage_settings metadata; using default remote MinIO settings."
		)
		storage_settings = StorageSettings(
			storage_name="minio-caching-storage",
			bucket_or_base_path="prefect-results",
			is_remote=True,
		)

	filesystem = load_file_system(storage_settings)
	local_dir = os.path.join(temp_root, hashlib.md5(output_dir_in_fs.encode("utf-8")).hexdigest())
	os.makedirs(local_dir, exist_ok=True)
	filesystem.get_directory(output_dir_in_fs, local_dir)

	recon_path = os.path.join(local_dir, "recon.pt")
	reference_path = os.path.join(local_dir, "reference_video.pt")
	pseudoinv_path = os.path.join(local_dir, "pseudoinv_video.pt")

	if not os.path.exists(recon_path):
		print(f"[WARN] Reconstruction file missing for '{output_dir_in_fs}', skipping visual entry.")
		return None

	payload = {
		"recon": torch.load(recon_path, map_location="cpu"),
		"reference": torch.load(reference_path, map_location="cpu") if os.path.exists(reference_path) else None,
		"pseudoinv": torch.load(pseudoinv_path, map_location="cpu") if os.path.exists(pseudoinv_path) else None,
		"output_dir_in_fs": output_dir_in_fs,
		"storage_settings": storage_settings,
	}
	cache[output_dir_in_fs] = payload
	return payload


def _extract_storage_settings_from_storage_path(storage_path_obj: Any) -> Optional[StorageSettings]:
	if isinstance(storage_path_obj, StoragePath):
		return storage_path_obj.storage_settings
	if isinstance(storage_path_obj, Mapping):
		storage_settings_dict = storage_path_obj.get("storage_settings")
		if storage_settings_dict is not None:
			return StorageSettings(**storage_settings_dict)
	return None


def _infer_row_col_params(
	df: pd.DataFrame,
	sweep_params: List[str],
	row_param: Optional[str],
	col_param: Optional[str],
) -> Tuple[Optional[str], Optional[str], List[str]]:
	varying_params = [param for param in sweep_params if param in df.columns and df[param].nunique(dropna=False) > 1]

	resolved_row = _resolve_column_name(row_param, sweep_params)
	resolved_col = _resolve_column_name(col_param, sweep_params)

	if resolved_row is None and resolved_col is None:
		sorted_params = sorted(varying_params, key=lambda param: df[param].nunique(dropna=False), reverse=True)
		if len(sorted_params) >= 1:
			resolved_col = sorted_params[0]
		if len(sorted_params) >= 2:
			resolved_row = sorted_params[1]
	elif resolved_row is None and resolved_col is not None:
		remaining = [param for param in varying_params if param != resolved_col]
		if remaining:
			resolved_row = sorted(remaining, key=lambda param: df[param].nunique(dropna=False), reverse=True)[0]
	elif resolved_col is None and resolved_row is not None:
		remaining = [param for param in varying_params if param != resolved_row]
		if remaining:
			resolved_col = sorted(remaining, key=lambda param: df[param].nunique(dropna=False), reverse=True)[0]

	if resolved_row == resolved_col:
		resolved_row = None

	figure_group_params = [param for param in varying_params if param not in {resolved_row, resolved_col}]
	return resolved_row, resolved_col, figure_group_params


def _build_auto_figsize(
	num_rows: int,
	num_cols: int,
	explicit_figsize: Optional[Sequence[float]],
) -> Tuple[float, float]:
	if explicit_figsize is not None:
		return float(explicit_figsize[0]), float(explicit_figsize[1])
	return max(4.0, 2.8 * max(1, num_cols)), max(3.5, 2.8 * max(1, num_rows) + 0.7)


def _resolve_frame_indices(
	visual_examples_frame_idx: int,
	visual_examples_frame_idxs_cfg: Optional[CacheableListConfig],
) -> List[int]:
	if visual_examples_frame_idxs_cfg is None:
		return [int(visual_examples_frame_idx)]
	frame_idxs_container = OmegaConf.to_container(visual_examples_frame_idxs_cfg.cfg, resolve=True)
	if isinstance(frame_idxs_container, list) and len(frame_idxs_container) > 0:
		return [int(frame_idx) for frame_idx in frame_idxs_container]
	return [int(visual_examples_frame_idx)]


def _build_comparative_grid_video(
	figure_df: pd.DataFrame,
	figure_name: str,
	row_values: List[Any],
	col_values: List[Any],
	row_param: Optional[str],
	col_param: Optional[str],
	row_param_is_bool: bool,
	col_param_is_bool: bool,
	auxiliary_items: List[Tuple[str, str]],
	auxiliary_layout: str,
	auxiliary_position: str,
	payload_cache: Dict[str, Dict[str, Any]],
	temp_dir: str,
	normalize_brightness: bool,
	metric_overlay_enabled: bool,
	overlay_metrics: List[str],
	overlay_text_color: str,
	overlay_fontsize: float,
	overlay_decimals: int,
	overlay_anchor_x: float,
	overlay_anchor_y: float,
	dpi: int,
	explicit_figsize: Optional[Sequence[float]],
	max_frames: int,
	frame_stride: int,
) -> Optional[Tuple[torch.Tensor, int, List[Dict[str, Any]]]]:
	"""Render one figure group as a synchronized grid animation laid out like the static figure.

	Returns (frames (T,H,W,3) float32 in [0,1], num_frames, cell_videos) or None; ``cell_videos``
	holds per-cell dicts (``name``, ``frames``, row/col metadata) for standalone export.
	"""
	base_num_rows = max(1, len(row_values))
	base_num_cols = max(1, len(col_values))
	num_auxiliary_items = len(auxiliary_items)
	if auxiliary_layout == "columns":
		num_rows = base_num_rows
		num_cols = base_num_cols + num_auxiliary_items
		recon_col_offset = num_auxiliary_items if auxiliary_position == "start" else 0
		auxiliary_col_start = 0 if auxiliary_position == "start" else base_num_cols
		auxiliary_col_index = None
	else:
		num_rows = max(base_num_rows, num_auxiliary_items)
		num_cols = base_num_cols + (1 if num_auxiliary_items > 0 else 0)
		recon_col_offset = 1 if auxiliary_position == "start" and num_auxiliary_items > 0 else 0
		auxiliary_col_start = None
		auxiliary_col_index = 0 if auxiliary_position == "start" and num_auxiliary_items > 0 else num_cols - 1

	figsize = _build_auto_figsize(num_rows=num_rows, num_cols=num_cols, explicit_figsize=explicit_figsize)
	fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, dpi=dpi, squeeze=False)

	# (axis, prepared (T,H,W,3) clip, cell metadata for the standalone export)
	cell_specs: List[Tuple[Any, torch.Tensor, Dict[str, Any]]] = []

	for row_idx, row_value in enumerate(row_values):
		row_df = figure_df
		if row_param is not None:
			row_df = row_df[row_df[row_param].map(lambda value: _value_matches(value, row_value))]
		_row_slug = _sanitize_for_filename(row_value) if row_value is not None else f"r{row_idx:02d}"

		row_label_text = _format_param_label(None, row_value, row_param_is_bool) if row_param is not None else None
		if row_label_text is not None:
			axes[row_idx][0].text(
				-0.22, 0.5, row_label_text, transform=axes[row_idx][0].transAxes, ha="right", va="center", fontsize=10,
			)

		for col_idx, col_value in enumerate(col_values):
			cell_df = row_df
			if col_param is not None:
				cell_df = cell_df[cell_df[col_param].map(lambda value: _value_matches(value, col_value))]
			ax = axes[row_idx][col_idx + recon_col_offset]
			ax.axis("off")
			if cell_df.empty:
				ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=10)
				continue
			result_row = cell_df.iloc[0].to_dict()
			payload = _load_visual_payload(result_row, payload_cache, temp_dir)
			if payload is None:
				ax.text(0.5, 0.5, "missing file", ha="center", va="center", fontsize=10)
				continue
			recon_video = (
				_brightness_match_video(payload["recon"], payload["reference"])
				if normalize_brightness and payload.get("reference") is not None
				else payload["recon"]
			)
			_col_slug = _sanitize_for_filename(_format_param_label(None, col_value, col_param_is_bool)) if col_value is not None else f"c{col_idx:02d}"
			cell_specs.append((ax, _prepare_video_thwc_normalized(recon_video, _reference_display_window(payload.get("reference"))), {
				"name": f"recon__{_col_slug}__row{_row_slug}",
				"panel_type": "recon",
				"row_index": row_idx,
				"col_index": col_idx + recon_col_offset,
				"row_value": row_value,
				"col_value": col_value,
				"source_output_dir_in_fs": payload["output_dir_in_fs"],
			}))
			if row_idx == 0:
				if col_param is not None:
					ax.set_title(_format_param_label(None, col_value, col_param_is_bool), fontsize=10)
				else:
					ax.set_title("Reconstruction", fontsize=10)
			if metric_overlay_enabled:
				_draw_metric_overlay(
					ax,
					_metric_overlay_lines_with_precision(result_row, overlay_metrics, overlay_decimals),
					overlay_text_color,
					overlay_fontsize,
					anchor_x=overlay_anchor_x,
					anchor_y=overlay_anchor_y,
				)

		if auxiliary_layout == "columns":
			exemplar_df = row_df
			if col_param is not None and len(col_values) > 0:
				anchor_col_value = col_values[0]
				exemplar_df = exemplar_df[exemplar_df[col_param].map(lambda value: _value_matches(value, anchor_col_value))]
			if exemplar_df.empty:
				exemplar_df = row_df
			exemplar_row = exemplar_df.iloc[0].to_dict() if not exemplar_df.empty else None
			exemplar_payload = _load_visual_payload(exemplar_row, payload_cache, temp_dir) if exemplar_row is not None else None
			for aux_idx, (panel_key, panel_title) in enumerate(auxiliary_items):
				actual_col_idx = cast(int, auxiliary_col_start) + aux_idx
				ax = axes[row_idx][actual_col_idx]
				ax.axis("off")
				panel_value = exemplar_payload.get(panel_key) if exemplar_payload is not None else None
				if normalize_brightness and panel_key == "pseudoinv" and exemplar_payload is not None and exemplar_payload.get("reference") is not None and panel_value is not None:
					panel_value = _brightness_match_video(panel_value, exemplar_payload["reference"])
				if panel_value is not None:
					panel_window = _reference_display_window(exemplar_payload.get("reference")) if panel_key in _INTENSITY_PANEL_KEYS else None
					cell_specs.append((ax, _prepare_video_thwc_normalized(panel_value, panel_window), {
						"name": f"{panel_key}__row{_row_slug}",
						"panel_type": panel_key,
						"row_index": row_idx,
						"col_index": actual_col_idx,
						"row_value": row_value,
						"col_value": None,
						"source_output_dir_in_fs": exemplar_payload["output_dir_in_fs"],
					}))
					if row_idx == 0:
						ax.set_title(panel_title, fontsize=10)
				else:
					ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=10)

	if auxiliary_layout == "rows" and num_auxiliary_items > 0:
		shared_panel_row_value = row_values[0] if len(row_values) > 0 else None
		for aux_idx, (panel_key, panel_title) in enumerate(auxiliary_items):
			if aux_idx >= num_rows:
				break
			ax = axes[aux_idx][cast(int, auxiliary_col_index)]
			ax.axis("off")
			panel_df = figure_df
			if row_param is not None and shared_panel_row_value is not None:
				panel_df = panel_df[panel_df[row_param].map(lambda value: _value_matches(value, shared_panel_row_value))]
			exemplar_row = panel_df.iloc[0].to_dict() if not panel_df.empty else None
			exemplar_payload = _load_visual_payload(exemplar_row, payload_cache, temp_dir) if exemplar_row is not None else None
			panel_value = exemplar_payload.get(panel_key) if exemplar_payload is not None else None
			if normalize_brightness and panel_key == "pseudoinv" and exemplar_payload is not None and exemplar_payload.get("reference") is not None and panel_value is not None:
				panel_value = _brightness_match_video(panel_value, exemplar_payload["reference"])
			if panel_value is not None:
				panel_window = _reference_display_window(exemplar_payload.get("reference")) if panel_key in _INTENSITY_PANEL_KEYS else None
				cell_specs.append((ax, _prepare_video_thwc_normalized(panel_value, panel_window), {
					"name": panel_key,
					"panel_type": panel_key,
					"row_index": aux_idx,
					"col_index": auxiliary_col_index,
					"row_value": shared_panel_row_value,
					"col_value": None,
					"source_output_dir_in_fs": exemplar_payload["output_dir_in_fs"],
				}))
				ax.set_title(panel_title, fontsize=10)
			else:
				ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=10)

	if len(cell_specs) == 0:
		plt.close(fig)
		return None

	num_frames_available = min(int(prepared.shape[0]) for _, prepared, _ in cell_specs)
	if num_frames_available <= 0:
		plt.close(fig)
		return None
	stride = max(1, int(frame_stride))
	t_indices = list(range(0, num_frames_available, stride))
	if max_frames > 0:
		t_indices = t_indices[: int(max_frames)]
	if len(t_indices) == 0:
		plt.close(fig)
		return None

	fig.suptitle(figure_name, fontsize=12)
	frame_counter_text = fig.text(0.99, 0.01, "", ha="right", va="bottom", fontsize=9)
	# tight margins, only room for labels
	plt.subplots_adjust(
		left=0.14 if row_param is not None else 0.01,
		right=0.99,
		top=0.88,
		bottom=0.04,
		wspace=0.03,
		hspace=0.03,
	)

	im_handles = [ax.imshow(prepared[t_indices[0]].numpy()) for ax, prepared, _ in cell_specs]

	rendered_frames: List[np.ndarray] = []
	for t in t_indices:
		for im_obj, (_, prepared, _) in zip(im_handles, cell_specs):
			im_obj.set_data(prepared[t].numpy())
		frame_counter_text.set_text(f"frame {t}")
		fig.canvas.draw()
		frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
		frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3].copy()
		rendered_frames.append(frame)

	plt.close(fig)
	frames_float = torch.from_numpy(np.stack(rendered_frames, axis=0).astype(np.float32) / 255.0)
	cell_videos = [dict(meta, frames=prepared[t_indices]) for _, prepared, meta in cell_specs]
	return frames_float, len(t_indices), cell_videos


@task(name="Visual Examples Plot", tags=["plot", "visual"], version="1.0", retries=0)
def plot_visual_examples(
	results: List[CacheableDict],
	df_sweep: pd.DataFrame,
	wandb_params_task: WandbParamsTask,
	visual_examples_enabled: bool = False,
	visual_examples_frame_idx: int = 0,
	visual_examples_frame_idxs_cfg: Optional[CacheableListConfig] = None,
	visual_examples_stochastic_params_cfg: Optional[CacheableListConfig] = None,
	visual_examples_row_param: Optional[str] = None,
	visual_examples_col_param: Optional[str] = None,
	visual_examples_max_rows: int = 2,
	visual_examples_max_cols: int = 4,
	visual_examples_figsize_cfg: Optional[CacheableListConfig] = None,
	visual_examples_formats_cfg: Optional[CacheableListConfig] = None,
	visual_examples_auxiliary_layout: str = "columns",
	visual_examples_auxiliary_position: str = "end",
	visual_examples_metric_overlay_enabled: bool = False,
	visual_examples_metric_overlay_metrics_cfg: Optional[CacheableListConfig] = None,
	visual_examples_metric_overlay_text_color: str = "white",
	visual_examples_metric_overlay_decimals: int = 2,
	visual_examples_metric_overlay_fontsize: float = 8.0,
	visual_examples_time_slice_mode: str = "none",
	visual_examples_time_slice_size_ratio: float = 0.25,
	visual_examples_time_slice_center_row: int = -1,
	visual_examples_time_slice_center_col: int = -1,
	visual_examples_xt_slice_enabled: bool = False,
	visual_examples_xt_slice_height_ratio: float = 0.25,
	visual_examples_xt_slice_center_row: int = -1,
	visual_examples_include_reference: bool = True,
	visual_examples_include_pseudoinv: bool = True,
	visual_examples_include_flow: bool = False,
	visual_examples_include_flow_pseudorec: bool = False,
	visual_examples_include_confidence_mask: bool = False,
	visual_examples_include_warp_residual: bool = False,
	visual_examples_normalize_brightness: bool = False,
	visual_examples_flow_enabled: bool = False,
	visual_examples_flow_source: str = "reference",
	visual_examples_confidence_enabled: bool = False,
	visual_examples_storage_subdir: str = "visual_examples",
	visual_examples_dpi: int = 300,
	visual_examples_video_enabled: bool = False,
	visual_examples_video_fps: int = 8,
	visual_examples_video_max_frames: int = 0,
	visual_examples_video_frame_stride: int = 1,
	visual_examples_video_dpi: Optional[int] = None,
	visual_examples_zoom_enabled: bool = False,
	visual_examples_zoom_factor: float = 2.0,
	visual_examples_zoom_box_cfg: Optional[CacheableListConfig] = None,
	visual_examples_zoom_box_size: float = 0.2,
	visual_examples_zoom_auto_frame: bool = False,
	visual_examples_zoom_criterion: str = "lpips",
	visual_examples_zoom_border_margin: float = 0.0,
	visual_examples_zoom_shared_across_frames: bool = False,
	visual_examples_zoom_crop_upscale: int = 4,
	visual_examples_frames_as_rows: bool = False,
	visual_examples_row_values_cfg: Optional[CacheableListConfig] = None,
	visual_examples_col_labels_cfg: Optional[CacheableDictConfig] = None,
	visual_examples_auxiliary_end_keys_cfg: Optional[CacheableListConfig] = None,
	visual_examples_include_anchor_frame: bool = False,
	visual_examples_include_next_frame_confidence: bool = False,
	visual_examples_next_frame_prediction_target: str = "next",
	visual_examples_next_frame_confidence_tau_pix: float = 1.0,
) -> None:
	if not visual_examples_enabled:
		print("[INFO] Visual example plotting disabled; skipping task.")
		return

	if len(results) == 0:
		print("[WARN] No results provided for visual example plotting.")
		return

	df_results = pd.DataFrame([_result_to_dict(result) for result in results])
	df_plot = pd.concat([df_results, df_sweep.reset_index(drop=True)], axis=1)
	sweep_params = df_sweep.columns.tolist()
	print(f"[INFO] Visual examples: received {len(df_plot)} result rows before filtering.")

	if "save_results" in df_plot.columns:
		save_results_col = df_plot["save_results"].map(lambda value: str(value).strip().lower() in ["true", "1", "yes"])
		if bool(save_results_col.any()):
			df_plot = df_plot[save_results_col]
		else:
			if "storage_path" in df_plot.columns and df_plot["storage_path"].notna().any():
				print(
					"save_results column is present but no rows evaluate to True; using rows with storage_path metadata instead."
				)
				df_plot = df_plot[df_plot["storage_path"].notna()]
			else:
				df_plot = df_plot.iloc[0:0]
	if df_plot.empty:
		print("[WARN] No saved reconstructions available after filtering save_results=True.")
		return

	requested_stochastic_params = []
	if visual_examples_stochastic_params_cfg is not None:
		stochastic_container = OmegaConf.to_container(visual_examples_stochastic_params_cfg.cfg, resolve=True)
		if isinstance(stochastic_container, list):
			requested_stochastic_params = stochastic_container

	df_plot, resolved_stochastic_params = _collapse_stochastic_params(
		df=df_plot,
		sweep_params=sweep_params,
		requested_stochastic_params=requested_stochastic_params,
	)
	if df_plot.empty:
		print("[WARN] Visual examples: stochastic collapse removed all rows; nothing to plot.")
		return

	effective_sweep_params = [
		param for param in sweep_params if param in df_plot.columns and param not in resolved_stochastic_params
	]
	row_param, col_param, figure_group_params = _infer_row_col_params(
		df=df_plot,
		sweep_params=effective_sweep_params,
		row_param=visual_examples_row_param,
		col_param=visual_examples_col_param,
	)

	row_values = _ordered_unique(df_plot[row_param].tolist()) if row_param is not None else [None]
	col_values = _ordered_unique(df_plot[col_param].tolist()) if col_param is not None else [None]
	row_param_is_bool = bool(row_param is not None and _is_bool_like_param_values(row_values))
	col_param_is_bool = bool(col_param is not None and _is_bool_like_param_values(col_values))

	if visual_examples_row_values_cfg is not None and row_param is not None:
		requested_row_values = list(OmegaConf.to_container(visual_examples_row_values_cfg.cfg, resolve=True))
		if len(requested_row_values) > 0:
			selected_row_values = []
			for requested in requested_row_values:
				matches = [value for value in row_values if _value_matches(value, requested)]
				if len(matches) == 0:
					print(f"[WARN] Visual examples: requested row value {requested!r} not found for {row_param}.")
				selected_row_values += matches[:1]
			row_values = selected_row_values

	if visual_examples_max_rows > 0 and len(row_values) > visual_examples_max_rows:
		print(f"[INFO] Visual examples: truncating rows from {len(row_values)} to max_rows={visual_examples_max_rows}.")
		row_values = row_values[:visual_examples_max_rows]
	if visual_examples_max_cols > 0 and len(col_values) > visual_examples_max_cols:
		print(f"[INFO] Visual examples: truncating cols from {len(col_values)} to max_cols={visual_examples_max_cols}.")
		col_values = col_values[:visual_examples_max_cols]
	if row_param is not None:
		row_desc = [_format_param_label(row_param, row_value, row_param_is_bool) for row_value in row_values]
	else:
		row_desc = row_values
	if col_param is not None:
		col_desc = [_format_param_label(col_param, col_value, col_param_is_bool) for col_value in col_values]
	else:
		col_desc = col_values
	print(f"[INFO] Visual examples: plotting rows={row_desc}, cols={col_desc}.")

	# Rows are frames instead: every selected row value becomes its own figure.
	if visual_examples_frames_as_rows and row_param is not None:
		df_plot = df_plot[df_plot[row_param].map(lambda value: _value_matches(value, row_values))]
		figure_group_params = figure_group_params + [row_param]
		row_param, row_values, row_param_is_bool = None, [None], False

	col_labels: Dict[str, str] = {}
	if visual_examples_col_labels_cfg is not None:
		col_labels = {str(key): str(value) for key, value in OmegaConf.to_container(visual_examples_col_labels_cfg.cfg, resolve=True).items()}
		# labelled columns first, in label order
		col_label_order = list(col_labels)
		col_values = sorted(col_values, key=lambda value: col_label_order.index(str(value)) if str(value) in col_label_order else len(col_label_order))

	def _col_title(col_value: Any) -> str:
		return col_labels.get(str(col_value), _format_param_label(None, col_value, col_param_is_bool))

	if len(figure_group_params) > 0:
		grouped_figures = list(df_plot.groupby(figure_group_params, dropna=False))
	else:
		grouped_figures = [("all", df_plot)]

	formats = ["png", "svg"]
	if visual_examples_formats_cfg is not None and len(list(visual_examples_formats_cfg.cfg)) > 0:
		formats = [str(fmt).lower() for fmt in visual_examples_formats_cfg.cfg]
	overlay_text_color = str(visual_examples_metric_overlay_text_color).strip().lower()
	if overlay_text_color not in ["white", "black"]:
		print(
			f"[WARN] Unsupported visual_examples_metric_overlay_text_color='{visual_examples_metric_overlay_text_color}', falling back to 'white'."
		)
		overlay_text_color = "white"
	overlay_decimals = int(visual_examples_metric_overlay_decimals)
	if overlay_decimals < 0:
		print(
			f"[WARN] Unsupported visual_examples_metric_overlay_decimals='{visual_examples_metric_overlay_decimals}', falling back to 2."
		)
		overlay_decimals = 2
	overlay_fontsize = float(visual_examples_metric_overlay_fontsize)
	if overlay_fontsize <= 0:
		print(
			f"[WARN] Unsupported visual_examples_metric_overlay_fontsize='{visual_examples_metric_overlay_fontsize}', falling back to 8.0."
		)
		overlay_fontsize = 8.0
	time_slice_mode = str(visual_examples_time_slice_mode).strip().lower()
	if visual_examples_xt_slice_enabled and time_slice_mode == "none":
		time_slice_mode = "x-t"
	if time_slice_mode not in ["none", "x-t", "y-t"]:
		print(
			f"[WARN] Unsupported visual_examples_time_slice_mode='{visual_examples_time_slice_mode}', falling back to 'none'."
		)
		time_slice_mode = "none"
	time_slice_ratio = float(visual_examples_time_slice_size_ratio)
	if visual_examples_xt_slice_enabled and visual_examples_time_slice_mode == "none":
		time_slice_ratio = float(visual_examples_xt_slice_height_ratio)
	if time_slice_ratio <= 0:
		print(
			f"[WARN] Unsupported visual_examples_time_slice_size_ratio='{visual_examples_time_slice_size_ratio}', falling back to 0.25."
		)
		time_slice_ratio = 0.25
	overlay_anchor_x, overlay_anchor_y = _overlay_anchor_for_time_slice(time_slice_mode, time_slice_ratio)
	if visual_examples_zoom_enabled:
		# The lower-left corner holds the zoom inset, so the metric overlay moves to the right.
		overlay_anchor_x = 0.98
	fixed_zoom_box: Optional[Tuple[float, float, float, float]] = None
	if visual_examples_zoom_box_cfg is not None:
		zoom_box_values = [float(v) for v in OmegaConf.to_container(visual_examples_zoom_box_cfg.cfg, resolve=True)]
		if len(zoom_box_values) == 4:
			fixed_zoom_box = (zoom_box_values[0], zoom_box_values[1], zoom_box_values[2], zoom_box_values[3])
		else:
			print(f"[WARN] visual_examples_zoom_box must be [y0, x0, h, w], got {zoom_box_values}; selecting the region automatically.")
	overlay_metrics: List[str] = []
	if visual_examples_metric_overlay_enabled:
		overlay_metrics = _resolve_overlay_metrics(
			df_plot=df_plot,
			sweep_params=sweep_params,
			metrics_filter_cfg=visual_examples_metric_overlay_metrics_cfg,
			max_metrics=2,
		)
		if len(overlay_metrics) == 0:
			print("[WARN] Visual metric overlay enabled but no numeric metrics were resolved.")
		else:
			print(f"[INFO] Visual metric overlay enabled with metrics: {overlay_metrics}")
	auxiliary_layout = str(visual_examples_auxiliary_layout).strip().lower()
	if auxiliary_layout not in ["columns", "rows"]:
		print(f"[WARN] Unsupported visual_examples_auxiliary_layout='{visual_examples_auxiliary_layout}', falling back to 'columns'.")
		auxiliary_layout = "columns"
	auxiliary_position = str(visual_examples_auxiliary_position).strip().lower()
	if auxiliary_position not in ["start", "end"]:
		print(f"[WARN] Unsupported visual_examples_auxiliary_position='{visual_examples_auxiliary_position}', falling back to 'end'.")
		auxiliary_position = "end"
	if visual_examples_frames_as_rows and auxiliary_layout != "columns":
		print("[WARN] visual_examples_frames_as_rows needs auxiliary_layout=columns; using columns.")
		auxiliary_layout = "columns"
	zoom_criterion = str(visual_examples_zoom_criterion).strip().lower()
	if zoom_criterion not in ["lpips", "flow_magnitude"]:
		print(f"[WARN] Unsupported visual_examples_zoom_criterion='{visual_examples_zoom_criterion}', falling back to 'lpips'.")
		zoom_criterion = "lpips"
	nfp_target = str(visual_examples_next_frame_prediction_target).strip().lower()
	if nfp_target not in ["next", "previous"]:
		print(f"[WARN] Unsupported visual_examples_next_frame_prediction_target='{visual_examples_next_frame_prediction_target}', falling back to 'next'.")
		nfp_target = "next"
	auxiliary_end_keys: List[str] = []
	if visual_examples_auxiliary_end_keys_cfg is not None:
		auxiliary_end_keys = [str(key) for key in OmegaConf.to_container(visual_examples_auxiliary_end_keys_cfg.cfg, resolve=True)]
	frame_indices = _resolve_frame_indices(
		visual_examples_frame_idx=visual_examples_frame_idx,
		visual_examples_frame_idxs_cfg=visual_examples_frame_idxs_cfg,
	)
	print(f"[INFO] Visual examples: plotting frame indices {frame_indices}.")

	wandb_kwargs = wandb_kwargs_for_prefect_task(wandb_params_task)
	output_base_dir = os.path.join(
		wandb_kwargs["config"]["flow_name"] + "_" + wandb_kwargs["config"]["flow_run_name"],
		visual_examples_storage_subdir,
		datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + wandb_kwargs["name"],
	)

	first_storage_settings = None
	if "storage_path" in df_plot.columns:
		first_storage_settings = _extract_storage_settings_from_storage_path(df_plot.iloc[0].get("storage_path"))
	if first_storage_settings is None and "storage_settings" in df_plot.columns:
		storage_settings_dict = df_plot.iloc[0].get("storage_settings")
		if storage_settings_dict is not None:
			first_storage_settings = StorageSettings(**storage_settings_dict)

	if first_storage_settings is None:
		output_storage_settings = StorageSettings(
			storage_name="minio-caching-storage",
			bucket_or_base_path="prefect-results",
			is_remote=True,
		)
	else:
		output_storage_settings = first_storage_settings
	output_filesystem = load_file_system(output_storage_settings)

	with wandb.init(**wandb_kwargs):
		with tempfile.TemporaryDirectory() as temp_dir:
			local_output_dir = os.path.join(temp_dir, "visual_examples")
			os.makedirs(local_output_dir, exist_ok=True)
			local_figures_dir = os.path.join(local_output_dir, "figures")
			local_panels_dir = os.path.join(local_output_dir, "panels")
			os.makedirs(local_figures_dir, exist_ok=True)
			os.makedirs(local_panels_dir, exist_ok=True)

			payload_cache: Dict[str, Dict[str, Any]] = {}
			manifest_rows: List[Dict[str, Any]] = []
			auxiliary_items: List[Tuple[str, str]] = []
			if visual_examples_include_anchor_frame:
				auxiliary_items.append(("anchor_frame", "Previous frame" if nfp_target == "next" else "Next frame"))
			if visual_examples_include_pseudoinv:
				auxiliary_items.append(("pseudoinv", "Pseudoinv"))
			if visual_examples_include_reference:
				auxiliary_items.append(("reference", "Ground truth"))
			if visual_examples_include_flow or visual_examples_flow_enabled:
				auxiliary_items.append(("flow_rgb", "Optical flow"))
			if visual_examples_include_flow_pseudorec:
				auxiliary_items.append(("flow_rgb_pseudorec", "Flow (pseudorec)"))
			if visual_examples_include_confidence_mask:
				auxiliary_items.append(("confidence_mask", "Confidence mask"))
			if visual_examples_include_warp_residual:
				auxiliary_items.append(("warp_residual", "Warp residual"))
			if visual_examples_confidence_enabled:
				auxiliary_items.append(("error_map", "Error map"))
			if visual_examples_include_next_frame_confidence:
				auxiliary_items.append(("next_frame_confidence", "Confidence"))

			# Column of every auxiliary panel (columns layout): auxiliary_end_keys always follow the
			# reconstructions, the other panels sit at auxiliary_position.
			start_aux_keys = [key for key, _ in auxiliary_items if auxiliary_position == "start" and key not in auxiliary_end_keys]
			end_aux_keys = [key for key, _ in auxiliary_items if key not in start_aux_keys]
			end_aux_keys = [key for key in end_aux_keys if key not in auxiliary_end_keys] + [key for key in end_aux_keys if key in auxiliary_end_keys]
			aux_column_index = {key: idx for idx, key in enumerate(start_aux_keys)}
			aux_column_index.update({key: len(start_aux_keys) + max(1, len(col_values)) + idx for idx, key in enumerate(end_aux_keys)})

			_need_raft = (
				visual_examples_flow_enabled
				or visual_examples_include_flow
				or visual_examples_include_flow_pseudorec
				or visual_examples_include_confidence_mask
				or visual_examples_include_warp_residual
				or visual_examples_include_next_frame_confidence
				or (visual_examples_zoom_enabled and fixed_zoom_box is None and zoom_criterion == "flow_magnitude")
			)
			# Load RAFT once for the entire plotting session if flow or confidence is requested
			_raft_model = None
			if _need_raft:
				try:
					from src.regularizers.raft_based_regularizers import load_raft_model as _load_raft
					_raft_dev = "cuda" if torch.cuda.is_available() else "cpu"
					_raft_model = _load_raft(device=_raft_dev)
					print(f"[INFO] RAFT model loaded on {_raft_dev} for flow/confidence visualisation.")
				except Exception as _raft_load_err:
					print(f"[WARN] Could not load RAFT for flow/confidence visualisation: {_raft_load_err}")

			def _add_next_frame_panels(payload: Optional[Dict[str, Any]]) -> None:
				if payload is None or payload.get("reference") is None:
					return
				if visual_examples_include_anchor_frame and "anchor_frame" not in payload:
					payload["anchor_frame"] = _anchor_frame_video(payload["reference"], nfp_target)
				if visual_examples_include_next_frame_confidence and "next_frame_confidence" not in payload:
					flows = _reference_flows(payload, _raft_model)
					if flows is not None:
						payload["next_frame_confidence"] = _next_frame_confidence_video(
							flows, nfp_target, visual_examples_next_frame_confidence_tau_pix
						)

			# Spatial LPIPS: zoom selection (criterion "lpips") and the per-frame metric overlay of
			# figures whose rows are frames.
			_lpips_fn = None
			_zoom_needs_lpips = visual_examples_zoom_enabled and fixed_zoom_box is None and zoom_criterion == "lpips"
			if _zoom_needs_lpips or (visual_examples_frames_as_rows and visual_examples_metric_overlay_enabled):
				try:
					import lpips as _lpips_lib
					_lpips_fn = _lpips_lib.LPIPS(net="alex", spatial=True, verbose=False).eval()
					_lpips_fn = _lpips_fn.to("cuda" if torch.cuda.is_available() else "cpu")
				except Exception as _lpips_err:
					print(f"[WARN] Could not load LPIPS, using squared error for zoom selection and no LPIPS overlay: {_lpips_err}")

			# Zoom region per (figure, frame) when no fixed box is configured, by zoom_criterion:
			# "lpips" picks the window where the reconstructions differ most from the reference
			# perceptually (spatial LPIPS averaged over all recon cells of the figure),
			# "flow_magnitude" the fastest ground-truth motion into the frame (next-frame prediction:
			# the flow from the frame to its anchor, see visual_examples_next_frame_prediction_target).
			zoom_boxes: Dict[Tuple[int, int], Tuple[float, float, float, float]] = {}
			zoom_best_frames: Dict[int, int] = {}
			if visual_examples_zoom_enabled and fixed_zoom_box is None:
				for figure_index, (_, figure_df) in enumerate(grouped_figures):
					cells_df = figure_df
					if row_param is not None:
						cells_df = cells_df[cells_df[row_param].map(lambda value: _value_matches(value, row_values))]
					if col_param is not None:
						cells_df = cells_df[cells_df[col_param].map(lambda value: _value_matches(value, col_values))]
					error_maps = []
					for _, cell_row in cells_df.iterrows():
						payload = _load_visual_payload(cell_row.to_dict(), payload_cache, temp_dir)
						if payload is None or payload.get("reference") is None:
							continue
						if zoom_criterion == "flow_magnitude":
							flows = _reference_flows(payload, _raft_model)
							if flows is not None:
								flow_to_anchor, _ = _next_frame_flows_to_anchor(*flows, nfp_target)
								error_maps.append(_pad_to_video(flow_to_anchor.norm(dim=1), nfp_target, 0.0))
								break
							continue
						recon_video = (
							_brightness_match_video(payload["recon"], payload["reference"])
							if visual_examples_normalize_brightness
							else payload["recon"]
						)
						error_maps.append(_perceptual_error_map(recon_video, payload["reference"], _lpips_fn))
					if len(error_maps) == 0:
						print(f"[WARN] Zoom: no reconstruction with a reference in figure {figure_index}; no zoom inset.")
						continue
					num_frames = min(error_map.shape[0] for error_map in error_maps)
					mean_error_map = torch.stack([error_map[:num_frames] for error_map in error_maps]).mean(dim=0)
					frame_boxes = _select_zoom_boxes(
						mean_error_map,
						frame_indices,
						visual_examples_zoom_box_size,
						visual_examples_zoom_factor,
						border_margin=visual_examples_zoom_border_margin,
						shared_across_frames=visual_examples_zoom_shared_across_frames,
					)
					for frame_idx, (box, score) in frame_boxes.items():
						zoom_boxes[(figure_index, frame_idx)] = box
						print(f"[INFO] Zoom: figure {figure_index}, frame {frame_idx}: box (y0, x0, h, w)={tuple(round(v, 3) for v in box)}, error={score:.4f}")
					zoom_best_frames[figure_index] = max(frame_boxes, key=lambda frame_idx: frame_boxes[frame_idx][1])
					print(f"[INFO] Zoom: figure {figure_index}: largest error in frame {zoom_best_frames[figure_index]}.")

			# With frames_as_rows every figure is rendered once, with one row per frame index.
			outer_frame_indices = frame_indices[:1] if visual_examples_frames_as_rows else frame_indices
			for frame_log_step, frame_idx in enumerate(outer_frame_indices):
				for figure_index, (group_key, figure_df) in enumerate(grouped_figures):
					if not visual_examples_frames_as_rows and visual_examples_zoom_auto_frame and figure_index in zoom_best_frames and frame_idx != zoom_best_frames[figure_index]:
						continue
					zoom_box = zoom_boxes.get((figure_index, frame_idx), fixed_zoom_box) if visual_examples_zoom_enabled else None
					if visual_examples_frames_as_rows:
						row_specs = [(row_values[0], row_frame_idx) for row_frame_idx in frame_indices]
					else:
						row_specs = [(row_value, frame_idx) for row_value in row_values]
					if len(figure_group_params) > 0:
						if not isinstance(group_key, tuple):
							group_key = (group_key,)
						figure_group_dict = {
							param: value for param, value in zip(figure_group_params, group_key)
						}
					else:
						figure_group_dict = {}

					base_num_rows = max(1, len(row_specs))
					base_num_cols = max(1, len(col_values))
					num_auxiliary_items = len(auxiliary_items)
					if auxiliary_layout == "columns":
						num_rows = base_num_rows
						num_cols = base_num_cols + num_auxiliary_items
						recon_col_offset = len(start_aux_keys)
						auxiliary_col_index = None
					else:
						num_rows = max(base_num_rows, num_auxiliary_items)
						num_cols = base_num_cols + (1 if num_auxiliary_items > 0 else 0)
						recon_col_offset = 1 if auxiliary_position == "start" and num_auxiliary_items > 0 else 0
						auxiliary_col_index = 0 if auxiliary_position == "start" and num_auxiliary_items > 0 else num_cols - 1
					figsize = _build_auto_figsize(
						num_rows=num_rows,
						num_cols=num_cols,
						explicit_figsize=visual_examples_figsize_cfg.cfg if visual_examples_figsize_cfg is not None else None,
					)

					fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, dpi=visual_examples_dpi, squeeze=False)
					if visual_examples_frames_as_rows:
						figure_name_parts = ["frames_" + "-".join(str(row_frame_idx) for row_frame_idx in frame_indices)]
					else:
						figure_name_parts = [f"frame_{frame_idx:04d}"]
					for key, value in figure_group_dict.items():
						figure_name_parts.append(f"{_make_safe_name(key)}_{_make_safe_name(value)}")
					figure_name = "__".join(figure_name_parts) if len(figure_name_parts) > 0 else f"figure_{figure_index:03d}"

					for row_idx, (row_value, row_frame_idx) in enumerate(row_specs):
						row_df = figure_df
						if row_param is not None:
							row_df = row_df[row_df[row_param].map(lambda value: _value_matches(value, row_value))]
						row_zoom_box = zoom_boxes.get((figure_index, row_frame_idx), fixed_zoom_box) if visual_examples_zoom_enabled else None

						if visual_examples_frames_as_rows:
							row_label_text = f"frame {row_frame_idx}"
						else:
							row_label_text = _format_param_label(None, row_value, row_param_is_bool) if row_param is not None else None
						if row_label_text is not None:
							axes[row_idx][0].text(
								-0.22,
								0.5,
								row_label_text,
								transform=axes[row_idx][0].transAxes,
								ha="right",
								va="center",
								fontsize=10,
							)

						for col_idx, col_value in enumerate(col_values):
							cell_df = row_df
							if col_param is not None:
								cell_df = cell_df[cell_df[col_param].map(lambda value: _value_matches(value, col_value))]

							ax = axes[row_idx][col_idx + recon_col_offset]
							ax.axis("off")
							if cell_df.empty:
								ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=10)
								continue

							if len(cell_df) > 1:
								print(
									f"Multiple reconstructions found for visual cell (row={row_value}, col={col_value}); using the first one."
								)
							result_row = cell_df.iloc[0].to_dict()
							payload = _load_visual_payload(result_row, payload_cache, temp_dir)
							if payload is not None:
								_enrich_payload_in_place(
									payload,
									_raft_model,
									visual_examples_flow_source,
									compute_error=visual_examples_confidence_enabled,
									compute_confidence=visual_examples_include_confidence_mask,
									compute_pseudorec_flow=visual_examples_include_flow_pseudorec,
									compute_warp_residual=visual_examples_include_warp_residual,
								)
							if payload is None:
								ax.text(0.5, 0.5, "missing file", ha="center", va="center", fontsize=10)
								continue

							_recon_video_display = (
								_brightness_match_video(payload["recon"], payload["reference"])
								if visual_examples_normalize_brightness and payload.get("reference") is not None
								else payload["recon"]
							)
							recon_window = _reference_display_window(payload.get("reference"))
							raw_recon_frame = _select_frame(_recon_video_display, row_frame_idx, recon_window)
							recon_frame = raw_recon_frame
							if row_zoom_box is not None:
								recon_frame = _draw_zoom_inset(recon_frame, row_zoom_box, visual_examples_zoom_factor)
							display_frame = recon_frame
							if time_slice_mode != "none":
								time_slice = _select_time_slice(
									payload["recon"],
									mode=time_slice_mode,
									center_row=visual_examples_time_slice_center_row,
									center_col=visual_examples_time_slice_center_col,
									window=recon_window,
								)
								display_frame = _compose_frame_with_time_slice(recon_frame, time_slice, time_slice_mode, time_slice_ratio)
							ax.imshow(display_frame)
							if row_idx == 0:
								if col_param is not None:
									ax.set_title(_col_title(col_value), fontsize=10)
								else:
									ax.set_title("Reconstruction", fontsize=10)
							if visual_examples_metric_overlay_enabled:
								# Rows are frames: the clip-level metrics of the result row would repeat in
								# every row, so show the metrics of the displayed frame instead.
								if visual_examples_frames_as_rows and payload.get("reference") is not None:
									overlay_lines = _frame_metric_lines(_recon_video_display, payload["reference"], row_frame_idx, _lpips_fn, overlay_decimals)
								else:
									overlay_lines = _metric_overlay_lines_with_precision(result_row, overlay_metrics, overlay_decimals)
								_draw_metric_overlay(
									ax,
									overlay_lines,
									overlay_text_color,
									overlay_fontsize,
									anchor_x=overlay_anchor_x,
									anchor_y=overlay_anchor_y,
								)

							_col_slug = _sanitize_for_filename(_format_param_label(None, col_value, col_param_is_bool)) if col_value is not None else f"c{col_idx:02d}"
							if visual_examples_frames_as_rows:
								_row_slug = f"frame{row_frame_idx}"
							else:
								_row_slug = _sanitize_for_filename(row_value) if row_value is not None else f"r{row_idx:02d}"
							panel_rel_path = os.path.join("panels", figure_name, f"recon__{_col_slug}__row{_row_slug}.png")
							panel_abs_path = os.path.join(local_output_dir, panel_rel_path)
							os.makedirs(os.path.dirname(panel_abs_path), exist_ok=True)
							if time_slice_mode == "none":
								_save_panel_with_zoom_variants(panel_abs_path, raw_recon_frame, row_zoom_box, visual_examples_zoom_factor, visual_examples_zoom_crop_upscale)
							else:
								plt.imsave(panel_abs_path, display_frame)
							if time_slice_mode != "none":
								noslice_rel_path = os.path.join("panels_noslice", figure_name, f"recon__{_col_slug}__row{_row_slug}.png")
								noslice_abs_path = os.path.join(local_output_dir, noslice_rel_path)
								os.makedirs(os.path.dirname(noslice_abs_path), exist_ok=True)
								plt.imsave(noslice_abs_path, recon_frame)
							manifest_rows.append({
								"figure_name": figure_name,
								"panel_type": "recon",
								"row_index": row_idx,
								"col_index": col_idx + recon_col_offset,
								"row_value": row_value,
								"col_value": col_value,
								"relative_path": panel_rel_path,
								"source_output_dir_in_fs": payload["output_dir_in_fs"],
							})

						if auxiliary_layout == "columns":
							exemplar_df = row_df
							if col_param is not None and len(col_values) > 0:
								anchor_col_value = col_values[0]
								exemplar_df = exemplar_df[
									exemplar_df[col_param].map(lambda value: _value_matches(value, anchor_col_value))
								]
							if exemplar_df.empty:
								exemplar_df = row_df
							exemplar_row = exemplar_df.iloc[0].to_dict() if not exemplar_df.empty else None
							exemplar_payload = _load_visual_payload(exemplar_row, payload_cache, temp_dir) if exemplar_row is not None else None
							if exemplar_payload is not None:
								_enrich_payload_in_place(
									exemplar_payload,
									_raft_model,
									visual_examples_flow_source,
									compute_error=visual_examples_confidence_enabled,
									compute_confidence=visual_examples_include_confidence_mask,
									compute_pseudorec_flow=visual_examples_include_flow_pseudorec,
									compute_warp_residual=visual_examples_include_warp_residual,
								)
								_add_next_frame_panels(exemplar_payload)
							for aux_idx, (panel_key, panel_title) in enumerate(auxiliary_items):
								actual_col_idx = aux_column_index[panel_key]
								ax = axes[row_idx][actual_col_idx]
								ax.axis("off")
								panel_value = exemplar_payload.get(panel_key) if exemplar_payload is not None else None
								if visual_examples_normalize_brightness and panel_key == "pseudoinv" and exemplar_payload is not None and exemplar_payload.get("reference") is not None and panel_value is not None:
									panel_value = _brightness_match_video(panel_value, exemplar_payload["reference"])
								if panel_value is not None:
									panel_window = _reference_display_window(exemplar_payload.get("reference")) if panel_key in _INTENSITY_PANEL_KEYS else None
									raw_panel_frame = _select_frame(panel_value, row_frame_idx, panel_window)
									panel_frame = raw_panel_frame
									if row_zoom_box is not None:
										panel_frame = _draw_zoom_inset(panel_frame, row_zoom_box, visual_examples_zoom_factor)
									display_panel_frame = panel_frame
									if time_slice_mode != "none":
										time_slice = _select_time_slice(
											panel_value,
											mode=time_slice_mode,
											center_row=visual_examples_time_slice_center_row,
											center_col=visual_examples_time_slice_center_col,
											window=panel_window,
										)
										display_panel_frame = _compose_frame_with_time_slice(panel_frame, time_slice, time_slice_mode, time_slice_ratio)
									ax.imshow(display_panel_frame)
									if row_idx == 0:
										ax.set_title(panel_title, fontsize=10)
									source_output_dir_in_fs = exemplar_payload["output_dir_in_fs"] if exemplar_payload is not None else None
									_aux_row_slug = f"_frame{row_frame_idx}" if visual_examples_frames_as_rows else f"_r{row_idx:02d}"
									panel_rel_path = os.path.join("panels", figure_name, f"{panel_key}{_aux_row_slug}.png")
									panel_abs_path = os.path.join(local_output_dir, panel_rel_path)
									os.makedirs(os.path.dirname(panel_abs_path), exist_ok=True)
									if time_slice_mode == "none":
										_save_panel_with_zoom_variants(panel_abs_path, raw_panel_frame, row_zoom_box, visual_examples_zoom_factor, visual_examples_zoom_crop_upscale)
									else:
										plt.imsave(panel_abs_path, display_panel_frame)
									if time_slice_mode != "none":
										noslice_rel_path = os.path.join("panels_noslice", figure_name, f"{panel_key}_r{row_idx:02d}.png")
										noslice_abs_path = os.path.join(local_output_dir, noslice_rel_path)
										os.makedirs(os.path.dirname(noslice_abs_path), exist_ok=True)
										plt.imsave(noslice_abs_path, panel_frame)
									manifest_rows.append({
										"figure_name": figure_name,
										"panel_type": panel_key,
										"row_index": row_idx,
										"col_index": actual_col_idx,
										"row_value": row_value,
										"col_value": None,
										"relative_path": panel_rel_path,
										"source_output_dir_in_fs": source_output_dir_in_fs,
									})
								else:
									ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=10)

					if auxiliary_layout == "rows" and num_auxiliary_items > 0:
						shared_panel_row_value = row_values[0] if len(row_values) > 0 else None
						for aux_idx, (panel_key, panel_title) in enumerate(auxiliary_items):
							if aux_idx >= num_rows:
								break
							ax = axes[aux_idx][cast(int, auxiliary_col_index)]
							ax.axis("off")
							panel_row_value = shared_panel_row_value
							panel_df = figure_df
							if row_param is not None and panel_row_value is not None:
								panel_df = panel_df[panel_df[row_param].map(lambda value: _value_matches(value, panel_row_value))]
							exemplar_row = panel_df.iloc[0].to_dict() if not panel_df.empty else None
							exemplar_payload = _load_visual_payload(exemplar_row, payload_cache, temp_dir) if exemplar_row is not None else None
							if exemplar_payload is not None:
								_enrich_payload_in_place(
									exemplar_payload,
									_raft_model,
									visual_examples_flow_source,
									compute_error=visual_examples_confidence_enabled,
									compute_confidence=visual_examples_include_confidence_mask,
									compute_pseudorec_flow=visual_examples_include_flow_pseudorec,
									compute_warp_residual=visual_examples_include_warp_residual,
								)
							panel_value = exemplar_payload.get(panel_key) if exemplar_payload is not None else None
							if visual_examples_normalize_brightness and panel_key == "pseudoinv" and exemplar_payload is not None and exemplar_payload.get("reference") is not None and panel_value is not None:
								panel_value = _brightness_match_video(panel_value, exemplar_payload["reference"])
							if panel_value is not None:
								panel_window = _reference_display_window(exemplar_payload.get("reference")) if panel_key in _INTENSITY_PANEL_KEYS else None
								panel_frame = _select_frame(panel_value, frame_idx, panel_window)
								if zoom_box is not None:
									panel_frame = _draw_zoom_inset(panel_frame, zoom_box, visual_examples_zoom_factor)
								display_panel_frame = panel_frame
								if time_slice_mode != "none":
									time_slice = _select_time_slice(
										panel_value,
										mode=time_slice_mode,
										center_row=visual_examples_time_slice_center_row,
										center_col=visual_examples_time_slice_center_col,
										window=panel_window,
									)
									display_panel_frame = _compose_frame_with_time_slice(panel_frame, time_slice, time_slice_mode, time_slice_ratio)
								ax.imshow(display_panel_frame)
								ax.set_title(panel_title, fontsize=10)
								source_output_dir_in_fs = exemplar_payload["output_dir_in_fs"] if exemplar_payload is not None else None
								_aux_row_row_slug = _sanitize_for_filename(panel_row_value) if panel_row_value is not None else f"r{aux_idx:02d}"
								_aux_row_title_slug = _sanitize_for_filename(panel_title)
								panel_rel_path = os.path.join("panels", figure_name, f"{panel_key}__{_aux_row_title_slug}__row{_aux_row_row_slug}.png")
								panel_abs_path = os.path.join(local_output_dir, panel_rel_path)
								os.makedirs(os.path.dirname(panel_abs_path), exist_ok=True)
								plt.imsave(panel_abs_path, display_panel_frame)
								if time_slice_mode != "none":
									noslice_rel_path = os.path.join("panels_noslice", figure_name, f"{panel_key}__{_aux_row_title_slug}__row{_aux_row_row_slug}.png")
									noslice_abs_path = os.path.join(local_output_dir, noslice_rel_path)
									os.makedirs(os.path.dirname(noslice_abs_path), exist_ok=True)
									plt.imsave(noslice_abs_path, panel_frame)
								manifest_rows.append({
									"figure_name": figure_name,
									"panel_type": panel_key,
									"row_index": aux_idx,
									"col_index": cast(int, auxiliary_col_index),
									"row_value": panel_row_value,
									"col_value": None,
									"relative_path": panel_rel_path,
									"source_output_dir_in_fs": source_output_dir_in_fs,
								})
							else:
								ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=10)

					subtitle_parts = [f"frames={list(frame_indices)}" if visual_examples_frames_as_rows else f"frame={frame_idx}"]
					subtitle_parts += [_format_caption_value(value) for _, value in figure_group_dict.items()]
					fig.suptitle(" | ".join(subtitle_parts), fontsize=12)
					axis_caption_parts: List[str] = []
					if row_param is not None:
						axis_caption_parts.append(f"rows: {row_param}")
					if col_param is not None:
						axis_caption_parts.append(f"cols: {col_param}")
					if len(axis_caption_parts) > 0:
						fig.text(0.5, 0.01, " | ".join(axis_caption_parts), ha="center", va="bottom", fontsize=9)
					plt.tight_layout()
					plt.subplots_adjust(left=0.14, top=0.88, bottom=0.08)

					for fmt in formats:
						figure_rel_path = os.path.join("figures", f"{figure_name}.{fmt}")
						figure_abs_path = os.path.join(local_output_dir, figure_rel_path)
						fig.savefig(figure_abs_path, dpi=visual_examples_dpi, bbox_inches="tight", format=fmt)
						manifest_rows.append({
							"figure_name": figure_name,
							"panel_type": "combined_figure",
							"row_index": None,
							"col_index": None,
							"row_value": None,
							"col_value": None,
							"relative_path": figure_rel_path,
							"source_output_dir_in_fs": None,
						})

					wandb.log(
						{
							f"visual_examples/{figure_name}": wandb.Image(fig),
							"visual_examples/frame_idx": frame_idx,
						},
						step=frame_log_step,
					)
					plt.close(fig)

			if visual_examples_video_enabled:
				videos_dir = os.path.join(local_output_dir, "videos")
				os.makedirs(videos_dir, exist_ok=True)
				for figure_index, (group_key, figure_df) in enumerate(grouped_figures):
					if len(figure_group_params) > 0:
						group_key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
						figure_group_dict = {
							param: value for param, value in zip(figure_group_params, group_key_tuple)
						}
					else:
						figure_group_dict = {}
					figure_name_parts = ["comparative"]
					for key, value in figure_group_dict.items():
						figure_name_parts.append(f"{_make_safe_name(key)}_{_make_safe_name(value)}")
					video_figure_name = "__".join(figure_name_parts)
					group_videos_rel_dir = os.path.join("videos", video_figure_name)

					# Two synchronized grids: the full one with every auxiliary panel (flow, confidence, ...)
					# and a videos-only one (recons + ground truth + pseudoinv) that gives the actual
					# clips the whole frame, which reads much better for judging temporal artifacts.
					videos_only_items = [item for item in auxiliary_items if item[0] in ("pseudoinv", "reference")]
					grid_variants: List[Tuple[str, str, List[Tuple[str, str]]]] = [
						("", "synchronized_grid", auxiliary_items),
					]
					if len(videos_only_items) < len(auxiliary_items):
						grid_variants.append(("__videos_only", "synchronized_grid_videos_only", videos_only_items))

					cell_videos: List[Dict[str, Any]] = []
					for variant_suffix, variant_file_stem, variant_aux_items in grid_variants:
						variant_name = video_figure_name + variant_suffix
						video_result = _build_comparative_grid_video(
							figure_df=figure_df,
							figure_name=variant_name,
							row_values=row_values,
							col_values=col_values,
							row_param=row_param,
							col_param=col_param,
							row_param_is_bool=row_param_is_bool,
							col_param_is_bool=col_param_is_bool,
							auxiliary_items=variant_aux_items,
							auxiliary_layout=auxiliary_layout,
							auxiliary_position=auxiliary_position,
							payload_cache=payload_cache,
							temp_dir=temp_dir,
							normalize_brightness=visual_examples_normalize_brightness,
							metric_overlay_enabled=visual_examples_metric_overlay_enabled,
							overlay_metrics=overlay_metrics,
							overlay_text_color=overlay_text_color,
							overlay_fontsize=overlay_fontsize,
							overlay_decimals=overlay_decimals,
							# No time-slice strip in the video, so the overlay always sits bottom-left.
							overlay_anchor_x=0.02,
							overlay_anchor_y=0.02,
							# The grid is meant to be played back inline in wandb; a lower dpi than the
							# static figures keeps the encoded frame within what browsers decode.
							dpi=int(visual_examples_video_dpi) if visual_examples_video_dpi is not None else visual_examples_dpi,
							explicit_figsize=visual_examples_figsize_cfg.cfg if visual_examples_figsize_cfg is not None else None,
							max_frames=visual_examples_video_max_frames,
							frame_stride=visual_examples_video_frame_stride,
						)
						if video_result is None:
							print(f"[WARN] Comparative video '{variant_name}' skipped: no synchronized frames available.")
							continue
						frames_float, num_frames_rendered, variant_cell_videos = video_result
						if variant_suffix == "":
							cell_videos = variant_cell_videos  # full grid covers every panel; export cells once
						os.makedirs(os.path.join(local_output_dir, group_videos_rel_dir), exist_ok=True)

						# Synchronized grid: played back inline in wandb.
						video_rel_path = os.path.join(group_videos_rel_dir, f"{variant_file_stem}.mp4")
						video_abs_path = os.path.join(local_output_dir, video_rel_path)
						if not _save_tensor_video_to_file(frames_float, video_abs_path, fps=int(visual_examples_video_fps)):
							continue
						print(f"[INFO] Saved comparative video: {video_abs_path} ({num_frames_rendered} frames)")
						wandb.log({
							f"visual_examples_video/{variant_name}": wandb.Video(
								video_abs_path, format="mp4"
							),
						})
						manifest_rows.append({
							"figure_name": variant_name,
							"panel_type": "comparative_video",
							"row_index": None,
							"col_index": None,
							"row_value": None,
							"col_value": None,
							"relative_path": video_rel_path,
							"source_output_dir_in_fs": None,
						})

					# One standalone clip per grid cell (each method's recon, ground truth, pseudoinv, ...)
					# at native resolution, so they can be downloaded individually from the artifact.
					for cell_video in cell_videos:
						cell_rel_path = os.path.join(group_videos_rel_dir, f"{cell_video['name']}.mp4")
						cell_abs_path = os.path.join(local_output_dir, cell_rel_path)
						if not _save_tensor_video_to_file(cell_video["frames"], cell_abs_path, fps=int(visual_examples_video_fps)):
							continue
						manifest_rows.append({
							"figure_name": video_figure_name,
							"panel_type": f"video_{cell_video['panel_type']}",
							"row_index": cell_video["row_index"],
							"col_index": cell_video["col_index"],
							"row_value": cell_video["row_value"],
							"col_value": cell_video["col_value"],
							"relative_path": cell_rel_path,
							"source_output_dir_in_fs": cell_video["source_output_dir_in_fs"],
						})
					print(f"[INFO] Saved {len(cell_videos)} per-cell videos under {group_videos_rel_dir}")

			manifest_df = pd.DataFrame(manifest_rows)
			manifest_df.to_csv(os.path.join(local_output_dir, "manifest.csv"), index=False)
			manifest_df.to_json(os.path.join(local_output_dir, "manifest.json"), orient="records", indent=2)

			# Save optical flow and confidence mask videos for each loaded payload into panels/videos/
			_video_keys_to_save: List[Tuple[str, str]] = []
			if visual_examples_include_flow or visual_examples_flow_enabled:
				_video_keys_to_save.append(("flow_rgb", "optical_flow"))
			if visual_examples_include_flow_pseudorec:
				_video_keys_to_save.append(("flow_rgb_pseudorec", "optical_flow_pseudorec"))
			if visual_examples_include_warp_residual:
				_video_keys_to_save.append(("warp_residual", "warp_residual"))
			if visual_examples_include_confidence_mask:
				_video_keys_to_save.append(("confidence_mask", "confidence_mask"))
			if _video_keys_to_save and payload_cache:
				videos_dir = os.path.join(local_output_dir, "panels", "videos")
				os.makedirs(videos_dir, exist_ok=True)
				for _payload_path, _payload_data in payload_cache.items():
					_safe_id = hashlib.md5(_payload_path.encode("utf-8")).hexdigest()[:8]
					for _vid_key, _vid_label in _video_keys_to_save:
						_vid_tensor = _payload_data.get(_vid_key)
						if _vid_tensor is not None:
							_vid_path = os.path.join(videos_dir, f"{_vid_label}_{_safe_id}.mp4")
							if _save_tensor_video_to_file(_vid_tensor, _vid_path):
								print(f"[INFO] Saved {_vid_label} video: {_vid_path}")

			output_filesystem.put_directory(local_output_dir, output_base_dir)

			# Upload the synchronized grids and per-cell clips as individual artifact files (not zipped),
			# so each video can be downloaded on its own from the wandb artifact browser.
			videos_root = os.path.join(local_output_dir, "videos")
			if visual_examples_video_enabled and os.path.isdir(videos_root):
				videos_artifact = wandb.Artifact(name="visual_videos", type="videos")
				videos_artifact.add_dir(videos_root)
				wandb.log_artifact(videos_artifact)

			# Zip individual panels (including videos/) and upload as a wandb artifact for easy download
			panels_dir = os.path.join(local_output_dir, "panels")
			if os.path.isdir(panels_dir):
				panels_zip_path = os.path.join(local_output_dir, "panels.zip")
				import zipfile
				with zipfile.ZipFile(panels_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
					for root, _, files in os.walk(panels_dir):
						for fname in files:
							full_path = os.path.join(root, fname)
							arcname = os.path.relpath(full_path, panels_dir)
							zf.write(full_path, arcname)
				artifact = wandb.Artifact(name="visual_panels", type="panels")
				artifact.add_file(panels_zip_path, name="panels.zip")
				wandb.log_artifact(artifact)

				# Also zip the no-slice panels (plain frames) if time slice was active
				if time_slice_mode != "none":
					noslice_dir = os.path.join(local_output_dir, "panels_noslice")
					if os.path.isdir(noslice_dir):
						noslice_zip_path = os.path.join(local_output_dir, "panels_noslice.zip")
						with zipfile.ZipFile(noslice_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf_ns:
							for root, _, files in os.walk(noslice_dir):
								for fname in files:
									full_path = os.path.join(root, fname)
									arcname = os.path.relpath(full_path, noslice_dir)
									zf_ns.write(full_path, arcname)
						artifact_ns = wandb.Artifact(name="visual_panels_noslice", type="panels")
						artifact_ns.add_file(noslice_zip_path, name="panels_noslice.zip")
						wandb.log_artifact(artifact_ns)

			wandb.log({
				"visual_examples_manifest": wandb.Table(dataframe=manifest_df),
				"visual_examples_output_dir_in_fs": output_base_dir,
			})
