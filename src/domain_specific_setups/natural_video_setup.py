import torch
import math
from types import SimpleNamespace
from typing import Callable, Optional, Tuple
from src.domain_specific_setups.base_domain_specific_setup import BaseDomainSpecificSetup

import tqdm
import wandb

from skimage.metrics import peak_signal_noise_ratio
import numpy as np

try:
    import lpips  # type: ignore[import-not-found]
except ImportError:
    lpips = None

from src.utils.metrics import nmse, ssim_slicewise_mag, tssim_temporal_mag
from src.dataset.vimeo_dataset import VimeoDataset
from src.representations.base_representation import BaseRepresentation
from src.warp_methods.base_warp_method import BaseWarpMethod

from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode
from torchvision.models.optical_flow._utils import make_coords_grid, grid_sample

from src.domain_specific_setups.natural_video_forward_models import get_forward_model, LinearForwardModel
from src.dataset.davis_dataset import DAVIS2017Dataset
from src.dataset.davis_svi_dataset import DAVISSVIDataset
from src.dataset.reds_dataset import RedsDataset

def dataset_resolver(name: str, **kwargs):
    if name == "vimeo":
        return VimeoDataset(**kwargs)
    elif name == "davis":
        # DAVIS 2017 raw frames cut into 16-frame 256x256 clips
        return DAVIS2017Dataset(**kwargs)
    elif name == "davis_svi":
        # pre-cut DAVIS-SVI .npy clip set
        return DAVISSVIDataset(**kwargs)
    elif name == "reds":
        return RedsDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset name: {name}")

class NaturalVideoSetup(BaseDomainSpecificSetup):

    def __init__(self,
            base_path,
            target_as_pseudoinv_in_reg : bool,
            sample_index : int,
            video_length_in_s : float,
            path_resolver,
            set_forward_model : bool,
            device : str,
            forward_model_cfg,
            dataset_cfg,
            additive_gaussian_noise_std : Optional[float] = None,
            scale_measurement : bool = False,
            per_frame_metrics : bool = False,
            flow_reference_forward_model_overrides : Optional[dict] = None,
            flow_noise_std : Optional[float] = None,
            flow_measurement_noise_std : Optional[float] = None,
            next_frame_prediction_target : Optional[str] = None,
            # soft fb confidence exp(-|fb error|^2 / (2 tau^2)); reliable if above threshold
            next_frame_prediction_confidence_tau_pix : float = 1.0,
            next_frame_prediction_confidence_threshold : float = 0.5,
        ):
        super().__init__()
        self.per_frame_metrics = per_frame_metrics
        # next-frame prediction eval: must match PropagationBaseline's single_step_target; None disables nfp_* metrics
        if next_frame_prediction_target not in (None, "previous", "next"):
            raise ValueError(f"next_frame_prediction_target must be None, 'previous' or 'next', got {next_frame_prediction_target!r}")
        self.next_frame_prediction_target = next_frame_prediction_target
        self.next_frame_prediction_confidence_tau_pix = float(next_frame_prediction_confidence_tau_pix)
        self.next_frame_prediction_confidence_threshold = float(next_frame_prediction_confidence_threshold)

        self.forward_model = get_forward_model(**forward_model_cfg) if forward_model_cfg is not None else None        

        self.base_path = path_resolver(base_path)
        dataset = dataset_resolver(**dataset_cfg, base_path=self.base_path)
        if set_forward_model:
            print("Setting forward model in dataset.")
            dataset.forward_model = self.forward_model

        if len(dataset) == 0:
            raise ValueError("Vimeo dataset is empty; check path and list file.")

        print(f"Dataset loaded with {len(dataset)} videos. Sample index: {sample_index}")

        self._measurement, self._target = dataset[sample_index] 
        self._measurement = self._measurement.movedim(1,-1) # (T, C, H, W) -> (T, H, W, C)

        self.additive_gaussian_noise_std = additive_gaussian_noise_std
        if additive_gaussian_noise_std is not None:
            assert self.forward_model is not None, "additive_gaussian_noise_std needs a forward model (noise is restricted to its acquired entries)."
            self._measurement = self.forward_model.add_noise(self._measurement, additive_gaussian_noise_std, cdim=-1)

        if scale_measurement:
            assert self.forward_model is not None
            self._scale_factor = self.forward_model.get_scale_factor(self._measurement, cdim=-1)
            self._measurement = self._measurement / self._scale_factor
        else:
            self._scale_factor = torch.ones( (1,), device=device)
        self.scale_measurement = scale_measurement

        self._target = self._target.movedim(1,-1) # (T, C, H, W) -> (T, H, W, C)

        T, H, W,_ = self._target.shape  # (T, C, H, W)
        self.device = device
        self.eval_cropping = SimpleNamespace(lx=0, rx=None, ty=0, by=None)

        from src.dataset.cinemri_dataset import get_meshes_and_corrections_for_matrix_size, get_spatial_mesh_for_matrix_size
        self.spatial_mesh = get_spatial_mesh_for_matrix_size((H, W), device=device)
        self.time_mesh = torch.linspace(0, video_length_in_s, T).to(device)
        self.target_as_pseudoinv_in_reg = target_as_pseudoinv_in_reg
        self._lpips_fn = None
        self._gt_flow_fwd = None
        self._gt_flow_bwd = None

        # Optional separate degradation (forward_model_cfg + overrides) that the method's flow estimates are taken from.
        self._flow_reference_video = None
        self._flow_forward_model = None
        self._flow_degradation_metrics = None
        # Reporting only (noise is injected by the consumer); must be swept in lockstep with the consumer's flow_noise_std.
        self.flow_noise_std = flow_noise_std
        # Measurement noise for the flow-reference video only; None = same as additive_gaussian_noise_std.
        self.flow_measurement_noise_std = flow_measurement_noise_std
        if flow_reference_forward_model_overrides or flow_measurement_noise_std:
            assert forward_model_cfg is not None, (
                "flow_reference_forward_model_overrides/flow_measurement_noise_std need a "
                "forward_model_cfg to build the flow-reference operator from."
            )
            self._flow_forward_model = get_forward_model(
                **{**forward_model_cfg, **(flow_reference_forward_model_overrides or {})}
            )
            flow_measurement = self._flow_forward_model(self._target, cdim=-1)
            # independent noise draw
            flow_measurement_noise = (
                flow_measurement_noise_std if flow_measurement_noise_std is not None else additive_gaussian_noise_std
            )
            if flow_measurement_noise:
                flow_measurement = self._flow_forward_model.add_noise(flow_measurement, flow_measurement_noise, cdim=-1)
            self._flow_reference_video = self._flow_forward_model.pseudoinv(
                flow_measurement, cdim=-1
            ) / self._scale_factor.to(self._target.device)


        from src.regularizers.raft_based_regularizers import load_raft_model, estimate_flow_for_video
        from src.regularizers.model_initializers.video_frame_init import process_video_for_raft

        with torch.no_grad():
            raft_model = load_raft_model(device=device)
            video = process_video_for_raft(self._target).to(device).movedim(-1,1)  # shape (T, 3, H, W)
            self._gt_flow_fwd = estimate_flow_for_video(raft_model, video, forward_flow=True)
            self._gt_flow_bwd = estimate_flow_for_video(raft_model, video, forward_flow=False)

    def cache_datasets_on_gpu(self, device):
        print(f"Preloading element to device {device}")
        self._measurement = self._measurement.to(device)
        self._target = self._target.to(device)
        self._scale_factor = self._scale_factor.to(device)
        if self._flow_reference_video is not None:
            self._flow_reference_video = self._flow_reference_video.to(device)
        if self._gt_flow_fwd is not None:
            self._gt_flow_fwd = self._gt_flow_fwd.to(device)
        if self._gt_flow_bwd is not None:
            self._gt_flow_bwd = self._gt_flow_bwd.to(device)

    def get_train_dataset(self, scaled : bool = True) -> torch.utils.data.Dataset:
        ret = self.forward_model.pseudoinv(self._measurement, cdim=-1) if self.forward_model is not None else self._measurement
        if not scaled:
            ret = ret * self._scale_factor.view(1, *[1]*(ret.ndim-1))
        return ret

    def get_flow_reference_video(self, scaled : bool = True) -> Optional[torch.Tensor]:
        """Video the method's flow estimates should be derived from, or None to use the pseudoinverse.
        Same scaling convention as get_train_dataset()."""
        if self._flow_reference_video is not None:
            ret = self._flow_reference_video
        elif self.target_as_pseudoinv_in_reg:
            ret = self._target / self._scale_factor
        else:
            return None
        if not scaled:
            ret = ret * self._scale_factor.view(1, *[1]*(ret.ndim-1))
        return ret

    def get_flow_source_video(self, scaled : bool = True) -> torch.Tensor:
        """Flow-reference video if configured, else the pseudoinverse."""
        ret = self.get_flow_reference_video(scaled=scaled)
        return ret if ret is not None else self.get_train_dataset(scaled=scaled)

    @staticmethod
    def _flow_error_metrics(flow_est: torch.Tensor, flow_gt: torch.Tensor) -> dict:
        """Optical-flow error metrics (EPE, outlier rates, angular error) between two (N, 2, H, W) pixel fields."""
        diff = flow_est - flow_gt
        epe = torch.linalg.norm(diff, dim=1)  # (N, H, W), pixels
        gt_magnitude = torch.linalg.norm(flow_gt, dim=1)  # (N, H, W), pixels

        est_h = torch.cat([flow_est, torch.ones_like(flow_est[:, :1])], dim=1)
        gt_h = torch.cat([flow_gt, torch.ones_like(flow_gt[:, :1])], dim=1)
        cos = ((est_h * gt_h).sum(dim=1) / (est_h.norm(dim=1) * gt_h.norm(dim=1)).clamp_min(1e-8)).clamp(-1.0, 1.0)

        return {
            "flow_epe": epe.mean().item(),
            "flow_epe_outlier_1px": (epe > 1.0).float().mean().item(),
            "flow_epe_outlier_3px": (epe > 3.0).float().mean().item(),
            # EPE normalized by GT motion magnitude
            "flow_epe_relative": (epe.sum() / gt_magnitude.sum().clamp_min(1e-8)).item(),
            "flow_angular_error_deg": torch.rad2deg(torch.arccos(cos)).mean().item(),
            "flow_gt_magnitude_px": gt_magnitude.mean().item(),
        }

    @torch.no_grad()
    def get_flow_degradation_metrics(self) -> dict:
        """RAFT flow on the flow-source video vs. on the clean target, averaged over fwd/bwd.
        Cached after the first call; includes flow_noise_std if set."""
        if self._flow_degradation_metrics is not None:
            return self._flow_degradation_metrics
        if self._gt_flow_fwd is None or self._gt_flow_bwd is None:
            return {}

        from src.regularizers.raft_based_regularizers import load_raft_model, estimate_flow_for_video
        from src.regularizers.model_initializers.video_frame_init import process_video_for_raft

        flow_source = self.get_flow_source_video(scaled=False)
        device = flow_source.device
        raft_model = load_raft_model(device=str(device))
        video = process_video_for_raft(flow_source).to(device).movedim(-1, 1)  # (T, 3, H, W)
        source_flow_fwd = estimate_flow_for_video(raft_model, video, forward_flow=True)
        source_flow_bwd = estimate_flow_for_video(raft_model, video, forward_flow=False)

        if self.flow_noise_std is not None and self.flow_noise_std > 0:
            source_flow_fwd = source_flow_fwd + torch.randn_like(source_flow_fwd) * self.flow_noise_std
            source_flow_bwd = source_flow_bwd + torch.randn_like(source_flow_bwd) * self.flow_noise_std

        per_direction = [
            self._flow_error_metrics(source_flow_fwd, self._gt_flow_fwd.to(device)),
            self._flow_error_metrics(source_flow_bwd, self._gt_flow_bwd.to(device)),
        ]
        metrics = {key: float(np.mean([d[key] for d in per_direction])) for key in per_direction[0]}

        # NMSE rather than PSNR: a second *psnr* key would make tune_metric="psnr" ambiguous
        metrics["flow_source_nmse"] = float(
            nmse(torch.linalg.norm(flow_source, dim=-1), torch.linalg.norm(self._target, dim=-1))
        )

        # oracle case (target used directly) counts as fully sampled
        if self._flow_forward_model is not None:
            flow_forward_model = self._flow_forward_model
        elif self.target_as_pseudoinv_in_reg:
            flow_forward_model = None
        else:
            flow_forward_model = self.forward_model
        undersampling_factor = 1.0 if flow_forward_model is None else getattr(flow_forward_model, "undersampling_factor", None)
        if undersampling_factor is not None:
            metrics["flow_undersampling_factor"] = float(undersampling_factor)
        metrics["flow_noise_std"] = float(self.flow_noise_std or 0.0)
        metrics["flow_measurement_noise_std"] = float(
            self.flow_measurement_noise_std if self.flow_measurement_noise_std is not None
            else (self.additive_gaussian_noise_std or 0.0)
        )

        self._flow_degradation_metrics = metrics
        return metrics

    def get_measurement(self, scaled : bool = False) -> torch.Tensor:
        if not scaled:
            return self._measurement * self._scale_factor.view(1, *[1]*(self._measurement.ndim-1))
        else:
            return self._measurement

    def get_forward_model(self) -> LinearForwardModel:
        return self.forward_model

    def get_reference_video(self, scaled: bool = False) -> torch.Tensor:
        """scaled=True returns the target in the measurement-scaled domain of get_train_dataset()."""
        return self._target / self._scale_factor if scaled else self._target

    def get_meshes_dict(self):
        return {
            "spatial_mesh": self.spatial_mesh,
            "kspace_mesh" : None,
            "time_mesh": self.time_mesh,
        }


    def get_fps(self):
        return self.time_mesh.shape[0] / (self.time_mesh[-1] - self.time_mesh[0]).item()

    def get_spatial_mesh(self):
        return self.spatial_mesh

    def get_time_mesh(self):
        return self.time_mesh

    def get_domain_specific_info(self):
        return {}

    def get_ground_truth_flow_pair(self):
        return self._gt_flow_fwd, self._gt_flow_bwd

    def unscale_video(self, video: torch.Tensor) -> torch.Tensor:
        """Undo the measurement scaling: multiply by _scale_factor to restore
        the video to the original amplitude domain."""
        sf = self._scale_factor.to(video.device)
        return video * sf.view(*([1] * video.ndim))


    def get_dataset_params(self):
        return None # there are no parameters..

    def get_video_processor_for_visualization(
        self,
    ) -> Callable[[torch.Tensor], np.ndarray]:
        def _to_unit_interval(video: torch.Tensor) -> torch.Tensor:
            video = torch.nan_to_num(video)
            v_min = float(video.min().item())
            v_max = float(video.max().item())

            # Common natural-video preprocessing mode.
            if v_min >= -1.1 and v_max <= 1.1:
                return ((video + 1.0) / 2.0).clamp(0.0, 1.0)

            # Already displayable.
            if v_min >= -0.05 and v_max <= 1.05:
                return video.clamp(0.0, 1.0)

            # Robust fallback for arbitrary ranges.
            lo = torch.quantile(video.flatten(), 0.01)
            hi = torch.quantile(video.flatten(), 0.99)
            return ((video - lo) / (hi - lo + 1e-8)).clamp(0.0, 1.0)

        return lambda x: (_to_unit_interval(x) * 255.0).to(torch.uint8).permute(0, 3, 1, 2).cpu().numpy()


    @staticmethod
    def _lai_occlusion_mask(flow_fwd: torch.Tensor, flow_bwd_warped: torch.Tensor) -> torch.Tensor:
        """Binary occlusion mask of Lai et al. 2018 (ECCV, Eq. 5) / Ruder et al. 2016.
        Based on detect_occlusion in https://github.com/phoenix104104/fast_blind_video_consistency

            flow_fwd:        (N,2,H,W) flow t->t+1, in pixels
            flow_bwd_warped: (N,2,H,W) flow t+1->t sampled at x + flow_fwd(x)

        Returns ``occluded`` as (N,H,W) bool.
        """
        def sq_mag(f: torch.Tensor) -> torch.Tensor:
            return f[:, 0].square() + f[:, 1].square()

        # (1) forward-backward consistency ("disocclusion") check
        fb_sum_mag = sq_mag(flow_fwd + flow_bwd_warped)
        mask_fb = fb_sum_mag > 0.01 * (sq_mag(flow_bwd_warped) + sq_mag(flow_fwd)) + 0.5

        # (2) motion-boundary check on the t->t+1 flow (forward differences)
        grad_du = torch.zeros_like(flow_fwd)  # d/dx of (u, v)
        grad_dv = torch.zeros_like(flow_fwd)  # d/dy of (u, v)
        grad_du[..., :, :-1] = flow_fwd[..., :, :-1] - flow_fwd[..., :, 1:]
        grad_dv[..., :-1, :] = flow_fwd[..., :-1, :] - flow_fwd[..., 1:, :]
        grad_mag = sq_mag(grad_du) + sq_mag(grad_dv)  # |grad u|^2 + |grad v|^2
        mask_mb = grad_mag > 0.01 * sq_mag(flow_fwd) + 0.002

        return mask_fb | mask_mb

    def _compute_warping_error(self, video: torch.Tensor) -> Optional[dict]:
        """Flow warping error between consecutive frames, using RAFT flow of the clean target.

        warping_error_l1/l2: unmasked mean |V_t - W(V_{t+1})|.
        warping_error_occ_l1/l2: occlusion-masked warping error of Lai et al. 2018, Eq. (5)-(6).
        """
        if self._gt_flow_fwd is None or self._gt_flow_bwd is None:
            return None
        if video.shape[0] < 2:
            return None

        cdim = video.shape[-1]
        if cdim == 2:
            video_for_warp = torch.linalg.norm(video, dim=-1, keepdim=True)
        elif cdim in (1, 3):
            video_for_warp = video
        elif cdim > 3:
            video_for_warp = video[..., :3]
        else:
            return None

        video_cf = video_for_warp.permute(0, 3, 1, 2).float()
        flow_fwd = self._gt_flow_fwd.to(device=video_cf.device, dtype=video_cf.dtype)
        flow_bwd = self._gt_flow_bwd.to(device=video_cf.device, dtype=video_cf.dtype)

        if flow_fwd.shape[0] != video_cf.shape[0] - 1 or flow_bwd.shape[0] != flow_fwd.shape[0]:
            return None

        _, _, h, w = video_cf.shape
        base_grid = make_coords_grid(batch_size=flow_fwd.shape[0], h=h, w=w, device=str(video_cf.device)).movedim(1, -1)
        grid_fwd = base_grid + flow_fwd.movedim(1, -1)
        warped_next = grid_sample(video_cf[1:], grid_fwd, align_corners=True, mode="bilinear")
        diff = (warped_next - video_cf[:-1])  # (T-1,C,H,W), frame-t coordinates

        result = {
            "warping_error_l1": diff.abs().mean().item(),
            "warping_error_l2": diff.square().mean().item(),
        }

        # occlusion-masked variant (Lai et al. 2018), mask from GT flow only
        flow_bwd_warped = grid_sample(flow_bwd, grid_fwd, align_corners=True, mode="bilinear")
        occluded = self._lai_occlusion_mask(flow_fwd, flow_bwd_warped)  # (T-1,H,W)
        noc = (~occluded).to(diff.dtype).unsqueeze(1)  # (T-1,1,H,W)

        # Per-pair masked mean over channel-pixels (Eq. 5), then mean over pairs (Eq. 6).
        # Like the reference code, a pair with no unoccluded pixel contributes zero error.
        n_valid = noc.sum(dim=(1, 2, 3)) * diff.shape[1]  # (T-1,)
        n_valid = torch.where(n_valid > 0, n_valid, torch.full_like(n_valid, float(diff[0].numel())))
        per_pair_l1 = (diff.abs() * noc).sum(dim=(1, 2, 3)) / n_valid
        per_pair_l2 = (diff.square() * noc).sum(dim=(1, 2, 3)) / n_valid
        result["warping_error_occ_l1"] = per_pair_l1.mean().item()
        result["warping_error_occ_l2"] = per_pair_l2.mean().item()
        # fraction of pixels kept by the mask
        result["warping_error_occ_valid_frac"] = noc.mean().item()

        return result

    def _compute_next_frame_prediction_metrics(
        self,
        video: torch.Tensor,
        gt: torch.Tensor,
        video_mag_np: np.ndarray,
        gt_mag_np: np.ndarray,
        mag_data_range: float,
        lpips_per_frame: Optional[torch.Tensor],
    ) -> dict:
        """Next-frame prediction metrics over the predicted frames only (anchor frame excluded).

        Not named *psnr* so tune_metric="psnr" stays unambiguous. MSE is also split into
        reliable/unreliable pixels by the occlusion mask and (``*_conf``) by the soft fb confidence.
        """
        num_frames = video.shape[0]
        if num_frames < 2:
            return {}
        previous = self.next_frame_prediction_target == "previous"
        pred = slice(0, num_frames - 1) if previous else slice(1, num_frames)

        sq_err = (video[pred].float() - gt[pred].float()).square()  # (T-1, X, Y, C)
        frame_mse = sq_err.flatten(1).mean(dim=1).clamp_min(1e-12)
        result = {
            "nfp_mse": sq_err.mean().item(),
            "nfp_peak_snr": float(peak_signal_noise_ratio(gt_mag_np[pred], video_mag_np[pred], data_range=mag_data_range)),
            # RGB PSNR (data range 1) per predicted frame, averaged
            "nfp_rgb_peak_snr": (-10.0 * torch.log10(frame_mse)).mean().item(),
        }
        if lpips_per_frame is not None:
            result["nfp_lpips"] = lpips_per_frame[pred].mean().item()

        if self._gt_flow_fwd is None or self._gt_flow_bwd is None:
            return result
        flow_fwd = self._gt_flow_fwd.to(device=sq_err.device, dtype=sq_err.dtype)
        flow_bwd = self._gt_flow_bwd.to(device=sq_err.device, dtype=sq_err.dtype)
        if flow_fwd.shape[0] != num_frames - 1 or tuple(flow_fwd.shape[-2:]) != tuple(sq_err.shape[1:3]):
            return result

        # x_t ~ W(x_{t+1}, flow_fwd[t]) for "previous", x_{t+1} ~ W(x_t, flow_bwd[t]) for "next"
        flow_to_anchor, flow_from_anchor = (flow_fwd, flow_bwd) if previous else (flow_bwd, flow_fwd)
        h, w = flow_to_anchor.shape[-2:]
        base_grid = make_coords_grid(batch_size=num_frames - 1, h=h, w=w, device=str(sq_err.device)).movedim(1, -1)
        flow_from_anchor_warped = grid_sample(
            flow_from_anchor, base_grid + flow_to_anchor.movedim(1, -1), align_corners=True, mode="bilinear"
        )
        unreliable = self._lai_occlusion_mask(flow_to_anchor, flow_from_anchor_warped).unsqueeze(-1).to(sq_err.dtype)  # (T-1, X, Y, 1)
        result.update(self._split_mse(sq_err, unreliable, suffix=""))

        # soft forward-backward confidence of the flow that drives the warp
        fb_error_sq = (flow_to_anchor + flow_from_anchor_warped).square().sum(dim=1)  # (T-1, X, Y)
        confidence = torch.exp(-fb_error_sq / (2.0 * self.next_frame_prediction_confidence_tau_pix ** 2))
        unreliable_conf = (confidence <= self.next_frame_prediction_confidence_threshold).unsqueeze(-1).to(sq_err.dtype)
        result.update(self._split_mse(sq_err, unreliable_conf, suffix="_conf"))
        return result

    @staticmethod
    def _split_mse(sq_err: torch.Tensor, unreliable: torch.Tensor, suffix: str) -> dict:
        """nfp_unreliable_frac / nfp_mse_reliable / nfp_mse_unreliable (+ suffix) of the squared
        error (T-1, X, Y, C) inside / outside the binary mask ``unreliable`` (T-1, X, Y, 1)."""
        reliable = 1.0 - unreliable
        channels = sq_err.shape[-1]
        n_unreliable = unreliable.sum().item() * channels
        n_reliable = reliable.sum().item() * channels
        return {
            f"nfp_unreliable_frac{suffix}": unreliable.mean().item(),
            f"nfp_mse_reliable{suffix}": (sq_err * reliable).sum().item() / n_reliable if n_reliable > 0 else float("nan"),
            f"nfp_mse_unreliable{suffix}": (sq_err * unreliable).sum().item() / n_unreliable if n_unreliable > 0 else float("nan"),
        }

    @torch.no_grad()
    def evaluate_video(self, video: torch.Tensor, scaled: bool = False, extensive_eval: bool = False) -> dict:
        """
        Evaluate a reconstructed video against ground truth.

        Computes:
        - NMSE (Normalized Mean Squared Error) and PSNR in magnitude domain
        - Slicewise SSIM (mean SSIM across all frames)
        - Temporal SSIM (SSIM along time axis) — optional, expensive

        Args:
            video: Reconstructed video of shape (T, X, Y, C) or (T, 1, X, Y, C)
            extensive_eval: also compute temporal SSIM (expensive)
        """
        compute_temporal_ssim = extensive_eval

        video = video.detach()

        video_for_dc = video
        if scaled and not self.scale_measurement:
            video_for_dc = video_for_dc * self._scale_factor.view(1, *[1] * (video_for_dc.ndim - 1))
        elif (not scaled) and self.scale_measurement:
            video_for_dc = video_for_dc / self._scale_factor.view(1, *[1] * (video_for_dc.ndim - 1))

        data_consistency_l2 = None
        if self.forward_model is not None:
            observation_pred = self.forward_model(video_for_dc, cdim=-1)
            data_consistency_l2 = (observation_pred - self._measurement).abs().square().mean().item()

        if scaled:
            video = video * self._scale_factor.view(1, *[1]*(video.ndim-1)) # rescale back to original scale if it was scaled
        gt = self._target.detach()

        # Normalize shapes: accept complex or real inputs without trailing channel
        if video.is_complex():
            video = torch.view_as_real(video)
        if gt.is_complex():
            gt = torch.view_as_real(gt)

        # squeeze coil dim if present
        if video.ndim == 5:
            video = video.squeeze(dim=1)
        if gt.ndim == 5:
            gt = gt.squeeze(dim=1)

        if video.shape != gt.shape:
            raise ValueError(f"Shape mismatch: video {video.shape} vs gt {gt.shape}")

        # magnitudes (stay on device)
        video_mag = torch.norm(video, dim=-1)  # (T, X, Y)
        gt_mag = torch.norm(gt, dim=-1)        # (T, X, Y)

        # NMSE
        nmse_mag = nmse(video_mag, gt_mag)

        # PSNR (needs numpy)
        gt_mag_np = gt_mag.cpu().numpy()
        video_mag_np = video_mag.cpu().numpy()
        mag_data_range = float(gt_mag.max()) # - gt_mag.min())
        if mag_data_range == 0:
            mag_data_range = 1e-8
        psnr_mag = peak_signal_noise_ratio(gt_mag_np, video_mag_np, data_range=mag_data_range)

        # Slicewise SSIM
        ssim_slicewise = ssim_slicewise_mag(gt_mag, video_mag, data_range=mag_data_range)

        # Temporal SSIM (optional, sampled for speed)
        ssim_temporal = None
        if compute_temporal_ssim:
            ssim_temporal = tssim_temporal_mag(gt_mag, video_mag, sample_stride=None, win_size=None)

        # LPIPS (optional if dependency is installed)
        lpips_score = None
        lpips_per_frame = None
        if lpips is not None:
            if self._lpips_fn is None:
                self._lpips_fn = lpips.LPIPS(net='alex').to(video.device).eval()

            def _to_lpips_input(x: torch.Tensor) -> torch.Tensor:
                if x.shape[-1] == 2:
                    x = torch.linalg.norm(x, dim=-1, keepdim=True)
                elif x.shape[-1] not in (1, 3):
                    x = x.mean(dim=-1, keepdim=True)

                x = x.permute(0, 3, 1, 2).float()
                if x.shape[1] == 1:
                    x = x.repeat(1, 3, 1, 1)
                elif x.shape[1] > 3:
                    x = x[:, :3]
                return x

            gt_lpips = _to_lpips_input(gt)
            video_lpips = _to_lpips_input(video)

            gt_min = gt_lpips.amin()
            gt_max = gt_lpips.amax()
            gt_range = (gt_max - gt_min).clamp_min(1e-8)

            gt_lpips = 2.0 * ((gt_lpips - gt_min) / gt_range) - 1.0
            video_lpips = 2.0 * ((video_lpips - gt_min) / gt_range) - 1.0
            video_lpips = video_lpips.clamp(-1.0, 1.0)

            lpips_raw = self._lpips_fn(video_lpips, gt_lpips)
            lpips_score = lpips_raw.mean().item()
            if self.per_frame_metrics:
                lpips_per_frame = lpips_raw.view(-1).tolist()

        warping_error = self._compute_warping_error(video)

        result = {
            "nmse_mag": nmse_mag,
            "psnr_mag": psnr_mag,
            "ssim_slicewise": ssim_slicewise,
        }
        if data_consistency_l2 is not None:
            result["data_consistency_l2"] = data_consistency_l2
        if lpips_score is not None:
            result["lpips"] = lpips_score
        if self.per_frame_metrics:
            # per-frame metrics; frame 0 is the propagation anchor and is skipped
            num_frames = gt_mag_np.shape[0]
            for t in range(1, num_frames):
                result[f"psnr_frame_{t}"] = float(
                    peak_signal_noise_ratio(gt_mag_np[t], video_mag_np[t], data_range=mag_data_range)
                )
            if lpips_per_frame is not None:
                for t in range(1, min(num_frames, len(lpips_per_frame))):
                    result[f"lpips_frame_{t}"] = lpips_per_frame[t]
        if warping_error is not None:
            result.update(warping_error)
        if self.next_frame_prediction_target is not None:
            result.update(self._compute_next_frame_prediction_metrics(
                video, gt, video_mag_np, gt_mag_np, mag_data_range,
                lpips_raw.view(-1) if lpips_score is not None else None,
            ))
        # quality of the motion estimate the method was given (independent of the recon)
        result.update(self.get_flow_degradation_metrics())
        if ssim_temporal is not None:
            result["ssim_temporal"] = ssim_temporal
            result["ssim_temp_x_xy"] = ssim_temporal * ssim_slicewise
        return result

    @torch.no_grad()
    def evaluate(self, warp_method : BaseRepresentation) -> dict:
        return self.evaluate_video(
            warp_method.full_warp(),
            scaled=True
        )

    def fit(self,
            # passed objects from recon_task
            loss_fn, regularizer_fn,
            sampler, optimizer,
            model, accelerator,
            # train specific parameters
            iterations, scheduler_kwargs, chunks=5, img_eval_every_iters = 5, skip_temp=False,
            lam_reg_init_frame=0.0, lam_reg_temporal_change=0.0, gradient_acc_steps=1,
            data_consistency_weight_start: float = 1.0,
            data_consistency_weight_end: float = 1.0,
            data_consistency_weight_schedule: str = "fixed",
            # wandb logging interval (logging costs extra full_warp() passes)
            log_every_iters: int = 1,
            # logging infos passed from recon_task
            is_main_process=True, **kwargs
        ):

        device = self.device if accelerator is None else accelerator.device
        print(f"Using device: {device}")

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)

        if accelerator is not None:    
            optimizer, model, scheduler = accelerator.prepare(
                optimizer, model, scheduler
            )

        dc_schedule = str(data_consistency_weight_schedule).lower()
        if dc_schedule not in {"fixed", "linear", "cosine"}:
            raise ValueError(
                "trainer.data_consistency_weight_schedule must be one of {'fixed','linear','cosine'}"
            )

        def _dc_weight(progress: float) -> float:
            p = max(0.0, min(1.0, float(progress)))
            if dc_schedule == "fixed":
                alpha = 0.0
            elif dc_schedule == "linear":
                alpha = p
            else:  # cosine
                alpha = 0.5 * (1.0 - math.cos(math.pi * p))
            return float(data_consistency_weight_start + alpha * (data_consistency_weight_end - data_consistency_weight_start))

        wandb.log(
            {
                "pseudoinv_psnr": peak_signal_noise_ratio(
                    self._target.cpu().numpy(),
                    self.get_train_dataset(scaled=False).cpu().numpy(),
                    data_range=self._target.max().item(),
                )
            }
        )

        with tqdm.tqdm(range(iterations), desc='optim') as pbar:

            for j in pbar: # for each iteration

                optimization_progress = float(j) / max(1, iterations - 1)
                dc_weight_j = _dc_weight(optimization_progress)

                optimizer.zero_grad()

                for k in range(gradient_acc_steps):

                    video = model.full_warp()
                    result = self.forward_model(video, cdim=-1) if self.forward_model is not None else video

                    if k == 0:
                        data_consistency_loss = loss_fn(result, self._measurement, kcoords=None, iteration=j)
                        train_loss = dc_weight_j * data_consistency_loss
                    else:
                        data_consistency_loss = torch.zeros(1, device=device)
                        train_loss = torch.zeros(1, device=device)

                    if regularizer_fn is not None:
                        flow_reference_video = self.get_flow_reference_video()
                        pseudoinverse_for_reg = flow_reference_video if flow_reference_video is not None else self.get_train_dataset()
                        reg_loss = regularizer_fn(model, result, pseudoinverse=pseudoinverse_for_reg, kspace=None, kcoords=None, timesteps=self.time_mesh, measurement=self._measurement, scale_factor=self._scale_factor, optimization_progress=optimization_progress) / gradient_acc_steps
                    else:
                        reg_loss = torch.zeros(1, device=device)

                    loss = train_loss + reg_loss

                    if accelerator is not None:
                        accelerator.backward(loss)
                    else:
                        loss.backward()

                optimizer.step()
                scheduler.step( loss.item() )

                if is_main_process:
                    if j % img_eval_every_iters == 0:
                        eval_results = self.evaluate(model)
                        last_eval_iter = j

                    pbar.set_description(f"train_loss: { (loss).item() } (@it: {last_eval_iter})" + str(eval_results), refresh=True)

                    if wandb.run is not None and j % log_every_iters == 0:
                        wandb.log({
                            "train_loss": train_loss.item(),
                            "data_consistency_loss": data_consistency_loss.item(),
                            "data_consistency_weight": float(dc_weight_j),
                            "target_std" : self._target.std().item(),
                            "pseudoinv_std" : self.get_train_dataset().std().item(),
                            "measurement_std" : self._measurement.std().item(),
                            "recon_std" : model.full_warp().std().item(),
                            "recon_scaled_std" : (model.full_warp() * self._scale_factor).std().item(),
                            "reg_loss": reg_loss.item(),
                            "global_step": j,
                        })

                        for param_group in optimizer.param_groups:
                            wandb.log({param_group["name"]: param_group["lr"], "global_step": j})

                        if j % img_eval_every_iters == 0:
                            wandb.log({
                                **eval_results,
                                "global_step": j,
                            })
