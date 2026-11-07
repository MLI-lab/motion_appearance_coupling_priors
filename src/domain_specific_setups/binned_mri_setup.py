
from typing import Callable
import torch
from types import SimpleNamespace
from src.domain_specific_setups.base_domain_specific_setup import BaseDomainSpecificSetup
from src.domain_specific_setups.natural_video_forward_models import LinearForwardModel

import tqdm
import wandb

from skimage.metrics import peak_signal_noise_ratio
import numpy as np

from src.utils.metrics import nmse, ssim_slicewise_mag, tssim_temporal_mag, psnr_compl, ssim3d
from src.representations.representation_mri_wrapper import RepresentationMRIWrapper

from src.dataset.mri import kspace2image, fftnc, ifftnc


class MulticoilMRIForwardModel(LinearForwardModel):
    """Masked, coil-weighted, centered 2D Fourier forward model for binned/unbinned CMR k-space.

    Operates on real-valued (real/imag as trailing dim of size 2) tensors, matching the
    convention used throughout `BinnedMRISetup`/`BinnedCMRxDataset`.
    """

    def __init__(self, smaps: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            smaps: coil sensitivity maps, shape (Ncoils, X, Y, 2)
            mask: undersampling mask, shape (T, X, Y)
        """
        super().__init__()
        self.smaps = torch.view_as_complex(smaps.contiguous())  # (Ncoils, X, Y)
        self.mask = mask  # (T, X, Y)

    def __call__(self, img: torch.Tensor, cdim: int = -1) -> torch.Tensor:
        """img: (T, X, Y, 2) -> masked multicoil k-space: (T, Ncoils, X, Y, 2)"""
        img_c = torch.view_as_complex(img.contiguous())  # (T, X, Y)
        coil_imgs = img_c.unsqueeze(1) * self.smaps.unsqueeze(0)  # (T, Ncoils, X, Y)
        kspace = fftnc(coil_imgs, dim=(2, 3))
        kspace = kspace * self.mask.unsqueeze(1)  # broadcast over coil dim
        return torch.view_as_real(kspace)

    def adjoint(self, meas: torch.Tensor, cdim: int = -1) -> torch.Tensor:
        """meas: (T, Ncoils, X, Y, 2) -> coil-combined image: (T, X, Y, 2)"""
        meas_c = torch.view_as_complex(meas.contiguous())  # (T, Ncoils, X, Y)
        meas_c = meas_c * self.mask.unsqueeze(1)
        coil_imgs = ifftnc(meas_c, dim=(2, 3))
        img = (coil_imgs * self.smaps.conj().unsqueeze(0)).sum(dim=1)  # (T, X, Y)
        return torch.view_as_real(img)

    def pseudoinv(self, meas: torch.Tensor, cdim: int = -1) -> torch.Tensor:
        return self.adjoint(meas, cdim=cdim)


class BinnedMRISetup(BaseDomainSpecificSetup):
    
    def __init__(self,
            base_path,
            target_as_psuedoinv_in_reg : bool,
            sample_index,
            video_length_in_s,
            path_resolver,
            use_smaps,
            sensmaps_in_base,
            device,
            **dataset_cfg,
        ):
        super().__init__()

        from src.dataset.binned_cmr_dataset import BinnedCMRxDataset
        self.base_path = path_resolver(base_path)
        dataset = BinnedCMRxDataset(data_path=self.base_path, return_original_kspace=False, **dataset_cfg)
        self._masked_kspace, self._mask, self._smaps, self._ground_truth = dataset[sample_index]
        self.device = device
        
        # kspace shape: (T, C, X, Y, 2)
        # smaps shape: (C, X, Y, 2)
        self._pseudoinverse = dataset.inv(self._masked_kspace, self._smaps, return_complex=False, adaptive=True).to(device)
        print(f"Pseudoinverse shape: {self._pseudoinverse.shape}")

        self.eval_cropping = SimpleNamespace(lx=0, rx=None, ty=0, by=None)
        self.eval_rot90 = False

        from src.dataset.cinemri_dataset import get_meshes_and_corrections_for_matrix_size
        F, X, Y, C = self._ground_truth.shape
        self.spatial_mesh, self.kspace_mesh, self.dft_normalization_factor, self.dft_phase_correction = get_meshes_and_corrections_for_matrix_size( (X, Y), device=device)
        self.time_mesh = torch.linspace(0, video_length_in_s, F).to(device)

        self.representation_wrapper = RepresentationMRIWrapper(
            spatial_mesh=self.spatial_mesh,
            kspace_mesh=self.kspace_mesh,
            time_mesh=self.time_mesh,
            smaps=self._smaps.squeeze(dim=1), # has shape (NC, 1, Ny, Nx, 2) -> (NC, Ny, Nx, 2)
            use_smaps=use_smaps,
            sensmaps_in_base=sensmaps_in_base
        )
        self.target_as_psuedoinv_in_reg = target_as_psuedoinv_in_reg

    def cache_datasets_on_gpu(self, device):
        print(f"Preloading element to device {device}")
        self._masked_kspace = self._masked_kspace.to(device)
        self._mask = self._mask.to(device)
        self._smaps = self._smaps.to(device)
        self._ground_truth = self._ground_truth.to(device)
        self._pseudoinverse = self._pseudoinverse.to(device)

    def get_train_dataset(self) -> torch.utils.data.Dataset:
        return self._pseudoinverse

    def get_reference_video(self) -> torch.Tensor:
        return self._ground_truth

    def get_meshes_dict(self):
        return {
            "spatial_mesh": self.spatial_mesh,
            "kspace_mesh": self.kspace_mesh,
            "time_mesh": self.time_mesh,
        }


    def get_fps(self):
        return self.time_mesh.shape[0] / (self.time_mesh[-1] - self.time_mesh[0]).item()

    def get_spatial_mesh(self):
        return self.spatial_mesh

    def get_time_mesh(self):
        return self.time_mesh

    def get_domain_specific_info(self):
        return {
            "smaps": self._smaps # should have (NC, _, Ny, Nx, _)
        }


    def get_dataset_params(self):
        return None # there are no parameters..

    def get_video_processor_for_visualization(
        self,
    ) -> Callable[[torch.Tensor], np.ndarray]:
        def processor(x : torch.Tensor) -> np.ndarray:
            x = x.norm(dim=-1,keepdim=True) # get magnitude
            x = x / x.max()
            x[x < 0] = 0.0
            x[x > 1.0] = 1.0
            if self.eval_cropping is None:
                lx, rx, ty, by = 0, -1, 0, -1
            else:
                lx, rx, ty, by = self.eval_cropping.lx, self.eval_cropping.rx, self.eval_cropping.ty, self.eval_cropping.by
            x_proc = x[:, lx:rx, ty:by]  # select region of interest
            return (x_proc * 255).to(torch.uint8).permute(0, 3, 1, 2).repeat(1,3,1,1).cpu().numpy()
        return processor

    def get_forward_model(self) -> LinearForwardModel:
        # built lazily so it follows the current device of self._smaps/self._mask
        mask = self._mask
        if mask.dim() == 5:
            # (T, 1, X, Y, 1) -> (T, X, Y)
            mask = mask.squeeze(1).squeeze(-1)
        elif mask.dim() == 2:
            # a single static (X, Y) mask shared across all frames
            mask = mask.unsqueeze(0).expand(self._masked_kspace.shape[0], -1, -1)
        return MulticoilMRIForwardModel(smaps=self._smaps.squeeze(dim=1), mask=mask)

    def get_measurement(self, scaled: bool = False) -> torch.Tensor:
        # real MRI k-space is already in a consistent physical scale; no rescaling needed
        return self._masked_kspace


    @torch.no_grad()
    def evaluate_video(self, video: torch.Tensor, scaled: bool = False, extensive_eval = False) -> dict:
        """
        Evaluate a reconstructed video against ground truth.

        Computes:
        - NMSE (Normalized Mean Squared Error) and PSNR in magnitude domain
        - Slicewise SSIM (mean SSIM across all frames)
        - Temporal SSIM (SSIM along time axis) — optional, expensive

        Args:
            video: Reconstructed video of shape (T, X, Y, 2) or (T, 1, X, Y, 2)
            scaled: unused (interface parity with NaturalVideoSetup.evaluate_video)
        """
        compute_temporal_ssim = extensive_eval

        video = video.detach()
        gt = self._ground_truth.detach()

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

        # Complex-domain PSNR (penalizes phase error, unlike psnr_mag)
        psnr_compl_val = psnr_compl(video, gt, data_range=mag_data_range)

        # Slicewise SSIM
        ssim_slicewise = ssim_slicewise_mag(gt_mag, video_mag, data_range=mag_data_range)

        # Volumetric (3D) SSIM: one joint window over (T, X, Y) instead of per-frame averaging
        ssim3d_val = ssim3d(gt_mag, video_mag, data_range=mag_data_range)

        # Temporal SSIM (optional, sampled for speed)
        ssim_temporal = None
        if compute_temporal_ssim:
            ssim_temporal = tssim_temporal_mag(gt_mag, video_mag, sample_stride=None, win_size=None)

        result = {
            "nmse_mag": nmse_mag,
            "psnr_mag": psnr_mag,
            "psnr_compl": psnr_compl_val,
            "ssim_slicewise": ssim_slicewise,
            "ssim3d": ssim3d_val,
        }
        if ssim_temporal is not None:
            result["ssim_temporal"] = ssim_temporal
            result["ssim_temp_x_xy"] = ssim_temporal * ssim_slicewise
        return result

    @torch.no_grad()
    def evaluate(self, warp_method : torch.nn.Module) -> dict:
        self.representation_wrapper.set_model(warp_method)
        video = self.representation_wrapper.full_warp_spatial(
            spatial_mesh=self.spatial_mesh,
            timesteps=self.time_mesh,
            skip_temp=False,
            skip_smaps=True, # we compare the base image
            device=self.device,
            chunks=1) # (T, X, Y, C1, C2) # here C1 is number of coils, C2 is 2 for real/imag
        
        # evaluate in complex domain
        return self.evaluate_video(video.movedim(-2, 1))

    def fit(self,
            # passed objects from recon_task
            loss_fn, regularizer_fn,
            sampler, optimizer,
            model, accelerator,
            # train specific parameters
            iterations, scheduler_kwargs, chunks=5, img_eval_every_iters = 5, skip_temp=False,
            lam_reg_init_frame=0.0, lam_reg_temporal_change=0.0, gradient_acc_steps=1,
            # logging infos passed from recon_task
            is_main_process=True, **kwargs
        ):
    
        device = self.device if accelerator is None else accelerator.device
        self.representation_wrapper.set_model(model)

        print(f"Using device: {device}")

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)

        if accelerator is not None:    
            optimizer, model, scheduler = accelerator.prepare(
                optimizer, model, scheduler
            )

        with tqdm.tqdm(range(iterations), desc='optim') as pbar:

            for j in pbar: # for each iteration

                optimizer.zero_grad()
                
                for k in range(gradient_acc_steps):
                    result = self.representation_wrapper.full_warp_kspace(
                        self.kspace_mesh, self.time_mesh, skip_temp, device=device, chunks=chunks
                    ).movedim(-2, 1) # T, X, Y, C1, C2 -> T, C1, X, Y, C2, where C1 is coils and C2 is real/imag

                    if k == 0:
                        train_loss = loss_fn(result * self._mask, self._masked_kspace, kcoords=None)
                    else:
                        train_loss = torch.zeros(1, device=device)

                    if regularizer_fn is not None:
                        if self.target_as_psuedoinv_in_reg:
                            reg_loss = regularizer_fn(model, result, pseudoinverse=self._ground_truth, kspace=None, kcoords=None, timesteps=self.time_mesh, measurement=self._masked_kspace, optimization_progress = float(j) / iterations) / gradient_acc_steps
                        else:
                            reg_loss = regularizer_fn(model, result, pseudoinverse=self._pseudoinverse, kspace=None, kcoords=None, timesteps=self.time_mesh, measurement=self._masked_kspace, optimization_progress = float(j) / iterations) / gradient_acc_steps
                    else:
                        reg_loss = torch.zeros(1, device=device)

                    loss = train_loss + reg_loss # + lam_reg_init_frame * loss_init_frame + lam_reg_temporal_change * loss_temporal_change

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

                    if wandb.run is not None:
                        wandb.log({
                            "train_loss": train_loss.item(),
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

        # finally, we also log the ground truth video since we have it available
        if is_main_process:
            wandb.log({
                "ground_truth": wandb.Video(self.get_video_processor_for_visualization()(self._ground_truth), fps=self.get_fps(), format="gif"),
                "global_step": iterations
            })