import h5py
import torch
import numpy as np

from functools import lru_cache
from torch.utils import data
from pathlib import Path

from src.dataset.mri import fftnc, ifftnc, kspace2image
from src.dataset.masking import ktMaskGenerator

from torchvision.transforms import CenterCrop
from torch.nn.functional import interpolate


@lru_cache(maxsize=64)
def _sampling_inclusion_prob(nx, ny, nt, ncalib, R, mask_type, n_trials=500):
    """Monte-Carlo marginal inclusion probability pi(ky, t) of ktMaskGenerator masks (averaged over kx),
    for density-compensated estimation of the full-kspace norm. Add-one smoothed so 1/pi stays finite.

    Each trial is reseeded from OS entropy (ktGaussianSampling reseeds the global RNG and cycles
    otherwise); the global RNG state is restored afterwards.
    """
    rng_state = np.random.get_state()
    try:
        counts = np.zeros((ny, nt), dtype=np.float64)
        for _ in range(n_trials):
            np.random.seed(None)
            m = ktMaskGenerator(nx, ny, nt, ncalib, R, mask_type)  # (nx, ny, nt)
            counts += m.mean(axis=0)
    finally:
        np.random.set_state(rng_state)
    return (counts + 1.0) / (n_trials + 2.0)  # (ny, nt), in (0, 1)

class BinnedCMRxDataset(data.Dataset):
    def __init__(
            self, 
            data_path, 
            filter=None, 
            multicoil=True, 
            image_only=False, 
            repeat_cycle=0,
            resolution_interpolation=False,
            magnitude_only=False,
            undersampling_factor=4, 
            sample_mask=False, 
            mask_type='ktUniform',
            return_fname=False,
            return_original_kspace=True,
            num_center_slices=None,
            crop_and_downscale_cfg = {
                "enabled": False,
                "size_before_downscale": [256, 256],
                "size_after_downscale": [128, 128]
            }
        ):
        super().__init__()
        self.data_path = data_path
        self.image_only = image_only
        self.repeat_cycle = repeat_cycle
        self.resolution_interpolation = resolution_interpolation
        self.magnitude_only = magnitude_only
        self.multicoil = multicoil
        self.undersampling_factor = undersampling_factor
        self.sample_mask = sample_mask
        self.mask_type = mask_type
        self.return_fname = return_fname
        self.mask_key = f"mask{undersampling_factor:02}"
        filter = '' if filter is None else filter

        if num_center_slices is not None and num_center_slices <= 0:
            raise ValueError(f"num_center_slices must be a positive int or None, got {num_center_slices}.")
        self.num_center_slices = num_center_slices

        self.return_original_kspace = return_original_kspace

        self.crop_and_downscale_cfg = crop_and_downscale_cfg

        # find all files
        if isinstance(data_path, (list, tuple)):
            paths = []
            for single_dir in data_path:
                single_dir = Path(single_dir)
                if single_dir.is_dir():
                    paths.extend(single_dir.glob(f'**/*{filter}.h5'))
                else:
                    print(f"[Warning] {single_dir} is not a valid directory. Skipping.")
        else:
            p = Path(data_path)
            if p.is_dir():
                paths = p.glob(f'**/*{filter}.h5')
            elif p.is_file():
                paths = [p]
            else:
                raise ValueError(f"{data_path} is not a valid directory or file.")

        # split each file to return slices individually
        self.raw_samples = []
        for fname in paths:
            with h5py.File(fname, "r") as hf:
                num_slices = hf['kspace'].shape[1]  # count slices in [t, s, ky, kx] or [t, s, coil, ky, kx]

            if self.num_center_slices is None:
                slice_range = range(num_slices)
            else:
                # N slices around the middle of the stack (clamped to the available count)
                n = min(self.num_center_slices, num_slices)
                start = max(0, (num_slices - n) // 2)
                slice_range = range(start, start + n)

            new_raw_samples = []
            for slice_ind in slice_range:
                raw_sample = (fname, slice_ind)
                new_raw_samples.append(raw_sample)
            self.raw_samples += new_raw_samples

    def _normalize(self, x, kspace_norm, volume_shape):
        x = x * np.sqrt(np.prod(volume_shape)*2) / (kspace_norm)
        return x


    def inv(self, kspace, sensitivities=None, return_complex=True, adaptive=True):
        # make sure kspace and sensmaps are in complex format
        if kspace.dtype in [torch.float32, torch.float64]:
            kspace = torch.view_as_complex(kspace)
        if sensitivities is not None and sensitivities.dtype in [torch.float32, torch.float64]:
            sensitivities = torch.view_as_complex(sensitivities)

        if self.multicoil:
            assert sensitivities != None, "Sensitivities must be provided for multicoil data."
            result = kspace2image(kspace, sens_maps=sensitivities, fdim=(2, 3), cdim=1, adaptive=adaptive)
        else:
            result = ifftnc(kspace, dim=(1, 2))

        if return_complex:
            return result
        else:
            return torch.view_as_real(result)

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, idx):
        fname, dataslice = self.raw_samples[idx]

        with h5py.File(fname, "r") as hf:
            kspace = hf['kspace'][:, dataslice]                                                                # [frame, coil, ky, kx] or [frame, ky, kx]
            ground = hf['image'][:, dataslice] if 'image' in hf else -1                                        # [frame, ky, kx]
            sensitivities = hf['sensitivities'][dataslice] if self.multicoil and not self.image_only else None # [coil, ky, kx]
            if self.mask_key in hf:
                mask = hf[self.mask_key][:] if not self.sample_mask and not self.image_only else None          # [ky, kx] or [frame, slice, ky, kx]
            else:
                mask = hf['mask'][:] if not self.sample_mask and not self.image_only else None                 # [ky, kx] or [frame, slice, ky, kx]
            mask = mask[:, dataslice, None]  if mask is not None and mask.ndim == 4 else mask                  # [frame, 1, ky, kx]

        kspace = torch.from_numpy(kspace)
        ground = torch.from_numpy(ground) if ground is not -1 else -1
        sensitivities = torch.from_numpy(sensitivities) if sensitivities is not None else None
        mask = torch.from_numpy(mask) if mask is not None else None

        ##### simulation at lower resoluion ######

        # currently the shapes are:
        #  -  kspace (F, C, Ny, Nx) or (F, Ny, Nx)
        #  -  ground (F, Ny, Nx)
        #  -  sensitivities (C, Ny, Nx)
        # -> mask will be generated afterwards

        print(f"Volume and kspace norms: {ground.norm().item():.2f}, {kspace.norm().item():.2f}")
        print(f"Volume and kspace std: {ground.std().item():.2f}, {kspace.std().item():.2f}")

        if self.crop_and_downscale_cfg["enabled"]:
            from src.utils.video_crop_transforms import SpatialCenterCropAndDownscale
            size_before_downscale = self.crop_and_downscale_cfg["size_before_downscale"]
            size_after_downscale = self.crop_and_downscale_cfg["size_after_downscale"]
            transform = SpatialCenterCropAndDownscale(tuple(size_before_downscale), tuple(size_after_downscale), normalization="none")
            # follow the format a, b, h, w -> cropping etc in last dim
            result = transform(torch.view_as_real(ground).movedim(-1,0), torch.view_as_real(sensitivities).movedim(-1,0))
            ground = torch.view_as_complex(result["video"].movedim(0,-1).contiguous())
            sensitivities = torch.view_as_complex(result["warped_noise_sample"].movedim(0,-1).contiguous())
            from fastmri import fft2c
            kspace = torch.view_as_complex(
                fft2c(
                    torch.view_as_real(
                        ground[:, None] * sensitivities[None]
                    )
                )
            )
            torch.testing.assert_close(ground, self.inv(kspace, sensitivities=sensitivities), atol=1e-5, rtol=1e-5)
            print(f"Info: Downsampled data from {size_before_downscale} to {size_after_downscale}.")

        print(f"Volume and kspace norms: {ground.norm().item():.2f}, {kspace.norm().item():.2f}")
        print(f"Volume and kspace std: {ground.std().item():.2f}, {kspace.std().item():.2f}")

        if self.repeat_cycle > 1:
            kspace = kspace.repeat(self.repeat_cycle, *[1]*(kspace.ndim-1))  # repeat along the temporal dimension
            if ground is not -1:
                ground = ground.repeat(self.repeat_cycle, *[1]*(ground.ndim-1))
            if (not self.sample_mask) and (mask is not None):
                mask = mask.repeat(self.repeat_cycle, *[1]*(mask.ndim-1))

        if self.image_only:
            if self.magnitude_only:
                ground = abs(ground).unsqueeze(0)  
                ground = (ground - ground.min()) / (ground.max() - ground.min())
                ground = (ground * 2) - 1
                return ground
            elif self.resolution_interpolation:
                crop = CenterCrop((298, 422)) # (298, 422) (279, 396)
                ground = torch.stack((ground.real, ground.imag), dim=0)
                ground = crop(ground)
                frames = ground.shape[1]
                ground = interpolate(ground.unsqueeze(0), size=(frames, 186, 264), mode='trilinear').squeeze(0) # 'nearest-exact' 'trilinear' # 334, 480
                return ground
            else:
                return torch.stack((ground.real, ground.imag), dim=0)  

        if self.sample_mask:
            nt, ny, nx = ground.shape
            ncalib = 0
            mask = ktMaskGenerator(nx, ny, nt, ncalib, self.undersampling_factor, self.mask_type)
            mask = mask.transpose((2, 1, 0))
            mask = torch.from_numpy(mask)
            mask = mask.unsqueeze(1)            # [frame, 1, ky, kx]

        # masking, transformation, normalization, and make complex dim explicit [frame, coil, ky, kx] -> [2, frame, coil, ky, kx]
        # fair setting, as typically one would not have access to the fully sampled kspace
        masked_kspace = kspace * mask

        # density-compensated (Horvitz-Thompson) estimate of the full-kspace norm
        F, Ny, Nx = ground.shape
        _ncalib = ncalib if self.sample_mask else 0
        pi_map = _sampling_inclusion_prob(nx=Nx, ny=Ny, nt=F, ncalib=_ncalib, R=self.undersampling_factor, mask_type=self.mask_type)
        pi_bcast = torch.from_numpy(np.ascontiguousarray(pi_map.T)).clamp_min(1e-6).reshape(F, 1, Ny, 1)  # (ny,nt)->(F,Ny) -> broadcast (F,1,Ny,1)
        full_kspace_norm_est = (masked_kspace.abs().double().square() / pi_bcast).sum().sqrt().item()

        ground = self._normalize(ground, full_kspace_norm_est, volume_shape=ground.shape) if ground is not -1 else -1
        masked_kspace = self._normalize(masked_kspace, full_kspace_norm_est, volume_shape=ground.shape)

        print(f"Volume and kspace norms: {ground.norm().item():.2f}, {kspace.norm().item():.2f}")
        print(f"Volume and kspace std: {ground.std().item():.2f}, {kspace.std().item():.2f}")

        masked_kspace = torch.view_as_real(masked_kspace) # (F, C, X, Y, 2)
        sensitivities = torch.view_as_real(sensitivities) # (C, X, Y, 2)
        ground = torch.view_as_real(ground) # (F, X, Y, 2)
        mask = mask.float().unsqueeze(dim=-1) # (F, 1, X, Y, 1)

        if self.return_fname:
            name = f'{fname.parent.stem}_{dataslice}'
            return masked_kspace, mask, sensitivities, ground, name
        elif self.return_original_kspace:
            return masked_kspace, mask, sensitivities, ground, kspace
        else:
            return masked_kspace, mask, sensitivities, ground
