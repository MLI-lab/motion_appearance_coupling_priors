import torch
from typing import Callable, Optional, Tuple

import numpy as np
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

class LinearForwardModel():
    def __init__(self):
        pass

    def __call__(self, img: torch.Tensor, cdim : int) -> torch.Tensor:
        raise NotImplementedError("Implement __call__ in subclass if needed.")

    def adjoint(self, meas: torch.Tensor, cdim : int) -> torch.Tensor:
        raise NotImplementedError("Implement adjoint in subclass if needed.")

    def pseudoinv(self, meas: torch.Tensor, cdim: int = 1) -> torch.Tensor:
        raise NotImplementedError("Implement pseudoinv in subclass if needed.")

    def measurement_support(self, meas: torch.Tensor, cdim: int = 1) -> torch.Tensor:
        """0/1 mask of the acquired entries of `meas` (same shape); additive noise is only applied where it is 1."""
        return torch.ones_like(meas)

    def add_noise(self, meas: torch.Tensor, std: float, cdim: int = 1) -> torch.Tensor:
        """Add i.i.d. Gaussian noise of the given std to the acquired entries of `meas` only."""
        return meas + torch.randn_like(meas) * std * self.measurement_support(meas, cdim=cdim)

    def get_scale_factor(self, meas: torch.Tensor, cdim: int = 1, random_probes : bool = False, num_probes: int = 10) -> torch.Tensor:
        if not random_probes:
            return self.pseudoinv(meas, cdim=cdim).std()
        else:
            return torch.stack(
                [
                    self.pseudoinv(torch.randn_like(meas), cdim=cdim).square().mean()
                    for _ in range(num_probes)
                ]
            ).mean().sqrt()


class TemporalBicubicDownsamplingModel(LinearForwardModel):
    def __init__(self, scale_factor: float, kernel_size_t: int, deconv_eps: float = 1e-6):
        self.scale_factor = scale_factor
        self.kernel_size_t = kernel_size_t
        self.deconv_eps = deconv_eps

    def _resize(self, img: torch.Tensor, scale: float) -> torch.Tensor:
        h, w = img.shape[-2:]
        new_h, new_w = int(h * scale), int(w * scale)
        return TF.resize(img, size=[new_h, new_w], interpolation=InterpolationMode.BICUBIC)

    def _temporal_blur(self, img_tchw: torch.Tensor) -> torch.Tensor:
        img_bcthw = img_tchw.permute(1, 0, 2, 3).unsqueeze(0)
        blurred = temporal_blur(img_bcthw, self.kernel_size_t)
        return blurred.squeeze(0).permute(1, 0, 2, 3)

    def _temporal_deconv(self, meas_tchw: torch.Tensor) -> torch.Tensor:
        meas_bcthw = meas_tchw.permute(1, 0, 2, 3).unsqueeze(0)
        _, _, t, _, _ = meas_bcthw.shape
        kernel = uniform_kernel_1d(self.kernel_size_t, dtype=meas_bcthw.dtype).to(meas_bcthw.device)
        kernel_padded = torch.zeros(t, dtype=meas_bcthw.dtype, device=meas_bcthw.device)
        kernel_padded[: self.kernel_size_t] = kernel
        kernel_padded = torch.roll(kernel_padded, -(self.kernel_size_t // 2))

        kernel_fft = torch.fft.fft(kernel_padded)
        meas_fft = torch.fft.fft(meas_bcthw, dim=2)
        denom = kernel_fft.abs() ** 2 + self.deconv_eps
        inv_filter = kernel_fft.conj() / denom
        deconv = torch.fft.ifft(meas_fft * inv_filter[None, None, :, None, None], dim=2).real
        return deconv.squeeze(0).permute(1, 0, 2, 3)

    def __call__(self, img: torch.Tensor, cdim: int = 1) -> torch.Tensor:
        assert img.ndim == 4, "Input image must be 4D tensor (T, H, W, C) or (T, C, H, W)."
        img = img.movedim(cdim, 1)  # (T, C, H, W)
        blurred = self._temporal_blur(img)
        out = self._resize(blurred, 1.0 / self.scale_factor)
        return out.movedim(1, cdim)

    def adjoint(self, meas: torch.Tensor, cdim: int = 1) -> torch.Tensor:
        assert meas.ndim == 4, "Measurement must be 4D tensor (T, H, W, C) or (T, C, H, W)."
        meas = meas.movedim(cdim, 1)  # (T, C, H, W)
        up = self._resize(meas, self.scale_factor)
        adj = self._temporal_blur(up)
        return adj.movedim(1, cdim)

    def pseudoinv(self, meas: torch.Tensor, cdim: int = 1) -> torch.Tensor:
        assert meas.ndim == 4, "Measurement must be 4D tensor (T, H, W, C) or (T, C, H, W)."
        meas = meas.movedim(cdim, 1)  # (T, C, H, W)
        deconv_t = self._temporal_deconv(meas)
        out = self._resize(deconv_t, self.scale_factor)
        return out.movedim(1, cdim)

class IdentityForwardModel(LinearForwardModel):
    """No-op forward model: measurement == target (e.g. for propagation baselines on GT frames)."""

    def __init__(self):
        pass

    def __call__(self, img: torch.Tensor, cdim: int = 1) -> torch.Tensor:
        return img

    def adjoint(self, meas: torch.Tensor, cdim: int = 1) -> torch.Tensor:
        return meas

    def pseudoinv(self, meas: torch.Tensor, cdim: int = 1) -> torch.Tensor:
        return meas



from src.domain_specific_setups.utils import Blurkernel

from src.dataset.mri import fftnc, ifftnc
from fastmri import ifft2c, fft2c
from src.dataset.masking import ktMaskGenerator

class CMRSimForwardModel(LinearForwardModel):
    """
        This model simulates taking an undersampled k-space measurenet in lieu of the binned_cmr_dataset setup. 
    """
    def __init__(
        self,
        undersampling_factor: int,
        mask_type: str = 'ktGaussian',
        mask_seed: int = 42,
        ncalib: int = 0,          # number of always-sampled central ky lines (ACS)
        static_mask: bool = False, # True: use frame 0's line set for every frame (time-constant operator)
    ):
        self.undersampling_factor = undersampling_factor
        self.mask_type = mask_type
        self.mask_seed = mask_seed
        self.ncalib = ncalib
        self.static_mask = static_mask
        self._mask_cache = {}

    def _get_mask(self, shape: Tuple[int, int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        nt, ny, nx = shape
        cache_key = (nt, ny, nx, device.type, dtype)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        rng_state = np.random.get_state()
        np.random.seed(self.mask_seed)
        mask = ktMaskGenerator(nx, ny, nt, self.ncalib, self.undersampling_factor, self.mask_type)
        mask = mask.transpose((2, 1, 0))  # (nt, ny, nx)
        np.random.set_state(rng_state)
        if self.static_mask:
            # same k-space lines in every frame (time-constant operator)
            mask = np.repeat(mask[:1], nt, axis=0)
        mask = torch.from_numpy(mask).to(device=device, dtype=dtype)
        mask = mask[:, None, :, :, None]
        self._mask_cache[cache_key] = mask
        return mask

    def __call__(self, img: torch.Tensor, cdim: int = 1) -> torch.Tensor:
        # image has shape (T, H, W, C) (or T, C, H, W), we want (T, H, W, C, 2) (or T, C, H, W, 2) where the last dim element is is zero phase
        assert img.ndim == 4, "Input image must be 4D tensor (T, H, W, C) or (T, C, H, W)"
        img = img.movedim(cdim,1) # (T, 3, H, W)
        img_compl = torch.stack([img, torch.zeros_like(img)], dim=-1)
        kspace = fft2c(img_compl) # assume img is (T, H, W, C)
        # createk mask
        nt, _, ny, nx = img.shape
        mask = self._get_mask((nt, ny, nx), kspace.device, kspace.dtype)
        masked_kspace = kspace * mask
        return masked_kspace.movedim(1, cdim)  # move back coil dimi

    def measurement_support(self, meas: torch.Tensor, cdim: int = 1) -> torch.Tensor:
        # Only the sampled k-t lines are acquired: noise must not leak into the zero-filled
        # entries (pseudoinv does not re-mask, so unmasked noise would otherwise pollute it).
        meas = meas.movedim(cdim, 1)  # (T, C, H, W, 2)
        nt, _, ny, nx, _ = meas.shape
        mask = self._get_mask((nt, ny, nx), meas.device, meas.dtype)
        return mask.expand_as(meas).movedim(1, cdim)

    def adjoint(self, meas: torch.Tensor, cdim: int = 1) -> torch.Tensor:
        meas = meas.movedim(cdim, 1)  # (T, C, H, W, 2)
        nt, _, ny, nx, _ = meas.shape
        mask = self._get_mask((nt, ny, nx), meas.device, meas.dtype)

        masked_meas = meas * mask
        img_rec = ifft2c(masked_meas)
        return img_rec[..., 0].movedim(1, cdim)

    def pseudoinv(self, meas: torch.Tensor, cdim: int = 1) -> torch.Tensor:
        # meas has shape (T, H, W, C, 2) (or T, C, H, W, 2), we want (T, H, W, C) or (T, C, H, W)
        img_rec = ifft2c(meas.movedim(cdim,1))  # (T, H, W, C, 2)
        return img_rec[...,0].movedim(1, cdim)  # return real part


class TemporalBlurringModel(LinearForwardModel):
    def __init__(
        self,
        blur_type: str = "gaussian",
        kernel_size: int = 31,
        std: float = 3.0,
        kernel_size_t: int = 3,
        deconv_eps: float = 1e-6,
        pseudoinv_mode: str = "auto",
        cond_threshold: float = 1e6,
        device: Optional[torch.device] = None,
    ) -> None:
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve spatial shape.")
        if kernel_size_t % 2 == 0:
            raise ValueError("kernel_size_t must be odd to preserve temporal shape.")
        self.blur_kernel = Blurkernel(
            blur_type=blur_type,
            kernel_size=kernel_size,
            std=std,
            device=device,
        )
        self.kernel_size = kernel_size
        self.kernel_size_t = kernel_size_t
        self.deconv_eps = deconv_eps
        self.pseudoinv_mode = pseudoinv_mode
        self.cond_threshold = cond_threshold

    def _get_kernel(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        kernel = self.blur_kernel.get_kernel().to(device=device, dtype=dtype)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)

    def _apply_blur(self, img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        # img: (T, C, H, W)
        t, c, h, w = img.shape
        pad = self.kernel_size // 2
        img_4d = img.view(t, c, h, w)
        img_4d = torch.nn.functional.pad(img_4d, (pad, pad, pad, pad), mode="reflect")
        weight = kernel.expand(c, 1, self.kernel_size, self.kernel_size)
        out = torch.nn.functional.conv2d(img_4d, weight, stride=1, padding=0, groups=c)
        return out

    def _temporal_blur(self, img_tchw: torch.Tensor) -> torch.Tensor:
        img_bcthw = img_tchw.permute(1, 0, 2, 3).unsqueeze(0)
        blurred = temporal_blur(img_bcthw, self.kernel_size_t)
        return blurred.squeeze(0).permute(1, 0, 2, 3)

    def _temporal_deconv(self, meas_tchw: torch.Tensor) -> torch.Tensor:
        meas_bcthw = meas_tchw.permute(1, 0, 2, 3).unsqueeze(0)
        _, _, t, _, _ = meas_bcthw.shape
        kernel = uniform_kernel_1d(self.kernel_size_t, dtype=meas_bcthw.dtype).to(meas_bcthw.device)
        kernel_padded = torch.zeros(t, dtype=meas_bcthw.dtype, device=meas_bcthw.device)
        kernel_padded[: self.kernel_size_t] = kernel
        kernel_padded = torch.roll(kernel_padded, -(self.kernel_size_t // 2))

        kernel_fft = torch.fft.fft(kernel_padded)
        meas_fft = torch.fft.fft(meas_bcthw, dim=2)
        denom = kernel_fft.abs() ** 2 + self.deconv_eps
        inv_filter = kernel_fft.conj() / denom
        deconv = torch.fft.ifft(meas_fft * inv_filter[None, None, :, None, None], dim=2).real
        return deconv.squeeze(0).permute(1, 0, 2, 3)

    def __call__(self, img: torch.Tensor, cdim: int = 1) -> torch.Tensor:
        assert img.ndim == 4, "Input image must be 4D tensor (T, H, W, C) or (T, C, H, W)."
        img = img.movedim(cdim, 1)  # (T, C, H, W)
        blurred_t = self._temporal_blur(img)
        kernel = self._get_kernel(img.device, img.dtype)
        out = self._apply_blur(blurred_t, kernel)
        return out.movedim(1, cdim)

    def adjoint(self, meas: torch.Tensor, cdim: int = 1) -> torch.Tensor:
        assert meas.ndim == 4, "Measurement must be 4D tensor (T, H, W, C) or (T, C, H, W)."
        meas = meas.movedim(cdim, 1)  # (T, C, H, W)
        kernel = self._get_kernel(meas.device, meas.dtype)
        kernel = torch.flip(kernel, dims=(-2, -1))
        spatial_adj = self._apply_blur(meas, kernel)
        out = self._temporal_blur(spatial_adj)
        return out.movedim(1, cdim)

    def pseudoinv(self, meas: torch.Tensor, cdim: int = 1) -> torch.Tensor:
        assert meas.ndim == 4, "Measurement must be 4D tensor (T, H, W, C) or (T, C, H, W)."
        meas = meas.movedim(cdim, 1)  # (T, C, H, W)
        deconv_t = self._temporal_deconv(meas)
        t, c, h, w = deconv_t.shape
        pad = self.kernel_size // 2
        padded = torch.nn.functional.pad(deconv_t, (pad, pad, pad, pad), mode="reflect")
        hp, wp = padded.shape[-2:]

        kernel = self._get_kernel(deconv_t.device, deconv_t.dtype).squeeze(0).squeeze(0)
        kernel_padded = torch.zeros((hp, wp), device=deconv_t.device, dtype=deconv_t.dtype)
        kh, kw = kernel.shape
        kernel_padded[:kh, :kw] = kernel
        kernel_padded = torch.roll(kernel_padded, shifts=(-(kh // 2), -(kw // 2)), dims=(0, 1))

        kernel_fft = torch.fft.fft2(kernel_padded)
        if self.pseudoinv_mode not in {"auto", "fft", "adjoint"}:
            raise ValueError(f"Unknown pseudoinv_mode: {self.pseudoinv_mode}")

        if self.pseudoinv_mode in {"auto", "fft"}:
            denom = kernel_fft.abs() ** 2 + self.deconv_eps
            cond = (denom.max() / denom.min()).item()
            if self.pseudoinv_mode == "fft" or cond <= self.cond_threshold:
                meas_fft = torch.fft.fft2(padded, dim=(-2, -1))
                inv_filter = kernel_fft.conj() / denom
                deconv = torch.fft.ifft2(meas_fft * inv_filter[None, None, :, :], dim=(-2, -1)).real
                deconv = deconv[..., pad:pad + h, pad:pad + w]
                return deconv.movedim(1, cdim)

        adj = self.adjoint(meas.movedim(1, cdim), cdim=cdim)
        return adj


def uniform_kernel_1d(kernel_size: int, dtype=torch.float32):
    """Generate a 1D uniform blur kernel."""
    if kernel_size <= 0:
        raise ValueError("Kernel size must be positive")
    
    kernel = torch.ones(kernel_size, dtype=dtype)
    kernel = kernel / kernel.sum()
    return kernel


def temporal_blur(video_tensor: torch.Tensor, kernel_size_t: int):
    """
    Apply spatio-temporal Gaussian blur to a video tensor.

    Parameters:
    - video_tensor: A tensor of shape (B, C, T, H, W) where B is the batch size,
      T is the number of frames, C is the number of channels, H is the height, and W is the width.
    - kernel_size_t: Size of the Gaussian kernel in the temporal dimension.

    Returns:
    - Blurred video tensor of the same shape as the input.
    """
    device = video_tensor.device
    dtype = video_tensor.dtype
    B, C, T, H, W = video_tensor.shape

    # Generate Gaussian kernels for each dimension
    kernel_t = uniform_kernel_1d(kernel_size_t, dtype=dtype).to(device).view(1, 1, kernel_size_t, 1, 1)

    padding_t = kernel_size_t // 2

    # Apply temporal blur
    video_tensor = torch.nn.functional.pad(video_tensor, (0, 0, 0, 0, padding_t, padding_t), mode='circular')
    video_tensor = torch.nn.functional.conv3d(video_tensor.view(B * C, 1, T + 2 * padding_t, H, W), kernel_t, padding=0, groups=1).view(B, C, T, H, W)

    return video_tensor


def get_forward_model(name: str, **kwargs) -> Optional[LinearForwardModel]:
    if name == "bicubic_downsampling":
        return TemporalBicubicDownsamplingModel(**kwargs)
    elif name == "blurring":
        return TemporalBlurringModel(**kwargs)
    elif name == "cmr_sim":
        return CMRSimForwardModel(**kwargs)
    elif name == "identity":
        return IdentityForwardModel(**kwargs)
    else:
        raise ValueError(f"Unknown simulation model: {name}")
