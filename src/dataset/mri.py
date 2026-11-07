import torch


def fftnc(data: torch.Tensor, norm='ortho', dim=(1, 2), use_rfft: bool = False) -> torch.Tensor:
    """
    Apply centered n-dimensional Fast Fourier Transform.

    Parameters:
        data: Complex valued input data
        norm: Normalization mode
        dim: dimensions to be shifted (i.e. `ndim`-dimensional Fourier shift)

    Returns:
        The FFT of the input.
    """
    data = torch.fft.ifftshift(data, dim=dim)
    if use_rfft:
        data = torch.fft.rfftn(data, norm=norm, dim=dim)
    else:
        data = torch.fft.fftn(data, norm=norm, dim=dim)
    data = torch.fft.fftshift(data, dim=dim)

    return data

def ifftnc(data: torch.Tensor, norm='ortho', dim=(1, 2), use_rfft: bool = False) -> torch.Tensor:
    """
    Apply centered n-dimensional Inverse Fast Fourier Transform.

    Parameters:
        data: Complex valued input data
        norm: Normalization mode
        dim: dimensions to be shifted (i.e. `ndim`-dimensional Fourier shift)

    Returns:
        The IFFT of the input.
    """
    data = torch.fft.ifftshift(data, dim=dim)
    if use_rfft:
        data = torch.fft.irfftn(data, norm=norm, dim=dim)
    else:
        data = torch.fft.ifftn(data, norm=norm, dim=dim)
    data = torch.fft.fftshift(data, dim=dim)

    return data

def optimal_svd_phase_correction(masked_kspace: torch.Tensor) -> torch.Tensor:
    """
    Removes global per-frame phase drift using a temporal SVD of the k-space center (DC) across coils.
    Assumes centered k-space (fftnc/ifftnc convention). Does not change magnitude images.

    Args:
        masked_kspace: [B, C, T, RO, PH, 2] (or complex [B, C, T, RO, PH])
    Returns:
        corrected_kspace: same shape and dtype convention as the input
    """
    if not masked_kspace.is_complex():
        k_complex = torch.view_as_complex(masked_kspace)
    else:
        k_complex = masked_kspace

    B, C, T, RO, PH = k_complex.shape
    center_ro, center_ph = RO // 2, PH // 2

    # 1. Extract the DC Component -> Shape: [B, C, T]
    dc_components = k_complex[:, :, :, center_ro, center_ph]

    # 2. Perform SVD natively in PyTorch
    U, S, Vh = torch.linalg.svd(dc_components, full_matrices=False)

    # 3. Extract the 1st Temporal Principal Component
    dominant_temporal = Vh[:, 0, :]
    raw_phase = torch.angle(dominant_temporal)

    # 4. Anchor to Frame 0 (cancels the SVD's arbitrary overall phase-gauge freedom)
    phase_drift = raw_phase - raw_phase[:, 0:1]

    # 5. Apply the Inverse Phase Rotation
    phase_correction = torch.exp(-1j * phase_drift).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    corrected_k_complex = k_complex * phase_correction

    if not masked_kspace.is_complex():
        return torch.view_as_real(corrected_k_complex)
    return corrected_k_complex


def rss(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS).

    Args:
        data: The complex input tensor.
        dim: The dimensions along which to apply the RSS transform (coil dimension).

    Returns:
        The RSS image.
    """

    return torch.sqrt((data.abs() ** 2).sum(dim))

def adaptive_combine(data: torch.Tensor, sens_maps: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute a MVUE image by combining coil images with coil sensitivities.

    Args:
        data: The individual coil images.
        coil_sens: The coil sensitivity maps.
        dim: The dimension along which to combine the images (coil dimension).

    Returns:
        The MVUE image.
    """

    return (data * torch.conj(sens_maps)).sum(dim)


def kspace2image(kspace: torch.Tensor, norm='ortho', fdim=(1, 2), cdim=0, adaptive=True, sens_maps: torch.Tensor=None) -> torch.Tensor:
    """
    Compute the inverse Fourier transform of the k-space data and combine these coil images to one image.

    Args:
        kspace: The input k-space data.
        norm: The normalization mode.
        fdim: The dimensions to be shifted (i.e. `ndim`-dimensional Fourier shift).
        cdim: The coil dimension.
        adaptive: Whether to use adaptive coil combination or RSS
        sens_maps: The coil sensitivity maps.
    Returns:
        The image representation of the input k-space data.
    """
    coil_images = ifftnc(kspace, norm=norm, dim=fdim)

    if adaptive:
        return adaptive_combine(coil_images, sens_maps, dim=cdim)
    else:
        return rss(coil_images, dim=cdim)