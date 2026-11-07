"""Video normalization (fit / normalize / denormalize / serialize) shared by preprocessing, training viz and inference."""

from __future__ import annotations

import glob
import json
import os
from typing import Optional

import numpy as np
import torch

__all__ = ["VideoNormalization"]

_CHANNEL_MODES = {"video_channel_mean_div_std", "frame0_channel_mean_div_std"}
_SCALAR_MODES  = {"video_div_std", "video_div_max"}
_STD_FLOOR_DEFAULT = 1e-3


class VideoNormalization:
    """Encapsulates video normalization: fit, normalize, denormalize, serialize."""

    SUPPORTED_MODES = [
        "none",
        "video_div_std",
        "video_div_max",
        "video_channel_mean_div_std",
        "frame0_channel_mean_div_std",
        "times2_minus1",
    ]

    def __init__(
        self,
        mode: str,
        scaling_factor: float = 1.0,
        channel_mean: Optional[np.ndarray] = None,
        channel_std:  Optional[np.ndarray] = None,
        std_floor: float = _STD_FLOOR_DEFAULT,
    ):
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unknown normalization mode {mode!r}. "
                f"Supported: {self.SUPPORTED_MODES}"
            )
        self.mode           = mode
        self.scaling_factor = float(scaling_factor)
        self.channel_mean   = np.asarray(channel_mean, dtype=np.float32) if channel_mean is not None else None
        self.channel_std    = np.asarray(channel_std,  dtype=np.float32) if channel_std  is not None else None
        self.std_floor      = float(std_floor)

    @classmethod
    def fit_fchw(
        cls,
        mode: str,
        video_fchw: torch.Tensor,
        std_floor: float = _STD_FLOOR_DEFAULT,
    ) -> "VideoNormalization":
        """Fit from a (F, C, H, W) tensor."""
        if mode == "none" or mode == "times2_minus1":
            return cls(mode=mode, std_floor=std_floor)

        if mode == "video_div_std":
            sf = 1.0 / max(float(video_fchw.std().item()), std_floor)
            return cls(mode=mode, scaling_factor=sf, std_floor=std_floor)

        if mode == "video_div_max":
            sf = 1.0 / max(float(video_fchw.abs().max().item()), std_floor)
            return cls(mode=mode, scaling_factor=sf, std_floor=std_floor)

        if mode == "video_channel_mean_div_std":
            # mean/std over (F, H, W) per channel
            mean = video_fchw.mean(dim=(0, 2, 3))                                       # (C,)
            std  = video_fchw.std(dim=(0, 2, 3), unbiased=False).clamp_min(std_floor)   # (C,)
            return cls(
                mode=mode,
                channel_mean=mean.detach().cpu().numpy(),
                channel_std =std.detach().cpu().numpy(),
                std_floor=std_floor,
            )

        if mode == "frame0_channel_mean_div_std":
            # mean/std only from frame 0, over (H, W) per channel
            f0   = video_fchw[0]                                                         # (C, H, W)
            mean = f0.mean(dim=(1, 2))                                                   # (C,)
            std  = f0.std(dim=(1, 2), unbiased=False).clamp_min(std_floor)               # (C,)
            return cls(
                mode=mode,
                channel_mean=mean.detach().cpu().numpy(),
                channel_std =std.detach().cpu().numpy(),
                std_floor=std_floor,
            )

        raise ValueError(f"Unknown normalization mode: {mode!r}")

    @classmethod
    def fit_thwc(
        cls,
        mode: str,
        video_thwc: torch.Tensor,
        std_floor: float = _STD_FLOOR_DEFAULT,
    ) -> "VideoNormalization":
        """Fit from a (T, H, W, C) tensor (preprocessing layout)."""
        return cls.fit_fchw(mode=mode, video_fchw=video_thwc.movedim(-1, 1), std_floor=std_floor)

    def _mean_std_tensors(self, device, dtype, layout_4d: str = "fchw"):
        """Return (mean, std) tensors broadcast-ready for a 4-D tensor."""
        mean = torch.as_tensor(self.channel_mean, device=device, dtype=dtype)
        std  = torch.as_tensor(self.channel_std,  device=device, dtype=dtype)
        if layout_4d == "fchw":           # (1, C, 1, 1)
            return mean.view(1, -1, 1, 1), std.view(1, -1, 1, 1)
        elif layout_4d == "thwc":         # (1, 1, 1, C)
            return mean.view(1, 1, 1, -1), std.view(1, 1, 1, -1)
        raise ValueError(f"Unknown layout: {layout_4d!r}")

    def normalize_fchw(self, video: torch.Tensor) -> torch.Tensor:
        """Normalize a (F, C, H, W) tensor."""
        if self.mode == "none":
            return video
        if self.mode in _SCALAR_MODES:
            return video * self.scaling_factor
        if self.mode in _CHANNEL_MODES:
            mean, std = self._mean_std_tensors(video.device, video.dtype, "fchw")
            return (video - mean) / std
        if self.mode == "times2_minus1":
            return video * 2.0 - 1.0
        raise ValueError(f"Unsupported mode: {self.mode!r}")

    def denormalize_fchw(self, video: torch.Tensor) -> torch.Tensor:
        """Invert normalization on a (F, C, H, W) tensor."""
        if self.mode == "none":
            return video
        if self.mode in _SCALAR_MODES:
            return video / (self.scaling_factor + 1e-12)
        if self.mode in _CHANNEL_MODES:
            mean, std = self._mean_std_tensors(video.device, video.dtype, "fchw")
            return video * std + mean
        if self.mode == "times2_minus1":
            return (video + 1.0) / 2.0
        raise ValueError(f"Unsupported mode: {self.mode!r}")

    def normalize_thwc(self, video: torch.Tensor) -> torch.Tensor:
        """Normalize a (T, H, W, C) tensor."""
        if self.mode == "none":
            return video
        if self.mode in _SCALAR_MODES:
            return video * self.scaling_factor
        if self.mode in _CHANNEL_MODES:
            mean, std = self._mean_std_tensors(video.device, video.dtype, "thwc")
            return (video - mean) / std
        if self.mode == "times2_minus1":
            return video * 2.0 - 1.0
        raise ValueError(f"Unsupported mode: {self.mode!r}")


    def denormalize_chw(self, frame: torch.Tensor) -> torch.Tensor:
        """Invert normalization on a single (C, H, W) frame."""
        return self.denormalize_fchw(frame.unsqueeze(0)).squeeze(0)

    def to_dict(self) -> dict:
        d: dict = {"normalization_type": self.mode, "std_floor": self.std_floor}
        if self.scaling_factor != 1.0:
            d["scaling_factor"] = self.scaling_factor
        if self.channel_mean is not None:
            d["channel_mean"] = self.channel_mean.tolist()
        if self.channel_std is not None:
            d["channel_std"] = self.channel_std.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "VideoNormalization":
        return cls(
            mode=str(d["normalization_type"]),
            scaling_factor=float(d.get("scaling_factor", 1.0)),
            channel_mean=np.array(d["channel_mean"]) if "channel_mean" in d else None,
            channel_std =np.array(d["channel_std"])  if "channel_std"  in d else None,
            std_floor=float(d.get("std_floor", _STD_FLOOR_DEFAULT)),
        )

    def save_to_h5_attrs(self, f) -> None:
        """Persist into an open h5py file's root attrs.
        
        Also writes legacy keys so old analysis code still finds values.
        """
        d = self.to_dict()
        f.attrs["normalization_type"]      = d["normalization_type"]
        f.attrs["normalization_std_floor"] = d["std_floor"]
        if "scaling_factor" in d:
            f.attrs["normalization_scaling_factor"] = d["scaling_factor"]
            f.attrs["scaling_factor"] = d["scaling_factor"]   # legacy key
        if "channel_mean" in d:
            arr = np.array(d["channel_mean"], dtype=np.float32)
            f.attrs["normalization_channel_mean"] = arr
            # legacy keys
            if d["normalization_type"] == "video_channel_mean_div_std":
                f.attrs["video_channel_mean"]      = arr
                f.attrs["video_channel_std"]       = np.array(d["channel_std"], dtype=np.float32)
                f.attrs["video_channel_std_floor"] = d["std_floor"]
            elif d["normalization_type"] == "frame0_channel_mean_div_std":
                f.attrs["frame0_channel_mean"]      = arr
                f.attrs["frame0_channel_std"]       = np.array(d["channel_std"], dtype=np.float32)
                f.attrs["frame0_channel_std_floor"] = d["std_floor"]
            f.attrs["normalization_channel_std"] = np.array(d["channel_std"], dtype=np.float32)

    @classmethod
    def from_h5_attrs(cls, attrs: dict) -> "VideoNormalization":
        """Reconstruct from an attrs dict (as returned by h5py or dataset sample)."""
        if "normalization_type" in attrs:
            mode      = str(attrs["normalization_type"])
            std_floor = float(attrs.get("normalization_std_floor", _STD_FLOOR_DEFAULT))
            sf        = float(attrs.get("normalization_scaling_factor", attrs.get("scaling_factor", 1.0)))
            mean = np.array(attrs["normalization_channel_mean"]) if "normalization_channel_mean" in attrs else None
            std  = np.array(attrs["normalization_channel_std"])  if "normalization_channel_std"  in attrs else None
            return cls(mode=mode, scaling_factor=sf, channel_mean=mean, channel_std=std, std_floor=std_floor)

        # --- legacy fallback: pre-VideoNormalization attr layout ---
        if "video_channel_mean" in attrs and "video_channel_std" in attrs:
            return cls(
                mode="video_channel_mean_div_std",
                channel_mean=np.array(attrs["video_channel_mean"]),
                channel_std =np.array(attrs["video_channel_std"]),
                std_floor=float(attrs.get("video_channel_std_floor", _STD_FLOOR_DEFAULT)),
            )
        if "frame0_channel_mean" in attrs and "frame0_channel_std" in attrs:
            return cls(
                mode="frame0_channel_mean_div_std",
                channel_mean=np.array(attrs["frame0_channel_mean"]),
                channel_std =np.array(attrs["frame0_channel_std"]),
                std_floor=float(attrs.get("frame0_channel_std_floor", _STD_FLOOR_DEFAULT)),
            )
        sf = float(attrs.get("scaling_factor", 1.0))
        if sf != 1.0:
            return cls(mode="video_div_std", scaling_factor=sf)
        return cls(mode="none")

    _FILENAME = "normalization.json"

    def save_to_dir(self, directory: str) -> None:
        """Write normalization.json into *directory*."""
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, self._FILENAME)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def from_dir(cls, directory: str) -> "VideoNormalization":
        """Load from normalization.json in *directory*."""
        path = os.path.join(directory, cls._FILENAME)
        with open(path) as fh:
            return cls.from_dict(json.load(fh))


    @classmethod
    def read_from_h5_dir(cls, data_dir: str) -> Optional["VideoNormalization"]:
        """Peek at the first .h5 file in *data_dir* and read normalization attrs."""
        import h5py
        files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
        if not files:
            return None
        with h5py.File(files[0], "r") as f:
            return cls.from_h5_attrs(dict(f.attrs))

    def __repr__(self) -> str:
        extra = ""
        if self.channel_mean is not None:
            extra = f", channel_mean={self.channel_mean}, channel_std={self.channel_std}"
        elif self.scaling_factor != 1.0:
            extra = f", scaling_factor={self.scaling_factor}"
        return f"VideoNormalization(mode={self.mode!r}{extra})"
