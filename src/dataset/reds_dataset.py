"""
REDS (Nah et al., NTIRE 2019) video dataset loader, https://huggingface.co/datasets/snah/REDS.
Layout: <base_path>/{train/train_sharp, val/val_sharp}/<3-digit video_id>/<8-digit frame>.png
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

from torchvision.transforms import functional as TF
from src.domain_specific_setups.natural_video_forward_models import get_forward_model


def get_simulation_model(name: str, **kwargs):
    return get_forward_model(name, **kwargs)


def _frame_sort_key(p: Path):
    """Numeric-aware sort key (handles non-zero-padded frame names); lexicographic if no digits."""
    m = re.search(r"(\d+)", p.stem)
    return (0, int(m.group(1))) if m else (1, p.stem)


class RedsDataset(Dataset):
    """REDS sequences for natural video reconstruction.

    Args:
        base_path: Dataset root containing split folders (train/val).
        split: Which split to load ("train" or "val").
        list_file: Optional text file listing video-id folders, one per line.
        target_folder: Optional subfolder under split for HR frames (e.g. "val_sharp").
        measurement_folder: Optional subfolder under split for precomputed LR frames.
        cache: If True, stores decoded frames in RAM after first access.
        center_only: If True, returns only the temporal center frame.
        crop_and_downscale_cfg: Optional spatial crop/downscale config.
        frame_glob: Glob pattern for frames inside each video folder.

    Returns per sample:
        - ``lr``: Tensor (T, 3, H/scale, W/scale) in [0, 1].
        - ``hr``: Tensor (T, 3, H, W) in [0, 1].
    """

    def __init__(
        self,
        base_path: str,
        simulation_cfg: Optional[dict] = None,
        split: str = "val",
        list_file: Optional[str] = None,
        target_folder: Optional[str] = None,
        measurement_folder: Optional[str] = None,
        cache: bool = False,
        center_only: bool = False,
        crop_and_downscale_cfg: Optional[dict] = None,
        temporal_interpolation_factor: Optional[float] = None,
        temporal_cropping_lb: Optional[int] = None,
        temporal_cropping_ub: Optional[int] = None,
        frame_glob: str = "*.png",
    ) -> None:
        super().__init__()

        self.root = Path(base_path)
        self.cache = cache
        self.center_only = center_only
        self.split = split
        self.target_folder = target_folder
        self.measurement_folder = measurement_folder
        self.frame_glob = frame_glob

        self.temporal_interpolation_factor = temporal_interpolation_factor
        self.temporal_cropping_lb = temporal_cropping_lb
        self.temporal_cropping_ub = temporal_cropping_ub

        split_root = self.root / self.split
        self.target_sequences_path = split_root / self.target_folder if self.target_folder else split_root
        if not self.target_sequences_path.exists():
            raise FileNotFoundError(
                f"Expected video folders under {self.target_sequences_path}. Please place the extracted REDS data there."
            )

        if list_file is not None:
            list_path = Path(list_file)
            if not list_path.is_absolute():
                list_path = self.root / list_file
            if not list_path.exists():
                raise FileNotFoundError(f"List file {list_path} not found.")

            with open(list_path, "r") as f:
                self.sequence_names = [line.strip() for line in f if line.strip()]
        else:
            self.sequence_names = sorted(
                [p.name for p in self.target_sequences_path.iterdir() if p.is_dir()]
            )

        self._cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self.forward_model = get_simulation_model(**simulation_cfg) if simulation_cfg is not None else None
        self.crop_and_downscale_cfg = crop_and_downscale_cfg

    def __len__(self) -> int:
        return len(self.sequence_names)

    def _load_rgb(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return TF.to_tensor(img)  # (3, H, W) in [0, 1]

    def _load_sequence(self, seq_name: str) -> Dict[str, torch.Tensor]:
        if self.cache and seq_name in self._cache:
            return self._cache[seq_name]

        folder = self.target_sequences_path / seq_name
        frame_paths = sorted(folder.glob(self.frame_glob), key=_frame_sort_key)
        if len(frame_paths) == 0:
            raise FileNotFoundError(f"No frames matching '{self.frame_glob}' in {folder}.")

        hr_frames: List[torch.Tensor] = [self._load_rgb(p) for p in frame_paths]
        hr = torch.stack(hr_frames, dim=0)  # (T, 3, H, W)

        if self.forward_model is None:
            assert self.measurement_folder is not None, "Either measurement_folder or simulation_cfg must be provided."
            assert self.temporal_cropping_lb is None, "temporal_cropping cannot be used with precomputed measurements."
            assert self.temporal_cropping_ub is None, "temporal_cropping cannot be used with precomputed measurements."
            assert self.temporal_interpolation_factor is None, "temporal_interpolation_factor cannot be used with precomputed measurements."
            assert self.crop_and_downscale_cfg is None or self.crop_and_downscale_cfg.get("enabled", False) is False, "crop_and_downscale_cfg cannot be used with precomputed measurements."
            measurement_sequences_path = (self.root / self.split / self.measurement_folder)
            measurement_folder = measurement_sequences_path / seq_name
            lr_frame_paths = sorted(measurement_folder.glob(self.frame_glob), key=_frame_sort_key)
            if len(lr_frame_paths) == 0:
                raise FileNotFoundError(f"No LR frames matching '{self.frame_glob}' in {measurement_folder}.")
            lr_frames: List[torch.Tensor] = [self._load_rgb(p) for p in lr_frame_paths]
            lr = torch.stack(lr_frames, dim=0)  # (T, 3, H/scale, W/scale)
        else:
            # simulation model
            if self.crop_and_downscale_cfg is not None and self.crop_and_downscale_cfg.get("enabled", False):
                # apply downscaling on hr before applying fwd model
                from src.utils.video_crop_transforms import SpatialCenterCropAndDownscale
                transform = SpatialCenterCropAndDownscale(**self.crop_and_downscale_cfg)
                hr = transform(hr, torch.zeros_like(hr))["video"]  # cropping and downscaling

            if self.temporal_interpolation_factor is not None:
                hr = torch.nn.functional.interpolate(hr.movedim(0, 1)[None], size=(int(hr.shape[0] * self.temporal_interpolation_factor), hr.shape[2], hr.shape[3]), mode='trilinear', align_corners=True)[0].movedim(1, 0)

            if self.temporal_cropping_lb is not None and self.temporal_cropping_ub is not None:
                hr = hr[self.temporal_cropping_lb:self.temporal_cropping_ub]  # crop in time

            lr = self.forward_model(hr, cdim=1)

        if self.center_only:
            center = hr.shape[0] // 2
            hr = hr[center: center + 1]
            lr = lr[center: center + 1]

        sample = {"hr": hr, "lr": lr}
        if self.cache:
            self._cache[seq_name] = sample
        return sample

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_name = self.sequence_names[idx]
        sample = self._load_sequence(seq_name)
        return sample["lr"], sample["hr"]
