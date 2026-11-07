"""
Vimeo-90K septuplet dataset loader (http://toflow.csail.mit.edu/index.html#septuplet).
Layout: <root>/sequences/<folder>/<subfolder>/im1..im7.png plus sep_{train,test}list.txt.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode
from src.domain_specific_setups.natural_video_forward_models import get_forward_model

def get_simulation_model(name: str, **kwargs):
    return get_forward_model(name, **kwargs)

class VimeoDataset(Dataset):
    """Vimeo-90K septuplet sequences for video super-resolution.

    Args:
            root: Base directory containing ``sequences`` and the split list file.
            list_file: Name or path to a split file (e.g., ``sep_testlist.txt``).
            scale_factor: Downscale factor used to create LR inputs if no explicit
                    low-resolution folder exists. Defaults to 4.
            cache: If True, stores decoded frames in RAM after first access.
            center_only: If True, returns only the center frame (im4) for both HR
                    and LR; useful for single-frame SR baselines.

    Returns per sample (dict):
            - ``lr``: Tensor (T, 3, H/scale, W/scale) in [0, 1].
            - ``hr``: Tensor (T, 3, H, W) in [0, 1].
            - ``name``: Sequence identifier string ``<folder>/<subfolder>``.
    """

    def __init__(
        self,
        base_path: str,
        simulation_cfg: Optional[dict] = None,
        list_file: str = "seq_testlist.txt",
        target_folder: str = "sequences",  # "target" for the SR task
        measurement_folder: Optional[str] = None,  # "low_resolution" for the SR task
        cache: bool = False,
        center_only: bool = False,
        crop_and_downscale_cfg : Optional[dict] = None,
    ) -> None:
        super().__init__()

        self.root = Path(base_path)
        self.cache = cache
        self.center_only = center_only
        self.target_folder = target_folder
        self.measurement_folder = measurement_folder

        self.target_sequences_path = self.root / self.target_folder
        if not self.target_sequences_path.exists():
            raise FileNotFoundError(
                f"Expected sequences under {self.target_sequences_path}. Please place the extracted Vimeo-90K septuplet data there."
            )

        list_path = Path(list_file)
        if not list_path.is_absolute():
            list_path = self.root / list_file
        if not list_path.exists():
            raise FileNotFoundError(f"List file {list_path} not found.")

        with open(list_path, "r") as f:
            self.sequence_names = [line.strip() for line in f if line.strip()]

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
        frame_paths = [folder / f"im{i}.png" for i in range(1, 8)]
        if not all(p.exists() for p in frame_paths):
            missing = [str(p) for p in frame_paths if not p.exists()]
            raise FileNotFoundError(f"Missing frames in {folder}: {missing}")

        hr_frames: List[torch.Tensor] = [self._load_rgb(p) for p in frame_paths]
        hr = torch.stack(hr_frames, dim=0)  # (T, 3, H, W)

        if self.forward_model is None:
            assert self.measurement_folder is not None, "Either measurement_folder or simulation_cfg must be provided."
            assert self.crop_and_downscale_cfg is None or self.crop_and_downscale_cfg.get("enabled", False) is False, "crop_and_downscale_cfg cannot be used with precomputed measurements."
            measurement_sequences_path = self.root / self.measurement_folder
            measurement_folder = measurement_sequences_path / seq_name
            lr_frame_paths = [measurement_folder / f"im{i}.png" for i in range(1, 8)]
            if not all(p.exists() for p in lr_frame_paths):
                missing = [str(p) for p in lr_frame_paths if not p.exists()]
                raise FileNotFoundError(f"Missing LR frames in {measurement_folder}: {missing}")
            lr_frames: List[torch.Tensor] = [self._load_rgb(p) for p in lr_frame_paths]
            lr = torch.stack(lr_frames, dim=0)  # (T, 3, H/scale, W/scale)
        else:
            # simulation model
            if self.crop_and_downscale_cfg is not None and self.crop_and_downscale_cfg.get("enabled", False):
                # apply downscaling on hr before applying fwd model
                pass
                from src.utils.video_crop_transforms import SpatialCenterCropAndDownscale
                transform = SpatialCenterCropAndDownscale(**self.crop_and_downscale_cfg)
                hr = transform(hr, torch.zeros_like(hr))["video"] # cropping and downscaling
            lr = self.forward_model(hr, cdim=1)

        sample = {"hr": hr, "lr": lr}
        if self.cache:
            self._cache[seq_name] = sample
        return sample

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        seq_name = self.sequence_names[idx]
        sample = self._load_sequence(seq_name)
        return sample["lr"], sample["hr"]