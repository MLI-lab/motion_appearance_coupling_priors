"""DAVIS 2017 (trainval, 480p) cut into fixed-length natural-video clips.

Each video is split into `clip_length`-frame windows, resized and center-cropped; hr is (T, 3, H, W) in [0, 1].
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from src.dataset.reds_dataset import _frame_sort_key, get_simulation_model


class DAVIS2017Dataset(Dataset):
    def __init__(
        self,
        base_path: str,
        simulation_cfg: Optional[dict] = None,
        split: str = "trainval",
        clip_length: int = 16,
        clip_stride: Optional[int] = None,
        resize_short_side: Optional[int] = 256,
        center_crop: Optional[int] = 256,
        resolution: str = "480p",
        frame_glob: str = "*.jpg",
        crop_and_downscale_cfg: Optional[dict] = None,
        cache: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(base_path)
        self.frames_root = self.root / "JPEGImages" / resolution
        self.clip_length = int(clip_length)
        self.clip_stride = int(clip_stride or clip_length)
        self.resize_short_side = resize_short_side
        self.center_crop = center_crop
        self.frame_glob = frame_glob
        self.crop_and_downscale_cfg = crop_and_downscale_cfg
        self.cache = cache
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}

        if not self.frames_root.is_dir():
            raise FileNotFoundError(
                f"Expected DAVIS frames under {self.frames_root} (JPEGImages/{resolution}/<video>/*.jpg). "
                "Run the davis_2017 download config, or check base_path."
            )
        video_names: List[str] = []
        splits = ["train", "val"] if split == "trainval" else [split]
        for s in splits:
            list_path = self.root / "ImageSets" / "2017" / f"{s}.txt"
            if not list_path.is_file():
                raise FileNotFoundError(f"DAVIS split file {list_path} not found.")
            with open(list_path) as f:
                video_names += [line.strip() for line in f if line.strip()]

        # (video, first frame index) per clip, so __len__/__getitem__ are cheap.
        self.clips: List[Tuple[str, int]] = []
        self.frame_paths: Dict[str, List[Path]] = {}
        for name in video_names:
            paths = sorted((self.frames_root / name).glob(self.frame_glob), key=_frame_sort_key)
            if not paths:
                raise FileNotFoundError(f"No frames matching '{self.frame_glob}' in {self.frames_root / name}.")
            self.frame_paths[name] = paths
            for start in range(0, len(paths) - self.clip_length + 1, self.clip_stride):
                self.clips.append((name, start))

        self.forward_model = get_simulation_model(**simulation_cfg) if simulation_cfg is not None else None

    def __len__(self) -> int:
        return len(self.clips)

    def _load_frame(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        if self.resize_short_side:
            img = TF.resize(img, int(self.resize_short_side), interpolation=InterpolationMode.BICUBIC, antialias=True)
        if self.center_crop:
            img = TF.center_crop(img, int(self.center_crop))
        return TF.to_tensor(img)  # (3, H, W) in [0, 1]

    def _load_clip(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cache and idx in self._cache:
            return self._cache[idx]
        name, start = self.clips[idx]
        paths = self.frame_paths[name][start:start + self.clip_length]
        hr = torch.stack([self._load_frame(p) for p in paths], dim=0)  # (T, 3, H, W)

        if self.crop_and_downscale_cfg is not None and self.crop_and_downscale_cfg.get("enabled", False):
            from src.utils.video_crop_transforms import SpatialCenterCropAndDownscale

            transform = SpatialCenterCropAndDownscale(**self.crop_and_downscale_cfg)
            hr = transform(hr, torch.zeros_like(hr))["video"]

        sample = {"hr": hr}
        if self.forward_model is not None:
            sample["lr"] = self.forward_model(hr, cdim=1)
        if self.cache:
            self._cache[idx] = sample
        return sample

    def __getitem__(self, idx: int):
        sample = self._load_clip(idx)
        if self.forward_model is not None:
            return sample["lr"], sample["hr"]
        return sample["hr"]
