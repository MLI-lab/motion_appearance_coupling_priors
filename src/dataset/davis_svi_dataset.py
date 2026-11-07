"""The DAVIS-SVI clip set: 338 pre-cut 16-frame 256x256 DAVIS 2017 clips, one (3, T, H, W) .npy in [0, 1] each.

Different clip set from DAVIS2017Dataset (indices do not correspond); sample_index i is the i-th .npy in sorted order.
"""

import os
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset.reds_dataset import get_simulation_model


def _npy_files(root: str) -> List[str]:
    return sorted(
        os.path.join(dirpath, f)
        for dirpath, _, files in os.walk(root)
        for f in files
        if f.endswith(".npy")
    )


class DAVISSVIDataset(Dataset):
    def __init__(
        self,
        base_path: str,
        simulation_cfg: Optional[dict] = None,
        crop_and_downscale_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.video_paths = _npy_files(base_path)
        if not self.video_paths:
            raise FileNotFoundError(f"No .npy clips under {base_path} (expected the DAVIS-SVI dump, see davis_svi.yaml).")
        self.crop_and_downscale_cfg = crop_and_downscale_cfg
        self.forward_model = get_simulation_model(**simulation_cfg) if simulation_cfg is not None else None

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int):
        hr = torch.from_numpy(np.load(self.video_paths[idx], allow_pickle=False)).permute(1, 0, 2, 3)  # (C,T,H,W) -> (T,C,H,W)

        if self.crop_and_downscale_cfg is not None and self.crop_and_downscale_cfg.get("enabled", False):
            from src.utils.video_crop_transforms import SpatialCenterCropAndDownscale

            transform = SpatialCenterCropAndDownscale(**self.crop_and_downscale_cfg)
            hr = transform(hr, torch.zeros_like(hr))["video"]

        if self.forward_model is not None:
            return self.forward_model(hr, cdim=1), hr
        return hr
