import os
import glob
import h5py
import torch
from collections import OrderedDict
from torch.utils.data import Dataset
import numpy as np
import re
from typing import Optional

class H5FileDataset(Dataset):

    def __init__(self, data_dir, file_filter = None, transform=None, device='cuda:0', key_filter : Optional[str] = None):
        self.data_dir = data_dir
        self.transform = transform
        self.device = device
        self.key_filter = key_filter

        # Parse files and group by case_id
        self.case_dict = {}
        h5_files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
        # prefilter h5 files if file_filter is provided
        if file_filter:
            h5_files_filtered = [f for f in h5_files if re.match(file_filter, os.path.basename(f))]
            if len(h5_files_filtered) < len(h5_files):
                print(f"Filtered {len(h5_files) - len(h5_files_filtered)} files using filter: {file_filter}")
            self.h5_files = h5_files_filtered
        else:
            self.h5_files = h5_files

    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, idx):
        fpath = self.h5_files[idx]
        with h5py.File(fpath, 'r') as f:
            if self.key_filter:
                keys = [k for k in f.keys() if re.match(self.key_filter, k)]
            else:
                keys = list(f.keys())

            # make sure types are ndarray
            for key in keys:
                if not isinstance(f[key][()], np.ndarray):
                    raise ValueError(f"Expected dataset {key} in file {fpath} to be a numpy array, but got {type(f[key][()])}.")
            data = {
                key : torch.from_numpy(f[key][()]).float().to(self.device) for key in keys
            }

            data["attrs"] = dict(f.attrs)

        if self.transform:
            return self.transform(**data)
        return data

from tqdm import tqdm
class ImagesFromH5VideoFileDataset(Dataset):
    # each h5 file holds a video (num_frames, height, width, num_coils); samples are individual frames
    def __init__(
        self,
        data_dir,
        file_filter=None,
        transform=None,
        key_filter: Optional[str] = None,
        device='cuda:0',
        frame_index_transform=None,
        time_slice_selector=None,
        video_key="video",
        index_cache_path: Optional[str] = None,
        index_cache_refresh: bool = False,
        index_cache_limit: int = -1,
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.device = device
        self.key_filter = key_filter
        self.video_key = video_key
        self.time_slice_selector = time_slice_selector
        self.index_cache_path = Path(index_cache_path) if index_cache_path else None
        self.index_cache_limit = int(index_cache_limit)

        # per-worker H5 handle cache (LRU); must start empty so forked workers inherit nothing
        self._open_files: OrderedDict[str, h5py.File] = OrderedDict()
        self._open_files_pid: Optional[int] = None

        # Parse files and group by case_id
        self.frame_index = []  # list of (file_path, frame_idx)
        if self.index_cache_path and self.index_cache_path.exists() and not index_cache_refresh:
            print(f"[ImagesFromH5VideoFileDataset][INDEX_CACHE_HIT] path={self.index_cache_path}")
            self.frame_index = torch.load(self.index_cache_path)
            if self.index_cache_limit > 0:
                self.frame_index = self.frame_index[: self.index_cache_limit]
                print(
                    f"[ImagesFromH5VideoFileDataset] Applied index_cache_limit={self.index_cache_limit} "
                    f"after cache load; entries={len(self.frame_index)}"
                )
            print(f"[ImagesFromH5VideoFileDataset] Loaded {len(self.frame_index)} frame-index entries from cache.")
        else:
            if self.index_cache_path:
                if index_cache_refresh and self.index_cache_path.exists():
                    print(f"[ImagesFromH5VideoFileDataset][INDEX_CACHE_REFRESH] path={self.index_cache_path}")
                else:
                    print(f"[ImagesFromH5VideoFileDataset][INDEX_CACHE_MISS] path={self.index_cache_path}")

            h5_files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
            # prefilter h5 files if file_filter is provided
            if file_filter:
                h5_files_filtered = [f for f in h5_files if re.match(file_filter, os.path.basename(f))]
                if len(h5_files_filtered) < len(h5_files):
                    print(f"Filtered {len(h5_files) - len(h5_files_filtered)} files using filter: {file_filter}")
                h5_files = h5_files_filtered

            for fpath in tqdm(h5_files, desc="Indexing H5 video files"):
                with h5py.File(fpath, 'r') as f:
                    data_shape = f[self.video_key].shape  # here we assume the first key has shape (T,...)
                    num_frames = data_shape[0]
                    for frame_idx in range(num_frames):
                        self.frame_index.append((fpath, frame_idx))

            if frame_index_transform:
                nr_before = len(self.frame_index)
                self.frame_index = frame_index_transform(self.frame_index)
                print(f"Transformed frame index with {frame_index_transform}, reduced from {nr_before} to {len(self.frame_index)} samples.")

            if self.index_cache_limit > 0:
                nr_before_limit = len(self.frame_index)
                self.frame_index = self.frame_index[: self.index_cache_limit]
                print(
                    f"Applied index_cache_limit={self.index_cache_limit}, "
                    f"reduced from {nr_before_limit} to {len(self.frame_index)} samples."
                )

            if self.index_cache_path:
                os.makedirs(self.index_cache_path.parent, exist_ok=True)
                torch.save(self.frame_index, self.index_cache_path)
                print(f"[ImagesFromH5VideoFileDataset][INDEX_CACHE_SAVE] path={self.index_cache_path}, entries={len(self.frame_index)}")

    def __len__(self):
        return len(self.frame_index)

    def _get_file(self, fpath: str) -> "h5py.File":
        """Return a per-process cached H5 file handle for *fpath* (LRU-capped to avoid fd exhaustion)."""
        _MAX_OPEN = 64  # keep at most this many H5 files open per worker

        current_pid = os.getpid()
        if self._open_files_pid != current_pid:
            # new process (e.g. forked worker): drop inherited handles
            for f in self._open_files.values():
                try:
                    f.close()
                except Exception:
                    pass
            self._open_files: OrderedDict[str, h5py.File] = OrderedDict()
            self._open_files_pid = current_pid

        if fpath in self._open_files:
            # Move to end (most-recently used) for LRU ordering
            self._open_files.move_to_end(fpath)
            return self._open_files[fpath]

        # Evict least-recently-used handles if at capacity
        while len(self._open_files) >= _MAX_OPEN:
            _, oldest = self._open_files.popitem(last=False)
            try:
                oldest.close()
            except Exception:
                pass

        handle = h5py.File(fpath, 'r')
        self._open_files[fpath] = handle
        return handle

    def __del__(self):
        for f in self._open_files.values():
            try:
                f.close()
            except Exception:
                pass
        self._open_files = {}

    def __getstate__(self):
        # h5py handles cannot be pickled; workers reopen them lazily
        state = self.__dict__.copy()
        for f in state.get("_open_files", {}).values():
            try:
                f.close()
            except Exception:
                pass
        state["_open_files"] = OrderedDict()
        state["_open_files_pid"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getitem__(self, idx):
        fpath, frame_idx = self.frame_index[idx]
        f = self._get_file(fpath)
        if self.key_filter:
            keys = [k for k in f.keys() if re.match(self.key_filter, k)]
        else:
            keys = list(f.keys())

        if self.time_slice_selector is None:
            data = {
                key: torch.from_numpy(f[key][frame_idx]).float().to(self.device)
                for key in keys
            }
        else:
            data = {}
            for key in keys:
                try:
                    if key == self.video_key:
                        data[key] = torch.from_numpy(f[self.video_key][list(frame_idx)]).float().to(self.device)
                    else:
                        data[key] = torch.from_numpy(self.time_slice_selector(param=f[key], name=key, slice=list(frame_idx))).float().to(self.device)
                except Exception as e:
                    print(f"Error processing key {key} in file {fpath} at frame {frame_idx}: {e}, shape is: {f[key].shape}")

        data["attrs"] = dict(f.attrs)

        # Emit frame distance when loading pairs (time_slice_selector mode).
        if self.time_slice_selector is not None and isinstance(frame_idx, (tuple, list)) and len(frame_idx) >= 2:
            data["__frame_distance__"] = torch.tensor(int(frame_idx[1]) - int(frame_idx[0]), dtype=torch.long)

        if self.transform:
            return self.transform(**data)
        return data


import torch
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm

class CachedH5FileDatasetNew(Dataset):
    """
    Wraps an H5FileDataset and caches all samples in memory (optionally GPU).
    Supports saving/loading the cached data to/from a .pt file to skip slow I/O on future runs.
    """

    def __init__(self, h5_dataset=None, limit_cache_size: int = -1,
                 cache_path: Optional[str] = None, device: str = "cuda:0",
                 save_after_build: bool = True, verbose: bool = True,
                 cache_key: Optional[str] = None):
        """
        Args:
            h5_dataset: The H5FileDataset to wrap.
            limit_cache_size: Limit number of samples to cache (use -1 for all).
            cache_path: Optional .pt file to load/save the cache.
            device: Device to place the cached tensors on ('cuda:0', 'cpu', etc.).
            save_after_build: Whether to save to `cache_path` after caching (if not found).
            verbose: Print progress and info messages.  
        """
        self.h5_dataset = h5_dataset
        self.device = torch.device(device)
        self.limit_cache_size = limit_cache_size
        self.cache_key = cache_key
        print(
            f"[CachedH5FileDataset] Initializing with device={self.device}, "
            f"limit_cache_size={self.limit_cache_size}, cache_key={self.cache_key}, cache_path={cache_path}"
        )
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache = []

        # Try to load pre-cached dataset if available
        if self.cache_path and self.cache_path.exists():
            if verbose:
                print(
                    f"[CachedH5FileDataset][CACHE_HIT] key={self.cache_key} "
                    f"path={self.cache_path}"
                )
                print(f"[CachedH5FileDataset] Loading cache from {self.cache_path} ...")
            # load to CPU; batches are moved to the device by the DataLoader
            self.cache = self.load_cache(self.cache_path, map_location="cpu")
            if verbose:
                print(f"[CachedH5FileDataset] Loaded {len(self.cache)} samples from cache.")
        else:
            if verbose:
                print(
                    f"[CachedH5FileDataset][CACHE_MISS] key={self.cache_key} "
                    f"path={self.cache_path}"
                )
            if h5_dataset is None:
                raise ValueError(
                    "CachedH5FileDatasetNew requires h5_dataset when cache file is missing. "
                    "Provide h5_dataset or an existing cache_path."
                )
            # Build the cache from the original dataset
            if verbose:
                print("[CachedH5FileDataset] Building in-memory cache from HDF5 files...")
            for idx in tqdm(range(len(h5_dataset)), desc="Caching H5 files", disable=not verbose):
                if 0 < self.limit_cache_size <= len(self.cache):
                    break
                self.cache.append(h5_dataset[idx])

            if verbose:
                print(f"[CachedH5FileDataset] Cached {len(self.cache)} samples in memory (device={self.device}).")

            # Save cache for next run
            if self.cache_path and save_after_build:
                if verbose:
                    print(
                        f"[CachedH5FileDataset][CACHE_SAVE] key={self.cache_key} "
                        f"path={self.cache_path}"
                    )
                    print(f"[CachedH5FileDataset] Saving cache to {self.cache_path} ...")
                self.save_cache(self.cache, self.cache_path)
                if verbose:
                    print("[CachedH5FileDataset] Cache saved successfully.")

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self.cache[i] for i in idx]
        return self.cache[idx]

    # -----------------------------
    # Static helper functions
    # -----------------------------

    @staticmethod
    def save_cache(cache_data, path: str, to_cpu: bool = True):
        """
        Save a cached dataset list (list of dicts) to a .pt file.

        Args:
            cache_data: The list of samples to save.
            path: Output path to save to.
            to_cpu: Move tensors to CPU before saving (for portability).
        """
        safe_cache = []
        for sample in cache_data:
            safe_sample = {
                k: (v.cpu() if to_cpu and isinstance(v, torch.Tensor) else v)
                for k, v in sample.items()
            }
            safe_cache.append(safe_sample)

        torch.save(safe_cache, path)

    @staticmethod
    def load_cache(path: str, map_location="cpu"):
        """
        Load a cached dataset from a .pt file.

        Args:
            path: Path to the saved cache.
            map_location: Device to load tensors onto.
        """
        cache = torch.load(path, map_location=map_location)
        return cache
