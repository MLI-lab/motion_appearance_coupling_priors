import numpy as np
import os
import torch
from prefect import task
from prefect.cache_policies import TASK_SOURCE, INPUTS, NONE
from src.prefect.caching import CacheableDictConfig, CacheableListConfig
from src.utils.wandb_utils import wandb_kwargs_for_prefect_task, WandbParamsTask
from functools import partial
import wandb
import tqdm
import h5py
from pathlib import Path
import torch.multiprocessing as mp
from PIL import Image
import torchvision.transforms.functional as TF

from src.dataset.reds_dataset import _frame_sort_key
from src.utils.data_roots import DataRoots, new_run_output_dir


def _make_slab_windows(num_frames: int, slab_size: int, slab_stride: int) -> list:
    """(start, stop) windows of length slab_size every slab_stride frames (default: non-overlapping);
    remainder frames are dropped. One whole-video window if slab_size <= 0 or >= num_frames.
    """
    if slab_size is None or slab_size <= 0 or slab_size >= num_frames:
        return [(0, num_frames)]
    stride = slab_stride if (slab_stride is not None and slab_stride > 0) else slab_size
    starts = range(0, num_frames - slab_size + 1, stride)
    return [(s, s + slab_size) for s in starts]


def process_case(args):
    """Process a single REDS video (folder of frames), splitting it into one
    or more fixed-length slabs and saving each as its own H5 file.

    Args:
        args: Tuple containing (video_id, split_frames_path, output_dir, h5_target_name,
            h5_attrs_max_name, frame_glob, slab_size, slab_stride)
    """
    video_id, split_frames_path, output_dir, h5_target_name, h5_attrs_max_name, frame_glob, slab_size, slab_stride = args

    try:
        split_frames_path = Path(split_frames_path)
        video_folder = split_frames_path / video_id
        frame_paths = sorted(video_folder.glob(frame_glob), key=_frame_sort_key)

        if len(frame_paths) == 0:
            raise FileNotFoundError(f"No frames matching '{frame_glob}' in {video_folder}")

        windows = _make_slab_windows(len(frame_paths), slab_size, slab_stride)

        for idx, (start, stop) in enumerate(windows):
            slab_frame_paths = frame_paths[start:stop]

            frames = []
            for frame_path in slab_frame_paths:
                img = Image.open(frame_path).convert("RGB")
                frame_tensor = TF.to_tensor(img)  # (3, H, W) in [0, 1]
                frames.append(frame_tensor)

            video = torch.stack(frames, dim=0)  # (T, 3, H, W) in [0, 1]
            video = video.movedim(1, -1).numpy()  # (T, H, W, 3)

            output_filename = f"{video_id}.h5" if len(windows) == 1 else f"{video_id}_slab{idx:03d}.h5"
            output_path = os.path.join(output_dir, output_filename)

            with h5py.File(output_path, "w") as data:
                data.create_dataset(h5_target_name, data=video)  # (T, H, W, 3)
                data.attrs.create(h5_attrs_max_name, data=video.max())
                data.attrs["video_id"] = video_id
                data.attrs["slab_index"] = idx
                data.attrs["slab_start_frame"] = start

            print(f'Saved file: {output_path}')

    except Exception as e:
        print(f'Problem with converting video {video_id}')
        print(e)


# data_roots is part of the cache key: the returned path is only valid under these roots
@task(cache_policy=TASK_SOURCE + INPUTS - "wandb_params_task" - "torch_mp_process_count", name="REDS to video", tags=["preprocess"], version="1.0", retries=0)
def convert_reds_to_video_dataset(
        seed: int,
        dataset_dir_in : str,
        base_output_dir : str,  # relative to the data cache root; each run writes a fresh timestamped subdirectory
        h5_target_name: str,
        h5_attrs_max_name: str,
        torch_mp_process_count: int,
        file_limit: int,
        sharp_subfolder_suffix: str,
        frame_glob: str,
        slab_size: int,
        slab_stride: int,
        splits_cfg: CacheableListConfig,
        wandb_params_task: WandbParamsTask,
        data_roots: DataRoots,) -> str:

    """
        This task takes the raw REDS dataset (as materialized by
        download_dataset) as input, converts each video (a folder of
        frames) into one or more fixed-length RGB video h5 files (splitting each raw
        100-frame video into slab_size-frame slabs, stepping by slab_stride - non-
        overlapping by default) and stores them in a new directory structure
        (path/split/video_id.h5, or path/split/video_id_slabNNN.h5 if a video was
        split into more than one slab).

        Expected input directory structure:
        <dataset_dir_in>/
            train/train_sharp/<video_id>/<frame files>
            val/val_sharp/<video_id>/<frame files>
    """
    wandb_kwargs = wandb_kwargs_for_prefect_task(wandb_params_task)
    path_resolver = data_roots.resolver()
    base_output_dir = new_run_output_dir(data_roots, base_output_dir, f"{wandb_params_task.flow_run_id}-{wandb_kwargs['config']['task_id']}")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    with wandb.init(**wandb_kwargs):

        dataset_dir_in_resolved = str(path_resolver(dataset_dir_in))

        wandb.log({
            "base_output_dir": base_output_dir,
        })

        for split_name in splits_cfg.cfg:

            split_path = Path(dataset_dir_in_resolved) / split_name / f"{split_name}{sharp_subfolder_suffix}"

            if not split_path.exists():
                print(f"Skipping split {split_name}: folder {split_path} not found")
                continue

            video_ids = sorted([p.name for p in split_path.iterdir() if p.is_dir()])
            print(f"Processing split {split_name} with {len(video_ids)} videos.")

            if file_limit > 0:
                video_ids = video_ids[:file_limit]
                print(f"Limiting to first {file_limit} videos for split {split_name}.")

            output_dir = os.path.join(base_output_dir, split_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f'Created directory: {output_dir}')

            args_list = [
                (video_id, split_path, output_dir, h5_target_name, h5_attrs_max_name, frame_glob, slab_size, slab_stride)
                for video_id in video_ids
            ]

            if torch_mp_process_count > 0:
                num_workers = min(torch_mp_process_count, len(video_ids))
                with mp.get_context("spawn").Pool(num_workers) as pool:
                    list(tqdm.tqdm(pool.imap(process_case, args_list), total=len(video_ids), desc=f"Preprocessing videos for split {split_name}"))
            else:
                for i in tqdm.trange(len(video_ids), desc=f"Preprocessing videos for split {split_name}"):
                    process_case(args_list[i])

        return data_roots.to_relative(base_output_dir)

