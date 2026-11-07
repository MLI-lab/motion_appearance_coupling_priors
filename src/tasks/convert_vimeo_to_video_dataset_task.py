import numpy as np
import os
import torch
from prefect import task
from prefect.cache_policies import TASK_SOURCE, INPUTS, NONE
from src.prefect.caching import CacheableDictConfig, CacheableListConfig
from src.utils.wandb_utils import wandb_kwargs_for_prefect_task, WandbParamsTask
from functools import partial
import socket
import wandb
import tqdm
import h5py
from pathlib import Path
import torch.multiprocessing as mp
from PIL import Image
import torchvision.transforms.functional as TF
from src.utils.data_roots import DataRoots, new_run_output_dir


def process_case(args):
    """Process a single Vimeo sequence and save as H5 file.
    
    Args:
        args: Tuple containing (sequence_name, sequences_path, output_dir, h5_target_name, h5_attrs_max_name)
    """
    sequence_name, sequences_path, output_dir, h5_target_name, h5_attrs_max_name = args
    
    try:
        # Construct paths
        sequences_path = Path(sequences_path)
        seq_folder = sequences_path / sequence_name
        frame_paths = [seq_folder / f"im{i}.png" for i in range(1, 8)]
        
        # Check if all frames exist
        if not all(p.exists() for p in frame_paths):
            missing = [str(p) for p in frame_paths if not p.exists()]
            raise FileNotFoundError(f"Missing frames in {seq_folder}: {missing}")
        
        # Load frames as RGB (shape: (T, 3, H, W) in [0, 1])
        frames = []
        for frame_path in frame_paths:
            img = Image.open(frame_path).convert("RGB")
            frame_tensor = TF.to_tensor(img)  # (3, H, W) in [0, 1]
            frames.append(frame_tensor)
        
        video = torch.stack(frames, dim=0)  # (T, 3, H, W) in [0, 1]

        video = video.movedim(1,-1).numpy()  # (T, H, W, 3)
        
        # Create output filename from sequence name (e.g., "00001/0001" -> "00001_0001.h5")
        output_filename = sequence_name.replace("/", "_") + ".h5"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save as H5
        with h5py.File(output_path, "w") as data:
            data.create_dataset(h5_target_name, data=video)  # (T, 3, H, W)
            data.attrs.create(h5_attrs_max_name, data=video.max())
            data.attrs["sequence_name"] = sequence_name
        
        print(f'Saved file: {output_path}')
        
    except Exception as e:
        print(f'Problem with converting sequence {sequence_name}')
        print(e)


# data_roots is part of the cache key: the returned path is only valid under these roots
@task(cache_policy=TASK_SOURCE + INPUTS - "wandb_params_task" - "torch_mp_process_count", name="Vimeo to video", tags=["preprocess"], version="1.0", retries=0)
def convert_vimeo_to_video_dataset(
        seed : int,
        dataset_dir_in : str,
        base_output_dir : str,  # relative to the data cache root; each run writes a fresh timestamped subdirectory
        h5_target_name : str,
        h5_attrs_max_name : str,
        torch_mp_process_count : int,
        file_limit : int,
        sequences_folder : str,
        train_list_file : str,
        test_list_file : str,
        splits_cfg : CacheableListConfig,
        wandb_params_task: WandbParamsTask,
        data_roots: DataRoots,) -> str:

    """
        This task takes the raw Vimeo-90K dataset as input, converts each sequence into an RGB video h5 file 
        and stores them in a new directory structure (path/split/sequence.h5).
        
        Expected input directory structure:
        <dataset_dir>/
            sequences/
                00001/0001/im1.png
                00001/0001/im2.png
                ...
                00001/0001/im7.png
                ...
            sep_trainlist.txt or sep_testlist.txt
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

        sequences_path = os.path.join(dataset_dir_in_resolved, sequences_folder)
        if not os.path.exists(sequences_path):
            raise FileNotFoundError(f"Sequences folder not found at {sequences_path}")

        # Process splits
        splits_mapping = [
            ("train", train_list_file),
            ("test", test_list_file),
        ]
        
        split_names_new = splits_cfg.cfg
        
        for (split_name, list_file), split_name_new in zip(splits_mapping, split_names_new):
            
            list_path = Path(dataset_dir_in_resolved) / list_file
            if not list_path.exists():
                print(f"Skipping split {split_name}: list file {list_path} not found")
                continue
            
            # Read sequence names from list file
            with open(list_path, "r") as f:
                sequence_names = [line.strip() for line in f if line.strip()]
            
            print(f"Processing split {split_name} (new: {split_name_new}) with {len(sequence_names)} sequences.")

            if file_limit > 0:
                sequence_names = sequence_names[:file_limit]
                print(f"Limiting to first {file_limit} sequences for split {split_name}.")

            output_dir = os.path.join(base_output_dir, split_name_new)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f'Created directory: {output_dir}')

            args_list = [
                (seq_name, sequences_path, output_dir, h5_target_name, h5_attrs_max_name) 
                for seq_name in sequence_names
            ]

            if torch_mp_process_count > 0:
                num_workers = min(torch_mp_process_count, len(sequence_names))
                with mp.get_context("spawn").Pool(num_workers) as pool:
                    list(tqdm.tqdm(pool.imap(process_case, args_list), total=len(sequence_names), desc=f"Preprocessing sequences for split {split_name}"))
            else:
                for i in tqdm.trange(len(sequence_names), desc=f"Preprocessing sequences for split {split_name}"):
                    process_case(args_list[i])

        return data_roots.to_relative(base_output_dir)

