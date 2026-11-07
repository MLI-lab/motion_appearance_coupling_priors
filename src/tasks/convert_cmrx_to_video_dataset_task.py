import numpy as np
import os
import torch
from prefect import task
from prefect.cache_policies import TASK_SOURCE, INPUTS, NONE
from src.prefect.caching import CacheableDictConfig
from src.utils.wandb_utils import wandb_kwargs_for_prefect_task, WandbParamsTask
from functools import partial
import socket
import wandb
from tqdm import tqdm
import h5py
import fastmri
from fastmri.data import transforms
import glob
from pathlib import Path
import re

from src.dataset.mri import fftnc, ifftnc, optimal_svd_phase_correction
from src.utils.data_roots import DataRoots, new_run_output_dir


def get_attrs_from_filename(path : str) -> dict:
    # path is e.g. .../CMRxRecon2024/ChallengeData/MultiCoil/Cine/TrainingSet/FullSample/P001/cine_sax.mat
    return {
        "challenge": re.search(r'CMRxRecon(\d{4})', path).group(0), # e.g. CMRxRecon2024
        "year": re.search(r'CMRxRecon(\d{4})', path).group(1), # e.g. 2024 
        "patient_id": re.search(r'/P(\d{3})/', path).group(1), # e.g., P001
        "perspective": Path(path).stem # e.g., cine_sax
    }

def get_filename(path_old : str, slice_nr):
    #  output_dir is e.g. val or train
    attrs = get_attrs_from_filename(path_old)
    return f"{attrs['challenge']}_{attrs['year']}_{attrs['patient_id']}_{attrs['perspective']}_slice={slice_nr}.h5"

def process_case(args):
    _, fname, output_dir, h5_attrs_max_name, store_rss_target, store_rss_target_name, store_mvue_target, store_mvue_target_name, apply_phase_correction = args

    import numpy as np
    try:
        with h5py.File(fname, "r") as hf:

            kspace_full = hf["kspace_full"][()]

            kspace_full = torch.view_as_real(
                torch.from_numpy(
                    (kspace_full["real"] + 1j * kspace_full["imag"]).astype(
                        np.complex64
                    )
                )
            )  # shape is: (F, S, C, H, W, 2)

            if apply_phase_correction:
                # expects (B, C, T, RO, PH, 2): slices as batch, frames as time
                kspace_for_correction = kspace_full.permute(1, 2, 0, 3, 4, 5)  # (S, C, F, H, W, 2)
                kspace_full = optimal_svd_phase_correction(kspace_for_correction).permute(
                    2, 0, 1, 3, 4, 5
                ).contiguous()  # back to (F, S, C, H, W, 2)

            target_full = fastmri.ifft2c(kspace_full) # still (F, S, C, H, W, 2)
            # here we skipped swapping H and W

            if store_mvue_target:
                from src.utils.bart_utils import compute_dynamic_sensmaps
                kspace_np = torch.view_as_complex(kspace_full).numpy()
                smaps = torch.from_numpy(
                    compute_dynamic_sensmaps(kspace_np, m=1, c=0) # expects (F, S, C, H, W) complex dtype
                ) # output shape should be (F, S, C, H, W) complex dtype
                target_mvue = torch.view_as_real(
                    (torch.view_as_complex(target_full) * torch.conj(smaps)).sum(dim=2)
                ).numpy().astype(np.float32) # shape (F, S, H, W, 2)
            else:
                target_mvue = None

            if store_rss_target:
                target_rss = (
                    fastmri.rss_complex(target_full, dim=2).numpy().astype(np.float32)
                ) # (F, S, H, W)
            else:
                target_rss = None
            # target_full shape: F, S, H, W

        for j in range(0, target_full.shape[1]):

            target = target_full[:, j] # F, slice, H, W

            attrs_by_filename = get_attrs_from_filename(fname)
            save_file = os.path.join(output_dir, get_filename(fname, j))
            with h5py.File(save_file, 'w') as data:

                if store_rss_target:
                    assert target_rss is not None, "target_rss should not be None when store_rss_target is True"
                    data.create_dataset(store_rss_target_name, data=target_rss[:, j])

                if store_mvue_target:
                    assert target_mvue is not None, "target_mvue should not be None when store_mvue_target is True"
                    data.create_dataset(store_mvue_target_name, data=target_mvue[:, j])

                data.attrs.create(h5_attrs_max_name, data=target.max())
                data.attrs["kspace_norm"] = np.linalg.norm(kspace_full)
                data.attrs["kspace_slice_norm"] = np.linalg.norm(kspace_full[:, j])
                data.attrs["kspace_std"] = kspace_full.std()
                data.attrs["kspace_slice_std"] = kspace_full[:, j].std()
                data.attrs['original_filename'] = str(Path(fname).name)
                data.attrs["slicenr"] = j
                data.attrs.update(attrs_by_filename)
            print(f'Saved file: {save_file}')

        target_mvue = None
        target_rss = None
        kspace_tensor = None
        kspace_full = None

    except OSError as ose:
        print(f'Problem with converting file {fname}')
        print(ose)

# ---------------------------------------------------------------------------------------------------------------------------

# data_roots is part of the cache key: the returned path is only valid under these roots
@task(cache_policy=TASK_SOURCE + INPUTS - "wandb_params_task" - "torch_mp_process_count", name="CMRxRecon to video", tags=["preprocess"], version="1.0", retries=0)
def preprocess_cmrx_to_video_dataset(
        seed : int,
        dataset_dir_in : str,
        base_output_dir : str,  # relative to the data cache root; each run writes a fresh timestamped subdirectory
        h5_attrs_max_name : str,
        torch_mp_process_count : int,
        file_limit : int,
        store_rss_target : bool,
        store_rss_target_name : str,
        store_mvue_target : bool,
        store_mvue_target_name : str,
        apply_phase_correction : bool,
        splits_cfg : CacheableDictConfig,
        wandb_params_task: WandbParamsTask,
        data_roots: DataRoots,) -> str:

    """
        This task takes the raw CMRxRecon datasets as input, splits each volume into video h5 files and stores them in a new directory structure (path/split/volume.h5).
    """
    wandb_kwargs = wandb_kwargs_for_prefect_task(wandb_params_task)
    path_resolver = data_roots.resolver()
    base_output_dir = new_run_output_dir(data_roots, base_output_dir, f"{wandb_params_task.flow_run_id}-{wandb_kwargs['config']['task_id']}")


    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    with wandb.init(**wandb_kwargs):

        dataset_dir_in_resolved = path_resolver(dataset_dir_in)

        split_keys = splits_cfg.cfg.keys() # e.g., train_val, test

        wandb.log({
            "nr_splits": len(split_keys),
            "base_output_dir": base_output_dir,
        })

        for split in split_keys:
            
            file_filters = splits_cfg.cfg[split]
            print(f"Processing split {split} with {len(file_filters)} file filters.")
            file_lists = [glob.glob(os.path.join(dataset_dir_in_resolved, file_filter), recursive=True) for file_filter in file_filters]

            file_lists = [item for sublist in file_lists for item in sublist]  # flatten list of lists
            print(f"Found {len(file_lists)} files for split {split}.")

            wandb.log({
                f"nr_files_{split}": len(file_lists),
            })

            output_dir = os.path.join(base_output_dir, split)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f'Created directory: {output_dir}')

            # take only the first file_limit files if file_limit > 0
            if file_limit > 0:
                file_lists = file_lists[:file_limit]
                print(f"Limiting to first {file_limit} files for split {split}.")

            if torch_mp_process_count > 0:
                num_workers = min(torch_mp_process_count, len(file_lists))  # adjust as needed
                import torch.multiprocessing as mp
                with mp.get_context("spawn").Pool(num_workers) as pool:
                    args_list = [(i, fname, output_dir, h5_attrs_max_name, store_rss_target, store_rss_target_name, store_mvue_target, store_mvue_target_name, apply_phase_correction) for i, fname in enumerate(file_lists)]
                    list(tqdm(pool.imap(process_case, args_list), total=len(file_lists), desc=f"Preprocessing cases for split {split}"))
            else:
                for i, fname in tqdm(enumerate(file_lists), desc=f"Preprocessing cases for split {split}"):
                    process_case((i, fname, output_dir, h5_attrs_max_name, store_rss_target, store_rss_target_name, store_mvue_target, store_mvue_target_name, apply_phase_correction))
        return data_roots.to_relative(base_output_dir)

