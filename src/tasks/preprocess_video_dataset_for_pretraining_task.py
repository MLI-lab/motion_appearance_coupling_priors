import hydra
from omegaconf import DictConfig
from prefect import flow, task
import logging
import os
from accelerate import notebook_launcher, Accelerator
from diffusers import DDPMScheduler
import torch
from prefect import flow, task
from prefect.cache_policies import TASK_SOURCE, INPUTS, NONE
from src.prefect.caching import CacheableDict, CacheableDictConfig, CacheableListConfig
from src.utils.wandb_utils import wandb_kwargs_for_prefect_task, WandbParamsTask
from functools import partial
import socket
import wandb
import numpy as np
from src.dataset.cmr_ocmr_dataset import H5FileDataset
from tqdm import tqdm
from src.representations.representation_resolver import representation_resolver
import h5py
from src.utils.state_dict_h5 import save_module_h5_auto
from src.utils.video_crop_transforms import SpatialCenterCropAndDownscale
from src.utils.data_roots import DataRoots, new_run_output_dir

def process_case(args):
    (
        idx,
        dataset,
        video_duration_in_s,
        warp_method_cfg,
        warp_method_fit_cfg,
        base_output_dir,
        output_keys,
        noise_warp_upsample_factor,
        store_warp_parameters,
        key_in_filter,
        gpu_id,
        crop_cfg,
        normalization,
        fit_motion_model
    ) = args
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    video = dataset[idx][key_in_filter]  # shape: (total_frames, height, width)
    print(f"Case {idx}: video tensor shape {video.shape}")

    fps = None
    if "attrs" in dataset[idx].keys():
        attrs = dataset[idx]["attrs"]
        if "fps" in attrs:
            fps = attrs["fps"]
            video_duration_in_s = video.shape[0] / fps
    if fps is None and not video_duration_in_s is None:
        fps = video.shape[0] / video_duration_in_s
    elif fps is None and video_duration_in_s is None:
        raise ValueError("Could not determine fps from dataset attributes or ms_per_frame parameter.")

    if video.ndim == 3:
        video = video[:, :, :, None]  # add channel dim if missing (F, H, W, C)

    # here we could crop the video, if we want to use it
    video_orig_shape = video.shape
    video_norm_scaling = 1.0
    if crop_cfg is not None and crop_cfg.cfg["enabled"]:
        # downscaling and interpoation, required if it is performed at a later stage
        video_norm_before = video.norm()
        transform = SpatialCenterCropAndDownscale(**crop_cfg.cfg, normalization="none")
        video_resh = video.movedim(-1,0)
        video = transform(video_resh, torch.randn_like(video_resh))["video"].movedim(
            0, -1
        )
        # corrects the sqrt(numel) term below, which uses the pre-downscale video_orig_shape
        video_norm_scaling = (video.norm() / video_norm_before).item()

    # some preprocessing of the video (after cropping etc)
    # kspace modes need special handling before VideoNormalization (they use external attrs)
    if normalization in ("kspace_attr_norm", "kspace_attr_norm_video"):
        assert "attrs" in dataset[idx].keys()
        slice_kspace_norm = dataset[idx]["attrs"]["kspace_slice_norm"]
        if normalization == "kspace_attr_norm":
            scaling_factor = np.sqrt(np.prod(video_orig_shape[1:])) / slice_kspace_norm
        else:
            scaling_factor = np.sqrt(np.prod(video_orig_shape)) / slice_kspace_norm * video_norm_scaling
        video = video * scaling_factor
        # represent as a scalar-factor normalization so VideoNormalization can carry it
        from src.utils.video_normalization import VideoNormalization
        video_norm = VideoNormalization(mode="video_div_std", scaling_factor=float(scaling_factor))
    else:
        from src.utils.video_normalization import VideoNormalization
        video_norm = VideoNormalization.fit_thwc(normalization, video)
        video = video_norm.normalize_thwc(video)

    if fit_motion_model:
        _, X, Y, C = video.shape
        from src.dataset.cinemri_dataset import get_spatial_mesh_for_matrix_size
        spatial_mesh = get_spatial_mesh_for_matrix_size((X, Y), device=device)  # (X, Y, 2)
        time_mesh = torch.linspace(0.0, video_duration_in_s, steps=video.shape[0], device=device)  # (T,) # same as in the data setup
        from src.regularizers.model_initializers.base_model_initializer import model_initializer_resolver
        model_params = model_initializer_resolver(**warp_method_fit_cfg.cfg.init_params)(
            dataset = video, spatial_mesh=spatial_mesh, time_mesh=time_mesh, device=device # this might not work for several model initializers
        )
        warp_method = representation_resolver(**warp_method_cfg.cfg,
            spatial_mesh=spatial_mesh, time_mesh=time_mesh, device=device, **model_params)
        warp_method.fit_to_video(video.to(device), **warp_method_fit_cfg.cfg.fit_params)

        random_sample = torch.randn( (1, noise_warp_upsample_factor*X, noise_warp_upsample_factor*Y, C), device=device)
        warped_noise_sample = warp_method.full_warp(base_tensor_replacement=random_sample, spatial_mesh_replacement=spatial_mesh, noise_correction=True)
    else:
        warped_noise_sample = torch.randn_like(video)

    # base_tensor has shape: (1, C, X, Y)
    # defo_map has shape: (1, 2, X, Y, T)
    # warped_noise_sample has shape: (T, X, Y, C) -> (T, C, X, Y)
    video = video.movedim(-1, 1) # convention for training data should be (F, C, X, Y)
    warped_noise_sample = warped_noise_sample.movedim(-1, 1)

    out_path = os.path.join(base_output_dir, f"file_{idx}.h5")
    with h5py.File(out_path, 'w') as f:
        f.create_dataset(output_keys["video"], data=video.cpu().numpy())
        if store_warp_parameters:
            assert fit_motion_model, "warp_method must be fitted to video before saving warp parameters"
            save_module_h5_auto(warp_method, output_keys["warp_parameters"], f, constructor_args=dict(warp_method_cfg.cfg))
        f.create_dataset(output_keys["warped_noise_sample"], data=warped_noise_sample.detach().cpu().numpy())
        f.attrs.create(output_keys["fps"], data=fps)
        video_norm.save_to_h5_attrs(f)

# ---------------------------------------------------------------------------------------------------------------------------

# data_roots is part of the cache key: the returned path is only valid under these roots
@task(cache_policy=TASK_SOURCE + INPUTS - "wandb_params_task" - "torch_mp_process_count", name="PreprocessVideoDatasetForPretraining", tags=["preprocess"], version="1.0", retries=0)
def preprocess_video_dataset_for_pretraining_task(
        seed : int,
        base_input_dir : str, # base directory given by previous task
        base_output_dir : str,  # relative to the data cache root; each run writes a fresh timestamped subdirectory
        splits_cfg: CacheableListConfig,
        splits_filter_cfg: CacheableListConfig,
        key_in_filter : str,
        video_duration_in_s : float,
        file_limit : int,
        noise_warp_upsample_factor : int,
        store_warp_parameters: bool,
        torch_mp_process_count : int,
        crop_cfg : CacheableDictConfig,
        normalization : str,
        fit_motion_model : bool,
        output_keys_cfg : CacheableDictConfig,
        warp_method_cfg : CacheableDictConfig,
        warp_method_fit_cfg : CacheableDictConfig,
        wandb_params_task: WandbParamsTask,
        data_roots: DataRoots,) -> str:

    """
        This task preproceses CMR volumes. It assumes the following format for input:
        - folder containing .h5 files
        - each .h5 file follows the naming convention <case_id>_t=<?>_<case_suffix>.h5 
        - each .h5 files contains a number of attributes including the image data ...

        At a defined location it outputs:
        - a folder containing .h5 files
        - the files follow the convention <case_id>_<case_suffix>.h5
        - the files contain the preprocessed video (rss images for now) as well as the deformation model parameter fitted to the data.

        Goal: The data is loaded during training, random noise is sampled and warped according to the deformation model, and used for training the diffusion model.
    """
    wandb_kwargs = wandb_kwargs_for_prefect_task(wandb_params_task)
    path_resolver = data_roots.resolver()
    base_input_dir = path_resolver(base_input_dir)  # upstream tasks return cache-root-relative paths
    base_output_dir = new_run_output_dir(data_roots, base_output_dir, f"{wandb_params_task.flow_run_id}-{wandb_kwargs['config']['task_id']}")

    from torch.utils.data.dataset import ConcatDataset



    device = "cuda"

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    with wandb.init(**wandb_kwargs):

        splits = splits_cfg.cfg # no need to resolve anything

        datasets = []
        for split, file_filter in zip(splits, splits_filter_cfg.cfg):
            data_dir_full = os.path.join(base_input_dir, split)
            print(f"Loading dataset from {data_dir_full} with filter {file_filter}")
            dataset = H5FileDataset(data_dir=data_dir_full, file_filter=file_filter, key_filter=key_in_filter, device=device)
            datasets.append(dataset)
            print(f"Loaded dataset with {len(dataset)} cases from {data_dir_full}")

        full_dataset = ConcatDataset(datasets)
        print(f"Total dataset size: {len(full_dataset)} cases")

        print(f"Preloading finished. Total cases: {len(full_dataset)}")
        import torch.multiprocessing as mp

        file_indices = np.arange(len(full_dataset)) if file_limit <= 0 else np.arange(min(file_limit, len(full_dataset)))

        wandb.log({
            "num_cases": len(file_indices),
            "base_output_dir": base_output_dir,
        })

        if len(full_dataset) == 0:
            # Never mark an empty output as complete: it would be cached as a
            # valid (empty) dataset instead of surfacing the misconfiguration.
            raise ValueError(f"No cases found under {base_input_dir} for splits {list(splits)} -- nothing to preprocess.")


        num_gpus = torch.cuda.device_count()
        if torch_mp_process_count > 0:
            num_workers = min(torch_mp_process_count, len(full_dataset))  # adjust as needed
            with mp.get_context("spawn").Pool(num_workers) as pool:
                args_list = []
                for i, idx in enumerate(file_indices):
                    gpu_id = i % num_gpus if num_gpus > 0 else 0
                    args_list.append((idx, full_dataset, video_duration_in_s, warp_method_cfg, warp_method_fit_cfg, base_output_dir, output_keys_cfg.cfg, noise_warp_upsample_factor, store_warp_parameters, key_in_filter, gpu_id, crop_cfg, normalization, fit_motion_model))
                list(tqdm(pool.imap(process_case, args_list), total=len(file_indices), desc="Preprocessing cases"))
        else:
            for i, idx in enumerate(tqdm(file_indices, desc="Preprocessing cases")):
                gpu_id = i % num_gpus if num_gpus > 0 else 0
                process_case((idx, full_dataset, video_duration_in_s, warp_method_cfg, warp_method_fit_cfg, base_output_dir, output_keys_cfg.cfg, noise_warp_upsample_factor, store_warp_parameters, key_in_filter, gpu_id, crop_cfg, normalization, fit_motion_model))

        return data_roots.to_relative(base_output_dir)

