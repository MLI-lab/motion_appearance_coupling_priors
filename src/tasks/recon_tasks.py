import hydra
import logging
import torch
import wandb
import tqdm
from omegaconf import DictConfig, OmegaConf
import random
import numpy as np
import socket
import pickle



from src.utils.compute_stats import ReconComputeStats
from src.prefect.caching import CacheableDict, CacheableDictConfig

from typing import Dict, Optional, Tuple, cast

from prefect import flow, task
from prefect.cache_policies import TASK_SOURCE, INPUTS, NONE
import logging
from functools import partial
from src.utils.wandb_utils import wandb_kwargs_for_prefect_task, WandbParamsTask

import tempfile
import accelerate
from src.prefect.setup import StoragePath
from datetime import datetime
from src.prefect.setup import load_remote_storage, StorageSettings, StoragePath, load_file_system
from src.utils.data_roots import DataRoots


def _to_python_primitives(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_python_primitives(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python_primitives(v) for v in value]
    return value


def _sanitize_constructor_args_for_json(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {
            "__type__": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, torch.Tensor):
        return {
            "__type__": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, dict):
        return {str(k): _sanitize_constructor_args_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_constructor_args_for_json(v) for v in value]
    return str(value)


def _extract_recon_video_for_saving(domain_specific_setup, model):
    with torch.no_grad():
        if hasattr(model, "full_warp"):
            try:
                recon_video = model.full_warp()
                if isinstance(recon_video, torch.Tensor):
                    return recon_video.detach().cpu()
            except Exception:
                pass

        representation_wrapper = getattr(domain_specific_setup, "representation_wrapper", None)
        spatial_mesh = getattr(domain_specific_setup, "spatial_mesh", None)
        time_mesh = getattr(domain_specific_setup, "time_mesh", None)
        device = getattr(domain_specific_setup, "device", None)
        if representation_wrapper is not None and spatial_mesh is not None and time_mesh is not None and device is not None:
            try:
                representation_wrapper.set_model(model)
                video = representation_wrapper.full_warp_spatial(
                    spatial_mesh=spatial_mesh,
                    timesteps=time_mesh,
                    skip_temp=False,
                    skip_smaps=True,
                    device=device,
                    chunks=1,
                )
                if isinstance(video, torch.Tensor):
                    return video.movedim(-2, 1).detach().cpu()
            except Exception:
                pass

    return None

def recon_task_inner(
    seed : int,
    save_result_cfg: CacheableDictConfig,
    domain_specific_setup_cfg : CacheableDictConfig,
    model_initializer_cfg : CacheableDictConfig,
    representation_cfg : CacheableDictConfig,
    trainer_cfg : CacheableDictConfig,
    loss_fn_cfg : CacheableDictConfig,
    regularizer_fn_cfg : CacheableDictConfig,
    optimizer_cfg : CacheableDictConfig,
    animate_cfg : CacheableDictConfig,
    storage_settings_cfg: CacheableDictConfig,
    wandb_params_task : WandbParamsTask,
    data_roots: DataRoots,
    use_accelerate : bool = False,
    output_path: str = "recon_task_result.pkl",
    pretrained_model_path: Optional[StoragePath] = None,
    pretrained_model_path2: Optional[StoragePath] = None,
    sweep_cfg : Optional[CacheableDictConfig] = None
) -> Tuple[CacheableDict, Optional[StoragePath]] | None:

    import os
    os.environ["NCCL_P2P_LEVEL"] = "PXB"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_DEBUG"] = "INFO"

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    print(f"Creating accelerator")
    if use_accelerate:
        accelerator = accelerate.Accelerator(mixed_precision="fp16")
    else:
        accelerator = None
    print(f"Accelerator created")
    # init wandb and run the task
    is_main_process = not use_accelerate or accelerator.is_main_process
    wandb_kwargs = wandb_kwargs_for_prefect_task(wandb_params_task)
    if is_main_process:
        wandb_run = wandb.init(**wandb_kwargs)
    else:
        wandb_run = None

    device = "cuda" if not use_accelerate else accelerator.device

    print(f"Wandb logged in main process")
    # dataset loading (done multiple times per gpu, if necessary)
    path_resolver = data_roots.resolver()
    from src.domain_specific_setups.base_domain_specific_setup import resolve_base_domain_specific_setup, BaseDomainSpecificSetup

    domain_specific_setup : BaseDomainSpecificSetup = resolve_base_domain_specific_setup(**domain_specific_setup_cfg.cfg, path_resolver=path_resolver, device=device)
    domain_specific_setup.cache_datasets_on_gpu(device)

    # resolve the initializer
    from src.regularizers.model_initializers.base_model_initializer import model_initializer_resolver
    # reference_video is only used by the oracle-perturbation inits (target_plus_alias / target_plus_noise)
    try:
        reference_video = domain_specific_setup.get_reference_video(scaled=True)
    except (TypeError, NotImplementedError):
        reference_video = None  # setups without a (scaled) framewise reference video
    # runtime / peak memory of init + fit only (no data/model loading, animations or eval)
    compute_stats = ReconComputeStats(device)
    with compute_stats.segment("init"):
        model_params = model_initializer_resolver(**model_initializer_cfg.cfg)(device=device, dataset=domain_specific_setup.get_train_dataset(),
            motion_source_dataset=domain_specific_setup.get_flow_reference_video(), reference_video=reference_video,
            **domain_specific_setup.get_meshes_dict(), dataset_params=domain_specific_setup.get_dataset_params(), **domain_specific_setup.get_domain_specific_info(), base_path=domain_specific_setup.base_path) # Todo: This should only depend on the training dataset | used for ECG signal extraction

    # resolve the module
    from src.representations.representation_resolver import representation_resolver
    representation_contr_kwargs = {
        "spatial_mesh": domain_specific_setup.get_spatial_mesh(),
        "time_mesh": domain_specific_setup.get_time_mesh(),
        **representation_cfg.cfg, **model_params
    }
    model = representation_resolver(**representation_contr_kwargs)
    if is_main_process:
        wandb_run.log({
            "model_param_count":  sum(p.numel() for p in model.parameters() if p.requires_grad)
        })

    if is_main_process:
        if animate_cfg.cfg.init:
            from src.utils.wandb_utils import wandb_video_animation
            wandb.log({
                "init_animation": wandb_video_animation(
                    model,
                    domain_specific_setup.get_spatial_mesh(),
                    domain_specific_setup.get_time_mesh(),
                    frame_processor=domain_specific_setup.get_video_processor_for_visualization(),
                    fps=domain_specific_setup.get_fps(),
                    **animate_cfg.cfg.params,
                    device=device),
                "global_step": 0,
            })

    # setup the optimizer
    print(f"Preloading dataset")
    optimizer = model.setup_optimizer(**optimizer_cfg.cfg)

    # resolve the loss function
    print(f"Resolving loss_fn")
    from src.regularizers.loss_fns import loss_fn_resolver
    loss_fn = loss_fn_resolver(**domain_specific_setup.get_meshes_dict(), **loss_fn_cfg.cfg)

    sampler = None

    if is_main_process and pretrained_model_path is not None: # otherwise we don't do anything
        wandb.log({
            "pretrained_model_path": pretrained_model_path.storage_path,
            "pretrained_model_storage_settings_bbp": pretrained_model_path.storage_settings.bucket_or_base_path,
            "pretrained_model_storage_settings_name": pretrained_model_path.storage_settings.storage_name
        })

    if is_main_process and pretrained_model_path2 is not None: # otherwise we don't do anything
        wandb.log({
            "pretrained_model_path2": pretrained_model_path2.storage_path,
            "pretrained_model_storage_settings_bbp2": pretrained_model_path2.storage_settings.bucket_or_base_path,
            "pretrained_model_storage_settings_name2": pretrained_model_path2.storage_settings.storage_name
        })

    # resolve the regularizer
    print(f"Resolving regularizer")
    from src.regularizers.regularizers import regularizer_fn_resolver
    regularizer_fn = regularizer_fn_resolver(**regularizer_fn_cfg.cfg, **domain_specific_setup.get_meshes_dict(), device=device, path_resolver=path_resolver, target_image=None, frame_processor_for_visualization=domain_specific_setup.get_video_processor_for_visualization(), accelerator=accelerator, pretrained_model_path=pretrained_model_path, pretrained_model_path2=pretrained_model_path2, forward_model=domain_specific_setup.get_forward_model())

    if is_main_process: # otherwise we don't do anything
        if animate_cfg.cfg.init:
            wandb.log({
                "ground_truth": wandb.Video(domain_specific_setup.get_video_processor_for_visualization()(domain_specific_setup.get_reference_video()), fps=domain_specific_setup.get_fps(), format="gif"),
                "pseudorec" : wandb.Video(domain_specific_setup.get_video_processor_for_visualization()(domain_specific_setup.get_train_dataset()), fps=domain_specific_setup.get_fps(), format="gif"),
            })

    print(f"Started training with {len(domain_specific_setup.get_train_dataset())} samples, model params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}, device: {device}, accelerator: {accelerator}, is_main_process: {is_main_process}")
    # run the training method
    with compute_stats.segment("fit"):
        domain_specific_setup.fit(
            # passed objects from recon_task
            loss_fn, regularizer_fn,
            sampler, optimizer,
            model, accelerator,
            # logging infos passed from recon_task
            is_main_process=is_main_process,
            # train specific parameters
            **trainer_cfg.cfg
        )

    # final evaluation and animation run on the main process only
    if is_main_process: # otherwise we don't do anything
        if animate_cfg.cfg.final:
            # evaluate PSNR and SSIM on the target image
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            from src.utils.wandb_utils import wandb_video_animation
            print(f"Creating final animation, test, test")
            wandb.log({
                "final_animation": wandb_video_animation(
                        model,
                        domain_specific_setup.get_spatial_mesh(),
                        domain_specific_setup.get_time_mesh(),
                        frame_processor=domain_specific_setup.get_video_processor_for_visualization(),
                        fps=domain_specific_setup.get_fps(),
                        **animate_cfg.cfg.params,
                        device=device),
                "global_step": 1
            })
            if animate_cfg.cfg.warp_animation:
                wandb.log(
                    model.wandb_animation(
                        quiver_cfg=animate_cfg.cfg.warp_animation_quiver_cfg,
                        quiver_enable=animate_cfg.cfg.warp_animation_quiver_enable,
                        color_cfg=animate_cfg.cfg.warp_animation_color_cfg,
                        color_enable=animate_cfg.cfg.warp_animation_color_enable,
                        overlay_cfg=animate_cfg.cfg.warp_animation_overlay_cfg,
                        overlay_enable=animate_cfg.cfg.warp_animation_overlay_enable,
                        reference_video=domain_specific_setup.get_reference_video()
                    )
                )

        with torch.no_grad():
            eval_result_dict = domain_specific_setup.evaluate(model) | compute_stats.metrics()
            wandb.log(eval_result_dict)

        storage_path: Optional[StoragePath] = None
        if save_result_cfg.cfg.enable:

            output_dir_in_fs = os.path.join(
                wandb_kwargs["config"]["flow_name"] + "_" + wandb_kwargs["config"]["flow_run_name"],
                str(datetime.now().strftime('%Y%m%d_%H%M%S')) + "_" + wandb_kwargs["name"]
            ) 
            storage_settings_dict = cast(Dict, OmegaConf.to_container(storage_settings_cfg.cfg, resolve=True))
            storage_settings = StorageSettings(**storage_settings_dict)
            filesystem = load_file_system(storage_settings)

            with tempfile.TemporaryDirectory() as tmp_dir:
                file_name = "wandb_kwargs.yaml"
                OmegaConf.save(config=wandb_params_task.config, f=os.path.join(tmp_dir, file_name))

                from src.utils.state_dict_h5 import save_module_h5_auto
                import h5py

                model_comp = model.to(torch.float16) if save_result_cfg.cfg.save_in_fp16 else model
                constructor_args_sanitized = _sanitize_constructor_args_for_json(representation_contr_kwargs)
                with h5py.File(os.path.join(tmp_dir, "model.h5"), "w") as h5f:
                    save_module_h5_auto(
                        model_comp,
                        key_prefix="final_result",
                        open_file=h5f,
                        constructor_args=constructor_args_sanitized,
                    )

                recon_video = _extract_recon_video_for_saving(domain_specific_setup, model)
                if recon_video is not None:
                    # undo measurement scaling (save in original amplitude domain)
                    recon_video = domain_specific_setup.unscale_video(recon_video.to("cpu")).cpu()
                    if save_result_cfg.cfg.save_in_fp16:
                        recon_video = recon_video.to(torch.float16)
                    torch.save(recon_video, os.path.join(tmp_dir, "recon.pt"))
                else:
                    print("[WARN] Could not extract reconstruction tensor for this setup; recon.pt was not written.")

                reference_video = domain_specific_setup.get_reference_video()
                if isinstance(reference_video, torch.Tensor):
                    torch.save(reference_video.detach().cpu(), os.path.join(tmp_dir, "reference_video.pt"))

                pseudoinv_video = domain_specific_setup.get_train_dataset()
                if isinstance(pseudoinv_video, torch.Tensor):
                    # Undo measurement scaling so pseudoinv is in the same amplitude domain as reference_video
                    pseudoinv_video = domain_specific_setup.unscale_video(pseudoinv_video.detach().cpu()).cpu()
                    torch.save(pseudoinv_video, os.path.join(tmp_dir, "pseudoinv_video.pt"))

                filesystem.put_directory(tmp_dir, output_dir_in_fs)

            storage_path = StoragePath(
                storage_path=output_dir_in_fs,
                storage_settings=storage_settings,
            )

        eval_payload = cast(Dict, _to_python_primitives(eval_result_dict | {"save_results": save_result_cfg.cfg.enable}))
        return_cfg = CacheableDict(cfg=eval_payload)
        return_payload = (return_cfg, storage_path)

        with open(output_path, "wb") as f:
            pickle.dump(return_payload, f)

        wandb.finish()  # Ensure wandb uploads are complete before exiting
        logging.info(f"Saved reconstruction results to {output_path}")
        return return_payload

@task(cache_policy=TASK_SOURCE + INPUTS - "wandb_params_task" - "sweep_cfg" - "data_roots", name="Recon Task", tags=["recon"], version="1.0", retries=0)
def recon_task(
    seed : int,
    save_result_cfg: CacheableDictConfig,
    domain_specific_setup_cfg : CacheableDictConfig,
    model_initializer_cfg : CacheableDictConfig,
    representation_cfg : CacheableDictConfig,
    trainer_cfg : CacheableDictConfig,
    loss_fn_cfg : CacheableDictConfig,
    regularizer_fn_cfg : CacheableDictConfig,
    optimizer_cfg : CacheableDictConfig,
    data_roots: DataRoots,
    animate_cfg : CacheableDictConfig,
    storage_settings_cfg: CacheableDictConfig,
    wandb_params_task : WandbParamsTask,
    pretrained_model_path: Optional[StoragePath] = None,
    pretrained_model_path2: Optional[StoragePath] = None,
    sweep_cfg : Optional[CacheableDictConfig] = None
) -> Tuple[CacheableDict, Optional[StoragePath]]:

    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    #logging.info(f"Launching recon task with devices {torch.cuda.device_count()}")

    import os
    os.environ["NCCL_P2P_LEVEL"] = "PXB"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_DEBUG"] = "INFO"

    use_accelerate = False
    use_accelerate_launcher = False
    import pickle
    output_dir = tempfile.mkdtemp(prefix="recon_task_")
    output_path = os.path.join(output_dir, "result.pkl")

    func = partial(
        recon_task_inner,
            seed,
            save_result_cfg=save_result_cfg,
            domain_specific_setup_cfg=domain_specific_setup_cfg,
            model_initializer_cfg=model_initializer_cfg,
            representation_cfg=representation_cfg,
            trainer_cfg=trainer_cfg,
            loss_fn_cfg=loss_fn_cfg,
            regularizer_fn_cfg=regularizer_fn_cfg,
            optimizer_cfg=optimizer_cfg,
            data_roots=data_roots,
            animate_cfg=animate_cfg,
            storage_settings_cfg=storage_settings_cfg,
            wandb_params_task=wandb_params_task,
            use_accelerate=use_accelerate,
            output_path=output_path,
            pretrained_model_path=pretrained_model_path,
            pretrained_model_path2=pretrained_model_path2,
            sweep_cfg=sweep_cfg
        )

    if use_accelerate and use_accelerate_launcher:
        from accelerate import notebook_launcher
        print(f"Using notebook launcher")
        notebook_launcher(
            recon_task_inner,
            args = (
                seed,
                save_result_cfg,
                domain_specific_setup_cfg,
                model_initializer_cfg,
                representation_cfg,
                trainer_cfg,
                optimizer_cfg,
                data_roots,
                animate_cfg,
                storage_settings_cfg,
                wandb_params_task,
                loss_fn_cfg,
                regularizer_fn_cfg,
                use_accelerate,
                output_path,
                pretrained_model_path,
                pretrained_model_path2,
                sweep_cfg
            ),
            num_nodes=1, num_processes=torch.cuda.device_count()
        )
    else:
        print(f"Using no launcher")
        func()

    with open(output_path, "rb") as f:
        result = pickle.load(f)

    os.remove(output_path)
    
    return result
