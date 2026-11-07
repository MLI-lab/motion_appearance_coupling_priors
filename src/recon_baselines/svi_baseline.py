from src.recon_baselines.recon_baseline_resolver import BaseReconBaseline, EvaluatorFunction
from src.domain_specific_setups.base_domain_specific_setup import BaseDomainSpecificSetup
import torch
from src.domain_specific_setups.natural_video_forward_models import LinearForwardModel
from typing import Optional

def normalize_img(t):
    return t * 2 - 1

def unnormalize_img(t):
    return (t + 1) * 0.5


from src.recon_baselines.svi.script_util import (
    create_model_and_diffusion,
    NUM_CLASSES
)

from src.regularizers.baseline_regularizer_wrapper import BaselineRegularizerWrapper
from src.utils.model_cache import ensure_cached_download

class SVIBaseline(BaseReconBaseline):

    def __init__(self,
            path_resolver,
            cache_dir,
            recon_args,
            diffusion_args,
            nfe,
            model_args,
            model_path : str,
            noise_mode: str,
            noise_params: dict,
            device : str 
        ):
        super().__init__()

        print("creating model and diffusion...")
        model, diffusion = create_model_and_diffusion(
            **model_args,
            **diffusion_args,
        )

        cache_dir = path_resolver(cache_dir)
        print(f"Cache_dir resolved to: {cache_dir}")
        model_path_resolved = ensure_cached_download(model_path, cache_dir)
        model.load_state_dict(
            torch.load(model_path_resolved, map_location="cpu")
        )
        model.to(device)
        if model_args["use_fp16"]:
            model.convert_to_fp16()
        model.eval()

        self.recon_args = recon_args
        self.model = model
        self.diffusion = diffusion
        self.noise_mode = noise_mode
        self.noise_params = noise_params
        self.nfe = nfe

    def run(
        self,
        measurement: torch.Tensor,
        forward_model: LinearForwardModel,
        evaluator: EvaluatorFunction,
        regularizer_wrapper : Optional[BaselineRegularizerWrapper] = None,
        flow_reference_video: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs

        measurement = measurement.movedim(-1,1) # (T, C, X, Y)

        pseudoinv = forward_model.adjoint(measurement, cdim=1) # (C, T, X, Y)

        if flow_reference_video is not None and flow_reference_video.shape != pseudoinv.shape:
            raise ValueError(
                f"flow_reference_video shape {tuple(flow_reference_video.shape)} != pseudoinv shape {tuple(pseudoinv.shape)}"
            )
        flow_seed = flow_reference_video if flow_reference_video is not None else pseudoinv

        from src.warp_methods.noise_warp_util import get_noise_warping_method
        noise_warp_method = get_noise_warping_method(name=self.noise_mode, **dict(self.noise_params), target_shape=pseudoinv.shape, device=measurement.device)
        noise_warp_method.initialize_flow(flow_seed)
        noise_sampler = noise_warp_method.sample_noise

        if regularizer_wrapper is not None:
            regularizer_wrapper.set_pseudoinv(flow_seed)

        
        sample = self.diffusion.time_deconv_sample_loop(
            self.model,
            forward_model.adjoint(measurement, cdim=1).shape,
            measurement=measurement,
            forward_model=forward_model,
            evaluator=evaluator,
            noise_sampler=noise_sampler,
            baseline_regularizer_wrapper=regularizer_wrapper,
            T_sampling=int(self.nfe+1),
            **self.recon_args
        )

        return unnormalize_img(sample.movedim(1,-1)) # (T, X, Y, C)