import torch
from abc import ABC, abstractmethod

from src.domain_specific_setups.natural_video_forward_models import LinearForwardModel
from functools import partial
from typing import Dict, Any, Optional

class EvaluatorFunction:
    def __call__(self, video : torch.Tensor) -> dict[str, float]:
        return {}

class BaseReconBaseline(ABC):

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def run(
        self,
        measurement: torch.Tensor,
        forward_model: LinearForwardModel,
        evaluator: EvaluatorFunction | partial[Dict[Any, Any]],
        flow_reference_video: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
            Runs the reconstruction baseline.
            Returns a tensor of shape (T, X, Y, C).

            flow_reference_video: optional (T, C, X, Y) tensor to use instead of the pseudoinverse
            for RAFT flow estimation only (never for data consistency or network conditioning).
        """
        raise NotImplementedError()


# Baselines that use the pretrained model paths; keep in sync with resolve_recon_baseline below.
BASELINES_CONSUMING_PRETRAINED_MODELS = frozenset({
    "propagation_baseline",
})


def resolve_recon_baseline(name : str, pretrained_model_path=None, pretrained_model_path2=None, pretrained_vdm_model_path=None, **kwargs) -> BaseReconBaseline:
    if name == "svi_baseline":
        from src.recon_baselines.svi_baseline import SVIBaseline
        return SVIBaseline(**kwargs)
    elif name == "propagation_baseline":
        from src.recon_baselines.propagation_baseline import PropagationBaseline
        return PropagationBaseline(pretrained_model_path=pretrained_model_path, **kwargs)
    elif name == "lps_baseline":
        from src.recon_baselines.lps import LPSBaseline
        return LPSBaseline(**kwargs)
    elif name == "ldm_baseline":
        from src.recon_baselines.ldm_baseline import LDMBaseline
        return LDMBaseline(**kwargs)
    elif name == "pddr_native_baseline":
        from src.recon_baselines.pddr_native_baseline import PDDRNativeBaseline
        return PDDRNativeBaseline(**kwargs)
    else:
        raise ValueError(f"Unknown recon baseline: {name}")
