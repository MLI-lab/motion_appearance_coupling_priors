import torch

from abc import ABC, abstractmethod
from typing import Callable, Tuple
import numpy as np

from src.domain_specific_setups.natural_video_forward_models import LinearForwardModel

class BaseDomainSpecificSetup(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def cache_datasets_on_gpu(self, device):
        pass

    @abstractmethod
    def get_measurement(self, scaled : bool = False) -> torch.Tensor:
        pass

    @abstractmethod
    def get_forward_model(self) -> LinearForwardModel:
        pass

    @abstractmethod
    def get_train_dataset(self) -> torch.utils.data.Dataset:
        pass

    @abstractmethod
    def get_reference_video(self) -> torch.Tensor:
        pass


    @abstractmethod
    def get_fps(self) -> float:
        """
            Returns the frames per second for visualization purposes
        """
        pass

    @abstractmethod
    def get_meshes_dict(self):
        pass


    @abstractmethod
    def get_spatial_mesh(self):
        pass

    @abstractmethod
    def get_time_mesh(self):
        pass

    @abstractmethod
    def get_domain_specific_info(self):
        """
            E.g., for CMR this should return smaps
        """
        pass

    @abstractmethod
    def get_dataset_params(self):
        """
            E.g., returns domain specific infos like TE, TR, times etc
        """
        pass

    @abstractmethod
    def get_video_processor_for_visualization(
        self,
    ) -> Callable[[torch.Tensor], np.ndarray]:
        """
            Returns a function that processes a video for visualization purposes.
            The function should expect THWC torch.Tensor and return a numpy ndarray of TCHW.
        """
        pass

    @abstractmethod
    def evaluate(self, model : torch.nn.Module) -> dict:
        pass

    @abstractmethod
    def evaluate_video(self, video : torch.Tensor) -> dict:
        pass

    @abstractmethod
    def fit(self,
            # passed objects from recon_task
            loss_fn, regularizer_fn,
            sampler, optimizer,
            model, accelerator,
            # logging infos passed from recon_task
            is_main_process=True,
            # train specific parameters
            **kwargs
        ):
        pass

    def get_ground_truth_flow_pair(self):
        return None, None

    def get_flow_reference_video(self, scaled : bool = True):
        """Video the method's flow/motion estimates should be derived from, if it differs
        from the reconstruction's own pseudoinverse. None means "use the pseudoinverse"."""
        return None

    def unscale_video(self, video: torch.Tensor) -> torch.Tensor:
        """Undo any measurement scaling applied during setup (no-op by default)."""
        return video

def resolve_base_domain_specific_setup(name : str, **params):
    if name == "binned_cmr":
        
        from src.domain_specific_setups.binned_mri_setup import BinnedMRISetup
        return BinnedMRISetup(**params)
    elif name == "natural_video_setup":
        from src.domain_specific_setups.natural_video_setup import NaturalVideoSetup
        return NaturalVideoSetup(**params)
    else:
        raise ValueError(f"Unknown dataset {name}")
