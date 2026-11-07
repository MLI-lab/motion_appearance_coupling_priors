import h5py
import torch

def torch_module_to_h5(open_file, key_prefix : str, module: torch.nn.Module) -> None:
    """
    Save the state_dict of a PyTorch module to an HDF5 group.

    Args:
     - module (torch.nn.Module): The PyTorch module whose state_dict is to be saved.
    """
    state_dict = module.state_dict()
    for key, tensor in state_dict.items():
        open_file.create_dataset(f"{key_prefix}_{key}", data=tensor.detach().cpu().numpy())
 

def torch_module_from_h5dict(h5_dict, key_prefix : str, module: torch.nn.Module) -> None:
    """
    Load the state_dict of a PyTorch module from an HDF5 group.

    Args:
     - module (torch.nn.Module): The PyTorch module whose state_dict is to be loaded.
    """
    state_dict = {}
    for key in module.state_dict().keys():
        dataset_key = f"{key_prefix}_{key}"
        if dataset_key in h5_dict:
            tensor = torch.tensor(h5_dict[dataset_key][()])
            # if shape of tensor doesn't match the state_dict[key], try to interpolate
            # e.g., tensor could be temporal param of shape 1, 2, X, Y, F
            if tensor.numel() != module.state_dict()[key].numel():
                if tensor.ndim == 5:
                    A, B, C, D, E = tensor.shape
                    A2, B2, C2, D2, E2 = module.state_dict()[key].shape
                    assert A == A2 and B == B2, "First two dimensions must match for interpolation"
                    tensor = torch.nn.functional.interpolate(tensor, size=(C2, D2, E2), mode='trilinear', align_corners=True)
                elif tensor.ndim == 4:
                    A, B, C, D = tensor.shape
                    A2, B2, C2, D2 = module.state_dict()[key].shape
                    assert A == A2, "First dimension must match for interpolation"
                    tensor = torch.nn.functional.interpolate(tensor.movedim(-1,1), size=(B2, C2), mode='bilinear', align_corners=True).movedim(1,-1)
                else:
                    raise NotImplementedError(f"Interpolation for tensor with ndim {tensor.ndim} not implemented.")
            state_dict[key] = tensor
        else:
            raise KeyError(f"Key {dataset_key} not found in HDF5 file.")
    module.load_state_dict(state_dict)


import torch
import h5py
import inspect
import json
import importlib

def save_module_h5_auto(module, key_prefix : str, open_file, constructor_args=None):
    """
    Save a PyTorch nn.Module to an HDF5 file including:
    - Module parameters
    - Class path and name
    - Constructor arguments
    
    Args:
        module: nn.Module instance
        filename: path to .h5 file
        constructor_args: dict of arguments to reconstruct the module
    """
    torch_module_to_h5(open_file, key_prefix, module)
        
    # Save metadata: class path, class name, constructor args
    cls = module.__class__
    module_path = cls.__module__ + "." + cls.__name__
    open_file.attrs.create(f"{key_prefix}_class_path", module_path)
    open_file.attrs.create(f"{key_prefix}_constructor_args", json.dumps(constructor_args or {}))

def load_module_h5_auto_prefix(h5_dict, key_prefix : str, **extra_constructor_args) -> torch.nn.Module:
    """
    Load a PyTorch nn.Module saved with save_module_h5_auto using a key prefix.
    
    Args:
        key_prefix: The prefix used when saving the module.
        open_file: An already-open h5py.File object in read mode.
    
    Returns:
        Reconstructed nn.Module with loaded parameters.
    """
    # Read metadata
    class_path = h5_dict["attrs"][f"{key_prefix}_class_path"]
    constructor_args = json.loads(h5_dict["attrs"][f"{key_prefix}_constructor_args"])

    # Dynamically import the class
    module_name, class_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    
    # Instantiate the module
    constructor_args_wo_name = {k: v for k, v in constructor_args.items() if k != 'name'}
    module = cls(**constructor_args_wo_name, **extra_constructor_args)

    torch_module_from_h5dict(h5_dict, key_prefix, module)
    
    return module
