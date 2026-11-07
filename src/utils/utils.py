from typing import Tuple, Dict
from torch import Tensor


def clamp_upper_lower(im, cl : float =-6, cu : float =6):
    """
        Normalizes, clamps, then reapplies the mean and std.
    """
    im_std = im.std()
    im_mean = im.mean()
    im = (im-im_mean) / im_std
    im = im.clamp(cl,cu)
    return im * im_std + im_mean


def scale_to_unit_interval(tensor):
    """Scales all values in the tensor to be between 0 and 1"""
    min_val = tensor.min()
    max_val = tensor.max()
    scaled_tensor = (tensor - min_val) / (max_val - min_val)
    return scaled_tensor


# %%
import torch
import torch.nn.functional as F


# given a tensor of shape 1, C, T, H, W
# convert to grayscale and save the video to a file path