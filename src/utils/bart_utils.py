import sys
import os
from src.utils.bart_toolbox import DEFAULT_TOOLBOX_PATH, toolbox_path

# BART (https://github.com/mrirecon/bart, v0.6.00): $TOOLBOX_PATH, or the setup/user_setup.sh install in ./libs.
_toolbox_path = toolbox_path()
if _toolbox_path is None:
    raise ImportError(f"BART not found at {DEFAULT_TOOLBOX_PATH}; run setup/user_setup.sh or set TOOLBOX_PATH to a BART build directory.")
os.environ["TOOLBOX_PATH"] = _toolbox_path  # bart.py locates the binary through it
sys.path.insert(0, os.path.join(_toolbox_path, "python"))
import bart
import numpy as np
import matplotlib.pyplot as plt
import torch
import fastmri
from tqdm import tqdm
import multiprocessing as mp


# from https://github.com/MLI-lab/untrained-motion-correction/blob/master/only_trans_more_states.ipynb


def compute_dynamic_sensmaps(kspace: np.ndarray,  **kwargs):
    '''ESPIRiT sensitivity map estimation using BART ecalib.

    Inputs:
        - kspace            : accelerated kspace
        - kwargs            : for the usage of flags e.g. for flag -a give a='', for -m 3 give m=3, ...

    Output:
        - sensitivities     : sensitivity maps belonging to the given kspace

    _____________________________________________________________________________________________________________________________________________________

    Usage: ecalib [-t f] [-c f] [-k d:d:d] [-r d:d:d] [-m d] [-S] [-W] [-I] [-1] [-P] [-v f] [-a] [-d d] <kspace> <sensitivities> [<ev-maps>]

    Estimate coil sensitivities using ESPIRiT calibration.
    Optionally outputs the eigenvalue maps.

    -t threshold     This determined the size of the null-space. 
    -c crop_value    Crop the sensitivities if the eigenvalue is smaller than {crop_value}.
    -k ksize         kernel size
    -r cal_size      Limits the size of the calibration region.
    -m maps          Number of maps to compute.
    -S               create maps with smooth transitions (Soft-SENSE).
    -W               soft-weighting of the singular vectors.
    -I               intensity correction
    -1               perform only first part of the calibration
    -P               Do not rotate the phase with respect to the first principal component
    -v variance      Variance of noise in data.
    -a               Automatically pick thresholds.
    -d level         Debug level
    -h               help

    _____________________________________________________________________________________________________________________________________________________
    '''
    # transform input to BART
    # kspace: [frame, slice, coil, ky, kx] -> [kx, ky, coil, frame, slice] -> [kx, ky, kz=1, coil, map=1, ?, ?, ?, ?, ?, frame, ?, ?, slice]
    kspace_bart = kspace.transpose(4, 3, 2, 0, 1)
    # BART does not return different sensitivities for different slices -> especially bad for lax

    # do BART sensitivity estimation
    command = 'ecalib '
    for kw in kwargs:
        command += f'-{kw} {kwargs[kw]} '

    sliced_sensitivities = []
    for s in range(kspace_bart.shape[-1]):
        slice_kspace = kspace_bart[:, :, None, :, None, None, None, None, None, None, :, None, None, s]
        sensitvities = bart.bart(1, command, slice_kspace)
        sliced_sensitivities.append(sensitvities.transpose(3, 2, 1, 0).squeeze(1))

    # transform to output shape
    sensitvities = np.stack(sliced_sensitivities, axis=0)

    return sensitvities

