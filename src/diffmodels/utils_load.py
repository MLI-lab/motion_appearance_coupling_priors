from typing import Dict, Optional
from collections import OrderedDict

import torch
import logging

from src.diffmodels.networks.archs.unet import UNetModel
from src.diffmodels.networks.ema import ExponentialMovingAverage


from pathlib import Path
from omegaconf import OmegaConf


def create_dense_model(
    num_channels: int,
    in_channels: int,
    num_res_blocks: int,
    channel_mult: str = '',
    use_checkpoint: bool = False,
    attention_resolutions: str = '16',
    num_heads: int = 1,
    num_head_channels: int = -1,
    num_heads_upsample: int = -1,
    use_scale_shift_norm: bool = False,
    dropout: float = 0.,
    resblock_updown: bool = False,
    use_fp16: bool = False,
    use_new_attention_order: bool = False,
    resamp_with_conv : bool = True,
    learn_sigma : bool = False,
    **kwargs
):

    logging.info(f"Unused kwargs: {kwargs}")
    
    attention_ds = []
    for res in str(attention_resolutions).split(","):
        attention_ds.append(int(res)) # attention resolutions given directly, not as image_size // res

    return UNetModel(
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=(in_channels if not learn_sigma else in_channels * 2),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=tuple(channel_mult),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        conv_resample=resamp_with_conv
    )

def create_model(name: str, params, arch_cfg = None) -> UNetModel:
    if name == "dense":
        return create_dense_model(**params)
    else:
        raise ValueError(f"Unknown model name {name}")
