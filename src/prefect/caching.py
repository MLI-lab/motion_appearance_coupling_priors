
from pydantic import BaseModel, ConfigDict, model_serializer
from omegaconf import DictConfig, OmegaConf, ListConfig
from typing import Dict
import dict_hash
import hashlib


class CacheableDictConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    cfg: DictConfig

    @model_serializer
    def ser(self) -> dict:
        # Convert to a plain dict (with interpolation resolved if needed)
        return dict_hash.sha256(OmegaConf.to_container(self.cfg, resolve=True))

class CacheableDict(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    cfg: Dict

    @model_serializer
    def ser(self) -> dict:
        # Convert to a plain dict (with interpolation resolved if needed)
        return dict_hash.sha256(self.cfg)

class CacheableListConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    cfg: ListConfig

    @model_serializer
    def ser(self) -> str:
        # Convert to a plain dict (with interpolation resolved if needed)
        return hashlib.sha256(OmegaConf.to_yaml(self.cfg).encode("utf-8")).hexdigest()

class CacheableList(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    cfg: list

    @model_serializer
    def ser(self) -> str:
        # Convert to a plain dict (with interpolation resolved if needed)
        return hashlib.sha256(str(self.cfg).encode("utf-8")).hexdigest()


def hydra_config_to_cacheable_dict(cfg: DictConfig) -> Dict[str, CacheableDictConfig]:
    """
    Convert a Hydra DictConfig to a CacheableDictConfig.
    """
    ret = {}
    for key, value in cfg.items():
        if isinstance(value, DictConfig):
            ret[key + "_cfg"] = CacheableDictConfig(cfg=value)
        elif isinstance(value, ListConfig):
            ret[key + "_cfg"] = CacheableListConfig(cfg=value)
        else:
            ret[key] = value
    return ret

import os
import hashlib
import time
from pathlib import Path
from tqdm import tqdm
from prefect import task

from src.prefect.setup import StoragePath, load_file_system
from typing import Optional
