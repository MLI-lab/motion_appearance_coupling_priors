"""Cluster-independent data paths: relative paths resolve under data_cache_root (wipeable),
"manual/..." under data_manual_root (hand-staged), absolute paths are kept as-is.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import BaseModel

MANUAL_PREFIX = "manual/"
COMPLETE_MARKER = "_complete.json"


class DataRoots(BaseModel):
    cache: str
    manual: str

    def resolve(self, param: Any) -> Any:
        """Turn a relative data path (or a list of them) into an absolute one."""
        if param is None:
            return None
        if isinstance(param, (ListConfig, list, tuple)):
            return [self.resolve(item) for item in list(param)]
        if isinstance(param, (DictConfig, dict)):
            keys = list(param.keys())
            if keys == ["default"]:
                return self.resolve(param["default"])
            raise ValueError(
                f"Cluster-keyed path dicts are no longer supported (got keys {keys}). "
                "Use a path relative to data_cache_root, 'manual/...' for data_manual_root, "
                "or an absolute path -- see src/utils/data_roots.py."
            )
        if isinstance(param, os.PathLike):
            param = os.fspath(param)
        if not isinstance(param, str):
            raise TypeError(f"Cannot resolve a data path from {type(param).__name__}: {param!r}")
        if os.path.isabs(param):
            return param
        if param == MANUAL_PREFIX.rstrip("/"):
            return self.manual
        if param.startswith(MANUAL_PREFIX):
            return os.path.join(self.manual, param[len(MANUAL_PREFIX):])
        return os.path.join(self.cache, param)

    def resolver(self) -> Callable[[Any], Any]:
        """The `path_resolver` callable passed into domain setups, regularizers and baselines."""
        return self.resolve

    def to_relative(self, absolute: str | os.PathLike) -> str:
        """Inverse of resolve() for paths under either root (so returned paths are valid on every cluster)."""
        absolute = os.fspath(absolute)
        # Don't follow the leaf: a cache symlink to a hand-staged copy must still report its cache path.
        normalized = Path(absolute).parent.resolve() / Path(absolute).name
        for root, prefix in ((self.cache, ""), (self.manual, MANUAL_PREFIX)):
            try:
                rel = normalized.relative_to(Path(root).resolve())
            except ValueError:
                continue
            return prefix + rel.as_posix()
        return absolute


# --- output directories of data-producing tasks -----------------------------

def _jsonable(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, BaseModel):
        if hasattr(value, "cfg") and OmegaConf.is_config(getattr(value, "cfg")):
            # CacheableDictConfig / CacheableListConfig: hash the resolved content
            return OmegaConf.to_container(value.cfg, resolve=True)
        return value.model_dump()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return value


def is_complete(output_dir: str | os.PathLike) -> bool:
    return os.path.isfile(os.path.join(output_dir, COMPLETE_MARKER))


def mark_complete(output_dir: str | os.PathLike, meta: Optional[Dict[str, Any]] = None) -> str:
    """Atomically write the completion marker that makes the next download_dataset run skip."""
    marker = os.path.join(os.fspath(output_dir), COMPLETE_MARKER)
    tmp = f"{marker}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(_jsonable(meta or {}), f, indent=2, sort_keys=True, default=str)
    os.replace(tmp, marker)
    return marker


def new_run_output_dir(data_roots: DataRoots, base_output_dir: str, run_name: str) -> str:
    """Create and return `<root>/<base_output_dir>/<timestamp>_run-<run_name>` (absolute) for one run of a
    data-producing task."""
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = data_roots.resolve(os.path.join(base_output_dir, f"{stamp}_run-{run_name}"))
    os.makedirs(output_dir, exist_ok=False)
    return output_dir
