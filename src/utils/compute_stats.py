"""Wall-clock runtime and peak GPU memory of a reconstruction, reported as plain metrics."""
import time
from contextlib import contextmanager
from typing import Dict, Iterator, Optional, Union

import torch


_BYTES_PER_GB = float(2**30)


class ReconComputeStats:
    """Accumulates the wall-clock time of named segments and the peak GPU memory (allocated and reserved)
    since the first segment started. CUDA is synchronized at segment boundaries.
    """

    def __init__(self, device: Optional[Union[str, torch.device]]):
        self.device = torch.device(device) if device is not None else None
        self.segments: Dict[str, float] = {}
        self._peak_tracking_started = False

    @property
    def _cuda(self) -> bool:
        return self.device is not None and self.device.type == "cuda" and torch.cuda.is_available()

    def _sync(self) -> None:
        if self._cuda:
            torch.cuda.synchronize(self.device)

    @contextmanager
    def segment(self, name: str) -> Iterator[None]:
        self._sync()
        if self._cuda and not self._peak_tracking_started:
            torch.cuda.reset_peak_memory_stats(self.device)
            self._peak_tracking_started = True
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.segments[name] = self.segments.get(name, 0.0) + (time.perf_counter() - start)

    def metrics(self, prefix: str = "recon") -> Dict[str, float]:
        """`<prefix>_runtime_s` (sum over segments), per-segment runtimes if more than one, and peak GPU
        memory (GB) plus `<prefix>_gpu_name` when tracking ran on CUDA."""
        out: Dict[str, float] = {f"{prefix}_runtime_s": float(sum(self.segments.values()))}
        if len(self.segments) > 1:
            for name, seconds in self.segments.items():
                out[f"{prefix}_{name}_runtime_s"] = float(seconds)
        if self._cuda and self._peak_tracking_started:
            out[f"{prefix}_peak_gpu_memory_gb"] = torch.cuda.max_memory_allocated(self.device) / _BYTES_PER_GB
            out[f"{prefix}_peak_gpu_memory_reserved_gb"] = torch.cuda.max_memory_reserved(self.device) / _BYTES_PER_GB
            # non-numeric, so the tuning/plot tasks skip it as a metric; it is kept in the per-run
            # dataframe so runtimes measured on different GPU types can be told apart
            out[f"{prefix}_gpu_name"] = torch.cuda.get_device_name(self.device)
        return out
