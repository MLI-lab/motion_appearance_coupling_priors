"""Imported by flow files as `remote_options`: per-task Ray resources (num_gpus, cpus_per_task -> num_cpus).

Without a Ray task runner (task_runner=local) the options have no effect.
"""

from contextlib import contextmanager
from typing import Any, Iterator, Optional

from prefect_ray.context import remote_options as _ray_remote_options


@contextmanager
def task_options(num_gpus: int = 0, cpus_per_task: Optional[int] = None) -> Iterator[Any]:
    ray_options: dict = {"num_gpus": num_gpus}
    if cpus_per_task is not None:
        ray_options["num_cpus"] = int(cpus_per_task)
    with _ray_remote_options(**ray_options) as opts:
        yield opts
