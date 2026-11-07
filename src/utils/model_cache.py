"""Race-free download of a public checkpoint into <data_cache_root>/model_cache.

Each worker writes to its own temp file and os.replace()s it into place; a best-effort flock
makes concurrent workers wait instead of downloading again.
"""

from __future__ import annotations

import contextlib
import os
from typing import Iterator, Optional


@contextlib.contextmanager
def _best_effort_lock(lock_path: str) -> Iterator[None]:
    """Exclusive flock on lock_path; degrades to no lock where the FS does not support it
    (the temp-file + rename below stays safe either way, only duplicate downloads remain)."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX
        yield
        return
    with open(lock_path, "a") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        except OSError:
            yield
            return
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def ensure_cached_download(url: str, cache_dir: str, filename: Optional[str] = None, chunk_size: int = 1 << 20) -> str:
    """Returns the local path of `url` inside `cache_dir`, downloading it once if needed."""
    os.makedirs(cache_dir, exist_ok=True)
    target = os.path.join(cache_dir, filename or url.split("/")[-1])
    if os.path.exists(target):
        return target

    with _best_effort_lock(target + ".lock"):
        if os.path.exists(target):  # another worker finished it while we waited
            return target
        import requests

        print(f"Downloading model from {url} to {target}...")
        tmp = f"{target}.tmp.{os.getpid()}"
        try:
            with requests.get(url, stream=True, timeout=(30, 600)) as response:
                response.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    return target
