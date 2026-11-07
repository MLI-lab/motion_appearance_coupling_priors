"""Download (mode=url) or validate (mode=manual) a raw dataset, per hydra/task/download_dataset/*.yaml.

No Prefect caching: a cached "validated" result would outlive a deleted folder.
"""

from __future__ import annotations

import concurrent.futures
import glob
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import tqdm
from prefect import task
from prefect.cache_policies import NONE

from src.prefect.caching import CacheableListConfig
from src.utils.data_roots import COMPLETE_MARKER, DataRoots, is_complete, mark_complete


# --- HTTP ----------------------------------------------------------------------

def _download_file_sequential(url: str, dest: Path, chunk_size: int, pbar: tqdm.tqdm) -> None:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def _download_range_chunk(url: str, dest: Path, start: int, end: int, chunk_size: int, pbar: tqdm.tqdm, pbar_lock: threading.Lock) -> None:
    # identity encoding: ranged chunks must be the raw bytes at their file offset
    with requests.get(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "r+b") as f:  # own handle per thread: independent offsets
            f.seek(start)
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    with pbar_lock:
                        pbar.update(len(chunk))


def download_file(url: str, dest: Path, chunk_size: int = 1024 * 1024, num_connections: int = 1) -> None:
    """Download `url` to `dest`, splitting into `num_connections` concurrent HTTP
    range requests when the server supports them (HF, Zenodo and MIT CSAIL do),
    falling back to one sequential stream otherwise."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    head = requests.head(url, headers={"Accept-Encoding": "identity"}, timeout=30, allow_redirects=True)
    total = int(head.headers.get("Content-Length", 0))
    supports_ranges = head.headers.get("Accept-Ranges", "").lower() == "bytes"

    if total <= 0 or not supports_ranges:
        # Some CDN redirects don't echo these on HEAD -- probe with a 1-byte ranged GET.
        probe = requests.get(url, headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"}, timeout=30)
        content_range = probe.headers.get("Content-Range", "")
        if probe.status_code == 206 and "/" in content_range:
            total = int(content_range.rsplit("/", 1)[-1])
            supports_ranges = True

    if num_connections <= 1 or not supports_ranges or total <= 0:
        with tqdm.tqdm(total=total or None, unit="B", unit_scale=True, unit_divisor=1024, desc=f"Downloading {dest.name}") as pbar:
            _download_file_sequential(url, dest, chunk_size, pbar)
        return

    with open(dest, "wb") as f:
        f.truncate(total)
    part_size = -(-total // num_connections)
    ranges = [(s, min(s + part_size - 1, total - 1)) for s in range(0, total, part_size)]
    pbar_lock = threading.Lock()
    with tqdm.tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=f"Downloading {dest.name} ({len(ranges)} conns)") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(ranges)) as executor:
            futures = [executor.submit(_download_range_chunk, url, dest, s, e, chunk_size, pbar, pbar_lock) for s, e in ranges]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()
    actual = dest.stat().st_size
    if actual != total:
        raise RuntimeError(f"Downloaded size mismatch for {dest.name}: expected {total} bytes, got {actual}")


def _hashsum(path: Path, algo: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _checksum_ok(path: Path, checksum: Optional[str]) -> bool:
    if not checksum:
        return True
    algo, expected = str(checksum).split(":", 1)
    return _hashsum(path, algo) == expected


# --- archives -------------------------------------------------------------------

def _combine_parts(part_paths: List[Path], out_path: Path) -> None:
    """Join split zip parts. Plain concatenation first (a raw byte-split of one
    zip, the usual case); Info-Zip's archive-join or 7z as fallback for a
    genuine multi-volume archive."""
    with open(out_path, "wb") as out_f:
        for part_path in part_paths:
            with open(part_path, "rb") as part_f:
                shutil.copyfileobj(part_f, out_f)
    if zipfile.is_zipfile(out_path):
        return
    out_path.unlink(missing_ok=True)
    if shutil.which("zip") is not None:
        subprocess.run(["zip", "-FF", str(part_paths[0]), "--out", str(out_path)], check=True)
    elif shutil.which("7z") is not None:
        subprocess.run(["7z", "x", f"-o{out_path.parent}", str(part_paths[0])], check=True)
    else:
        raise RuntimeError("Could not join the split zip parts: plain concatenation is not a valid zip and neither 'zip' nor '7z' is installed.")
    if not zipfile.is_zipfile(out_path):
        raise RuntimeError(f"Failed to produce a valid zip archive at {out_path} from {part_paths}")


def _extract_archive(archive: Path, into: Path, kind: str) -> None:
    into.mkdir(parents=True, exist_ok=True)
    if kind == "zip":
        if shutil.which("unzip") is not None:
            subprocess.run(["unzip", "-q", "-o", str(archive), "-d", str(into)], check=True)
        else:
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(into)
    elif kind in ("tar", "tar.gz", "tgz"):
        subprocess.run(["tar", "-xf", str(archive), "-C", str(into)], check=True)
    else:
        raise ValueError(f"Unknown archive kind {kind!r} (zip | tar | tar.gz)")


def _move_tree_contents(src: Path, dst: Path) -> None:
    """Move every entry of src into dst (merging into existing directories)."""
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        target = dst / entry.name
        if entry.is_dir() and target.is_dir():
            _move_tree_contents(entry, target)
            entry.rmdir()
        else:
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(entry), str(target))


# --- layout validation ---------------------------------------------------------

def check_layout(target_dir: Path, expected_layout: List[Dict[str, Any]], quick: bool = False) -> List[str]:
    """Returns one message per unsatisfied `{glob, min_count}` entry (empty = OK).

    quick=True only requires one match per glob (for data that already passed the full check)."""
    problems = []
    for entry in expected_layout:
        pattern = str(entry["glob"])
        matches = glob.iglob(os.path.join(str(target_dir), pattern), recursive=True)
        if quick:
            if next(matches, None) is None:
                problems.append(f"{pattern!r}: no match")
            continue
        min_count = int(entry.get("min_count", 1))
        found = sum(1 for _ in matches)
        if found < min_count:
            problems.append(f"{pattern!r}: found {found}, need >= {min_count}")
    return problems


def _staged_marker(target: Path) -> Path:
    """Completion marker next to (not inside) a cache symlink to a hand-staged copy."""
    return target.parent / f"{target.name}.{COMPLETE_MARKER}"


def _is_present(target: Path, expected_layout: List[Dict[str, Any]]) -> bool:
    """Fast path: the full layout check passed before (marker) and the data is still there (quick check)."""
    marker_ok = _staged_marker(target).is_file() if target.is_symlink() else is_complete(target)
    return marker_ok and target.is_dir() and not check_layout(target, expected_layout, quick=True)


def _manual_help(dataset_name: str, target_dir: Path, source: str, instructions: str, problems: List[str]) -> str:
    lines = [
        f"Dataset '{dataset_name}' is not (fully) present at {target_dir}",
        "This dataset is staged by hand (download.mode=manual); it is never downloaded or written by the flows.",
    ]
    if source:
        lines.append(f"Source: {source}")
    if instructions:
        lines.append(f"Instructions: {instructions.strip()}")
    lines.append("Unsatisfied layout checks (relative to that folder):")
    lines += [f"  - {p}" for p in problems]
    return "\n".join(lines)


def _link_target_to_staged_copy(target: Path, staged: Path) -> None:
    """Make the cache `target` a symlink to the hand-staged `staged` folder.

    Anything already at `target` is an incomplete download and is removed.
    """
    if target.is_symlink():
        if target.resolve() == staged.resolve():
            return
        target.unlink()
        _staged_marker(target).unlink(missing_ok=True)
    elif target.is_dir():
        print(f"[download_dataset] removing partial cache directory {target}")
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(staged.resolve(), target)


# --- the task ------------------------------------------------------------------

@task(cache_policy=NONE, name="Download Dataset", tags=["preprocess"], version="1.0", retries=2, retry_delay_seconds=30)
def download_dataset(
    dataset_name: str,
    target_dir: str,
    mode: str,
    data_roots: DataRoots,
    expected_layout_cfg: Optional[CacheableListConfig] = None,
    files_cfg: Optional[CacheableListConfig] = None,
    source: str = "",
    instructions: str = "",
    verify_checksums: bool = True,
    cleanup_after_extract: bool = True,
    download_parallel_files: int = 2,
    download_connections_per_file: int = 8,
    manual_fallback_dir: Optional[str] = None,
) -> str:
    """Materialize (url) or validate (manual) a raw dataset; returns its root-relative path.

    url mode with `manual_fallback_dir` (a `manual/...` path): if the cache copy
    is not there but a hand-staged copy satisfying `expected_layout` is, the
    cache target becomes a symlink to it and nothing is downloaded -- so a
    dataset whose public URL is down can be rsynced from another cluster while
    every consumer keeps reading the `raw/<name>` cache path. The URL stays the
    primary source: a fresh cluster without the staged copy still downloads.
    """
    target = Path(data_roots.resolve(target_dir))
    expected_layout = [dict(e) for e in (expected_layout_cfg.cfg if expected_layout_cfg is not None else [])]
    rel_target = data_roots.to_relative(target)

    if mode == "manual":
        problems = check_layout(target, expected_layout) if target.is_dir() else [f"folder missing: {target}"]
        if problems:
            raise FileNotFoundError(_manual_help(dataset_name, target, source, instructions, problems))
        print(f"[download_dataset] '{dataset_name}' validated at {target}")
        return rel_target

    if mode != "url":
        raise ValueError(f"download.mode must be 'url' or 'manual', got {mode!r}")

    # Fast path: complete marker + layout still intact.
    if _is_present(target, expected_layout):
        print(f"[download_dataset] '{dataset_name}' already present at {target}, skipping.")
        return rel_target

    if manual_fallback_dir:
        staged = Path(data_roots.resolve(manual_fallback_dir))
        staged_problems = check_layout(staged, expected_layout) if staged.is_dir() else [f"folder missing: {staged}"]
        if not staged_problems:
            _link_target_to_staged_copy(target, staged)
            marker = _staged_marker(target)
            tmp = marker.with_name(f"{marker.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps({"task": "download_dataset", "dataset_name": dataset_name, "staged_copy": str(staged)}, indent=2))
            os.replace(tmp, marker)
            print(f"[download_dataset] '{dataset_name}' using the hand-staged copy at {staged} ({target} -> symlink); no download.")
            return rel_target
        if staged.is_dir():
            print(
                f"[download_dataset] '{dataset_name}': staged copy at {staged} does not satisfy the expected layout yet "
                f"(still copying?), falling back to the download:\n  - " + "\n  - ".join(staged_problems)
            )

    if target.is_symlink():
        # A stale link to a staged copy that no longer validates: download into
        # a real cache directory, never through the link into the manual root.
        target.unlink()
        _staged_marker(target).unlink(missing_ok=True)

    files = [dict(f) for f in (files_cfg.cfg if files_cfg is not None else [])]
    if not files:
        raise ValueError(f"'{dataset_name}': download.mode=url but no files listed")
    target.mkdir(parents=True, exist_ok=True)
    staging = target / "_download_staging"
    staging.mkdir(parents=True, exist_ok=True)

    def _fetch(entry: Dict[str, Any]) -> Path:
        dest = staging / str(entry["key"])
        if dest.exists() and (not verify_checksums or _checksum_ok(dest, entry.get("checksum"))):
            print(f"[download_dataset] {dest.name} already downloaded and verified -- skipping.")
            return dest
        print(f"[download_dataset] downloading {entry['url']} -> {dest} ({download_connections_per_file} connections)")
        download_file(str(entry["url"]), dest, num_connections=int(download_connections_per_file))
        if verify_checksums and entry.get("checksum") and not _checksum_ok(dest, entry["checksum"]):
            dest.unlink(missing_ok=True)
            raise ValueError(f"Checksum mismatch for {dest.name} (expected {entry['checksum']}); deleted so a retry re-fetches it.")
        return dest

    # Flatten multi-part entries into individual downloads.
    to_fetch: List[Dict[str, Any]] = []
    for entry in files:
        if entry.get("parts"):
            to_fetch += [dict(p) for p in entry["parts"]]
        else:
            to_fetch.append(entry)
    fetched: Dict[str, Path] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(download_parallel_files))) as executor:
        futures = {executor.submit(_fetch, e): str(e["key"]) for e in to_fetch}
        for fut in concurrent.futures.as_completed(futures):
            fetched[futures[fut]] = fut.result()

    # Extract (or place) each file.
    for entry in files:
        key = str(entry["key"])
        if entry.get("parts"):
            archive = staging / key
            if not (archive.exists() and zipfile.is_zipfile(archive)):
                _combine_parts([fetched[str(p["key"])] for p in entry["parts"]], archive)
        else:
            archive = fetched[key]
        kind = entry.get("extract")
        dest_subdir = target / str(entry.get("extract_to", "") or "")
        if not kind:
            # Not an archive: keep the file itself under the target directory.
            # `key` may carry a sub-path (e.g. "vae/config.json" for a diffusers
            # checkpoint), so the file lands at <target>/<extract_to>/<key>.
            final = dest_subdir / key
            final.parent.mkdir(parents=True, exist_ok=True)
            if not final.exists():
                shutil.copy2(archive, final)
            continue
        with tempfile.TemporaryDirectory(dir=staging, prefix=f"extract_{key}_") as tmp:
            print(f"[download_dataset] extracting {archive.name} ({kind})")
            _extract_archive(archive, Path(tmp), str(kind))
            strip_prefix = entry.get("strip_prefix")
            src_root = Path(tmp) / str(strip_prefix) if strip_prefix else Path(tmp)
            if not src_root.is_dir():
                raise RuntimeError(f"{archive.name}: strip_prefix {strip_prefix!r} not found after extraction (top level: {sorted(os.listdir(tmp))})")
            _move_tree_contents(src_root, dest_subdir)

    problems = check_layout(target, expected_layout)
    if problems:
        raise RuntimeError(
            f"'{dataset_name}' was downloaded and extracted to {target} but the expected layout is not there:\n  - "
            + "\n  - ".join(problems)
            + "\nThe archive may unpack into a different layout than the config assumes (check strip_prefix/extract_to)."
        )
    if cleanup_after_extract:
        shutil.rmtree(staging, ignore_errors=True)
    mark_complete(target, {"task": "download_dataset", "dataset_name": dataset_name, "files": [f.get("key") for f in files], "source": source})
    print(f"[download_dataset] '{dataset_name}' ready at {target}")
    return rel_target
