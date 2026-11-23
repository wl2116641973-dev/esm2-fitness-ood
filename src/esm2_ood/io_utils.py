"""Atomic staging and SHA256 readback verification for Colab <-> Drive persistence."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil


def sha256_file(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_copy_and_verify(src_path: Path, dst_path: Path) -> str:
    """Copy `src_path` to `dst_path` via a temp file + fsync + atomic replace, then verify SHA256."""
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    if not src_path.exists():
        raise FileNotFoundError(f"Source file does not exist: {src_path}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    src_hash = sha256_file(src_path)
    tmp_path = dst_path.with_name(f".{dst_path.name}.tmp.{os.getpid()}")

    try:
        with open(src_path, "rb") as fin, open(tmp_path, "wb") as fout:
            shutil.copyfileobj(fin, fout, length=1 << 20)
            fout.flush()
            os.fsync(fout.fileno())
        os.replace(tmp_path, dst_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    dst_hash = sha256_file(dst_path)
    if dst_hash != src_hash:
        raise IOError(
            f"Readback SHA256 mismatch after atomic copy to {dst_path}: "
            f"src={src_hash}, dst={dst_hash}"
        )
    return dst_hash
