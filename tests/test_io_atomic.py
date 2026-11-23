"""Tests for atomic Drive staging/verification and lazy GPU imports."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from esm2_ood import colab_gpu
from esm2_ood.io_utils import atomic_copy_and_verify, sha256_file


def test_atomic_copy_and_verify(tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    dst = tmp_path / "nested" / "dst.bin"
    payload = b"proteingym-wave0b-atomic-test" * 100
    src.write_bytes(payload)

    digest = atomic_copy_and_verify(src, dst)
    assert dst.exists()
    assert dst.read_bytes() == payload
    assert digest == sha256_file(dst)
    # Ensure no leftover temp files remain
    assert [p.name for p in dst.parent.iterdir()] == ["dst.bin"]


def test_colab_gpu_does_not_import_torch_at_module_load() -> None:
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
    assert "peft" not in sys.modules
    assert "650M" in colab_gpu.MODEL_IDS


def test_wave0b_notebook_has_exactly_five_code_cells() -> None:
    nb_path = Path(__file__).resolve().parents[1] / "notebooks" / "wave0b_colab_runner.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 5
