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
    assert callable(colab_gpu.train_eval_esm2_lora_llr)
    assert callable(colab_gpu.train_eval_esm2_lora_linear_head)


def test_fit_ols_affine_calibration_and_guards() -> None:
    import numpy as np

    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = 2.5 * x - 1.25
    slope, intercept = colab_gpu._fit_ols_affine(x, y)
    assert abs(slope - 2.5) < 1e-10
    assert abs(intercept - (-1.25)) < 1e-10

    # Zero-variance guard
    s_const, b_const = colab_gpu._fit_ols_affine(np.array([3.0, 3.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert (s_const, b_const) == (1.0, 0.0)

    # Insufficient sample guard
    s_one, b_one = colab_gpu._fit_ols_affine(np.array([1.0]), np.array([2.0]))
    assert (s_one, b_one) == (1.0, 0.0)


def test_wave0b_notebook_has_exactly_five_code_cells() -> None:
    nb_path = Path(__file__).resolve().parents[1] / "notebooks" / "wave0b_colab_runner.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 5
