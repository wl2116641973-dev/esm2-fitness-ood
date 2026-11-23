"""Smoke test for local CPU-only environment."""

import importlib.util
import esm2_ood


def test_package_version() -> None:
    assert esm2_ood.__version__ == "0.1.0"


def test_no_torch_installed_locally() -> None:
    assert importlib.util.find_spec("torch") is None, (
        "Local .venv must remain CPU-only without PyTorch/CUDA."
    )
