"""Evaluation metrics: constant-safe Spearman, tie-aware order-invariant Precision@k/EF, and block bootstrap."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def compute_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Compute Spearman rank correlation; return None (not 0.0) if either array is constant."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if len(yt) < 2 or np.ptp(yt) == 0.0 or np.ptp(yp) == 0.0:
        return None
    rho = float(spearmanr(yt, yp).statistic)
    if np.isnan(rho):
        return None
    return rho


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((yt - yp) ** 2))


def _elite_membership_weights(y_true: np.ndarray, elite_frac: float) -> np.ndarray:
    """Compute order-invariant fractional elite indicators summing to m = max(1, ceil(elite_frac * N))."""
    n = len(y_true)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    m = max(1, int(np.ceil(elite_frac * n)))
    sorted_vals = np.sort(y_true)[::-1]
    cutoff = sorted_vals[m - 1]

    strictly_above = y_true > cutoff
    at_cutoff = y_true == cutoff
    n_above = int(np.sum(strictly_above))
    n_tie = int(np.sum(at_cutoff))

    weights = np.zeros(n, dtype=np.float64)
    weights[strictly_above] = 1.0
    if n_tie > 0:
        weights[at_cutoff] = (m - n_above) / n_tie
    return weights


def compute_precision_at_k_and_ef(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 10,
    elite_frac: float = 0.10,
) -> dict[str, float]:
    """Compute exact tie-aware expected Precision@k and Enrichment Factor (invariant to row order)."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    n = len(yt)
    if n == 0 or k <= 0:
        raise ValueError(f"Invalid inputs: n={n}, k={k}")
    k_eff = min(k, n)

    elite_w = _elite_membership_weights(yt, elite_frac=elite_frac)
    random_expected_precision = float(np.sum(elite_w) / n)

    # Group by exact unique prediction values in descending order
    unique_scores = np.unique(yp)[::-1]
    rem = float(k_eff)
    expected_hits = 0.0
    for score in unique_scores:
        if rem <= 0.0:
            break
        mask = yp == score
        group_size = float(np.sum(mask))
        group_elite_sum = float(np.sum(elite_w[mask]))
        if group_size <= rem:
            expected_hits += group_elite_sum
            rem -= group_size
        else:
            expected_hits += (rem / group_size) * group_elite_sum
            rem = 0.0

    precision_at_k = expected_hits / k_eff
    ef = (
        precision_at_k / random_expected_precision
        if random_expected_precision > 0.0
        else float("nan")
    )
    return {
        "k": float(k_eff),
        "expected_hits": float(expected_hits),
        "precision_at_k": float(precision_at_k),
        "random_precision": float(random_expected_precision),
        "enrichment_factor": float(ef),
    }


def site_block_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positions: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float | None, float | None]:
    """Within-fold site-stratified block bootstrap 95% CI for Spearman correlation."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    pos = np.asarray(positions)
    unique_pos = np.unique(pos)
    if len(unique_pos) < 2:
        return None, None

    pos_indices = [np.where(pos == p)[0] for p in unique_pos]
    rng = np.random.default_rng(seed)
    boot_vals = []
    n_sites = len(pos_indices)
    for _ in range(n_boot):
        chosen = rng.integers(0, n_sites, size=n_sites)
        idx = np.concatenate([pos_indices[c] for c in chosen])
        rho = compute_spearman(yt[idx], yp[idx])
        if rho is not None:
            boot_vals.append(rho)
    if len(boot_vals) < 10:
        return None, None
    lo = float(np.quantile(boot_vals, alpha / 2.0))
    hi = float(np.quantile(boot_vals, 1.0 - alpha / 2.0))
    return lo, hi
