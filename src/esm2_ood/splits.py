"""Data loading, ProteinGym official 5-fold splitting, budget sampling, and ESM-2 index alignment."""

from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd

SPLIT_COLUMNS: dict[str, str] = {
    "random": "fold_random_5",
    "modulo": "fold_modulo_5",
    "contiguous": "fold_contiguous_5",
}

BUDGET_ALLOCATIONS: dict[int, tuple[int, int]] = {
    64: (48, 16),
    256: (192, 64),
    1024: (768, 256),
}

_MUTANT_RE = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")


def parse_mutant(mutant: str) -> tuple[str, int, str]:
    """Parse a single substitution code like 'A126C' into (wt_aa, pos_1based, mut_aa)."""
    match = _MUTANT_RE.match(mutant.strip())
    if not match:
        raise ValueError(f"Invalid single-substitution mutant string: {mutant!r}")
    wt_aa, pos_str, mut_aa = match.groups()
    return wt_aa, int(pos_str), mut_aa


def esm2_token_index(pos_1based: int, wt_seq: str | None = None, expected_wt: str | None = None) -> int:
    """Return 0-based token index in ESM-2 tokenized sequence (skipping <cls> at index 0).

    ESM-2 tokenizers prepend <cls> at token index 0, followed by residue 1 at index 1, ...,
    residue L at index L. Therefore 1-based position `p` maps directly to token index `p`.
    """
    if pos_1based < 1:
        raise ValueError(f"1-based residue position must be >= 1, got {pos_1based}")
    if wt_seq is not None:
        if pos_1based > len(wt_seq):
            raise ValueError(f"Position {pos_1based} exceeds sequence length {len(wt_seq)}")
        if expected_wt is not None and wt_seq[pos_1based - 1] != expected_wt:
            raise ValueError(
                f"Wild-type mismatch at pos {pos_1based}: seq has {wt_seq[pos_1based - 1]!r}, "
                f"expected {expected_wt!r}"
            )
    return pos_1based


def load_panel_dms(dms_id: str, cv_dir: Path) -> pd.DataFrame:
    """Load a ProteinGym single-substitution CV fold CSV and attach parsed mutation columns."""
    csv_path = cv_dir / f"{dms_id}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"DMS CV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df["mutant"].duplicated().any():
        raise ValueError(f"Duplicate mutant entries detected in {dms_id}")

    parsed = df["mutant"].map(parse_mutant)
    df = df.copy()
    df["dms_id"] = dms_id
    df["wt_aa"] = [p[0] for p in parsed]
    df["pos"] = [p[1] for p in parsed]
    df["mut_aa"] = [p[2] for p in parsed]
    return df


def get_outer_split(
    df: pd.DataFrame, split_scheme: str, test_fold: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DMS dataframe into (train_pool, val_pool, test_df) using official fold columns.

    Outer test fold is `test_fold` (0..4); inner validation fold is `(test_fold + 1) % 5`;
    remaining 3 folds form the training pool. For 'modulo' and 'contiguous', asserts that
    train, val, and test residue positions are pairwise disjoint.
    """
    if split_scheme not in SPLIT_COLUMNS:
        raise ValueError(f"Unknown split_scheme {split_scheme!r}; expected {list(SPLIT_COLUMNS)}")
    if not 0 <= test_fold <= 4:
        raise ValueError(f"test_fold must be in 0..4, got {test_fold}")

    fold_col = SPLIT_COLUMNS[split_scheme]
    val_fold = (test_fold + 1) % 5

    test_df = df[df[fold_col] == test_fold].copy().reset_index(drop=True)
    val_pool = df[df[fold_col] == val_fold].copy().reset_index(drop=True)
    train_pool = df[~df[fold_col].isin([test_fold, val_fold])].copy().reset_index(drop=True)

    if split_scheme in ("modulo", "contiguous"):
        train_pos = set(train_pool["pos"])
        val_pos = set(val_pool["pos"])
        test_pos = set(test_df["pos"])
        if (train_pos & val_pos) or (train_pos & test_pos) or (val_pos & test_pos):
            raise AssertionError(
                f"Position leakage detected in {split_scheme} fold {test_fold}: "
                f"|tr&va|={len(train_pos & val_pos)}, "
                f"|tr&te|={len(train_pos & test_pos)}, "
                f"|va&te|={len(val_pos & test_pos)}"
            )

    return train_pool, val_pool, test_df


def sample_budget_split(
    train_pool: pd.DataFrame,
    val_pool: pd.DataFrame,
    test_df: pd.DataFrame,
    n_total: int,
    seed: int = 42,
) -> dict:
    """Sample a total-budget-matched (train, val) pair using deterministic prefix ordering.

    Smaller budgets are strict prefixes of larger budgets for the same (fold, seed).
    """
    if n_total not in BUDGET_ALLOCATIONS:
        raise ValueError(f"Unsupported n_total={n_total}; expected {sorted(BUDGET_ALLOCATIONS)}")
    n_train_target, n_val_target = BUDGET_ALLOCATIONS[n_total]

    # Sort canonically by mutant before permuting so row order in source CSV never alters sampling
    train_sorted = train_pool.sort_values("mutant", kind="mergesort").reset_index(drop=True)
    val_sorted = val_pool.sort_values("mutant", kind="mergesort").reset_index(drop=True)

    rng_train = np.random.default_rng(seed)
    rng_val = np.random.default_rng(seed + 100_003)

    train_perm = rng_train.permutation(len(train_sorted))
    val_perm = rng_val.permutation(len(val_sorted))

    n_train = min(n_train_target, len(train_sorted))
    n_val = min(n_val_target, len(val_sorted))

    train_df = train_sorted.iloc[train_perm[:n_train]].copy().reset_index(drop=True)
    val_df = val_sorted.iloc[val_perm[:n_val]].copy().reset_index(drop=True)

    train_pos = np.array(sorted(train_df["pos"].unique()), dtype=int)
    test_pos = test_df["pos"].to_numpy(dtype=int)

    seen_frac = float(np.mean(np.isin(test_pos, train_pos))) if len(test_pos) > 0 else float("nan")
    if len(train_pos) > 0 and len(test_pos) > 0:
        min_1d = np.min(np.abs(test_pos[:, None] - train_pos[None, :]), axis=1)
        mean_min_1d_dist = float(np.mean(min_1d))
    else:
        mean_min_1d_dist = float("nan")

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "meta": {
            "n_total_requested": n_total,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": len(test_df),
            "p_unique_train": int(len(train_pos)),
            "p_unique_val": int(val_df["pos"].nunique()),
            "p_unique_test": int(test_df["pos"].nunique()),
            "test_seen_site_frac": seen_frac,
            "mean_min_1d_dist": mean_min_1d_dist,
            "budget_capped": (n_train < n_train_target) or (n_val < n_val_target),
        },
    }
