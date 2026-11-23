"""Run local CPU baselines on cached 650M zero-shot & fp16 diff embeddings:
Zero-Shot (650M), FrozenDiff-Ridge (650M), and Hsu-Aug (650M) across the core manifest.
"""

from __future__ import annotations

import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import psutil

from esm2_ood.baselines import FrozenDiffRidgeBaseline, HsuAugmentedRidgeBaseline
from esm2_ood.manifest import build_core_task_manifest
from esm2_ood.metrics import compute_mse, compute_precision_at_k_and_ef, compute_spearman
from esm2_ood.splits import get_outer_split, load_panel_dms, sample_budget_split

ROOT = Path(__file__).resolve().parents[1]
CV_DIR = ROOT / "data" / "cv_folds"
PANEL_JSON = ROOT / "configs" / "panel_10.json"
CACHE_DIR = ROOT / "outputs" / "wave1_cache"
OUT_DIR = ROOT / "outputs" / "tables"


def check_memory_sentinel() -> None:
    avail_gib = psutil.virtual_memory().available / (1024**3)
    rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    if avail_gib < 3.0:
        warnings.warn(f"Low available RAM: {avail_gib:.2f} GiB", RuntimeWarning)
    if rss_mb > 1000.0:
        warnings.warn(f"High single-process RSS: {rss_mb:.1f} MB", RuntimeWarning)


def main() -> None:
    check_memory_sentinel()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_core_task_manifest(PANEL_JSON)

    dms_cache: dict[str, pd.DataFrame] = {}
    emb_cache: dict[str, tuple[dict[str, int], np.ndarray]] = {}

    for dms_id in manifest["dms_id"].unique():
        zs_path = CACHE_DIR / f"{dms_id}_zeroshot.parquet"
        npz_path = CACHE_DIR / f"{dms_id}_650M_diff_fp16.npz"
        if not zs_path.exists() or not npz_path.exists():
            raise FileNotFoundError(
                f"Missing cached 650M artifacts for {dms_id} in {CACHE_DIR}. "
                "Run Wave 1 cache extraction first."
            )
        df = load_panel_dms(dms_id, CV_DIR)
        zs_df = pd.read_parquet(zs_path)[["mutant", "zs_650M"]]
        df = df.merge(zs_df, on="mutant", how="left", validate="one_to_one")
        if df["zs_650M"].isna().any():
            raise ValueError(f"NaN zs_650M scores after merge for {dms_id}")
        dms_cache[dms_id] = df

        with np.load(npz_path, allow_pickle=False) as npz:
            muts = npz["mutant"]
            diffs = npz["diffs"].astype(np.float32)
        emb_cache[dms_id] = ({str(m): i for i, m in enumerate(muts)}, diffs)

    records = []
    for row in manifest.itertuples(index=False):
        df = dms_cache[row.dms_id]
        mut_to_row, diffs_all = emb_cache[row.dms_id]

        train_pool, val_pool, test_df = get_outer_split(df, row.split_scheme, row.test_fold)
        sampled = sample_budget_split(
            train_pool, val_pool, test_df, n_total=row.n_total, seed=row.seed
        )
        tr = sampled["train_df"]
        va = sampled["val_df"]
        te = sampled["test_df"]
        y_tr = tr["DMS_score"].to_numpy(dtype=np.float64)
        y_va = va["DMS_score"].to_numpy(dtype=np.float64)
        y_te = te["DMS_score"].to_numpy(dtype=np.float64)
        meta = sampled["meta"]

        # 1. Zero-Shot (650M)
        zs_pred = te["zs_650M"].to_numpy(dtype=np.float64)
        zs_rho = compute_spearman(y_te, zs_pred)
        zs_mse = compute_mse(y_te, zs_pred)
        zs_p10 = compute_precision_at_k_and_ef(y_te, zs_pred, k=10, elite_frac=0.10)

        # 2. FrozenDiff-Ridge (650M)
        idx_tr = [mut_to_row[str(m)] for m in tr["mutant"]]
        idx_va = [mut_to_row[str(m)] for m in va["mutant"]]
        idx_te = [mut_to_row[str(m)] for m in te["mutant"]]
        fr = FrozenDiffRidgeBaseline().fit(diffs_all[idx_tr], y_tr, diffs_all[idx_va], y_va)
        fr_pred = fr.predict(diffs_all[idx_te])
        fr_rho = compute_spearman(y_te, fr_pred)
        fr_mse = compute_mse(y_te, fr_pred)
        fr_p10 = compute_precision_at_k_and_ef(y_te, fr_pred, k=10, elite_frac=0.10)

        # 3. Hsu-Aug (650M)
        hsu = HsuAugmentedRidgeBaseline().fit(
            tr, tr["zs_650M"].to_numpy(), va, va["zs_650M"].to_numpy()
        )
        hsu_pred = hsu.predict(te, te["zs_650M"].to_numpy())
        hsu_rho = compute_spearman(y_te, hsu_pred)
        hsu_mse = compute_mse(y_te, hsu_pred)
        hsu_p10 = compute_precision_at_k_and_ef(y_te, hsu_pred, k=10, elite_frac=0.10)

        for method, rho, mse, p10, alpha in (
            ("zeroshot_650m", zs_rho, zs_mse, zs_p10, None),
            ("frozen_diff_ridge", fr_rho, fr_mse, fr_p10, fr.best_alpha),
            ("hsu_aug", hsu_rho, hsu_mse, hsu_p10, hsu.best_alpha),
        ):
            records.append(
                {
                    "task_id": row.task_id,
                    "wave": row.wave,
                    "dms_id": row.dms_id,
                    "split_scheme": row.split_scheme,
                    "test_fold": row.test_fold,
                    "n_total": row.n_total,
                    "seed": row.seed,
                    "method": method,
                    "spearman": rho,
                    "spearman_status": "NA (constant)" if rho is None else "ok",
                    "mse": mse,
                    "precision_at_10": p10["precision_at_k"],
                    "ef_at_10pct": p10["enrichment_factor"],
                    "best_alpha": alpha,
                    **meta,
                }
            )

    check_memory_sentinel()
    res_df = pd.DataFrame(records)
    res_df.to_csv(OUT_DIR / "wave1_frozen_baselines_per_fold.csv", index=False)

    w1 = res_df[res_df["wave"] == "wave1_main"].copy()
    protein_means = (
        w1.groupby(["dms_id", "split_scheme", "method"], as_index=False)
        .agg(
            spearman_mean=("spearman", "mean"),
            mse_mean=("mse", "mean"),
            precision_at_10_mean=("precision_at_10", "mean"),
            ef_at_10pct_mean=("ef_at_10pct", "mean"),
        )
    )
    protein_means.to_csv(OUT_DIR / "wave1_frozen_baselines_protein_means.csv", index=False)

    macro = (
        protein_means.groupby(["split_scheme", "method"], as_index=False)
        .agg(
            n_proteins=("dms_id", "nunique"),
            spearman_macro_mean=("spearman_mean", "mean"),
            spearman_macro_median=("spearman_mean", "median"),
            mse_macro_mean=("mse_mean", "mean"),
        )
    )
    print("=== Wave 1 Frozen 650M Baselines (10 proteins, N=256, 5-fold mean per protein) ===")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
