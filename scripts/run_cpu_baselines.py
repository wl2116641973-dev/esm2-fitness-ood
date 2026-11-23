"""Run CPU-only baselines (AA-44 Ridge & Site-Mean) across the core manifest with memory guard."""

from __future__ import annotations

import json
import os
from pathlib import Path
import warnings
import pandas as pd
import psutil

from esm2_ood.baselines import AA44RidgeBaseline, SiteMeanBaseline
from esm2_ood.manifest import build_core_task_manifest
from esm2_ood.metrics import compute_mse, compute_precision_at_k_and_ef, compute_spearman
from esm2_ood.splits import get_outer_split, load_panel_dms, sample_budget_split

ROOT = Path(__file__).resolve().parents[1]
CV_DIR = ROOT / "data" / "cv_folds"
PANEL_JSON = ROOT / "configs" / "panel_10.json"
OUT_DIR = ROOT / "outputs" / "tables"


def check_memory_sentinel() -> None:
    """Enforce Section 6.1 memory guard: warn if free RAM < 3.0 GiB, check RSS < 400 MB."""
    avail_gib = psutil.virtual_memory().available / (1024**3)
    rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    if avail_gib < 3.0:
        warnings.warn(
            f"Low system available RAM: {avail_gib:.2f} GiB (< 3.0 GiB). Close browser tabs if needed.",
            RuntimeWarning,
        )
    if rss_mb > 400.0:
        warnings.warn(
            f"High single-process RSS: {rss_mb:.1f} MB (> 400 MB).",
            RuntimeWarning,
        )


def main() -> None:
    check_memory_sentinel()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_core_task_manifest(PANEL_JSON)

    # Cache loaded DMS dataframes in memory (10 files total < 50 MB)
    dms_cache = {
        dms_id: load_panel_dms(dms_id, CV_DIR)
        for dms_id in manifest["dms_id"].unique()
    }

    records = []
    for row in manifest.itertuples(index=False):
        df = dms_cache[row.dms_id]
        train_pool, val_pool, test_df = get_outer_split(df, row.split_scheme, row.test_fold)
        sampled = sample_budget_split(
            train_pool, val_pool, test_df, n_total=row.n_total, seed=row.seed
        )
        tr = sampled["train_df"]
        va = sampled["val_df"]
        te = sampled["test_df"]
        y_te = te["DMS_score"].to_numpy()
        meta = sampled["meta"]

        # 1. Site-Mean baseline
        sm = SiteMeanBaseline().fit(tr)
        sm_pred = sm.predict(te)
        sm_rho = compute_spearman(y_te, sm_pred)
        sm_mse = compute_mse(y_te, sm_pred)
        sm_p10 = compute_precision_at_k_and_ef(y_te, sm_pred, k=10, elite_frac=0.10)

        # 2. AA-44 Ridge baseline
        aa44 = AA44RidgeBaseline().fit(tr, va)
        aa_pred = aa44.predict(te)
        aa_rho = compute_spearman(y_te, aa_pred)
        aa_mse = compute_mse(y_te, aa_pred)
        aa_p10 = compute_precision_at_k_and_ef(y_te, aa_pred, k=10, elite_frac=0.10)

        for method, rho, mse, p10, alpha in (
            ("site_mean", sm_rho, sm_mse, sm_p10, None),
            ("aa44_ridge", aa_rho, aa_mse, aa_p10, aa44.best_alpha),
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
    res_df.to_csv(OUT_DIR / "cpu_baselines_per_fold.csv", index=False)

    # Main Wave-1 summary (N=256, seed=42): average 5 folds within each protein first!
    w1 = res_df[res_df["wave"] == "wave1_main"].copy()
    protein_means = (
        w1.groupby(["dms_id", "split_scheme", "method"], as_index=False)
        .agg(
            spearman_mean=("spearman", "mean"),
            mse_mean=("mse", "mean"),
            test_seen_site_frac=("test_seen_site_frac", "mean"),
            mean_min_1d_dist=("mean_min_1d_dist", "mean"),
            p_unique_train=("p_unique_train", "mean"),
        )
    )
    protein_means.to_csv(OUT_DIR / "cpu_baselines_wave1_protein_means.csv", index=False)

    macro = (
        protein_means.groupby(["split_scheme", "method"], as_index=False)
        .agg(
            n_proteins=("dms_id", "nunique"),
            spearman_macro_mean=("spearman_mean", "mean"),
            spearman_macro_median=("spearman_mean", "median"),
            mse_macro_mean=("mse_mean", "mean"),
            seen_site_frac=("test_seen_site_frac", "mean"),
            min_1d_dist=("mean_min_1d_dist", "mean"),
        )
    )
    print("=== Wave 1 CPU Baselines (10 proteins, N=256, 5-fold mean per protein) ===")
    print(macro.to_string(index=False))

    # Also print OXDA_RHOTO Q3 screening comparison at N=256
    oxda = w1[w1["dms_id"] == "OXDA_RHOTO_Vanella_2023_activity"].groupby(
        ["split_scheme", "method"], as_index=False
    )[["spearman", "precision_at_10", "ef_at_10pct"]].mean()
    print("\n=== OXDA_RHOTO CPU Baselines (N=256, 5-fold mean) ===")
    print(oxda.to_string(index=False))


if __name__ == "__main__":
    main()
