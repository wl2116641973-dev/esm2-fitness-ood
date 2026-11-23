"""Task registry generation and completeness / missingness audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pandas as pd

CORE_SPLITS = ("random", "modulo", "contiguous")
FOLDS = (0, 1, 2, 3, 4)


def build_core_task_manifest(
    panel_json: Path,
    learning_curve_ids: tuple[str, ...] = (
        "OXDA_RHOTO_Vanella_2023_activity",
        "ESTA_BACSU_Nutschel_2020",
        "TPMT_HUMAN_Matreyek_2018",
    ),
    main_seed: int = 42,
    robustness_seed: int = 1042,
) -> pd.DataFrame:
    """Build the deterministic core LoRA/baseline task manifest (285 LoRA tasks across Wave 1-3)."""
    with open(panel_json, "r", encoding="utf-8") as f:
        panel_ids = [p["dms_id"] for p in json.load(f)["proteins"]]

    rows = []
    # Wave 1: 10 proteins x 3 splits x 5 folds @ N=256, seed=main_seed (150 tasks)
    for dms_id in panel_ids:
        for split in CORE_SPLITS:
            for fold in FOLDS:
                rows.append((dms_id, split, fold, 256, main_seed, "wave1_main"))

    # Wave 2: 3 learning-curve proteins x 3 splits x 5 folds @ N in {64, 1024}, seed=main_seed (90 tasks)
    for dms_id in learning_curve_ids:
        for n_total in (64, 1024):
            for split in CORE_SPLITS:
                for fold in FOLDS:
                    rows.append((dms_id, split, fold, n_total, main_seed, "wave2_curve"))

    # Wave 3: 3 learning-curve proteins x 3 splits x 5 folds @ N=256, seed=robustness_seed (45 tasks)
    for dms_id in learning_curve_ids:
        for split in CORE_SPLITS:
            for fold in FOLDS:
                rows.append((dms_id, split, fold, 256, robustness_seed, "wave3_seed"))

    df = pd.DataFrame(
        rows, columns=["dms_id", "split_scheme", "test_fold", "n_total", "seed", "wave"]
    )
    df["task_id"] = [
        hashlib.sha1(
            f"{r.dms_id}|{r.split_scheme}|{r.test_fold}|{r.n_total}|{r.seed}".encode("utf-8")
        ).hexdigest()[:12]
        for r in df.itertuples(index=False)
    ]
    return df


def audit_manifest_completeness(
    manifest_df: pd.DataFrame, completed_task_ids: set[str]
) -> dict:
    """Audit completed tasks against the expected manifest and explicitly flag missing entries."""
    expected_ids = set(manifest_df["task_id"])
    missing_ids = sorted(expected_ids - completed_task_ids)
    extra_ids = sorted(completed_task_ids - expected_ids)

    missing_df = manifest_df[manifest_df["task_id"].isin(missing_ids)].copy()
    return {
        "n_expected": int(len(expected_ids)),
        "n_completed": int(len(expected_ids & completed_task_ids)),
        "n_missing": int(len(missing_ids)),
        "n_extra": int(len(extra_ids)),
        "is_complete": len(missing_ids) == 0,
        "missing_records": missing_df.to_dict(orient="records"),
    }
