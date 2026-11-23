"""Six mandatory automated anti-leakage pytest gates (Section 4.3 of PROJECT_PLAN_v3.1.md)."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from esm2_ood.baselines import HsuAugmentedRidgeBaseline, SiteMeanBaseline
from esm2_ood.manifest import audit_manifest_completeness, build_core_task_manifest
from esm2_ood.metrics import compute_precision_at_k_and_ef, compute_spearman
from esm2_ood.splits import esm2_token_index, get_outer_split, load_panel_dms, sample_budget_split

ROOT = Path(__file__).resolve().parents[1]
CV_DIR = ROOT / "data" / "cv_folds"
PANEL_JSON = ROOT / "configs" / "panel_10.json"
REF_CSV = ROOT / "data" / "raw" / "DMS_substitutions.csv"


def _load_panel_ids() -> list[str]:
    with open(PANEL_JSON, "r", encoding="utf-8") as f:
        return [p["dms_id"] for p in json.load(f)["proteins"]]


def test_gate1_disjoint_positions_and_sitemean_constant() -> None:
    """Gate 1: Train/Val/Test residue positions are pairwise disjoint under modulo & contiguous,
    and Site-Mean produces constant predictions (Spearman explicitly None, never 0.0).
    """
    panel_ids = _load_panel_ids()
    for dms_id in panel_ids:
        df = load_panel_dms(dms_id, CV_DIR)
        for split_scheme in ("modulo", "contiguous"):
            for fold in range(5):
                train_pool, val_pool, test_df = get_outer_split(df, split_scheme, fold)
                tr_pos = set(train_pool["pos"])
                va_pos = set(val_pool["pos"])
                te_pos = set(test_df["pos"])
                assert tr_pos.isdisjoint(va_pos), f"{dms_id} {split_scheme} fold {fold}: train & val overlap"
                assert tr_pos.isdisjoint(te_pos), f"{dms_id} {split_scheme} fold {fold}: train & test overlap"
                assert va_pos.isdisjoint(te_pos), f"{dms_id} {split_scheme} fold {fold}: val & test overlap"

        # Verify Site-Mean returns constant -> Spearman is None on unseen-site split
        train_pool, val_pool, test_df = get_outer_split(df, "contiguous", 0)
        sampled = sample_budget_split(train_pool, val_pool, test_df, n_total=256, seed=42)
        sm = SiteMeanBaseline().fit(sampled["train_df"])
        preds = sm.predict(sampled["test_df"])
        assert np.ptp(preds) == 0.0
        assert compute_spearman(sampled["test_df"]["DMS_score"].to_numpy(), preds) is None


def test_gate2_zeroshot_and_test_invariant_to_budget() -> None:
    """Gate 2: Outer test fold & zero-shot predictions are invariant to N; smaller budgets are prefixes."""
    df = load_panel_dms("OXDA_RHOTO_Vanella_2023_activity", CV_DIR)
    # Deterministic synthetic zero-shot scores keyed strictly by mutant
    rng = np.random.default_rng(123)
    zs_lookup = dict(zip(df["mutant"], rng.normal(size=len(df))))

    for split_scheme in ("random", "modulo", "contiguous"):
        train_pool, val_pool, test_df = get_outer_split(df, split_scheme, test_fold=2)
        runs = {
            n: sample_budget_split(train_pool, val_pool, test_df, n_total=n, seed=42)
            for n in (64, 256, 1024)
        }
        # Test variants & zero-shot Spearman must be bit-for-bit invariant across N
        base_muts = runs[64]["test_df"]["mutant"].tolist()
        base_zs = np.array([zs_lookup[m] for m in base_muts])
        base_rho = compute_spearman(runs[64]["test_df"]["DMS_score"].to_numpy(), base_zs)

        for n in (256, 1024):
            assert runs[n]["test_df"]["mutant"].tolist() == base_muts
            zs_n = np.array([zs_lookup[m] for m in runs[n]["test_df"]["mutant"]])
            assert compute_spearman(runs[n]["test_df"]["DMS_score"].to_numpy(), zs_n) == base_rho

        # Strict prefix nesting: N=64 subset is prefix of N=256, which is prefix of N=1024
        tr64 = runs[64]["train_df"]["mutant"].tolist()
        tr256 = runs[256]["train_df"]["mutant"].tolist()
        tr1024 = runs[1024]["train_df"]["mutant"].tolist()
        assert tr256[: len(tr64)] == tr64
        assert tr1024[: len(tr256)] == tr256

        va64 = runs[64]["val_df"]["mutant"].tolist()
        va256 = runs[256]["val_df"]["mutant"].tolist()
        va1024 = runs[1024]["val_df"]["mutant"].tolist()
        assert va256[: len(va64)] == va64
        assert va1024[: len(va256)] == va256


def test_gate3_esm2_cls_token_index_alignment() -> None:
    """Gate 3: ESM-2 <cls> offset (+1) aligns 1-based DMS positions to tokenized sequence indices."""
    ref = pd.read_csv(REF_CSV).set_index("DMS_id")
    for dms_id in _load_panel_ids():
        wt_seq = str(ref.loc[dms_id, "target_seq"])
        # Simulate ESM-2 tokenization: ['<cls>', res_1, res_2, ..., res_L, '<eos>']
        tokens = ["<cls>", *list(wt_seq), "<eos>"]
        df = load_panel_dms(dms_id, CV_DIR)
        for wt_aa, pos in zip(df["wt_aa"], df["pos"]):
            tok_idx = esm2_token_index(int(pos), wt_seq=wt_seq, expected_wt=wt_aa)
            assert tok_idx == int(pos)
            assert tok_idx >= 1
            assert tokens[tok_idx] == wt_aa

    # Verify OXDA shifted catalytic residues map cleanly
    oxda_seq = str(ref.loc["OXDA_RHOTO_Vanella_2023_activity", "target_seq"])
    oxda_tokens = ["<cls>", *list(oxda_seq), "<eos>"]
    for expected_aa, pos in (("Y", 222), ("Y", 237), ("R", 284)):
        idx = esm2_token_index(pos, wt_seq=oxda_seq, expected_wt=expected_aa)
        assert oxda_tokens[idx] == expected_aa


def test_gate4_hsu_degenerates_to_zeroshot_on_unseen_sites() -> None:
    """Gate 4: Hsu-Augmented Ridge reduces exactly to affine transform of zero-shot on unseen sites."""
    df = load_panel_dms("OXDA_RHOTO_Vanella_2023_activity", CV_DIR)
    # Construct correlated synthetic zero-shot signal so w_zs > 0
    rng = np.random.default_rng(7)
    df = df.copy()
    df["zs_mock"] = df["DMS_score"] * 0.8 + rng.normal(0, 0.2, size=len(df))

    for split_scheme in ("modulo", "contiguous"):
        train_pool, val_pool, test_df = get_outer_split(df, split_scheme, test_fold=1)
        sampled = sample_budget_split(train_pool, val_pool, test_df, n_total=256, seed=42)
        tr = sampled["train_df"]
        va = sampled["val_df"]
        te = sampled["test_df"]

        hsu = HsuAugmentedRidgeBaseline().fit(
            tr, tr["zs_mock"].to_numpy(), va, va["zs_mock"].to_numpy()
        )
        preds = hsu.predict(te, te["zs_mock"].to_numpy())
        zs_te = te["zs_mock"].to_numpy(dtype=np.float64)

        # Must be an exact affine function of zs_te (R^2 == 1.0, residual ~ 0)
        slope, intercept = np.polyfit(zs_te, preds, 1)
        assert slope > 0.0
        assert np.max(np.abs(preds - (slope * zs_te + intercept))) < 1e-10

        rho_hsu = compute_spearman(te["DMS_score"].to_numpy(), preds)
        rho_zs = compute_spearman(te["DMS_score"].to_numpy(), zs_te)
        assert rho_hsu is not None and rho_zs is not None
        assert rho_hsu == pytest.approx(rho_zs, abs=1e-12)


def test_gate5_tie_aware_precision_ef_order_invariant() -> None:
    """Gate 5: Exact tie-aware Precision@k and EF@10% are strictly invariant to row order."""
    # Dataset of 20 items: 2 elites (top 10% = 2), heavy ties in y_pred:
    # top group: 4 items tied at 0.9 containing 1 elite
    # second group: 10 items tied at 0.5 containing 1 elite
    # bottom group: 6 items tied at 0.1 containing 0 elites
    y_true = np.array(
        [10.0, 1.0, 2.0, 3.0] + [9.0] + [0.5] * 9 + [0.1] * 6, dtype=np.float64
    )
    y_pred = np.array(
        [0.9] * 4 + [0.5] * 10 + [0.1] * 6, dtype=np.float64
    )

    # Analytic expected hits for k=10:
    # First 4 slots take all of group 1 (1 elite). Remaining 6 slots take 6/10 of group 2 (0.6 elite).
    # Total expected hits = 1.6 -> Precision@10 = 0.16, random_precision = 2/20 = 0.10, EF = 1.6
    res = compute_precision_at_k_and_ef(y_true, y_pred, k=10, elite_frac=0.10)
    assert res["expected_hits"] == pytest.approx(1.6, abs=1e-12)
    assert res["precision_at_k"] == pytest.approx(0.16, abs=1e-12)
    assert res["random_precision"] == pytest.approx(0.10, abs=1e-12)
    assert res["enrichment_factor"] == pytest.approx(1.6, abs=1e-12)

    # Permute rows 20 times and assert exact invariance
    rng = np.random.default_rng(99)
    for _ in range(20):
        perm = rng.permutation(len(y_true))
        res_perm = compute_precision_at_k_and_ef(y_true[perm], y_pred[perm], k=10, elite_frac=0.10)
        for key in res:
            assert res_perm[key] == pytest.approx(res[key], abs=1e-12)


def test_gate6_manifest_completeness_and_missing_flag() -> None:
    """Gate 6: Task manifest verifies 285 core LoRA tasks and explicitly flags missing folds."""
    manifest = build_core_task_manifest(PANEL_JSON)
    assert len(manifest) == 285
    assert manifest["task_id"].nunique() == 285
    assert manifest["wave"].value_counts().to_dict() == {
        "wave1_main": 150,
        "wave2_curve": 90,
        "wave3_seed": 45,
    }

    all_ids = set(manifest["task_id"])
    full_audit = audit_manifest_completeness(manifest, all_ids)
    assert full_audit["is_complete"] is True
    assert full_audit["n_missing"] == 0

    dropped = set(manifest["task_id"].iloc[:3])
    partial_audit = audit_manifest_completeness(manifest, all_ids - dropped)
    assert partial_audit["is_complete"] is False
    assert partial_audit["n_missing"] == 3
    assert len(partial_audit["missing_records"]) == 3
