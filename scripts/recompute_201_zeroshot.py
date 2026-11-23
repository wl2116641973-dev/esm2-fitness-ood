"""Recompute official ProteinGym v1.3 zero-shot ESM-2 benchmarks and verify the 10-protein panel."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF_CSV = ROOT / "data" / "raw" / "DMS_substitutions.csv"
SP_CSV = ROOT / "data" / "raw" / "DMS_substitutions_Spearman_DMS_level.csv"
PANEL_JSON = ROOT / "configs" / "panel_10.json"
OUT_DIR = ROOT / "outputs" / "tables"

ESM2_COLS = [
    "ESM2 (8M)",
    "ESM2 (35M)",
    "ESM2 (150M)",
    "ESM2 (650M)",
    "ESM2 (3B)",
]


def summarize_subset(df: pd.DataFrame, name: str) -> dict:
    """Compute equal-weight and UniProt-equal-weight means and 650M vs 3B comparison."""
    means = {col: round(float(df[col].mean()), 4) for col in ESM2_COLS}
    uniprot_means = {
        col: round(float(df.groupby("UniProt_ID")[col].mean().mean()), 4)
        for col in ESM2_COLS
    }
    diff = df["ESM2 (3B)"] - df["ESM2 (650M)"]
    return {
        "subset": name,
        "n_assays": int(len(df)),
        "n_uniprot": int(df["UniProt_ID"].nunique()),
        "total_single_mutants": int(df["DMS_number_single_mutants"].sum()),
        "equal_weight_mean": means,
        "uniprot_weight_mean": uniprot_means,
        "3b_vs_650m": {
            "3b_higher": int((diff > 0).sum()),
            "3b_lower": int((diff < 0).sum()),
            "tie": int((diff == 0).sum()),
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(REF_CSV)
    sp = pd.read_csv(SP_CSV)
    merged = ref.merge(sp, left_on="DMS_id", right_on="DMS ID", how="inner")
    assert len(merged) == 217, f"Expected 217 assays, got {len(merged)}"

    le1022 = merged[merged["seq_len"] <= 1022].copy()
    pure_single = le1022[~le1022["includes_multiple_mutants"]].copy()
    with_multi = le1022[le1022["includes_multiple_mutants"]].copy()

    summary = {
        "all_217": summarize_subset(merged, "all_217"),
        "seq_len_le_1022": summarize_subset(le1022, "seq_len_le_1022"),
        "pure_single_le_1022": summarize_subset(pure_single, "pure_single_le_1022"),
        "with_multi_le_1022": summarize_subset(with_multi, "with_multi_le_1022"),
    }

    with open(PANEL_JSON, "r", encoding="utf-8") as f:
        panel_cfg = json.load(f)

    panel_ids = [p["dms_id"] for p in panel_cfg["proteins"]]
    expected_map = {p["dms_id"]: p["expected_single_mutants"] for p in panel_cfg["proteins"]}
    panel_df = merged[merged["DMS_id"].isin(panel_ids)].copy()
    assert len(panel_df) == 10, f"Expected 10 panel proteins, found {len(panel_df)}"

    # Verify exact single mutant counts match the locked plan
    panel_df["expected_single_mutants"] = panel_df["DMS_id"].map(expected_map)
    mismatches = panel_df[
        panel_df["DMS_number_single_mutants"] != panel_df["expected_single_mutants"]
    ]
    assert len(mismatches) == 0, f"Single mutant count mismatches: {mismatches[['DMS_id', 'DMS_number_single_mutants', 'expected_single_mutants']]}"

    cols_out = [
        "DMS_id",
        "UniProt_ID",
        "seq_len",
        "DMS_number_single_mutants",
        "includes_multiple_mutants",
        "coarse_selection_type",
        "DMS_filename",
        *ESM2_COLS,
    ]
    panel_out = panel_df[cols_out].set_index("DMS_id").loc[panel_ids].reset_index()
    panel_out.to_csv(OUT_DIR / "panel_10_zeroshot.csv", index=False)

    with open(OUT_DIR / "zeroshot_201_recomputed.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=== ProteinGym v1.3 Zero-Shot Recompute ===")
    for key in ("all_217", "seq_len_le_1022", "pure_single_le_1022"):
        s = summary[key]
        print(
            f"{key}: {s['n_assays']} assays, {s['n_uniprot']} UniProts, "
            f"{s['total_single_mutants']:,} single mutants | "
            f"650M={s['equal_weight_mean']['ESM2 (650M)']:.4f}, "
            f"3B={s['equal_weight_mean']['ESM2 (3B)']:.4f} "
            f"(3B win/loss/tie={s['3b_vs_650m']['3b_higher']}/{s['3b_vs_650m']['3b_lower']}/{s['3b_vs_650m']['tie']})"
        )
    print(
        f"\n10-protein panel verified: {int(panel_out['DMS_number_single_mutants'].sum()):,} single mutants total."
    )
    print(
        panel_out[["DMS_id", "seq_len", "DMS_number_single_mutants", "ESM2 (650M)", "ESM2 (3B)"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
