"""Verify extracted ProteinGym v1.3 CV fold CSVs and freeze data_manifest.json."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF_CSV = ROOT / "data" / "raw" / "DMS_substitutions.csv"
PANEL_JSON = ROOT / "configs" / "panel_10.json"
CV_DIR = ROOT / "data" / "cv_folds"
MANIFEST_OUT = ROOT / "configs" / "data_manifest.json"
ZIP_PATH = Path(r"D:\Antigravity\cache\proteingym\cv_folds_singles_substitutions.zip")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_pos(mutant: str) -> int:
    return int(mutant[1:-1])


def check_disjoint_folds(df: pd.DataFrame, col: str) -> tuple[bool, list[int]]:
    folds = sorted(df[col].unique().tolist())
    pos_sets = [set(df.loc[df[col] == f, "pos"]) for f in folds]
    counts = [int((df[col] == f).sum()) for f in folds]
    disjoint = True
    for i in range(len(pos_sets)):
        for j in range(i + 1, len(pos_sets)):
            if pos_sets[i] & pos_sets[j]:
                disjoint = False
    return (folds == [0, 1, 2, 3, 4] and all(c > 0 for c in counts) and disjoint), counts


def main() -> None:
    ref = pd.read_csv(REF_CSV).set_index("DMS_id")
    with open(PANEL_JSON, "r", encoding="utf-8") as f:
        panel_cfg = json.load(f)

    # 1. Verify OXDA catalytic residue shift (P80324 2..365 -> 1..364 in DMS)
    oxda_seq = str(ref.loc["OXDA_RHOTO_Vanella_2023_activity", "target_seq"])
    assert len(oxda_seq) == 364, f"Expected 364 aa, got {len(oxda_seq)}"
    assert (oxda_seq[221], oxda_seq[236], oxda_seq[283]) == ("Y", "Y", "R"), (
        f"Unexpected OXDA pocket residues at 222/237/284: "
        f"{oxda_seq[221]}/{oxda_seq[236]}/{oxda_seq[283]}"
    )

    records = []
    for item in panel_cfg["proteins"]:
        dms_id = item["dms_id"]
        csv_path = CV_DIR / f"{dms_id}.csv"
        assert csv_path.exists(), f"Missing {csv_path}"
        df = pd.read_csv(csv_path)
        assert len(df) == item["expected_single_mutants"], (
            f"{dms_id}: expected {item['expected_single_mutants']}, got {len(df)}"
        )
        assert not df["mutant"].duplicated().any(), f"{dms_id}: duplicate mutants found"

        df["pos"] = df["mutant"].map(parse_pos)
        wt_seq = str(ref.loc[dms_id, "target_seq"])
        # Verify 1-based position aligns with target_seq
        for m in df["mutant"]:
            wt, pos, mut = m[0], int(m[1:-1]), m[-1]
            assert wt_seq[pos - 1] == wt, f"{dms_id}: mismatch at {m}, seq has {wt_seq[pos - 1]}"

        rand_ok, rand_counts = check_disjoint_folds(df, "fold_random_5")
        assert sorted(df["fold_random_5"].unique().tolist()) == [0, 1, 2, 3, 4]
        assert all(c > 0 for c in rand_counts)

        mod_ok, mod_counts = check_disjoint_folds(df, "fold_modulo_5")
        contig_ok, contig_counts = check_disjoint_folds(df, "fold_contiguous_5")
        assert mod_ok, f"{dms_id}: fold_modulo_5 failed disjoint/non-empty check"
        assert contig_ok, f"{dms_id}: fold_contiguous_5 failed disjoint/non-empty check"

        records.append({
            "dms_id": dms_id,
            "n_rows": int(len(df)),
            "seq_len": int(len(wt_seq)),
            "n_unique_positions": int(df["pos"].nunique()),
            "sha256": sha256_file(csv_path),
            "fold_counts": {
                "random": rand_counts,
                "modulo": mod_counts,
                "contiguous": contig_counts,
            },
        })

    # Verify auxiliary datasets
    aux_records = {}
    for aux_id in ("OXDA_RHOTO_Vanella_2023_expression", "CASP3_HUMAN_Roychowdhury_2020"):
        aux_path = CV_DIR / f"{aux_id}.csv"
        aux_df = pd.read_csv(aux_path)
        aux_records[aux_id] = {
            "n_rows": int(len(aux_df)),
            "sha256": sha256_file(aux_path),
        }

    # Verify OXDA activity vs expression overlap
    oxda_act = pd.read_csv(CV_DIR / "OXDA_RHOTO_Vanella_2023_activity.csv")
    oxda_exp = pd.read_csv(CV_DIR / "OXDA_RHOTO_Vanella_2023_expression.csv")
    shared_muts = len(set(oxda_act["mutant"]) & set(oxda_exp["mutant"]))

    manifest = {
        "proteingym_version": "v1.3",
        "source_zip": str(ZIP_PATH),
        "source_zip_sha256": sha256_file(ZIP_PATH) if ZIP_PATH.exists() else None,
        "oxda_catalytic_pocket_1based_dms": {"Y222": oxda_seq[221], "Y237": oxda_seq[236], "R284": oxda_seq[283]},
        "oxda_activity_expression_shared_mutants": shared_muts,
        "panel_10": records,
        "auxiliary": aux_records,
    }
    with open(MANIFEST_OUT, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Verified all 10 panel proteins + 2 auxiliary datasets.")
    print(f"OXDA pocket 222/237/284 = {oxda_seq[221]}/{oxda_seq[236]}/{oxda_seq[283]}")
    print(f"OXDA activity ({len(oxda_act)}) vs expression ({len(oxda_exp)}) shared mutants = {shared_muts}")
    print(f"Manifest saved to {MANIFEST_OUT}")


if __name__ == "__main__":
    main()
