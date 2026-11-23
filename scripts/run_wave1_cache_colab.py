"""Wave 1 Colab L4 cache runner: extract 650M masked-marginals & fp16 diff embeddings.

Supports resumable per-protein execution with atomic file writes and --skip / --proteins
filtering so already-cached proteins (e.g., OXDA & GFP from Wave 0B) are not recomputed.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
import urllib.request
import zipfile
import numpy as np
import pandas as pd

from esm2_ood.colab_gpu import (
    compute_esm2_masked_marginals,
    extract_frozen_diff_embeddings,
)
from esm2_ood.metrics import compute_spearman
from esm2_ood.splits import load_panel_dms

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_WORK_DIR = pathlib.Path("/content/work/wave1_cache")


def stage_data(work_dir: pathlib.Path, panel_ids: list[str]) -> tuple[pd.DataFrame, pathlib.Path]:
    cv_dir = work_dir / "cv_folds"
    cv_dir.mkdir(parents=True, exist_ok=True)

    ref_csv = work_dir / "DMS_substitutions.csv"
    if not ref_csv.exists():
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv",
            ref_csv,
        )

    missing = [d for d in panel_ids if not (cv_dir / f"{d}.csv").exists()]
    if missing:
        zip_path = work_dir / "cv_folds_singles_substitutions.zip"
        if not zip_path.exists():
            urllib.request.urlretrieve(
                "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/cv_folds_singles_substitutions.zip",
                zip_path,
            )
        target_set = set(missing)
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                stem = pathlib.Path(member).stem
                if stem in target_set and member.endswith(".csv"):
                    (cv_dir / f"{stem}.csv").write_bytes(zf.read(member))

    return pd.read_csv(ref_csv).set_index("DMS_id"), cv_dir


def _atomic_write_parquet(df: pd.DataFrame, dst: pathlib.Path) -> None:
    tmp = dst.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, dst)


def _atomic_write_npz(dst: pathlib.Path, **arrays: np.ndarray) -> None:
    tmp = dst.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache 650M zero-shot & fp16 diff embeddings on Colab L4.")
    parser.add_argument("--work-dir", type=pathlib.Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--out-dir", type=pathlib.Path, default=None)
    parser.add_argument("--proteins", nargs="*", default=None, help="Optional subset of DMS IDs to process.")
    parser.add_argument("--skip", nargs="*", default=[], help="DMS IDs to skip (e.g., already cached).")
    parser.add_argument("--batch-size-zs", type=int, default=32)
    parser.add_argument("--batch-size-emb", type=int, default=64)
    args = parser.parse_args()

    panel_cfg = json.loads((REPO_ROOT / "configs" / "panel_10.json").read_text(encoding="utf-8"))
    all_panel_ids = [p["dms_id"] for p in panel_cfg["proteins"]]
    target_ids = args.proteins if args.proteins else all_panel_ids
    skip_set = set(args.skip)
    run_ids = [d for d in target_ids if d not in skip_set]

    out_dir = args.out_dir or (args.work_dir / "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_df, cv_dir = stage_data(args.work_dir, run_ids)

    summary_rows = []
    t_total = time.perf_counter()

    for idx, dms_id in enumerate(run_ids, start=1):
        zs_path = out_dir / f"{dms_id}_zeroshot.parquet"
        npz_path = out_dir / f"{dms_id}_650M_diff_fp16.npz"

        df = load_panel_dms(dms_id, cv_dir)
        wt_seq = str(ref_df.loc[dms_id, "target_seq"])

        if zs_path.exists():
            print(f"[{idx}/{len(run_ids)}] {dms_id}: zeroshot already cached, skipping.", flush=True)
            zs_df = pd.read_parquet(zs_path)
            rho_650 = compute_spearman(zs_df["DMS_score"].to_numpy(), zs_df["zs_650M"].to_numpy())
            dt_zs = 0.0
        else:
            t0 = time.perf_counter()
            zs_650 = compute_esm2_masked_marginals(
                df, wt_seq=wt_seq, model_name_or_size="650M", batch_size=args.batch_size_zs, device="cuda"
            )
            dt_zs = time.perf_counter() - t0
            rho_650 = compute_spearman(df["DMS_score"].to_numpy(), zs_650)
            out_df = df[["mutant", "DMS_score"]].copy()
            out_df["zs_650M"] = zs_650
            _atomic_write_parquet(out_df, zs_path)
            print(
                f"[{idx}/{len(run_ids)}] [ZS] {dms_id}: n={len(df)}, len={len(wt_seq)}, "
                f"rho={rho_650:.4f}, time={dt_zs:.2f}s",
                flush=True,
            )

        if npz_path.exists():
            print(f"[{idx}/{len(run_ids)}] {dms_id}: diff_fp16 already cached, skipping.", flush=True)
            dt_emb = 0.0
        else:
            t0 = time.perf_counter()
            diffs = extract_frozen_diff_embeddings(
                df, wt_seq=wt_seq, model_name_or_size="650M", batch_size=args.batch_size_emb, device="cuda"
            )
            dt_emb = time.perf_counter() - t0
            _atomic_write_npz(npz_path, mutant=df["mutant"].to_numpy(), diffs=diffs)
            print(
                f"[{idx}/{len(run_ids)}] [EMB] {dms_id}: shape={diffs.shape}, "
                f"size_mb={npz_path.stat().st_size / 1e6:.2f}, time={dt_emb:.2f}s",
                flush=True,
            )

        summary_rows.append(
            {
                "dms_id": dms_id,
                "seq_len": len(wt_seq),
                "n_mutants": len(df),
                "zs_650M_spearman": round(float(rho_650), 4) if rho_650 is not None else None,
                "zs_sec": round(dt_zs, 2),
                "emb_sec": round(dt_emb, 2),
                "npz_size_mb": round(npz_path.stat().st_size / 1e6, 2),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "wave1_cache_summary.csv", index=False)
    print(f"\n=== Wave 1 Cache Complete in {time.perf_counter() - t_total:.2f}s ===", flush=True)
    if not summary_df.empty:
        print(summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
