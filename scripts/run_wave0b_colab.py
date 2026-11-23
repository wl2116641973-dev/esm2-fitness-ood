"""Wave 0B end-to-end GPU verification runner for Google Colab L4.

Executes:
1. Zero-shot masked-marginals calibration (8M & 650M) on CASP3_HUMAN (out-of-panel dev) & GFP.
2. fp16 frozen mutation-site diff embedding extraction for CASP3, GFP, and OXDA_RHOTO.
3. Short (GFP, 238 aa) & Long (OXDA_RHOTO, 364 aa) Contiguous fold=0 timing & baseline comparison
   (Zero-Shot, FrozenDiff-Ridge, Hsu-Aug, LoRA-LLR).
"""

from __future__ import annotations

import json
import pathlib
import time
import urllib.request
import zipfile
import numpy as np
import pandas as pd

from esm2_ood.baselines import FrozenDiffRidgeBaseline, HsuAugmentedRidgeBaseline
from esm2_ood.colab_gpu import (
    compute_esm2_masked_marginals,
    extract_frozen_diff_embeddings,
    train_eval_esm2_lora_llr,
)
from esm2_ood.metrics import compute_mse, compute_spearman
from esm2_ood.splits import get_outer_split, load_panel_dms, sample_budget_split

WORK_DIR = pathlib.Path("/content/work/wave0b")
CV_DIR = WORK_DIR / "cv_folds"
OUT_DIR = WORK_DIR / "outputs"


def stage_data() -> pd.DataFrame:
    CV_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ref_csv = WORK_DIR / "DMS_substitutions.csv"
    if not ref_csv.exists():
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv",
            ref_csv,
        )

    zip_path = WORK_DIR / "cv_folds_singles_substitutions.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(
            "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/cv_folds_singles_substitutions.zip",
            zip_path,
        )

    targets = {
        "CASP3_HUMAN_Roychowdhury_2020",
        "GFP_AEQVI_Sarkisyan_2016",
        "OXDA_RHOTO_Vanella_2023_activity",
    }
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            stem = pathlib.Path(member).stem
            if stem in targets and member.endswith(".csv"):
                (CV_DIR / f"{stem}.csv").write_bytes(zf.read(member))

    return pd.read_csv(ref_csv).set_index("DMS_id")


def main() -> None:
    t_start = time.perf_counter()
    ref_df = stage_data()

    # 1. Zero-shot calibration on CASP3 (pure single dev) & GFP (smoke) + OXDA
    calib_rows = []
    zs_cache: dict[str, pd.DataFrame] = {}
    emb_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for dms_id in (
        "CASP3_HUMAN_Roychowdhury_2020",
        "GFP_AEQVI_Sarkisyan_2016",
        "OXDA_RHOTO_Vanella_2023_activity",
    ):
        df = load_panel_dms(dms_id, CV_DIR)
        wt_seq = str(ref_df.loc[dms_id, "target_seq"])
        for size in ("8M", "650M"):
            t0 = time.perf_counter()
            zs = compute_esm2_masked_marginals(
                df, wt_seq=wt_seq, model_name_or_size=size, batch_size=32, device="cuda"
            )
            dt = time.perf_counter() - t0
            rho = compute_spearman(df["DMS_score"].to_numpy(), zs)
            df[f"zs_{size}"] = zs
            calib_rows.append(
                {
                    "dms_id": dms_id,
                    "seq_len": len(wt_seq),
                    "n_mutants": len(df),
                    "model": size,
                    "spearman": round(float(rho), 4) if rho is not None else None,
                    "elapsed_sec": round(dt, 2),
                }
            )
        zs_cache[dms_id] = df
        df[["mutant", "DMS_score", "zs_8M", "zs_650M"]].to_parquet(
            OUT_DIR / f"{dms_id}_zeroshot.parquet", index=False
        )

        # Extract fp16 frozen diff embeddings (650M)
        t0 = time.perf_counter()
        diffs = extract_frozen_diff_embeddings(
            df, wt_seq=wt_seq, model_name_or_size="650M", batch_size=64, device="cuda"
        )
        dt_emb = time.perf_counter() - t0
        npz_path = OUT_DIR / f"{dms_id}_650M_diff_fp16.npz"
        np.savez_compressed(npz_path, mutant=df["mutant"].to_numpy(), diffs=diffs)
        emb_cache[dms_id] = (df["mutant"].to_numpy(), diffs)
        print(
            f"[EMB] {dms_id}: shape={diffs.shape}, size_mb={npz_path.stat().st_size / 1e6:.2f}, time={dt_emb:.2f}s"
        )

    calib_df = pd.DataFrame(calib_rows)
    calib_df.to_csv(OUT_DIR / "wave0b_zeroshot_calibration.csv", index=False)
    print("\n=== Wave 0B Zero-Shot Calibration ===")
    print(calib_df.to_string(index=False))

    # 2. Short (GFP, 238 aa) & Long (OXDA, 364 aa) + Dev (CASP3, 258 aa) Contiguous fold=0 timing
    bench_rows = []
    for dms_id in (
        "CASP3_HUMAN_Roychowdhury_2020",
        "GFP_AEQVI_Sarkisyan_2016",
        "OXDA_RHOTO_Vanella_2023_activity",
    ):
        df = zs_cache[dms_id]
        wt_seq = str(ref_df.loc[dms_id, "target_seq"])
        muts_all, diffs_all = emb_cache[dms_id]
        mut_to_row = {m: i for i, m in enumerate(muts_all)}

        tr_pool, va_pool, te_df = get_outer_split(df, "contiguous", test_fold=0)
        sp = sample_budget_split(tr_pool, va_pool, te_df, n_total=256, seed=42)
        tr_df, va_df, te_df = sp["train_df"], sp["val_df"], sp["test_df"]

        idx_tr = [mut_to_row[m] for m in tr_df["mutant"]]
        idx_va = [mut_to_row[m] for m in va_df["mutant"]]
        idx_te = [mut_to_row[m] for m in te_df["mutant"]]
        y_tr = tr_df["DMS_score"].to_numpy()
        y_va = va_df["DMS_score"].to_numpy()
        y_te = te_df["DMS_score"].to_numpy()

        # Zero-shot on this exact fold
        zs_rho = compute_spearman(y_te, te_df["zs_650M"].to_numpy())
        zs_mse = compute_mse(y_te, te_df["zs_650M"].to_numpy())

        # FrozenDiff-Ridge
        fr = FrozenDiffRidgeBaseline().fit(
            diffs_all[idx_tr].astype(np.float32), y_tr, diffs_all[idx_va].astype(np.float32), y_va
        )
        fr_pred = fr.predict(diffs_all[idx_te].astype(np.float32))
        fr_rho = compute_spearman(y_te, fr_pred)
        fr_mse = compute_mse(y_te, fr_pred)

        # Hsu-Aug
        hsu = HsuAugmentedRidgeBaseline().fit(
            tr_df, tr_df["zs_650M"].to_numpy(), va_df, va_df["zs_650M"].to_numpy()
        )
        hsu_pred = hsu.predict(te_df, te_df["zs_650M"].to_numpy())
        hsu_rho = compute_spearman(y_te, hsu_pred)
        hsu_mse = compute_mse(y_te, hsu_pred)

        # LoRA-LLR (650M)
        lora_res = train_eval_esm2_lora_llr(
            tr_df,
            va_df,
            te_df,
            wt_seq=wt_seq,
            model_name_or_size="650M",
            lora_rank=8,
            lora_alpha=16,
            lr=1e-4,
            max_epochs=12,
            patience=3,
            batch_size=8,
            seed=42,
            device="cuda",
        )

        bench_rows.append(
            {
                "dms_id": dms_id,
                "seq_len": len(wt_seq),
                "zs_650m_rho": round(float(zs_rho), 4) if zs_rho is not None else None,
                "hsu_aug_rho": round(float(hsu_rho), 4) if hsu_rho is not None else None,
                "frozen_ridge_rho": round(float(fr_rho), 4) if fr_rho is not None else None,
                "lora_llr_rho": round(float(lora_res["test_spearman"]), 4)
                if lora_res["test_spearman"] is not None
                else None,
                "frozen_ridge_mse": round(float(fr_mse), 4),
                "lora_llr_mse": round(float(lora_res["test_mse"]), 4),
                "lora_best_epoch": lora_res["best_epoch"],
                "lora_elapsed_sec": round(float(lora_res["elapsed_sec"]), 2),
            }
        )

    bench_df = pd.DataFrame(bench_rows)
    bench_df.to_csv(OUT_DIR / "wave0b_timing_benchmark.csv", index=False)
    total_sec = time.perf_counter() - t_start
    with open(OUT_DIR / "wave0b_meta.json", "w") as f:
        json.dump({"total_elapsed_sec": round(total_sec, 2)}, f, indent=2)

    print("\n=== Wave 0B Contiguous Fold=0 (N=256) Benchmark ===")
    print(bench_df.to_string(index=False))
    print(f"\nTotal Wave 0B GPU script elapsed: {total_sec:.2f}s")


if __name__ == "__main__":
    main()
