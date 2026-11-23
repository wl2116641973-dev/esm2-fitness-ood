"""Wave 1 Colab L4 LoRA runner: 10 proteins x 3 splits x 5 folds (N=256, seed=42).

Runs each fold with validated prediction Parquet followed by its JSON checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import time
import urllib.request
import zipfile
import pandas as pd

from esm2_ood.colab_gpu import (
    train_eval_esm2_lora_linear_head,
    train_eval_esm2_lora_llr,
)
from esm2_ood.io_utils import sha256_file
from esm2_ood.manifest import build_core_task_manifest
from esm2_ood.metrics import compute_precision_at_k_and_ef
from esm2_ood.predictions import frozen_test_rows, read_predictions, write_predictions
from esm2_ood.splits import get_outer_split, load_panel_dms, sample_budget_split

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_WORK_DIR = pathlib.Path("/content/work/wave1_lora")


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


def _atomic_write_json(obj: dict, dst: pathlib.Path) -> None:
    tmp = dst.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Wave 1 650M LoRA matrix on Colab L4.")
    parser.add_argument("--work-dir", type=pathlib.Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--out-dir", type=pathlib.Path, default=None)
    parser.add_argument("--proteins", nargs="*", default=None, help="Filter to specific DMS IDs.")
    parser.add_argument("--splits", nargs="*", default=None, help="Filter to specific split schemes.")
    parser.add_argument("--folds", nargs="*", type=int, default=None, help="Filter to specific folds.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["lora_llr", "lora_linear_head"],
        choices=["lora_llr", "lora_linear_head"],
    )
    parser.add_argument("--model-size", default="650M")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--git-commit", help="Base Git commit of a transferred source snapshot.")
    args = parser.parse_args()

    out_dir = args.out_dir or (args.work_dir / "outputs")
    rec_dir = out_dir / "records"
    rec_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_core_task_manifest(REPO_ROOT / "configs" / "panel_10.json")
    w1 = manifest[manifest["wave"] == "wave1_main"].copy()

    if args.proteins:
        w1 = w1[w1["dms_id"].isin(args.proteins)]
    if args.splits:
        w1 = w1[w1["split_scheme"].isin(args.splits)]
    if args.folds is not None:
        w1 = w1[w1["test_fold"].isin(args.folds)]

    panel_ids = sorted(w1["dms_id"].unique().tolist())
    if not panel_ids:
        print("No matching tasks in manifest. Exiting.", flush=True)
        return

    git_commit = args.git_commit or subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    frozen_manifest = REPO_ROOT / "configs" / "data_manifest.json"
    expected_hashes = {
        item["dms_id"]: item["sha256"]
        for item in json.loads(frozen_manifest.read_text(encoding="utf-8"))["panel_10"]
    }
    source_files = [pathlib.Path(__file__), *sorted((REPO_ROOT / "src" / "esm2_ood").glob("*.py"))]
    config = {
        "model_size": args.model_size, "batch_size": args.batch_size,
        "frozen_data": sha256_file(frozen_manifest),
        "panel": sha256_file(REPO_ROOT / "configs" / "panel_10.json"),
        "source": {f.name: sha256_file(f) for f in source_files},
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    ref_df, cv_dir = stage_data(args.work_dir, panel_ids)
    for dms_id in panel_ids:
        assert sha256_file(cv_dir / f"{dms_id}.csv") == expected_hashes[dms_id], (
            f"CV data differs from the frozen manifest: {dms_id}"
        )
    dms_cache = {dms_id: load_panel_dms(dms_id, cv_dir) for dms_id in panel_ids}

    total_jobs = len(w1) * len(args.methods)
    done_jobs = 0
    t_start = time.perf_counter()

    for row in w1.itertuples(index=False):
        df = dms_cache[row.dms_id]
        wt_seq = str(ref_df.loc[row.dms_id, "target_seq"])
        tr_pool, va_pool, te_pool = get_outer_split(df, row.split_scheme, row.test_fold)
        sp = sample_budget_split(tr_pool, va_pool, te_pool, n_total=row.n_total, seed=row.seed)
        tr_df, va_df, te_df = sp["train_df"], sp["val_df"], sp["test_df"]
        y_te = te_df["DMS_score"].to_numpy()

        for method in args.methods:
            done_jobs += 1
            rec_path = rec_dir / f"{row.task_id}__{method}.json"
            prediction_path = rec_path.with_suffix(".parquet")
            model_id = f"{method}:{args.model_size}"
            frozen = frozen_test_rows(
                te_pool, dms_id=row.dms_id, split=row.split_scheme, fold=int(row.test_fold),
                seed=int(row.seed), model_id=model_id,
            )
            if rec_path.exists():
                prior = json.loads(rec_path.read_text(encoding="utf-8"))
                assert prior["task_id"] == row.task_id and prior["method"] == method
                assert prior["model_size"] == args.model_size
                status = "legacy metrics only"
                if "prediction_sha256" in prior:
                    assert prior["config_hash"] == config_hash, "Checkpoint configuration changed"
                    assert sha256_file(prediction_path) == prior["prediction_sha256"]
                    read_predictions(prediction_path, frozen, config_hash=config_hash, git_commit=prior["git_commit"])
                    status = "prediction verified"
                print(f"[{done_jobs}/{total_jobs}] SKIP {row.task_id} ({method}) - {status}", flush=True)
                continue

            fn = train_eval_esm2_lora_llr if method == "lora_llr" else train_eval_esm2_lora_linear_head
            res = fn(
                tr_df,
                va_df,
                te_df,
                wt_seq=wt_seq,
                model_name_or_size=args.model_size,
                batch_size=args.batch_size,
                seed=row.seed,
                device="cuda",
            )
            evaluated = frozen_test_rows(
                te_df, dms_id=row.dms_id, split=row.split_scheme, fold=int(row.test_fold),
                seed=int(row.seed), model_id=model_id,
            )
            assert evaluated.equals(frozen), "Evaluation rows differ from the frozen test fold"
            write_predictions(
                prediction_path, frozen, res["test_pred"], config_hash=config_hash, git_commit=git_commit
            )
            p10 = compute_precision_at_k_and_ef(y_te, res["test_pred"], k=10, elite_frac=0.10)
            record = {
                "task_id": row.task_id,
                "wave": row.wave,
                "dms_id": row.dms_id,
                "split_scheme": row.split_scheme,
                "test_fold": int(row.test_fold),
                "n_total": int(row.n_total),
                "seed": int(row.seed),
                "method": method,
                "model_size": args.model_size,
                "config_hash": config_hash,
                "git_commit": git_commit,
                "prediction_sha256": sha256_file(prediction_path),
                "spearman": float(res["test_spearman"]) if res["test_spearman"] is not None else None,
                "spearman_status": "NA (constant)" if res["test_spearman"] is None else "ok",
                "mse": float(res["test_mse"]),
                "precision_at_10": float(p10["precision_at_k"]),
                "ef_at_10pct": float(p10["enrichment_factor"]),
                "best_val_spearman": float(res["best_val_spearman"]),
                "best_epoch": int(res["best_epoch"]),
                "elapsed_sec": round(float(res["elapsed_sec"]), 2),
                **sp["meta"],
            }
            _atomic_write_json(record, rec_path)
            print(
                f"[{done_jobs}/{total_jobs}] DONE {row.task_id} ({method}): "
                f"rho={record['spearman']}, mse={record['mse']:.4f}, "
                f"ep={record['best_epoch']}, {record['elapsed_sec']}s",
                flush=True,
            )

    print(f"\n=== Wave 1 LoRA batch finished in {time.perf_counter() - t_start:.2f}s ===", flush=True)


if __name__ == "__main__":
    main()
