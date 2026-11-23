"""CPU-only regression tests for prediction schema, checkpoints, and transport."""

import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import pandas as pd
import pytest

from esm2_ood.io_utils import sha256_file
from esm2_ood.predictions import frozen_test_rows, read_predictions, validate_predictions, write_predictions

ROOT = Path(__file__).resolve().parents[1]
META = {"config_hash": "a" * 64, "git_commit": "b" * 40}


def frozen(model="lora_llr:650M"):
    sample = pd.DataFrame({"mutant": ["A1C", "A2C"], "pos": [1, 2], "DMS_score": [0.1, 0.2]})
    return frozen_test_rows(sample, dms_id="fixture", split="modulo", fold=0, seed=42, model_id=model)


@pytest.mark.parametrize(("change", "error"), [
    (lambda x: x.iloc[:1], "missing=1"),
    (lambda x: x.assign(mutant=["A1C", "A3C"]), "extra=1"),
    (lambda x: pd.concat([x, x.iloc[:1]]), "duplicates=1"),
    (lambda x: x.assign(y_true=[0.0, 0.2]), "frozen raw DMS_score"),
    (lambda x: x.assign(y_true=[np.inf, 0.2]), "Non-finite"),
    (lambda x: x.assign(y_pred=[0.0, np.nan]), "Non-finite"),
    (lambda x: x.assign(pos=[1, 3]), "Position mismatch"),
])
def test_frozen_fold_rejects_mismatches(change, error):
    expected = frozen()
    with pytest.raises(AssertionError, match=error):
        validate_predictions(change(expected.assign(y_pred=[0.0, 1.0])), expected)


def test_prediction_roundtrip_and_constant_policy(tmp_path):
    path = tmp_path / "fold.parquet"
    expected = frozen()
    assert write_predictions(path, expected, [0.0, 1.0], **META) == "ok"
    assert read_predictions(path, expected.iloc[::-1], **META) == "ok"
    with pytest.raises(AssertionError, match="Configuration hash mismatch"):
        read_predictions(path, expected, **{**META, "config_hash": "c" * 64})
    original = sha256_file(path)
    with pytest.raises(AssertionError, match="Degenerate constant"):
        write_predictions(path, expected, [1.0, 1.0], **META)
    assert sha256_file(path) == original
    assert write_predictions(path, frozen("site_mean"), [1.0, 1.0], **META) == "NA (constant)"


def test_archive_roundtrip_and_conflict(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from sync_wave1_lora_archive import import_records, pack

    source, staging, dest = [tmp_path / name for name in ("src", "staging", "dest")]
    source.mkdir()
    pred = source / "fold__lora_llr.parquet"
    write_predictions(pred, frozen(), [0.0, 1.0], **META)
    record = pred.with_suffix(".json")
    record.write_text(json.dumps({"task_id": "fold", "method": "lora_llr",
                                  "prediction_sha256": sha256_file(pred)}), encoding="utf-8")
    archive = tmp_path / "records.tar.gz"
    pack(source, archive)
    with tarfile.open(archive) as bundle:
        assert bundle.getnames() == [pred.name, record.name]
        bundle.extractall(staging, filter="data")
    import_records(staging, dest)
    import_records(staging, dest)
    assert read_predictions(dest / pred.name, frozen(), **META) == "ok"
    (dest / pred.name).write_bytes(b"conflict")
    with pytest.raises(AssertionError, match="Existing fold conflicts"):
        import_records(staging, dest)


def test_runner_archives_and_verifies_resume(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import run_wave1_lora_colab as runner

    dms_id = "GFP_AEQVI_Sarkisyan_2016"
    ref = pd.read_csv(ROOT / "data" / "raw" / "DMS_substitutions.csv").set_index("DMS_id")
    monkeypatch.setattr(runner, "stage_data", lambda *_: (ref, ROOT / "data" / "cv_folds"))
    monkeypatch.setattr(runner, "train_eval_esm2_lora_llr", lambda *a, **k: {
        "test_pred": np.arange(len(a[2]), dtype=float), "test_spearman": 0.1,
        "test_mse": 0.2, "best_val_spearman": 0.3, "best_epoch": 1, "elapsed_sec": 0.01,
    })
    monkeypatch.setattr(sys, "argv", [
        "runner", "--out-dir", str(tmp_path), "--proteins", dms_id,
        "--splits", "modulo", "--folds", "0", "--methods", "lora_llr",
        "--git-commit", META["git_commit"],
    ])
    runner.main()
    record_path, = (tmp_path / "records").glob("*.json")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    pred = record_path.with_suffix(".parquet")
    assert sha256_file(pred) == record["prediction_sha256"]
    monkeypatch.setattr(runner, "train_eval_esm2_lora_llr", lambda *_a, **_k: pytest.fail("Unexpected training"))
    runner.main()
    pred.write_bytes(b"damaged")
    with pytest.raises(AssertionError):
        runner.main()
