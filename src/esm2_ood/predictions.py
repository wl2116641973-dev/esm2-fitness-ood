"""Validate and atomically archive one frozen outer fold of predictions."""

from __future__ import annotations

import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PREDICTION_KEY = ["dms_id", "split", "fold", "seed", "model_id", "mutant"]
COLUMNS = PREDICTION_KEY + ["pos", "y_true", "y_pred"]


def frozen_test_rows(test_df, *, dms_id, split, fold, seed, model_id):
    return test_df[["mutant", "pos", "DMS_score"]].rename(columns={"DMS_score": "y_true"}).assign(
        dms_id=dms_id, split=split, fold=fold, seed=seed, model_id=model_id
    )[COLUMNS[:-1]]


def validate_predictions(predictions, frozen):
    assert list(predictions.columns) == COLUMNS, "Unexpected prediction columns"
    assert len(frozen) and frozen[PREDICTION_KEY].notna().all().all(), "Invalid frozen fold"
    assert not frozen.duplicated(PREDICTION_KEY).any(), "Duplicate keys in frozen fold"
    assert predictions[PREDICTION_KEY].notna().all().all(), "Null prediction key"
    assert all(pd.api.types.is_integer_dtype(predictions[c]) for c in ("fold", "seed", "pos"))
    assert all(pd.api.types.is_numeric_dtype(predictions[c]) for c in ("y_true", "y_pred"))
    keys = pd.MultiIndex.from_frame(predictions[PREDICTION_KEY])
    expected = pd.MultiIndex.from_frame(frozen[PREDICTION_KEY])
    missing, extra = len(expected.difference(keys)), len(keys.difference(expected))
    duplicates = int(keys.duplicated().sum())
    assert (missing, extra, duplicates) == (0, 0, 0), (
        f"Frozen test keys: missing={missing}, extra={extra}, duplicates={duplicates}"
    )
    aligned = predictions.set_index(PREDICTION_KEY).loc[expected]
    assert np.array_equal(aligned["pos"].to_numpy(), frozen["pos"].to_numpy()), "Position mismatch"
    values = aligned[["y_true", "y_pred"]].to_numpy(dtype=np.float64)
    assert np.isfinite(values).all(), "Non-finite truth or prediction"
    assert np.array_equal(values[:, 0], frozen["y_true"].to_numpy(dtype=np.float64)), (
        "y_true must equal the frozen raw DMS_score"
    )
    if np.ptp(values[:, 1]) == 0:
        assert set(frozen["model_id"]) == {"site_mean"}, "Degenerate constant model predictions"
        return "NA (constant)"
    return "ok"


def read_predictions(path, frozen, *, config_hash, git_commit):
    table = pq.read_table(path)
    meta = table.schema.metadata or {}
    assert meta.get(b"config_hash") == config_hash.encode(), "Configuration hash mismatch"
    assert meta.get(b"git_commit") == git_commit.encode(), "Git commit mismatch"
    return validate_predictions(table.to_pandas(), frozen)


def write_predictions(path: Path, frozen, y_pred, *, config_hash, git_commit):
    assert re.fullmatch("[0-9a-f]{64}", config_hash), "Invalid configuration hash"
    assert re.fullmatch("[0-9a-f]{40}", git_commit), "Invalid Git commit"
    values = np.asarray(y_pred, dtype=np.float64)
    assert values.shape == (len(frozen),), "Prediction count or shape mismatch"
    frame = frozen.assign(y_pred=values)
    validate_predictions(frame, frozen)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    table = table.replace_schema_metadata({
        **(table.schema.metadata or {}),
        b"config_hash": config_hash.encode(),
        b"git_commit": git_commit.encode(),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as output:
        tmp = Path(output.name)
    try:
        pq.write_table(table, tmp)
        status = read_predictions(tmp, frozen, config_hash=config_hash, git_commit=git_commit)
        with tmp.open("r+b") as output:
            os.fsync(output.fileno())
        os.replace(tmp, path)
        return status
    finally:
        tmp.unlink(missing_ok=True)
