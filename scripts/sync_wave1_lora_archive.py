"""Pack and import complete Wave 1 fold records and their prediction files."""

import argparse
import json
from pathlib import Path
import tarfile

from esm2_ood.io_utils import atomic_copy_and_verify, sha256_file


def completed_files(records):
    for path in sorted(records.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        stem = f"{record['task_id']}__{record['method']}"
        assert path.name == f"{stem}.json", f"Checkpoint name mismatch: {path}"
        if "prediction_sha256" in record:
            prediction = path.with_suffix(".parquet")
            assert prediction.is_file(), f"Missing prediction: {prediction}"
            assert sha256_file(prediction) == record["prediction_sha256"], f"Invalid hash: {prediction}"
            yield prediction
        yield path


def pack(records, archive):
    files = list(completed_files(records))
    with tarfile.open(archive, "w:gz") as bundle:
        for path in files:
            bundle.add(path, arcname=path.name)


def import_records(source, destination):
    files = list(completed_files(source))
    for path in files:
        target = destination / path.name
        assert not target.exists() or sha256_file(target) == sha256_file(path), (
            f"Existing fold conflicts with downloaded file: {target}"
        )
    for path in files:
        target = destination / path.name
        if not target.exists():
            atomic_copy_and_verify(path, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["pack", "import"])
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    actions = {"pack": pack, "import": import_records}
    actions[args.action](args.source, args.target)
