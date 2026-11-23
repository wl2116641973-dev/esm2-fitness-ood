"""Export Wave 1 reports from accepted fold records without running any models."""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RANKING_COLUMNS = ["precision_at_10", "ef_at_10pct"]


def heldout_site_mean(frame: pd.DataFrame) -> pd.Series:
    return frame["method"].eq("site_mean") & frame["split_scheme"].isin(["modulo", "contiguous"])


def export_report(output: Path) -> None:
    tables = ROOT / "outputs" / "tables"
    sources = [tables / f"{stem}_baselines_per_fold.csv" for stem in ("cpu", "wave1_frozen")]
    frame = pd.concat([pd.read_csv(p, keep_default_na=False) for p in sources], ignore_index=True)
    wave = frame.loc[frame["wave"].eq("wave1_main")].copy()
    restricted = heldout_site_mean(wave)
    assert len(wave) == 750 and not wave.duplicated(["task_id", "method"]).any()
    assert restricted.sum() == 100 and wave.loc[restricted, "spearman_status"].eq("NA (constant)").all()
    assert wave.loc[restricted, "spearman"].eq("").all()
    assert pd.to_numeric(wave.loc[restricted, "mse"], errors="raise").notna().all()
    output.mkdir(parents=True, exist_ok=False)
    wave.loc[~restricted].to_csv(output / "ranked_per_fold.csv", index=False)
    wave.loc[restricted].drop(columns=RANKING_COLUMNS).to_csv(output / "site_mean_heldout_per_fold.csv", index=False)
    print(f"Exported {len(wave)} accepted fold records to {output}: 650 ranked and 100 Site-Mean held-out")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    export_report(parser.parse_args().out_dir)
