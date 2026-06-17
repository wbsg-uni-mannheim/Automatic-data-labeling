#!/usr/bin/env python3
"""Build chronological learning-curve training subsets for abt-buy.

For each method (similarity_selection, simple_active_learning, ditto_active_learning):
  - Concatenates active_labels_latest.csv (chronological labelling order) + random_profile_labels.csv
  - Dedupes by (id1, id2), keeping the first occurrence (preserves chronological order)
  - Writes labels_subset.csv (head N) for N in [1000, 2000, 3000, 4000, 5000, 6000]
  - Calls scripts/archive/ditto_internal/convert_active_labels_to_wdc.py to emit ditto train.json.gz

Does NOT filter test-set leakage (per user request — moving away from cleaned variant).
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/work/aasteine/Automatic-data-labeling")
OUT = ROOT / "output/learning_curve_abtbuy"
OUT.mkdir(parents=True, exist_ok=True)

SIZES = [1000, 2000, 3000, 4000, 5000, 6000]
FIELDS = "title,description,price"

METHODS = {
    "similarity_selection": {
        "run_dir": ROOT / "output/seed_round_only_profiles/benchmark_abt-buy_20260415_190530",
    },
    "simple_active_learning": {
        "run_dir": ROOT / "output/simple_active_learning_labeling/benchmark_abt-buy_1",
    },
    "ditto_active_learning": {
        # Newer run (091957) has +1000 random pool labels, so 6k point is reachable.
        # Older run (202820) matches the published 5,917 number but caps at ~5800.
        "run_dir": ROOT / "output/three_phase_labeling_ditto_only_v2/benchmark_abt-buy_20260424_091957",
    },
}


def load_chronological_labels(run_dir: Path) -> pd.DataFrame:
    """active_labels_latest.csv (AL order) + random_profile_labels.csv appended at end."""
    main = pd.read_csv(run_dir / "active_labels_latest.csv")
    random_path = run_dir / "random_profile_labels.csv"
    if random_path.exists():
        rnd = pd.read_csv(random_path)
        df = pd.concat([main, rnd], ignore_index=True, sort=False)
    else:
        df = main
    df["label"] = df["label"].astype(str).str.upper()
    df = df.drop_duplicates(subset=["id1", "id2"], keep="first").reset_index(drop=True)
    return df


def main():
    for method, cfg in METHODS.items():
        run_dir = cfg["run_dir"]
        left_canonical = run_dir / "canonical/left.csv"
        right_canonical = run_dir / "canonical/right.csv"
        if not left_canonical.exists() or not right_canonical.exists():
            print(f"[skip] {method}: missing canonical CSVs in {run_dir}")
            continue
        df = load_chronological_labels(run_dir)
        total = len(df)
        print(f"\n=== {method} ===")
        print(f"  run_dir: {run_dir}")
        print(f"  chronological labels (deduped): {total}")
        for N in SIZES:
            if N > total:
                print(f"  N={N}: SKIP (only {total} labels available)")
                continue
            sub_dir = OUT / method / f"N{N}"
            sub_dir.mkdir(parents=True, exist_ok=True)
            labels_csv = sub_dir / "labels_subset.csv"
            df.head(N).to_csv(labels_csv, index=False)
            out_jsongz = sub_dir / f"abt-buy_N{N}_train.json.gz"
            cmd = [
                sys.executable,
                str(ROOT / "scripts/archive/ditto_internal/convert_active_labels_to_wdc.py"),
                "--labels-csv", str(labels_csv),
                "--left-csv", str(left_canonical),
                "--right-csv", str(right_canonical),
                "--output-json-gz", str(out_jsongz),
                "--fields", FIELDS,
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            pos = int((df.head(N)["label"] == "TRUE").sum())
            neg = int((df.head(N)["label"] == "FALSE").sum())
            print(f"  N={N}: pos={pos} neg={neg} -> {out_jsongz.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
