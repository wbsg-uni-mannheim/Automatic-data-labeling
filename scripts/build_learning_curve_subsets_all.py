#!/usr/bin/env python3
"""Build chronological learning-curve subsets for all 4 non-abt-buy benchmarks
(abt-buy already done in output/learning_curve_abtbuy/).

For each (benchmark, method), take head N from chronological labels
(active_labels_latest.csv + random_profile_labels.csv concat, dedupe), at
sizes [1000, 2000, ..., up_to benchmark-size].

Output: output/learning_curve_<benchmark>/<method>/N<size>/<benchmark>_N<size>_train.json.gz
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/work/aasteine/Automatic-data-labeling")

# benchmark train-only size = max useful chronological N (we floor to 1k below benchmark size)
BENCHMARK_TRAIN_SIZE = {
    "abt-buy":        5743,
    "walmart-amazon": 6144,
    "dblp-acm":       7417,
    "dblp-scholar":   17223,
    "wdc":            19835,
}

# per (benchmark, method) → run_dir for chronological labels
SOURCES = {
    "walmart-amazon": {
        "similarity_selection": ROOT / "output/seed_round_only_profiles/benchmark_walmart-amazon_20260415_190530",
        "simple_active_learning": ROOT / "output/simple_active_learning_labeling/benchmark_walmart-amazon_20260302_113733",
        "ditto_active_learning":  ROOT / "output/three_phase_labeling_ditto_only_v2/benchmark_walmart-amazon_20260323_202820",
    },
    "dblp-acm": {
        "similarity_selection": ROOT / "output/seed_round_only_profiles/benchmark_dblp-acm_20260415_190530",
        "simple_active_learning": ROOT / "output/simple_active_learning_labeling/benchmark_dblp-acm_20260302_113733",
        "ditto_active_learning":  ROOT / "output/three_phase_labeling_ditto_only_v2/benchmark_dblp-acm_20260323_202820",
    },
    "dblp-scholar": {
        "similarity_selection": ROOT / "output/seed_round_only_profiles/benchmark_dblp-scholar_20260415_190530",
        "simple_active_learning": ROOT / "output/simple_active_learning_labeling/benchmark_dblp-scholar_20260302_113733",
        "ditto_active_learning":  ROOT / "output/three_phase_labeling_ditto_only_v2/benchmark_dblp-scholar_20260323_202820",
    },
    "wdc": {
        "similarity_selection": ROOT / "output/seed_round_only_profiles/benchmark_wdc_20260415_190530",
        "simple_active_learning": ROOT / "output/simple_active_learning_labeling/benchmark_wdc_20260413_152105",
        "ditto_active_learning":  ROOT / "output/three_phase_labeling_ditto_only_v2/benchmark_wdc_20260323_202820",
    },
}
FIELDS = {
    "abt-buy":        "title,description,price",
    "walmart-amazon": "title,category,brand,modelno,price",
    "dblp-acm":       "title,authors,venue,year",
    "dblp-scholar":   "title,authors,venue,year",
    "wdc":            "title,brand,description,price,priceCurrency",
}


def load_chronological_labels(run_dir: Path) -> pd.DataFrame:
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
    for bm, methods in SOURCES.items():
        tgt = BENCHMARK_TRAIN_SIZE[bm]
        max_size = (tgt // 1000) * 1000  # floor to nearest 1k
        sizes = list(range(1000, max_size + 1, 1000))
        out_root = ROOT / f"output/learning_curve_{bm}"
        for method, run_dir in methods.items():
            left_canonical = run_dir / "canonical/left.csv"
            right_canonical = run_dir / "canonical/right.csv"
            if not left_canonical.exists() or not right_canonical.exists():
                print(f"[skip] {bm}/{method}: missing canonical CSVs in {run_dir}")
                continue
            df = load_chronological_labels(run_dir)
            total = len(df)
            print(f"\n=== {bm} × {method}  (avail={total}) ===")
            for N in sizes:
                if N > total:
                    print(f"  N={N}: SKIP (only {total} labels)")
                    continue
                sub_dir = out_root / method / f"N{N}"
                sub_dir.mkdir(parents=True, exist_ok=True)
                labels_csv = sub_dir / "labels_subset.csv"
                df.head(N).to_csv(labels_csv, index=False)
                out_jsongz = sub_dir / f"{bm}_N{N}_train.json.gz"
                if out_jsongz.exists() and out_jsongz.stat().st_size > 0:
                    # skip if already built
                    pos = int((df.head(N)["label"] == "TRUE").sum())
                    print(f"  N={N}: exists ({pos}p)")
                    continue
                cmd = [
                    sys.executable,
                    str(ROOT / "scripts/ditto/convert_active_labels_to_wdc.py"),
                    "--labels-csv", str(labels_csv),
                    "--left-csv", str(left_canonical),
                    "--right-csv", str(right_canonical),
                    "--output-json-gz", str(out_jsongz),
                    "--fields", FIELDS[bm],
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                pos = int((df.head(N)["label"] == "TRUE").sum())
                neg = int((df.head(N)["label"] == "FALSE").sum())
                print(f"  N={N}: pos={pos} neg={neg}")


if __name__ == "__main__":
    main()
