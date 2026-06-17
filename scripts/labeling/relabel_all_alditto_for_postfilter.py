#!/usr/bin/env python3
"""Relabel all 5 AL(Ditto) training sets used in the ±5% benchmark-size
comparison so we can build postfilter variants downstream. Walmart-amazon is
already relabeled; this script reuses that output.

Output: output/postfilter_variants/<benchmark>/_relabel/{train__relabeled.json.gz, relabel_diff.csv, relabel_summary.json}
"""
from __future__ import annotations
import json
import subprocess
import shutil
import sys
from pathlib import Path

ROOT = Path("/work/aasteine/Automatic-data-labeling")
OUT_ROOT = ROOT / "output/postfilter_variants"

# AL Ditto train.json.gz per benchmark (matching the ±5% comparison config) + prompt fields
SOURCES = {
    "abt-buy": {
        "train": ROOT / "output/learning_curve_abtbuy/ditto_active_learning/N6000/abt-buy_N6000_train.json.gz",
        "fields": "title,description,price",
    },
    "walmart-amazon": {
        "train": ROOT / "output/three_phase_labeling_ditto_only_v2/benchmark_walmart-amazon_20260323_202820/profiles/all/active_labels_latest_walmart-amazon_all_train.json.gz",
        "fields": "title,category,brand,modelno,price",
    },
    "dblp-acm": {
        "train": ROOT / "output/three_phase_labeling_ditto_only_v2/benchmark_dblp-acm_20260323_202820/profiles/all_plus20random/active_labels_latest_dblp-acm_all_plus20random_train.json.gz",
        "fields": "title,authors,venue,year",
    },
    "dblp-scholar": {
        "train": ROOT / "output/three_phase_labeling_ditto_only_v2/benchmark_dblp-scholar_20260323_202820/profiles/large/active_labels_latest_dblp-scholar_large_train.json.gz",
        "fields": "title,authors,venue,year",
    },
    "wdc": {
        "train": ROOT / "output/benchmark_size_runs/al_ditto/wdc/wdc_train.json.gz",
        "fields": "title,brand,description,price,priceCurrency",
    },
}

# Pre-existing walmart-amazon relabel output
WALMART_EXISTING_RELABELED = ROOT / "output/three_phase_labeling_ditto_only_v2/benchmark_walmart-amazon_20260323_202820/profiles/all/active_labels_latest_walmart-amazon_all_train__relabeled__gpt-5-mini__agent-precision-system-prompt.json.gz"
WALMART_EXISTING_RESULTS = ROOT / "output/three_phase_labeling_ditto_only_v2/benchmark_walmart-amazon_20260323_202820/profiles/all/relabel_results__gpt-5-mini__agent-precision-system-prompt.csv"


def import_walmart_existing(out_dir):
    """Reuse the walmart-amazon relabel from earlier into the new layout."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rel_out = out_dir / "train__relabeled.json.gz"
    shutil.copy(WALMART_EXISTING_RELABELED, rel_out)

    import pandas as pd
    df = pd.read_csv(WALMART_EXISTING_RESULTS)
    # Existing results columns include pair_id, original_label, predicted_match, etc.
    # Normalize to the schema we want: pair_id, original_label, predicted_label, changed
    diff_df = pd.DataFrame({
        "pair_id": df["pair_id"].astype(str),
        "original_label": df["original_label"].apply(lambda x: 1 if str(x).strip().upper() in ("TRUE", "1") else 0),
        "predicted_label": df["predicted_match"].apply(lambda x: 1 if str(x).strip().upper() in ("TRUE", "1") else (0 if str(x).strip().upper() in ("FALSE", "0") else None)),
    })
    diff_df["changed"] = (diff_df["original_label"] != diff_df["predicted_label"]).astype(int)
    diff_df.to_csv(out_dir / "relabel_diff.csv", index=False)
    summary = {
        "src": str(WALMART_EXISTING_RELABELED),
        "n_pairs": int(len(diff_df)),
        "n_changed": int(diff_df["changed"].sum()),
        "model": "gpt-5-mini",
        "note": "Imported from earlier relabel run.",
        "relabeled_json_gz": str(rel_out),
    }
    (out_dir / "relabel_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  walmart-amazon: imported existing relabel ({len(diff_df)} pairs, {diff_df['changed'].sum()} changed)")


def relabel_one(benchmark, src_train, prompt_fields, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts/labeling/relabel_train_jsongz_realtime.py"),
        "--train-json-gz", str(src_train),
        "--out-dir", str(out_dir),
        "--prompt-fields", prompt_fields,
        "--model", "gpt-5-mini",
        "--workers", "10",
    ]
    print(f"\n=== {benchmark} ({src_train.name}) ===")
    subprocess.run(cmd, check=True)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for bm, spec in SOURCES.items():
        out_dir = OUT_ROOT / bm / "_relabel"
        if (out_dir / "train__relabeled.json.gz").exists():
            print(f"\n=== {bm}: skip, already done at {out_dir.relative_to(ROOT)} ===")
            continue
        if bm == "walmart-amazon" and WALMART_EXISTING_RELABELED.exists():
            print(f"\n=== walmart-amazon: import existing ===")
            import_walmart_existing(out_dir)
            continue
        relabel_one(bm, spec["train"], spec["fields"], out_dir)


if __name__ == "__main__":
    main()
