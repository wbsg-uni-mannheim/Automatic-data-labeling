#!/usr/bin/env python3
"""Export each (benchmark × method) labelled training set as a CSV for manual
error-rate annotation.

Reads the train.json.gz files referenced in
configs/ditto/benchmark_size_runs.yaml and writes one CSV per (benchmark, method)
into output/benchmark_size_runs/annotation_csvs/<benchmark>__<method>.csv

CSV columns:
  pair_id, label, correct_label, notes, <left fields with _left>, <right fields with _right>
The `correct_label` and `notes` columns are empty for the annotator to fill in.
"""
from __future__ import annotations
import gzip, json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path("/work/aasteine/Automatic-data-labeling")
CONFIG = ROOT / "configs/ditto/benchmark_size_runs.yaml"
OUT_DIR = ROOT / "output/benchmark_size_runs/annotation_csvs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl_gz(p):
    rows = []
    with gzip.open(p, "rt") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def main():
    cfg = yaml.safe_load(CONFIG.read_text())
    benchmarks = cfg["benchmarks"]
    summary = []
    for key, b in benchmarks.items():
        train_path = ROOT / b["train"]
        if not train_path.exists():
            print(f"[skip] {key}: train file missing -> {train_path}")
            continue
        rows = load_jsonl_gz(train_path)
        if not rows:
            print(f"[skip] {key}: empty")
            continue
        df = pd.DataFrame(rows)
        # Column ordering: pair_id, label, correct_label (empty), notes (empty), then _left then _right
        leading = ["pair_id", "label"]
        annotator = ["correct_label", "notes"]
        left = sorted([c for c in df.columns if c.endswith("_left")])
        right = sorted([c for c in df.columns if c.endswith("_right")])
        other = [c for c in df.columns if c not in leading + left + right]
        for ann in annotator:
            df[ann] = ""
        df = df[leading + annotator + left + right + other]
        out = OUT_DIR / f"{key}.csv"
        df.to_csv(out, index=False)
        pos = int((df["label"] == 1).sum() + (df["label"].astype(str).str.upper() == "TRUE").sum())
        neg = int(len(df) - pos)
        summary.append({"variant": key, "rows": len(df), "pos": pos, "neg": neg, "csv": str(out.relative_to(ROOT))})
        print(f"{key:25s}: {len(df):>6d} rows ({pos}p/{neg}n) -> {out.relative_to(ROOT)}")

    # write a manifest
    manifest = pd.DataFrame(summary)
    manifest.to_csv(OUT_DIR / "_manifest.csv", index=False)
    print(f"\nmanifest: {(OUT_DIR / '_manifest.csv').relative_to(ROOT)}")
    print(f"\nTotal {len(summary)} CSVs written to {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
