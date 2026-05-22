#!/usr/bin/env python3
"""Aggregate Qwen3.5 zero-shot baseline + fine-tune metrics across benchmarks.

Reads:
  output/qwen35_em/<bench>_test/baseline_zero_shot/metrics.json  (zero-shot)
  output/qwen35_em/<bench>_all/eval_on_test/metrics.json         (fine-tune)

Writes a combined CSV with one row per benchmark, columns for both variants.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BENCHMARKS = [
    "abt-buy",
    "amazon-google",
    "dblp-acm",
    "dblp-scholar",
    "walmart-amazon",
    "wdc",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="output/qwen35_em")
    parser.add_argument("--output", default="output/qwen35_em/baseline_summary.csv")
    args = parser.parse_args()

    root = Path(args.root)

    def load(p: Path):
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    rows = []
    for bench in BENCHMARKS:
        bl = load(root / f"{bench}_test" / "baseline_zero_shot" / "metrics.json")
        ft = load(root / f"{bench}_all" / "eval_on_test" / "metrics.json")
        row = {"benchmark": bench}
        for prefix, data in (("bl", bl), ("ft", ft)):
            if data is None:
                for k in ("rows", "parse_failures", "accuracy", "precision", "recall", "f1"):
                    row[f"{prefix}_{k}"] = None
                continue
            row[f"{prefix}_rows"] = data.get("rows")
            row[f"{prefix}_parse_failures"] = data.get("parse_failures")
            row[f"{prefix}_accuracy"] = data.get("accuracy")
            row[f"{prefix}_precision"] = data.get("precision")
            row[f"{prefix}_recall"] = data.get("recall")
            row[f"{prefix}_f1"] = data.get("f1")
        if row.get("bl_f1") is not None and row.get("ft_f1") is not None:
            row["delta_f1"] = row["ft_f1"] - row["bl_f1"]
        else:
            row["delta_f1"] = None
        rows.append(row)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    fieldnames = ["benchmark"]
    for prefix in ("bl", "ft"):
        for suffix in ("rows", "parse_failures", "accuracy", "precision", "recall", "f1"):
            fieldnames.append(f"{prefix}_{suffix}")
    fieldnames.append("delta_f1")
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote {out_path}")
    # Pretty summary
    print(f"\n{'benchmark':16} {'baseline F1':>12} {'fine-tune F1':>13} {'delta':>8}")
    for row in rows:
        bl = row.get("bl_f1")
        ft = row.get("ft_f1")
        d = row.get("delta_f1")
        bl_s = f"{bl:.4f}" if bl is not None else "—"
        ft_s = f"{ft:.4f}" if ft is not None else "—"
        d_s = f"{d:+.4f}" if d is not None else "—"
        print(f"{row['benchmark']:16} {bl_s:>12} {ft_s:>13} {d_s:>8}")


if __name__ == "__main__":
    main()
