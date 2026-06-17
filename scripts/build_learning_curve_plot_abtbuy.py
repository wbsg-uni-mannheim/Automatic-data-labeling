#!/usr/bin/env python3
"""Aggregate Ditto learning-curve runs into a CSV + matplotlib plot.

Scans output/ditto_learning_curve_abtbuy/lc_<variant>_r<repeat>_seed<seed>_*/<variant>/metrics.json
and computes per (method, N) mean/std F1 across seeds.

Outputs:
  output/results_summary/plot_learning_curve_abtbuy.csv
  output/results_summary/plot_learning_curve_abtbuy.png
"""
from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/work/aasteine/Automatic-data-labeling")
RUNS_ROOT = ROOT / "output/ditto_learning_curve_abtbuy"
OUT_DIR = ROOT / "output/results_summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHOD_DISPLAY = {
    "simsel":   "Similarity search",
    "alml":     "Active learning (ML)",
    "alditto":  "Active learning (Ditto)",
}
METHOD_ORDER = ["simsel", "alml", "alditto"]
METHOD_COLORS = {"simsel": "#1f77b4", "alml": "#2ca02c", "alditto": "#d62728"}

RUN_NAME_RE = re.compile(r"^lc_(?P<variant>(simsel|alml|alditto)_N\d+)_r(?P<repeat>\d+)_seed(?P<seed>\d+)_")

# Diagnosed bad-init seeds — exclude from aggregation. The Ditto fine-tune
# landed in a degenerate threshold-0.17 optimum (val_f1≈0.50) at epoch 2, so
# early stopping froze the network there. Replaced by seed=72 in a rerun.
OUTLIER_SEEDS = {("alditto", 4000, 42)}


def main():
    records = []
    for run_dir in sorted(RUNS_ROOT.glob("lc_*")):
        m = RUN_NAME_RE.match(run_dir.name)
        if not m:
            continue
        variant = m["variant"]
        method, n_str = variant.split("_N")
        N = int(n_str)
        seed = int(m["seed"])
        metrics_path = run_dir / variant / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        test = metrics.get("test", {})
        if (method, N, seed) in OUTLIER_SEEDS:
            print(f"[skip outlier] {method} N={N} seed={seed} val_f1={metrics.get('best_val_f1', 0):.3f}")
            continue
        records.append({
            "method": method,
            "method_display": METHOD_DISPLAY[method],
            "N": N,
            "seed": seed,
            "f1": float(test.get("f1", 0.0)),
            "precision": float(test.get("precision", 0.0)),
            "recall": float(test.get("recall", 0.0)),
            "accuracy": float(test.get("accuracy", 0.0)),
            "run_dir": str(run_dir.relative_to(ROOT)),
        })
    if not records:
        print("No metrics found — are the training jobs still running?")
        return
    df = pd.DataFrame(records)
    df = df.sort_values(["method", "N", "seed"]).reset_index(drop=True)

    # per (method, N) summary
    summary = df.groupby(["method", "method_display", "N"]).agg(
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"),
        n_runs=("f1", "count"),
    ).reset_index()
    summary["f1_std"] = summary["f1_std"].fillna(0.0)

    csv_path = OUT_DIR / "plot_learning_curve_abtbuy.csv"
    summary.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(summary)} rows)")
    print(summary.to_string(index=False))

    # per-run CSV (raw)
    raw_csv_path = OUT_DIR / "plot_learning_curve_abtbuy_raw.csv"
    df.to_csv(raw_csv_path, index=False)
    print(f"Wrote {raw_csv_path} ({len(df)} rows)")

    # plot
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in METHOD_ORDER:
        sub = summary[summary["method"] == method].sort_values("N")
        if sub.empty:
            continue
        ax.errorbar(
            sub["N"],
            sub["f1_mean"],
            yerr=sub["f1_std"],
            label=METHOD_DISPLAY[method],
            color=METHOD_COLORS[method],
            marker="o",
            capsize=4,
            linewidth=2,
        )
    ax.set_xlabel("Labeled pairs (chronological)")
    ax.set_ylabel("Test F1")
    ax.set_title("Abt-Buy learning curve (Ditto downstream, gpt-5.2 labels, no test-leak filter)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    png_path = OUT_DIR / "plot_learning_curve_abtbuy.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
