#!/usr/bin/env python3
"""Aggregate Ditto results for the ±5% benchmark-size comparison into a single CSV.

Sources:
  - abt-buy: reuse existing learning curve N=6000 metrics (6 seeds sim+alditto, 3 AL ML)
  - walmart-amazon AL Ditto: existing trained run (output/ditto_walmart_amazon_alditto_new)
  - everything else: output/ditto_benchmark_size_runs/bench_<variant>_r<rep>_seed<seed>_<ts>/<variant>/metrics.json

Outputs:
  output/results_summary/plot_benchmark_size_comparison.csv (long)
  output/results_summary/plot_benchmark_size_comparison_wide.csv (mean±std table)
"""
from __future__ import annotations
import json, re, statistics
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path("/work/aasteine/Automatic-data-labeling")
OUT = ROOT / "output/results_summary"
OUT.mkdir(parents=True, exist_ok=True)

METHOD_DISPLAY = {
    "sim":      "Similarity search",
    "al_ml":    "Active learning (ML)",
    "al_ditto": "Active learning (Ditto)",
}
METHOD_ORDER = ["sim", "al_ml", "al_ditto"]
BM_ORDER = ["abt-buy", "walmart-amazon", "dblp-acm", "dblp-scholar", "wdc"]
TARGETS = {"abt-buy": 5743, "walmart-amazon": 6144, "dblp-acm": 7417, "dblp-scholar": 17223, "wdc": 19835}

BENCH_RUNS_ROOT = ROOT / "output/ditto_benchmark_size_runs"
LC_ROOT = ROOT / "output/ditto_learning_curve_abtbuy"
WALMART_ALDITTO_DIR = ROOT / "output/ditto_walmart_amazon_alditto_new"
ABT_BUY_LC_MAP = {"sim": "simsel_N6000", "al_ml": "alml_N6000", "al_ditto": "alditto_N6000"}

OUTLIER_SEEDS = {("alditto", "abt-buy", 42)}  # bad init from learning curve N=4000 (not N=6000, but keep filter generic)


def collect_lc(method_key, variant, lc_root):
    """Reuse learning-curve metrics for a single N (N=6000)."""
    runs = []
    for run in sorted(lc_root.glob(f"lc_{variant}_r*_seed*_*")):
        m = re.match(rf"lc_{re.escape(variant)}_r\d+_seed(\d+)_", run.name)
        if not m: continue
        seed = int(m.group(1))
        metrics_path = run / variant / "metrics.json"
        if not metrics_path.exists(): continue
        d = json.loads(metrics_path.read_text())
        t = d.get("test", {})
        runs.append({"seed": seed, "f1": float(t.get("f1", 0)),
                     "precision": float(t.get("precision", 0)),
                     "recall": float(t.get("recall", 0)), "size": 6000})
    return runs


def collect_walmart_alditto():
    runs = []
    for run in sorted(WALMART_ALDITTO_DIR.glob("walmart_alditto_new_r*_seed*_*")):
        m = re.match(r"walmart_alditto_new_r\d+_seed(\d+)_", run.name)
        if not m: continue
        seed = int(m.group(1))
        metrics = run / "walmart-amazon/metrics.json"
        if not metrics.exists(): continue
        d = json.loads(metrics.read_text())
        t = d.get("test", {})
        runs.append({"seed": seed, "f1": float(t.get("f1", 0)),
                     "precision": float(t.get("precision", 0)),
                     "recall": float(t.get("recall", 0)), "size": 6117})
    return runs


def collect_bench(variant):
    """variant is e.g. 'dblp-acm_sim'; pick the NEWEST run per seed (timestamp
    suffix in dir name), so re-runs supersede older filtered runs."""
    by_seed = {}
    for run in sorted(BENCH_RUNS_ROOT.glob(f"bench_{variant}_r*_seed*_*")):
        m = re.match(rf"bench_{re.escape(variant)}_r\d+_seed(\d+)_(\d{{8}}_\d{{6}})", run.name)
        if not m: continue
        seed = int(m.group(1))
        ts = m.group(2)
        metrics = run / variant / "metrics.json"
        if not metrics.exists(): continue
        # keep newest by timestamp
        if seed not in by_seed or ts > by_seed[seed]["ts"]:
            d = json.loads(metrics.read_text())
            t = d.get("test", {})
            by_seed[seed] = {"seed": seed, "ts": ts,
                             "f1": float(t.get("f1", 0)),
                             "precision": float(t.get("precision", 0)),
                             "recall": float(t.get("recall", 0)), "size": None}
    return [{k: v for k, v in r.items() if k != "ts"} for r in by_seed.values()]


def main():
    records = []
    for bm in BM_ORDER:
        for method in METHOD_ORDER:
            if bm == "abt-buy":
                variant = ABT_BUY_LC_MAP[method]
                runs = collect_lc(method, variant, LC_ROOT)
            elif bm == "walmart-amazon" and method == "al_ditto":
                # walmart-amazon AL Ditto was trained in its own dedicated dir
                walmart_runs = collect_walmart_alditto()
                bench_runs = collect_bench(f"{bm}_{method}")
                runs = walmart_runs + bench_runs
            else:
                runs = collect_bench(f"{bm}_{method}")
            # filter outlier seeds
            runs = [r for r in runs if (method, bm, r["seed"]) not in OUTLIER_SEEDS]
            for r in runs:
                records.append({
                    "benchmark": bm,
                    "method": method,
                    "method_display": METHOD_DISPLAY[method],
                    "target_size": TARGETS[bm],
                    "actual_size": r.get("size"),
                    "seed": r["seed"],
                    "f1": r["f1"],
                    "precision": r["precision"],
                    "recall": r["recall"],
                })

    if not records:
        print("No metrics found yet — are the training jobs still running?")
        return

    df = pd.DataFrame(records).sort_values(["benchmark", "method", "seed"]).reset_index(drop=True)
    df.to_csv(OUT / "plot_benchmark_size_comparison_raw.csv", index=False)
    print(f"Wrote raw: {OUT/'plot_benchmark_size_comparison_raw.csv'} ({len(df)} rows)")

    # summary
    summary = df.groupby(["benchmark", "method", "method_display", "target_size"]).agg(
        n_runs=("f1", "count"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", lambda x: float(x.std()) if len(x) > 1 else 0.0),
        precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"),
    ).reset_index()
    summary.to_csv(OUT / "plot_benchmark_size_comparison.csv", index=False)
    print(f"Wrote summary: {OUT/'plot_benchmark_size_comparison.csv'} ({len(summary)} rows)")

    # wide CSV: benchmark × method, F1 mean ± std
    summary["cell"] = summary.apply(lambda r: f"{r.f1_mean*100:.2f} ± {r.f1_std*100:.2f}", axis=1)
    wide = summary.pivot(index="benchmark", columns="method_display", values="cell")
    wide = wide.reindex(BM_ORDER)
    wide = wide[[METHOD_DISPLAY[m] for m in METHOD_ORDER]]
    wide.to_csv(OUT / "plot_benchmark_size_comparison_wide.csv")
    print(f"Wrote wide: {OUT/'plot_benchmark_size_comparison_wide.csv'}")
    print()
    print(wide.to_string())


if __name__ == "__main__":
    main()
