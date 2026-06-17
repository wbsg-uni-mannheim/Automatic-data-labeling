#!/usr/bin/env python3
"""Aggregate learning curve results across all 5 benchmarks × 3 methods × all N.
Combines abt-buy results (from output/ditto_learning_curve_abtbuy/) with
new results from output/ditto_learning_curve_all/.

Outputs:
  output/results_summary/plot_learning_curve_all.csv  — long format
  output/results_summary/plot_learning_curve_all_wide.csv  — pivot
  output/results_summary/plot_learning_curve_all_raw.csv  — per-seed
"""
from __future__ import annotations
import json, re
from pathlib import Path
import pandas as pd

ROOT = Path("/work/aasteine/Automatic-data-labeling")
LC_OLD = ROOT / "output/ditto_learning_curve_abtbuy"        # abt-buy from before
LC_NEW = ROOT / "output/ditto_learning_curve_all"           # other 4
OUT = ROOT / "output/results_summary"
OUT.mkdir(parents=True, exist_ok=True)

METHOD_DISPLAY = {
    "sim":     "Similarity search",
    "simsel":  "Similarity search",
    "alml":    "Active learning (ML)",
    "alditto": "Active learning (Ditto)",
}
METHOD_ORDER = ["sim", "alml", "alditto"]
BM_ORDER = ["abt-buy", "walmart-amazon", "dblp-acm", "dblp-scholar", "wdc"]

# old uses "simsel" prefix, new uses "sim"
OLD_LC_RE = re.compile(r"^lc_(?P<v>(simsel|alml|alditto)_N\d+)_r\d+_seed(?P<seed>\d+)_")
NEW_LC_RE = re.compile(r"^lc_(?P<bm>[a-z-]+)_(?P<method>sim|alml|alditto)_N(?P<N>\d+)_seed(?P<seed>\d+)_")

OUTLIER_SEEDS = {("abt-buy", "alditto", 4000, 42)}  # bad-init


def collect_old():
    records = []
    if not LC_OLD.exists(): return records
    for d in LC_OLD.glob("lc_*"):
        m = OLD_LC_RE.match(d.name)
        if not m: continue
        variant = m["v"]
        seed = int(m["seed"])
        method, n_str = variant.split("_N")
        method = "sim" if method == "simsel" else method
        N = int(n_str)
        metrics = d / variant / "metrics.json"
        if not metrics.exists(): continue
        if ("abt-buy", method, N, seed) in OUTLIER_SEEDS: continue
        t = json.loads(metrics.read_text()).get("test", {})
        records.append({
            "benchmark": "abt-buy", "method": method, "N": N, "seed": seed,
            "f1": float(t.get("f1", 0)), "precision": float(t.get("precision", 0)),
            "recall": float(t.get("recall", 0)), "ts": d.name[-15:],
        })
    return records


def collect_new():
    records = []
    if not LC_NEW.exists(): return records
    for d in LC_NEW.glob("lc_*"):
        m = NEW_LC_RE.match(d.name)
        if not m: continue
        bm = m["bm"]
        method = m["method"]
        N = int(m["N"])
        seed = int(m["seed"])
        # find the actual metric path — variant key is f"{bm}_{method}_N{N}"
        v = f"{bm}_{method}_N{N}"
        metrics = d / v / "metrics.json"
        if not metrics.exists(): continue
        t = json.loads(metrics.read_text()).get("test", {})
        records.append({
            "benchmark": bm, "method": method, "N": N, "seed": seed,
            "f1": float(t.get("f1", 0)), "precision": float(t.get("precision", 0)),
            "recall": float(t.get("recall", 0)), "ts": d.name[-15:],
        })
    return records


def main():
    all_records = collect_old() + collect_new()
    if not all_records:
        print("No metrics found.")
        return
    df = pd.DataFrame(all_records)
    # dedup: keep newest timestamp per (benchmark, method, N, seed)
    df = df.sort_values(["benchmark","method","N","seed","ts"])
    df = df.groupby(["benchmark","method","N","seed"], as_index=False).tail(1)
    df = df.sort_values(["benchmark","method","N","seed"]).reset_index(drop=True)
    df["method_display"] = df["method"].map(METHOD_DISPLAY)
    df.to_csv(OUT/"plot_learning_curve_all_raw.csv", index=False)

    summary = df.groupby(["benchmark","method","method_display","N"]).agg(
        n_seeds=("f1", "count"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", lambda x: float(x.std()) if len(x) > 1 else 0.0),
    ).reset_index()
    summary["f1_mean_pct"] = (summary["f1_mean"] * 100).round(2)
    summary["f1_std_pct"] = (summary["f1_std"] * 100).round(2)
    summary.to_csv(OUT/"plot_learning_curve_all.csv", index=False)

    # Wide: benchmark+N rows × method cols
    summary["cell"] = summary.apply(lambda r: f"{r.f1_mean_pct:.2f}±{r.f1_std_pct:.2f} (n={r.n_seeds})", axis=1)
    wide = summary.pivot_table(index=["benchmark","N"], columns="method_display", values="cell", aggfunc="first")
    bench_order_map = {b:i for i,b in enumerate(BM_ORDER)}
    wide = wide.reset_index()
    wide["_o"] = wide["benchmark"].map(bench_order_map)
    wide = wide.sort_values(["_o","N"]).drop(columns="_o")
    wide.to_csv(OUT/"plot_learning_curve_all_wide.csv", index=False)

    print(f"Wrote 3 CSVs to {OUT.relative_to(ROOT)}/")
    print(f"  long  : plot_learning_curve_all.csv ({len(summary)} rows)")
    print(f"  wide  : plot_learning_curve_all_wide.csv ({len(wide)} rows)")
    print(f"  raw   : plot_learning_curve_all_raw.csv ({len(df)} rows)")
    print()
    # Per benchmark, show table
    for bm in BM_ORDER:
        sub = summary[summary["benchmark"] == bm]
        if sub.empty: continue
        print(f"=== {bm} ===")
        piv = sub.pivot_table(index="N", columns="method_display", values="f1_mean_pct", aggfunc="first")
        print(piv.to_string())
        print()


if __name__ == "__main__":
    main()
