#!/usr/bin/env python3
"""Aggregate postfilter variant results into CSV matching the user's published
table layout: Benchmark | Bench set | AL (Ditto) | +Relabel | +Relabel drop |
+Closure drop | +Cl AND Rel drop | +Cl OR Rel drop | Δ to bench.

AL (Ditto) baseline values come from output/results_summary/plot_benchmark_size_comparison.csv
Postfilter values come from output/ditto_postfilter_variants/.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import pandas as pd

ROOT = Path("/work/aasteine/Automatic-data-labeling")
PF_ROOT = ROOT / "output/ditto_postfilter_variants"
BENCH_SIZE_CSV = ROOT / "output/results_summary/plot_benchmark_size_comparison.csv"
OUT_DIR = ROOT / "output/results_summary"

BENCHMARK_SET = {
    "abt-buy":        (88.34, 1.54),
    "walmart-amazon": (85.66, 1.04),
    "dblp-acm":       (98.43, 0.34),
    "dblp-scholar":   (95.64, 0.26),
    "wdc":            (71.94, 0.97),
}
BM_DISPLAY = {
    "abt-buy": "Abt-Buy", "walmart-amazon": "Walmart-Amazon",
    "dblp-acm": "DBLP-ACM", "dblp-scholar": "DBLP-Scholar", "wdc": "WDC",
}
BM_ORDER = ["abt-buy", "walmart-amazon", "dblp-acm", "dblp-scholar", "wdc"]

VARIANT_ORDER = ["v_relabel", "v_relabel_drop", "v_closure_drop", "v_closure_and_relabel", "v_closure_or_relabel"]
VARIANT_DISPLAY = {
    "v_relabel":              "+Relabel",
    "v_relabel_drop":         "+Relabel drop",
    "v_closure_drop":         "+Closure drop",
    "v_closure_and_relabel":  "+Cl ∧ Rel drop",
    "v_closure_or_relabel":   "+Cl ∨ Rel drop",
}

RUN_NAME_RE = re.compile(r"^pf_(?P<v>.+?)_r\d+_seed(?P<seed>\d+)_(?P<ts>\d{8}_\d{6})$")


def collect_pf():
    """Pick newest run per (variant, seed)."""
    by_key = {}
    for d in PF_ROOT.glob("pf_*"):
        m = RUN_NAME_RE.match(d.name)
        if not m: continue
        variant = m["v"]
        seed = int(m["seed"])
        ts = m["ts"]
        metrics = d / variant / "metrics.json"
        if not metrics.exists(): continue
        key = (variant, seed)
        if key not in by_key or ts > by_key[key]["ts"]:
            t = json.loads(metrics.read_text()).get("test", {})
            by_key[key] = {"variant": variant, "seed": seed, "ts": ts, "f1": float(t.get("f1", 0))}
    return list(by_key.values())


def main():
    records = collect_pf()
    df = pd.DataFrame(records)
    print(f"Collected {len(df)} postfilter (variant, seed) results")

    # Parse variant into benchmark + post-variant
    df["benchmark"] = df["variant"].apply(lambda v: next((b for b in BM_ORDER if v.startswith(b + "_v_")), None))
    df["post_variant"] = df.apply(lambda r: r["variant"].replace(r["benchmark"] + "_", "") if r["benchmark"] else None, axis=1)
    df = df[df["benchmark"].notna()].copy()

    # mean/std per (benchmark, post_variant)
    summary = df.groupby(["benchmark", "post_variant"]).agg(
        n_seeds=("f1", "count"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", lambda x: float(x.std()) if len(x) > 1 else 0.0),
    ).reset_index()
    summary["f1_mean_pct"] = (summary["f1_mean"] * 100).round(2)
    summary["f1_std_pct"]  = (summary["f1_std"]  * 100).round(2)

    # AL (Ditto) baseline values from benchmark-size CSV
    bench = pd.read_csv(BENCH_SIZE_CSV)
    bench = bench[bench["method"] == "al_ditto"][["benchmark", "f1_mean", "f1_std"]]
    bench["alditto_mean_pct"] = (bench["f1_mean"] * 100).round(2)
    bench["alditto_std_pct"]  = (bench["f1_std"]  * 100).round(2)
    alditto_map = dict(zip(bench["benchmark"], zip(bench["alditto_mean_pct"], bench["alditto_std_pct"])))

    # Build wide table
    rows = []
    for bm in BM_ORDER:
        bm_mean, bm_std = BENCHMARK_SET[bm]
        alditto_mean, alditto_std = alditto_map.get(bm, (None, None))
        row = {
            "Benchmark":     BM_DISPLAY[bm],
            "Benchmark set": f"{bm_mean:.2f} ± {bm_std:.2f}",
            "AL (Ditto)":    f"{alditto_mean:.2f} ± {alditto_std:.2f}" if alditto_mean is not None else "—",
        }
        all_means = [alditto_mean] if alditto_mean is not None else []
        for v in VARIANT_ORDER:
            sub = summary[(summary["benchmark"] == bm) & (summary["post_variant"] == v)]
            if not sub.empty:
                r = sub.iloc[0]
                row[VARIANT_DISPLAY[v]] = f"{r.f1_mean_pct:.2f} ± {r.f1_std_pct:.2f}"
                all_means.append(r.f1_mean_pct)
            else:
                row[VARIANT_DISPLAY[v]] = "—"
        best = max(all_means) if all_means else None
        delta = (best - bm_mean) if best is not None else None
        row["Δ to bench."] = f"{delta:+.2f}" if delta is not None else "—"
        rows.append(row)

    wide = pd.DataFrame(rows)
    out_path = OUT_DIR / "postfilter_variants_comparison.csv"
    wide.to_csv(out_path, index=False)
    print(f"Wrote {out_path.relative_to(ROOT)}")
    print()
    print(wide.to_string(index=False))


if __name__ == "__main__":
    main()
