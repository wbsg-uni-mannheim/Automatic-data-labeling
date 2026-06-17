#!/usr/bin/env python3
"""Build a plot-ready CSV for the abt-buy size-scan experiment.

Combines:
  - the 21 new (labeller × profile) trainings in output/ditto_size_scan_abtbuy_runs/
  - the existing all_plus20random results (n=6) from output/ditto_cleaned_runs/

Output: output/results_summary/size_scan_abtbuy.csv
"""
from __future__ import annotations

import csv
import glob
import gzip
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path("/work/aasteine/Automatic-data-labeling")
OUT_DIR = ROOT / "output/results_summary"

PROFILES = [
    "small", "small_plus20random",
    "medium", "medium_plus20random",
    "large", "large_plus20random",
    "all", "all_plus20random",
]
LABELLERS = {"gpt": "gpt-5.2", "qwen": "qwen3.6-plus", "kimi": "kimi-k2.6"}


def cleaned_size(labeller: str, profile: str) -> int:
    fp = ROOT / f"output/labeling_cleaned_size_scan/{labeller}/abt-buy/profiles/{profile}/active_labels_latest_abt-buy_{profile}_train.json.gz"
    if not fp.exists():
        return 0
    with gzip.open(fp, "rt") as f:
        return sum(1 for line in f if line.strip())


def collect_size_scan_runs():
    pool: Dict[tuple, List[dict]] = defaultdict(list)
    for r in glob.glob(str(ROOT / "output/ditto_size_scan_abtbuy_runs/scan_*/abt-buy/metrics.json")):
        if "training_output" in r: continue
        name = Path(r).parts[-3]
        m = re.match(r"scan_(gpt|qwen|kimi)_(.+?)_r\d+_seed(\d+)_", name)
        if not m: continue
        labeller_key, profile, seed = m.group(1), m.group(2), int(m.group(3))
        d = json.loads(Path(r).read_text()).get("test", {}) or {}
        pool[(labeller_key, profile)].append({
            "seed": seed, "p": d.get("precision"), "r": d.get("recall"),
            "f1": d.get("f1"), "acc": d.get("accuracy"),
        })
    return pool


def collect_existing_all_plus20random():
    """The all_plus20random cells were trained earlier in output/ditto_cleaned_runs/."""
    pool: Dict[tuple, List[dict]] = defaultdict(list)
    for r in glob.glob(str(ROOT / "output/ditto_cleaned_runs/cleaned_*/abt-buy/metrics.json")):
        if "training_output" in r: continue
        name = Path(r).parts[-3]
        m = re.match(r"cleaned_(\w+)_abt-buy_r\d+_seed(\d+)_", name)
        if not m: continue
        labeller_key, seed = m.group(1), int(m.group(2))
        if labeller_key not in LABELLERS: continue
        d = json.loads(Path(r).read_text()).get("test", {}) or {}
        pool[(labeller_key, "all_plus20random")].append({
            "seed": seed, "p": d.get("precision"), "r": d.get("recall"),
            "f1": d.get("f1"), "acc": d.get("accuracy"),
        })
    return pool


def main() -> None:
    scan = collect_size_scan_runs()
    existing = collect_existing_all_plus20random()
    # Prefer existing (n=6) over scan (n=3) for all_plus20random
    for k, v in existing.items():
        scan[k] = v

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Per-run CSV
    run_rows: List[Dict] = []
    for (labeller_key, profile), seeds in scan.items():
        for s in seeds:
            run_rows.append({
                "benchmark": "abt-buy",
                "labeller": LABELLERS[labeller_key],
                "profile": profile,
                "training_size": cleaned_size(labeller_key, profile),
                "seed": s["seed"],
                "precision": s["p"], "recall": s["r"], "f1": s["f1"], "accuracy": s["acc"],
            })
    run_rows.sort(key=lambda r: (r["labeller"], PROFILES.index(r["profile"]) if r["profile"] in PROFILES else 99, r["seed"]))

    run_csv = OUT_DIR / "size_scan_abtbuy_runs.csv"
    cols = ["benchmark","labeller","profile","training_size","seed","precision","recall","f1","accuracy"]
    with run_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in run_rows: w.writerow(r)
    print(f"Wrote {run_csv}  ({len(run_rows)} rows)")

    # Summary CSV
    summary_rows: List[Dict] = []
    for (labeller_key, profile), seeds in sorted(scan.items(),
                                                  key=lambda x: (x[0][0], PROFILES.index(x[0][1]) if x[0][1] in PROFILES else 99)):
        ps = [s["p"] for s in seeds if s["p"] is not None]
        rs = [s["r"] for s in seeds if s["r"] is not None]
        f1s = [s["f1"] for s in seeds if s["f1"] is not None]
        accs = [s["acc"] for s in seeds if s["acc"] is not None]
        summary_rows.append({
            "benchmark": "abt-buy",
            "labeller": LABELLERS[labeller_key],
            "profile": profile,
            "training_size": cleaned_size(labeller_key, profile),
            "n_seeds": len(f1s),
            "precision_mean": round(sum(ps)/len(ps), 4) if ps else "",
            "precision_std":  round(st.stdev(ps) if len(ps)>1 else 0.0, 4),
            "recall_mean":    round(sum(rs)/len(rs), 4) if rs else "",
            "recall_std":     round(st.stdev(rs) if len(rs)>1 else 0.0, 4),
            "f1_mean":        round(sum(f1s)/len(f1s), 4) if f1s else "",
            "f1_std":         round(st.stdev(f1s) if len(f1s)>1 else 0.0, 4),
            "accuracy_mean":  round(sum(accs)/len(accs), 4) if accs else "",
            "accuracy_std":   round(st.stdev(accs) if len(accs)>1 else 0.0, 4),
        })

    sum_csv = OUT_DIR / "size_scan_abtbuy.csv"
    cols = ["benchmark","labeller","profile","training_size","n_seeds",
            "precision_mean","precision_std","recall_mean","recall_std",
            "f1_mean","f1_std","accuracy_mean","accuracy_std"]
    with sum_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in summary_rows: w.writerow(r)
    print(f"Wrote {sum_csv}  ({len(summary_rows)} rows)")

    # Pretty-print for human eyes
    print(f"\n{'labeller':<14} {'profile':<22} {'size':<6} {'n':<3} {'F1 mean ± std':<18}")
    for r in summary_rows:
        print(f"{r['labeller']:<14} {r['profile']:<22} {r['training_size']:<6} {r['n_seeds']:<3} {r['f1_mean']:.4f} ± {r['f1_std']:.4f}")


if __name__ == "__main__":
    main()
