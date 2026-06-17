#!/usr/bin/env python3
"""Build per-run and summary CSVs covering all four methods × all regimes.

Methods:
  - baseline (supervised Ditto on official train+valid splits)
  - similarity_selection  (seed_round_only_profiles)
  - simple_active_learning (simple_active_learning_labeling)
  - ditto_active_learning  (three_phase_labeling_ditto_only_v2, labelled by GPT-5.2)

Regimes:
  - supervised   (baseline only; official train+valid)
  - test_cleaned (test-leak + dedup removed; official valid)
  - llm_valid    (test_cleaned + 80/20 stratified split of LLM labels for valid)

Profile per (method × benchmark) matches the published table.

Outputs:
  output/results_summary/run_level_metrics.csv
  output/results_summary/summary_metrics.csv
"""
from __future__ import annotations

import csv
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path("/work/aasteine/Automatic-data-labeling")
OUT_DIR = ROOT / "output/results_summary"

BENCHMARKS = ["abt-buy", "dblp-acm", "dblp-scholar", "walmart-amazon", "wdc"]

# Per-cell profile choice matching the published table.
PROFILE = {
    ("similarity_selection",  "abt-buy"):        "all_plus20random",
    ("similarity_selection",  "dblp-acm"):       "all_plus20random",
    ("similarity_selection",  "dblp-scholar"):   "all_plus20random",
    ("similarity_selection",  "walmart-amazon"): "all",
    ("similarity_selection",  "wdc"):            "medium",
    ("simple_active_learning", "abt-buy"):        "large",
    ("simple_active_learning", "dblp-acm"):       "all",
    ("simple_active_learning", "dblp-scholar"):   "all",
    ("simple_active_learning", "walmart-amazon"): "all",
    ("simple_active_learning", "wdc"):            "large",
    ("ditto_active_learning", "abt-buy"):        "all_plus20random",
    ("ditto_active_learning", "dblp-acm"):       "all_plus20random",
    ("ditto_active_learning", "dblp-scholar"):   "all_plus20random",
    ("ditto_active_learning", "walmart-amazon"): "all_plus20random",
    ("ditto_active_learning", "wdc"):            "all",
}


def _read_metrics(p: Path) -> Tuple[float, float, float, float]:
    data = json.loads(p.read_text()).get("test", {}) or {}
    return data.get("precision"), data.get("recall"), data.get("f1"), data.get("accuracy")


def _parse_seed_from_run_dirname(name: str) -> int | None:
    m = re.search(r"_seed(\d+)_", name)
    return int(m.group(1)) if m else None


def collect_runs() -> List[Dict]:
    rows: List[Dict] = []

    # --- 1) BASELINE (supervised) ---
    base_root = ROOT / "output/training_from_generated_labels_3runs/ditto_baseline"
    for run_dir in sorted(base_root.glob("baseline_r*_seed*")):
        seed = int(re.search(r"seed(\d+)", run_dir.name).group(1))
        repeat = int(re.search(r"_r(\d+)_", run_dir.name).group(1))
        for bm in BENCHMARKS:
            mp = run_dir / bm / "metrics.json"
            if not mp.exists(): continue
            prec, rec, f1, acc = _read_metrics(mp)
            rows.append({
                "method": "baseline", "regime": "supervised", "benchmark": bm,
                "profile": "(official train+valid)", "labeller": "gold (official)",
                "seed": seed, "repeat": repeat,
                "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
                "metrics_path": str(mp.relative_to(ROOT)),
            })

    # --- 2) SIMILARITY SELECTION (seed_round_only_profiles) ---
    # test_cleaned, then llm_valid.
    def add_sr_runs(method_root: Path, regime: str):
        for mp in method_root.glob("cleaned_*/[!_]*/metrics.json"):
            if "training_output" in mp.parts: continue
            run_name = mp.parts[-3]
            bm = mp.parts[-2]
            seed = _parse_seed_from_run_dirname(run_name)
            if seed is None: continue
            prof = PROFILE[("similarity_selection", bm)]
            prec, rec, f1, acc = _read_metrics(mp)
            rows.append({
                "method": "similarity_selection", "regime": regime, "benchmark": bm,
                "profile": prof, "labeller": "gpt-5.2",
                "seed": seed, "repeat": -1,
                "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
                "metrics_path": str(mp.relative_to(ROOT)),
            })

    # test_cleaned: use the "_published" runs for cells whose profile differs
    # from the original config; otherwise use the original.
    # Original config: all_plus20random for everywhere
    for mp in (ROOT / "output/ditto_cleaned_seedround_runs").glob("cleaned_*/[!_]*/metrics.json"):
        if "training_output" in mp.parts: continue
        run_name = mp.parts[-3]
        bm = mp.parts[-2]
        # Skip cells whose published profile differs (walmart-amazon=all, wdc=medium)
        if bm in ("walmart-amazon", "wdc"): continue
        seed = _parse_seed_from_run_dirname(run_name)
        prec, rec, f1, acc = _read_metrics(mp)
        rows.append({
            "method": "similarity_selection", "regime": "test_cleaned", "benchmark": bm,
            "profile": PROFILE[("similarity_selection", bm)], "labeller": "gpt-5.2",
            "seed": seed, "repeat": -1,
            "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
            "metrics_path": str(mp.relative_to(ROOT)),
        })
    for mp in (ROOT / "output/ditto_cleaned_seedround_published_runs").glob("cleaned_*/[!_]*/metrics.json"):
        if "training_output" in mp.parts: continue
        run_name = mp.parts[-3]
        bm = mp.parts[-2]
        seed = _parse_seed_from_run_dirname(run_name)
        prec, rec, f1, acc = _read_metrics(mp)
        rows.append({
            "method": "similarity_selection", "regime": "test_cleaned", "benchmark": bm,
            "profile": PROFILE[("similarity_selection", bm)], "labeller": "gpt-5.2",
            "seed": seed, "repeat": -1,
            "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
            "metrics_path": str(mp.relative_to(ROOT)),
        })
    add_sr_runs(ROOT / "output/ditto_cleaned_seedround_llmvalid_runs", "llm_valid")

    # --- 3) SIMPLE ACTIVE LEARNING ---
    for mp in (ROOT / "output/ditto_cleaned_simpleactive_runs").glob("cleaned_*/[!_]*/metrics.json"):
        if "training_output" in mp.parts: continue
        run_name = mp.parts[-3]
        bm = mp.parts[-2]
        if bm == "wdc": continue  # use published-profile version instead
        seed = _parse_seed_from_run_dirname(run_name)
        prec, rec, f1, acc = _read_metrics(mp)
        rows.append({
            "method": "simple_active_learning", "regime": "test_cleaned", "benchmark": bm,
            "profile": PROFILE[("simple_active_learning", bm)], "labeller": "gpt-5.2",
            "seed": seed, "repeat": -1,
            "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
            "metrics_path": str(mp.relative_to(ROOT)),
        })
    for mp in (ROOT / "output/ditto_cleaned_simpleactive_published_runs").glob("cleaned_*/[!_]*/metrics.json"):
        if "training_output" in mp.parts: continue
        run_name = mp.parts[-3]
        bm = mp.parts[-2]
        seed = _parse_seed_from_run_dirname(run_name)
        prec, rec, f1, acc = _read_metrics(mp)
        rows.append({
            "method": "simple_active_learning", "regime": "test_cleaned", "benchmark": bm,
            "profile": PROFILE[("simple_active_learning", bm)], "labeller": "gpt-5.2",
            "seed": seed, "repeat": -1,
            "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
            "metrics_path": str(mp.relative_to(ROOT)),
        })
    for mp in (ROOT / "output/ditto_cleaned_simpleactive_llmvalid_runs").glob("cleaned_*/[!_]*/metrics.json"):
        if "training_output" in mp.parts: continue
        run_name = mp.parts[-3]
        bm = mp.parts[-2]
        seed = _parse_seed_from_run_dirname(run_name)
        prec, rec, f1, acc = _read_metrics(mp)
        rows.append({
            "method": "simple_active_learning", "regime": "llm_valid", "benchmark": bm,
            "profile": PROFILE[("simple_active_learning", bm)], "labeller": "gpt-5.2",
            "seed": seed, "repeat": -1,
            "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
            "metrics_path": str(mp.relative_to(ROOT)),
        })

    # --- 4) DITTO ACTIVE LEARNING (GPT-5.2, Qwen 3.6+, Kimi K2.6) ---
    LABELLER_NAMES = {
        "gpt":  "gpt-5.2",
        "qwen": "qwen3.6-plus",
        "kimi": "kimi-k2.6",
    }

    def add_ditto_al(method_root: Path, regime: str, prefix_re: str):
        for mp in method_root.glob("cleaned_*/[!_]*/metrics.json"):
            if "training_output" in mp.parts: continue
            run_name = mp.parts[-3]
            bm = mp.parts[-2]
            m = re.match(prefix_re + r"(\w+)_(" + "|".join(re.escape(b) for b in BENCHMARKS) + r")_r\d+_seed(\d+)_", run_name)
            if not m: continue
            labeller_key, _, seed = m.group(1), m.group(2), int(m.group(3))
            if labeller_key not in LABELLER_NAMES: continue
            prec, rec, f1, acc = _read_metrics(mp)
            rows.append({
                "method": "ditto_active_learning", "regime": regime, "benchmark": bm,
                "profile": PROFILE[("ditto_active_learning", bm)],
                "labeller": LABELLER_NAMES[labeller_key],
                "seed": seed, "repeat": -1,
                "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
                "metrics_path": str(mp.relative_to(ROOT)),
            })

    add_ditto_al(ROOT / "output/ditto_cleaned_runs",           "test_cleaned", r"cleaned_")
    add_ditto_al(ROOT / "output/ditto_cleaned_llmvalid_runs",  "llm_valid",   r"cleaned_llmv_")

    return rows


def write_run_csv(rows: List[Dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "run_level_metrics.csv"
    cols = ["method", "regime", "benchmark", "profile", "labeller",
            "seed", "repeat", "precision", "recall", "f1", "accuracy", "metrics_path"]
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"Wrote {p}  ({len(rows)} rows)")


def write_summary_csv(rows: List[Dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Group by (method, regime, benchmark, labeller) — labeller matters for ditto_active_learning
    buckets: Dict[Tuple[str, str, str, str], List[Dict]] = defaultdict(list)
    for r in rows:
        buckets[(r["method"], r["regime"], r["benchmark"], r["labeller"])].append(r)
    out_rows: List[Dict] = []
    for (method, regime, bm, labeller), seeds in sorted(buckets.items()):
        for m in ("precision", "recall", "f1", "accuracy"):
            pass
        prec = [r["precision"] for r in seeds if r["precision"] is not None]
        rec  = [r["recall"]    for r in seeds if r["recall"]    is not None]
        f1   = [r["f1"]        for r in seeds if r["f1"]        is not None]
        acc  = [r["accuracy"]  for r in seeds if r["accuracy"]  is not None]
        out_rows.append({
            "method": method, "regime": regime, "benchmark": bm,
            "profile": seeds[0]["profile"], "labeller": seeds[0]["labeller"],
            "n": len(f1),
            "precision_mean": sum(prec)/len(prec) if prec else "",
            "precision_std":  st.stdev(prec) if len(prec) > 1 else 0.0,
            "recall_mean":    sum(rec)/len(rec) if rec else "",
            "recall_std":     st.stdev(rec) if len(rec) > 1 else 0.0,
            "f1_mean":        sum(f1)/len(f1) if f1 else "",
            "f1_std":         st.stdev(f1) if len(f1) > 1 else 0.0,
            "accuracy_mean":  sum(acc)/len(acc) if acc else "",
            "accuracy_std":   st.stdev(acc) if len(acc) > 1 else 0.0,
        })
    p = OUT_DIR / "summary_metrics.csv"
    cols = ["method", "regime", "benchmark", "profile", "labeller", "n",
            "precision_mean", "precision_std", "recall_mean", "recall_std",
            "f1_mean", "f1_std", "accuracy_mean", "accuracy_std"]
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out_rows:
            # Round numeric to 4 d.p.
            for k in cols:
                if k.endswith("_mean") or k.endswith("_std"):
                    v = r.get(k, "")
                    if isinstance(v, float):
                        r[k] = round(v, 4)
            w.writerow(r)
    print(f"Wrote {p}  ({len(out_rows)} rows)")


def main() -> None:
    rows = collect_runs()
    write_run_csv(rows)
    write_summary_csv(rows)


if __name__ == "__main__":
    main()
