#!/usr/bin/env python3
"""Slim CSV: Benchmark set vs LLM-direct (GPT-5.2 / Qwen / Kimi) on the same
official test files. Δ column = best LLM direct - benchmark set.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path("/work/aasteine/Automatic-data-labeling")
TEST_PRED_ROOT = ROOT / "output/test_set_predictions"
OUT = ROOT / "output/results_summary/llm_direct_vs_benchmark.csv"

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
LABELLERS = [("gpt-5.2", "GPT-5.2"), ("qwen", "Qwen 3.6+"), ("kimi", "Kimi K2.6")]


def load_f1(bm, lab):
    p = TEST_PRED_ROOT / bm / lab / "summary.json"
    if not p.exists(): return None
    return json.loads(p.read_text())["f1"] * 100


def main():
    rows = []
    for bm in BM_ORDER:
        bm_mean, bm_std = BENCHMARK_SET[bm]
        f1s = {lab: load_f1(bm, lab) for lab, _ in LABELLERS}
        best = max([v for v in f1s.values() if v is not None], default=None)
        delta = (best - bm_mean) if best is not None else None
        rows.append({
            "Benchmark":      BM_DISPLAY[bm],
            "Benchmark set":  f"{bm_mean:.2f} ± {bm_std:.2f}",
            "GPT-5.2":        f"{f1s['gpt-5.2']:.2f}" if f1s['gpt-5.2'] is not None else "—",
            "Qwen 3.6+":      f"{f1s['qwen']:.2f}"    if f1s['qwen']    is not None else "—",
            "Kimi K2.6":      f"{f1s['kimi']:.2f}"    if f1s['kimi']    is not None else "—",
            "Δ best LLM":     f"{delta:+.2f}" if delta is not None else "—",
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
