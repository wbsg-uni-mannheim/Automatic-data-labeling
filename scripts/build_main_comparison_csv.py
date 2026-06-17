#!/usr/bin/env python3
"""Final main comparison CSV: benchmark set vs Similarity search vs AL ML vs AL Ditto
vs LLM-direct (gpt-5.2 / qwen / kimi). Includes Δ-to-benchmark for the best
auto-labeled method.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path("/work/aasteine/Automatic-data-labeling")
SRC = ROOT / "output/results_summary/plot_benchmark_size_comparison.csv"
TEST_PRED_ROOT = ROOT / "output/test_set_predictions"
OUT_DIR = ROOT / "output/results_summary"

# Published benchmark-set F1 means + stds (Ditto trained on the official benchmark train split)
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


def load_llm_direct_f1(benchmark: str, labeller: str):
    """labeller in {'gpt-5.2', 'qwen', 'kimi'}."""
    p = TEST_PRED_ROOT / benchmark / labeller / "summary.json"
    if not p.exists():
        return None, None, None
    d = json.loads(p.read_text())
    return d["f1"] * 100, d["precision"] * 100, d["recall"] * 100


def main():
    df = pd.read_csv(SRC)
    # pivot to method columns
    df["f1_mean_pct"] = df["f1_mean"] * 100
    df["f1_std_pct"] = df["f1_std"] * 100
    df["cell"] = df.apply(lambda r: f"{r.f1_mean_pct:.2f} ± {r.f1_std_pct:.2f}", axis=1)

    rows_long = []
    rows_wide = []
    for bm in BM_ORDER:
        bench_mean, bench_std = BENCHMARK_SET[bm]
        sub = df[df["benchmark"] == bm]
        method_cells = {}
        method_means = {}
        method_stds = {}
        for _, r in sub.iterrows():
            method_cells[r["method"]] = r["cell"]
            method_means[r["method"]] = r["f1_mean_pct"]
            method_stds[r["method"]] = r["f1_std_pct"]
        auto_means = [method_means.get(m) for m in ("sim", "al_ml", "al_ditto") if method_means.get(m) is not None]
        best_auto = max(auto_means) if auto_means else None
        delta = (best_auto - bench_mean) if best_auto is not None else None

        # LLM-direct on test set
        llm_means = {}
        for lab in ("gpt-5.2", "qwen", "kimi"):
            f1, p, r = load_llm_direct_f1(bm, lab)
            llm_means[lab] = f1
        best_llm = max([v for v in llm_means.values() if v is not None], default=None)
        delta_llm = (best_llm - bench_mean) if best_llm is not None else None

        rows_wide.append({
            "Benchmark":              BM_DISPLAY[bm],
            "Benchmark set":          f"{bench_mean:.2f} ± {bench_std:.2f}",
            "Similarity search":      method_cells.get("sim", "—"),
            "Active learning (ML)":   method_cells.get("al_ml", "—"),
            "Active learning (Ditto)":method_cells.get("al_ditto", "—"),
            "Δ to bench. (best auto)":f"{delta:+.2f}" if delta is not None else "—",
            "LLM direct (GPT-5.2)":   f"{llm_means['gpt-5.2']:.2f}" if llm_means.get("gpt-5.2") is not None else "—",
            "LLM direct (Qwen)":      f"{llm_means['qwen']:.2f}"    if llm_means.get("qwen")    is not None else "—",
            "LLM direct (Kimi)":      f"{llm_means['kimi']:.2f}"    if llm_means.get("kimi")    is not None else "—",
            "Δ to bench. (best LLM)": f"{delta_llm:+.2f}" if delta_llm is not None else "—",
        })
        rows_long.append({
            "benchmark": bm,
            "benchmark_set_mean": bench_mean, "benchmark_set_std": bench_std,
            "sim_mean": method_means.get("sim"), "sim_std": method_stds.get("sim"),
            "al_ml_mean": method_means.get("al_ml"), "al_ml_std": method_stds.get("al_ml"),
            "al_ditto_mean": method_means.get("al_ditto"), "al_ditto_std": method_stds.get("al_ditto"),
            "best_auto_mean": best_auto,
            "delta_best_auto_minus_bench": delta,
            "llm_direct_gpt_5_2": llm_means.get("gpt-5.2"),
            "llm_direct_qwen":    llm_means.get("qwen"),
            "llm_direct_kimi":    llm_means.get("kimi"),
            "best_llm_direct":    best_llm,
            "delta_best_llm_direct_minus_bench": delta_llm,
        })

    wide_df = pd.DataFrame(rows_wide)
    long_df = pd.DataFrame(rows_long)
    wide_path = OUT_DIR / "main_comparison_benchmark_size.csv"
    long_path = OUT_DIR / "main_comparison_benchmark_size_long.csv"
    wide_df.to_csv(wide_path, index=False)
    long_df.to_csv(long_path, index=False)
    print(f"Wrote {wide_path.relative_to(ROOT)}")
    print(f"Wrote {long_path.relative_to(ROOT)}")
    print()
    print(wide_df.to_string(index=False))


if __name__ == "__main__":
    main()
