#!/usr/bin/env python3
"""Produce focused per-plot CSVs from the run registry.

Outputs (in output/results_summary/):
  1. plot_main_comparison.csv        — baseline + 3 methods (similarity/simple_active/ditto_AL), GPT-5.2
  2. plot_llm_comparison.csv         — Ditto AL across GPT/Qwen/Kimi
  3. plot_postfilter_variants.csv    — Ditto AL post-processing variants vs baseline
  4. plot_active_learning_size.csv   — abt-buy F1 vs label budget (already exists; this is a convenience copy)
  5. plot_cleaning_impact.csv        — raw (leaky) vs test_cleaned across labellers (shows leak penalty)
  6. plot_delta_heatmap.csv          — (method, labeller) x benchmark matrix of Δ vs baseline (heatmap-ready)
"""
from __future__ import annotations
import csv, json, re, statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path("/work/aasteine/Automatic-data-labeling")
OUT = ROOT / "output/results_summary"

reg = list(csv.DictReader(open(OUT / "run_registry.csv")))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def agg(records, key):
    vals = [_f(r[key]) for r in records if _f(r[key]) is not None]
    if not vals:
        return None, None
    return sum(vals) / len(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def per_cell(records, n_label="n"):
    out = {"n": len(records)}
    for k in ["precision", "recall", "f1", "accuracy"]:
        m, s = agg(records, k)
        out[f"{k}_mean"] = round(m, 4) if m is not None else ""
        out[f"{k}_std"] = round(s, 4) if s is not None else ""
    return out


# ---- Baseline lookup (per benchmark, mean f1 — used for Δ columns) ----
baseline_rows = [r for r in reg if r["pipeline"] == "supervised_baseline" and r["benchmark"] in
                 ("abt-buy", "dblp-acm", "dblp-scholar", "walmart-amazon", "wdc")]
baseline_by_bm = defaultdict(list)
for r in baseline_rows:
    baseline_by_bm[r["benchmark"]].append(r)


def baseline_for(bm):
    recs = baseline_by_bm.get(bm, [])
    m, s = agg(recs, "f1")
    return m, s


# ---- 1) Main comparison: 4 methods on GPT-5.2 (test_cleaned regime) ----
# Baseline + similarity_selection + simple_active + ditto_active_learning, all on GPT-5.2.
# Use the canonical profile per cell (matches the published table; uses _published_profile rows where applicable).

PROFILE_BY_BM = {
    "similarity_selection": {"abt-buy": "all_plus20random", "dblp-acm": "all_plus20random",
                             "dblp-scholar": "all_plus20random", "walmart-amazon": "all", "wdc": "medium"},
    "simple_active_learning": {"abt-buy": "large", "dblp-acm": "all", "dblp-scholar": "all",
                               "walmart-amazon": "all", "wdc": "large"},
    "ditto_active_learning": {"abt-buy": "all_plus20random", "dblp-acm": "all_plus20random",
                              "dblp-scholar": "all_plus20random", "walmart-amazon": "all_plus20random", "wdc": "all"},
}

def collect_method_test_cleaned(method, labeller, bm):
    profile = PROFILE_BY_BM[method][bm]
    if method == "ditto_active_learning":
        return [r for r in reg if r["pipeline"] == "test_cleaned_llm_labeller_canonical"
                and r["method"] == "ditto_active_learning"
                and r["labeller"] == labeller and r["benchmark"] == bm]
    pipelines = ("method_test_cleaned", "method_test_cleaned_published_profile")
    return [r for r in reg if r["pipeline"] in pipelines and r["method"] == method
            and r["labeller"] == labeller and r["benchmark"] == bm and r["profile"] == profile]


BMS = ["abt-buy", "walmart-amazon", "wdc", "dblp-acm", "dblp-scholar"]
methods_main = [("baseline", "gold (official)"),
                ("similarity_selection", "gpt-5.2"),
                ("simple_active_learning", "gpt-5.2"),
                ("ditto_active_learning", "gpt-5.2")]

rows = []
for bm in BMS:
    bf1_m, _ = baseline_for(bm)
    for method, labeller in methods_main:
        if method == "baseline":
            recs = baseline_by_bm[bm]
            profile = "(official train+valid)"
        else:
            recs = collect_method_test_cleaned(method, labeller, bm)
            profile = PROFILE_BY_BM[method][bm]
        cell = per_cell(recs)
        delta = (cell["f1_mean"] - bf1_m) if (bf1_m is not None and cell["f1_mean"] != "") else ""
        rows.append({"benchmark": bm, "method": method, "labeller": labeller, "profile": profile,
                     **cell, "f1_delta_vs_baseline": round(delta, 4) if delta != "" else ""})

with (OUT / "plot_main_comparison.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"Wrote {OUT/'plot_main_comparison.csv'} ({len(rows)} rows)")


# ---- 2) LLM comparison: Ditto AL across GPT/Qwen/Kimi (test_cleaned regime) ----
rows = []
for bm in BMS:
    bf1_m, _ = baseline_for(bm)
    rows.append({"benchmark": bm, "labeller": "gold (official)", "method": "baseline",
                 "profile": "(official train+valid)", **per_cell(baseline_by_bm[bm]),
                 "f1_delta_vs_baseline": 0.0})
    for labeller in ["gpt-5.2", "qwen3.6-plus", "kimi-k2.6"]:
        recs = [r for r in reg if r["pipeline"] == "test_cleaned_llm_labeller_canonical"
                and r["labeller"] == labeller and r["benchmark"] == bm]
        cell = per_cell(recs)
        delta = (cell["f1_mean"] - bf1_m) if (bf1_m is not None and cell["f1_mean"] != "") else ""
        rows.append({"benchmark": bm, "labeller": labeller, "method": "ditto_active_learning",
                     "profile": PROFILE_BY_BM["ditto_active_learning"][bm], **cell,
                     "f1_delta_vs_baseline": round(delta, 4) if delta != "" else ""})

with (OUT / "plot_llm_comparison.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"Wrote {OUT/'plot_llm_comparison.csv'} ({len(rows)} rows)")


# ---- 3) Postfilter variants: AL(Ditto) + 5 post-processing variants, GPT-5.2 ----
VARIANT_PIPELINES = {
    "AL_Ditto": "test_cleaned_llm_labeller_canonical",
    "+Relabel": "ditto_variant_v_relabel",
    "+Relabel drop": "ditto_variant_v_relabel_drop",
    "+Closure drop": "ditto_variant_v_closure_drop",
    "+Closure AND relabel drop": "ditto_variant_v_closure_relabel",
    "+Closure OR relabel drop": "ditto_variant_v_closure_or_relabel",
}
rows = []
for bm in BMS:
    bf1_m, _ = baseline_for(bm)
    rows.append({"benchmark": bm, "variant": "Benchmark set (baseline)",
                 **per_cell(baseline_by_bm[bm]), "f1_delta_vs_baseline": 0.0})
    for variant_label, pipeline in VARIANT_PIPELINES.items():
        recs = [r for r in reg if r["pipeline"] == pipeline
                and r["labeller"] == "gpt-5.2" and r["benchmark"] == bm]
        cell = per_cell(recs)
        delta = (cell["f1_mean"] - bf1_m) if (bf1_m is not None and cell["f1_mean"] != "") else ""
        rows.append({"benchmark": bm, "variant": variant_label, **cell,
                     "f1_delta_vs_baseline": round(delta, 4) if delta != "" else ""})

with (OUT / "plot_postfilter_variants.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"Wrote {OUT/'plot_postfilter_variants.csv'} ({len(rows)} rows)")


# ---- 4) Active learning size plot: abt-buy F1 vs label budget across all 3 methods ----
size_rows = list(csv.DictReader(open(OUT / "size_scan_abtbuy_summary.csv")))
# Add Δ vs baseline column for convenience
bf_abt, _ = baseline_for("abt-buy")
for r in size_rows:
    if bf_abt is not None and r["f1_mean"]:
        r["f1_delta_vs_baseline"] = round(float(r["f1_mean"]) - bf_abt, 4)
    else:
        r["f1_delta_vs_baseline"] = ""

with (OUT / "plot_active_learning_size.csv").open("w", newline="") as f:
    cols = list(size_rows[0].keys())
    if "f1_delta_vs_baseline" not in cols:
        cols.append("f1_delta_vs_baseline")
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader(); w.writerows(size_rows)
print(f"Wrote {OUT/'plot_active_learning_size.csv'} ({len(size_rows)} rows)")


# ---- 5) Cleaning impact: raw (leaky) vs test_cleaned (per labeller × benchmark) ----
# Raw (leaky) numbers were in the user-provided published table at the start of the conversation;
# we can recover them per-run from the older 3-runs export.
import glob
def collect_raw_3runs(label_root: str, labeller_key: str):
    pool = defaultdict(list)
    for r in glob.glob(f"export_260418_1200_3runs_errorana/training_from_generated_labels_3runs/{label_root}/generated_ditto_*/*/metrics.json"):
        if "training_output" in r: continue
        nm = Path(r).parts[-3]
        bm_dir = Path(r).parts[-2]
        m = re.match(r"generated_ditto_benchmark_([a-z-]+)_\d+_\d+_(.+?)_r\d+_\d+_\d+", nm)
        if not m: continue
        bm, profile = m.group(1), m.group(2)
        if bm not in BMS: continue
        canonical = PROFILE_BY_BM["ditto_active_learning"][bm]
        if profile != canonical: continue
        f1 = json.loads(Path(r).read_text()).get("test", {}).get("f1")
        if f1 is not None:
            pool[bm].append(f1)
    return pool

raw_sources = {
    "gpt-5.2": "three_phase_labeling_ditto_only_v2",
    "qwen3.6-plus": "qwen",
    "kimi-k2.6": "kimi",
}
rows = []
for labeller_display, source_dir in raw_sources.items():
    raw_pool = collect_raw_3runs(source_dir, labeller_display)
    for bm in BMS:
        bf1, _ = baseline_for(bm)
        raw_vals = raw_pool.get(bm, [])
        raw_mean = sum(raw_vals)/len(raw_vals) if raw_vals else None
        raw_std = st.stdev(raw_vals) if len(raw_vals) > 1 else 0.0
        clean_recs = [r for r in reg if r["pipeline"] == "test_cleaned_llm_labeller_canonical"
                      and r["labeller"] == labeller_display and r["benchmark"] == bm]
        clean = per_cell(clean_recs)
        leak_penalty = (raw_mean - clean["f1_mean"]) if (raw_mean is not None and clean["f1_mean"] != "") else ""
        rows.append({
            "benchmark": bm, "labeller": labeller_display,
            "raw_leaky_f1_mean": round(raw_mean, 4) if raw_mean is not None else "",
            "raw_leaky_f1_std": round(raw_std, 4) if raw_vals else "",
            "raw_leaky_n": len(raw_vals),
            "test_cleaned_f1_mean": clean["f1_mean"], "test_cleaned_f1_std": clean["f1_std"], "test_cleaned_n": clean["n"],
            "leak_penalty_f1": round(leak_penalty, 4) if leak_penalty != "" else "",
            "baseline_f1": round(bf1, 4) if bf1 is not None else "",
        })

with (OUT / "plot_cleaning_impact.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"Wrote {OUT/'plot_cleaning_impact.csv'} ({len(rows)} rows)")


# ---- 6) Delta-vs-baseline heatmap matrix (long format, easy to pivot) ----
# Rows = (pipeline_label, labeller); columns = benchmark; value = Δ f1 vs baseline.
LONG_PIPELINES = [
    ("baseline (supervised)", "supervised_baseline", "gold (official)"),
    ("similarity_selection (cleaned)", "method_test_cleaned", "gpt-5.2"),
    ("similarity_selection (cleaned, pub profile)", "method_test_cleaned_published_profile", "gpt-5.2"),
    ("simple_active_learning (cleaned)", "method_test_cleaned", "gpt-5.2"),
    ("simple_active_learning (cleaned, pub profile)", "method_test_cleaned_published_profile", "gpt-5.2"),
    ("ditto_AL (cleaned)", "test_cleaned_llm_labeller_canonical", "gpt-5.2"),
    ("ditto_AL (cleaned, qwen)", "test_cleaned_llm_labeller_canonical", "qwen3.6-plus"),
    ("ditto_AL (cleaned, kimi)", "test_cleaned_llm_labeller_canonical", "kimi-k2.6"),
    ("ditto_AL (no_val)", "no_val_cleaned", "gpt-5.2"),
    ("ditto_AL (llm_valid)", "llm_valid", "gpt-5.2"),
    ("ditto_AL +Relabel", "ditto_variant_v_relabel", "gpt-5.2"),
    ("ditto_AL +Relabel drop", "ditto_variant_v_relabel_drop", "gpt-5.2"),
    ("ditto_AL +Closure drop", "ditto_variant_v_closure_drop", "gpt-5.2"),
    ("ditto_AL +Closure AND relabel drop", "ditto_variant_v_closure_relabel", "gpt-5.2"),
    ("ditto_AL +Closure OR relabel drop", "ditto_variant_v_closure_or_relabel", "gpt-5.2"),
]
rows = []
for display_name, pipeline, labeller in LONG_PIPELINES:
    for bm in BMS:
        bf1, _ = baseline_for(bm)
        if pipeline == "method_test_cleaned" and display_name.startswith("similarity_selection"):
            recs = [r for r in reg if r["pipeline"] == pipeline and r["method"] == "similarity_selection"
                    and r["labeller"] == labeller and r["benchmark"] == bm]
        elif pipeline == "method_test_cleaned_published_profile" and "similarity" in display_name:
            recs = [r for r in reg if r["pipeline"] == pipeline and r["method"] == "similarity_selection"
                    and r["labeller"] == labeller and r["benchmark"] == bm]
        elif pipeline == "method_test_cleaned" and display_name.startswith("simple_active"):
            recs = [r for r in reg if r["pipeline"] == pipeline and r["method"] == "simple_active_learning"
                    and r["labeller"] == labeller and r["benchmark"] == bm]
        elif pipeline == "method_test_cleaned_published_profile" and "simple_active" in display_name:
            recs = [r for r in reg if r["pipeline"] == pipeline and r["method"] == "simple_active_learning"
                    and r["labeller"] == labeller and r["benchmark"] == bm]
        else:
            recs = [r for r in reg if r["pipeline"] == pipeline
                    and r["labeller"] == labeller and r["benchmark"] == bm]
        cell = per_cell(recs)
        delta = (cell["f1_mean"] - bf1) if (bf1 is not None and cell["f1_mean"] != "") else ""
        rows.append({"pipeline_label": display_name, "pipeline_key": pipeline, "labeller": labeller,
                     "benchmark": bm, "f1_mean": cell["f1_mean"], "f1_std": cell["f1_std"], "n": cell["n"],
                     "baseline_f1": round(bf1, 4) if bf1 is not None else "",
                     "f1_delta_vs_baseline": round(delta, 4) if delta != "" else ""})

with (OUT / "plot_delta_heatmap.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"Wrote {OUT/'plot_delta_heatmap.csv'} ({len(rows)} rows)")
