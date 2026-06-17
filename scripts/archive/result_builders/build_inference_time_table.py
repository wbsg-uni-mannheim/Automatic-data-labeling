#!/usr/bin/env python3
"""Build inference time comparison table:
  4 inference methods × 5 test sets

Methods:
  - Ditto (roberta-base, batch=64, single A40 GPU)
  - GPT-5.2 (OpenAI API, 10 parallel workers)
  - Qwen 3.6+ (OpenRouter, 10 parallel workers)
  - Kimi K2.6 (OpenRouter, 10 parallel workers)

Sources:
  - Ditto: output/results_summary/ditto_inference_times.json (measured)
  - LLMs: parsed from logs/label-test-sets-249682.out, with extrapolation
    for the abt-buy gpt-5.2 portion that ran locally before slurm.
"""
from __future__ import annotations
import re, json
from pathlib import Path
import pandas as pd

ROOT = Path("/work/aasteine/Automatic-data-labeling")
OUT = ROOT / "output/results_summary"

N_TEST = {"abt-buy":1916,"walmart-amazon":2049,"dblp-acm":2473,"dblp-scholar":5742,"wdc":4500}

# Parse slurm log
slurm_log = open(ROOT / "logs/label-test-sets-249682.out").read()
sections = re.split(r"=== (\S+) × (\S+) \((\S+)\) ===", slurm_log)
llm_data = {}
for i in range(1, len(sections), 4):
    bm, lab, model = sections[i], sections[i+1], sections[i+2]
    body = sections[i+3]
    matches = re.findall(r"(\d+)/(\d+) pairs, ([\d.]+)s, [\d.]+ pair/s, tokens=(\d+)", body)
    if matches:
        done, total, elapsed, tokens = matches[-1]
        llm_data[(bm, lab)] = {"done": int(done), "total": int(total), "elapsed_s": float(elapsed), "tokens": int(tokens)}

# Per-labeller average rate (for extrapolating missing portions)
rates = {"gpt-5.2":(180, 0.090), "qwen":(525,0.770), "kimi":(605,1.190)}

# Build LLM rows
llm_rows = []
for bm in N_TEST:
    for lab in rates:
        n = N_TEST[bm]
        d = llm_data.get((bm, lab), {})
        slurm_done = d.get("done", 0)
        slurm_time = d.get("elapsed_s", 0)
        slurm_tok = d.get("tokens", 0)
        missing = n - slurm_done
        tok_per_pair, sec_per_pair = rates[lab]
        if slurm_done == 0:
            total_time = n * sec_per_pair; total_tok = n * tok_per_pair
        elif missing > 0:
            total_time = slurm_time + missing * sec_per_pair
            total_tok = slurm_tok + missing * tok_per_pair
        else:
            total_time = slurm_time; total_tok = slurm_tok
        llm_rows.append({
            "benchmark": bm, "method": {"gpt-5.2":"GPT-5.2","qwen":"Qwen 3.6+","kimi":"Kimi K2.6"}[lab],
            "n_pairs": n, "time_s": round(total_time,1), "tokens_total": int(total_tok),
            "ms_per_pair": round(1000*total_time/n,1), "tokens_per_pair": round(total_tok/n,1),
        })

# Ditto inference times (measured)
ditto = json.load(open(OUT / "ditto_inference_times.json"))
ditto_rows = [{
    "benchmark": d["benchmark"], "method": "Ditto (roberta-base, A40)",
    "n_pairs": d["n_pairs"], "time_s": round(d["time_s"], 2),
    "tokens_total": None, "ms_per_pair": round(d["ms_per_pair"],2), "tokens_per_pair": None,
} for d in ditto]

all_rows = ditto_rows + llm_rows
df = pd.DataFrame(all_rows)
df.to_csv(OUT/"inference_time_comparison.csv", index=False)
print(f"Wrote {OUT/'inference_time_comparison.csv'}")
print()
# pivot: rows = benchmark, cols = method
piv_time = df.pivot_table(index="benchmark", columns="method", values="time_s", aggfunc="first")
piv_time = piv_time.reindex(["abt-buy","walmart-amazon","dblp-acm","dblp-scholar","wdc"])
print("=== Total inference time (sec) per benchmark ===")
print(piv_time.to_string())
print()

# Average ms/pair per method (across all 5 benchmarks)
print("=== Average time + tokens per 1k examples ===")
for method in ["Ditto (roberta-base, A40)","GPT-5.2","Qwen 3.6+","Kimi K2.6"]:
    sub = df[df["method"] == method]
    avg_ms = sub["ms_per_pair"].mean()
    avg_tok = sub["tokens_per_pair"].mean() if sub["tokens_per_pair"].notna().any() else None
    sec_per_1k = avg_ms
    line = f"  {method:35s} {sec_per_1k:>8.1f} sec/1k pairs ({sec_per_1k/60:>5.2f} min)"
    if avg_tok is not None:
        line += f"   tokens/1k: {int(avg_tok*1000):>9,d}"
    print(line)

# Token input/output split (estimated from AL Ditto walmart-amazon detailed data: 95.6% prompt / 4.4% completion)
print()
print("=== Tokens/1k examples (estimated input/output split) ===")
print("Note: input:output ratio ~95:5 (prompt is the records, output is short JSON match flag).")
for method in ["GPT-5.2","Qwen 3.6+","Kimi K2.6"]:
    sub = df[df["method"] == method]
    avg_tok = sub["tokens_per_pair"].mean()
    total_per_1k = int(avg_tok * 1000)
    input_per_1k = int(total_per_1k * 0.955)
    output_per_1k = total_per_1k - input_per_1k
    print(f"  {method:15s}: input ≈ {input_per_1k:>8,d}  output ≈ {output_per_1k:>6,d}  total {total_per_1k:>9,d}")
