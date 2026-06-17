#!/usr/bin/env python3
"""Label each benchmark test set with GPT-5.2, Qwen 3.6+ and Kimi K2.6
using the same prompt as the AL labelling pipeline (run_simple_labeling._label_pair).

For each (benchmark × labeller):
  - Load <benchmark>-gs.json.gz (test split, fully-joined pairs with gold labels)
  - Parallel-label each pair with the configured model via OpenAI/OpenRouter
  - Write output/test_set_predictions/<benchmark>/<labeller>/predictions.csv +
    summary.json (precision/recall/f1 vs gold)

CLI:
  python scripts/labeling/label_test_sets_with_llms.py \
    --benchmarks abt-buy,walmart-amazon,dblp-acm,dblp-scholar,wdc \
    --labellers gpt-5.2,qwen,kimi \
    --workers 10
"""
from __future__ import annotations
import argparse
import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path("/work/aasteine/Automatic-data-labeling")
sys.path.insert(0, str(ROOT / "scripts" / "labeling"))
from run_simple_labeling import _label_pair, _make_openai_client  # type: ignore

OUT_ROOT = ROOT / "output/test_set_predictions"

# Labeller configs: model, api_base_url, api_key_env_var
LABELLERS = {
    "gpt-5.2": ("gpt-5.2", "", "OPENAI_API_KEY"),
    "qwen":    ("qwen/qwen3.6-plus",   "https://openrouter.ai/api/v1", "OPEN_ROUTER_API_KEY"),
    "kimi":    ("moonshotai/kimi-k2.6", "https://openrouter.ai/api/v1", "OPEN_ROUTER_API_KEY"),
}
# Test file path per benchmark
TEST_FILES = {
    "abt-buy":        ROOT / "data/abt-buy/abt-buy-gs.json.gz",
    "walmart-amazon": ROOT / "data/walmart-amazon/walmart-amazon-gs.json.gz",
    "dblp-acm":       ROOT / "data/dblp-acm/dblp-acm-gs.json.gz",
    "dblp-scholar":   ROOT / "data/dblp-scholar/dblp-scholar-gs.json.gz",
    "wdc":            ROOT / "data/wdc/wdcproducts80cc20rnd100un_gs.json.gz",
}


def load_test(path: Path):
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def split_left_right(row):
    left, right = {}, {}
    for k, v in row.items():
        if k.endswith("_left"):
            left[k[:-5]] = v
        elif k.endswith("_right"):
            right[k[:-6]] = v
    return left, right


def parse_pred(label):
    """Map _label_pair output to int 0/1."""
    s = str(label).strip().upper()
    if s == "TRUE":
        return 1
    if s == "FALSE":
        return 0
    return None


def run_one(benchmark, labeller_key, workers, resume):
    model, base_url, key_env = LABELLERS[labeller_key]
    test_path = TEST_FILES[benchmark]
    out_dir = OUT_ROOT / benchmark / labeller_key
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = out_dir / "predictions.csv"
    summary_json = out_dir / "summary.json"
    progress_jsonl = out_dir / "_in_progress.jsonl"  # for resume

    rows = load_test(test_path)
    print(f"\n=== {benchmark} × {labeller_key} ({model}) ===")
    print(f"test rows: {len(rows)}, output: {pred_csv.relative_to(ROOT)}")

    # resume logic: load already-labelled pair_ids
    done_map = {}
    if resume and progress_jsonl.exists():
        with progress_jsonl.open("r") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    done_map[d["pair_id"]] = d
        print(f"resuming: {len(done_map)} pairs already labelled")

    pending = [r for r in rows if r.get("pair_id") not in done_map]
    if not pending:
        print("all pairs already labelled, skipping")
    else:
        client = _make_openai_client(base_url, key_env)
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as ex, progress_jsonl.open("a") as flog:
            futs = {}
            for r in pending:
                left, right = split_left_right(r)
                futs[ex.submit(_label_pair, client, model, left, right)] = r
            done = 0
            for fut in as_completed(futs):
                r = futs[fut]
                try:
                    label, u = fut.result()
                except Exception as exc:
                    label, u = None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    print(f"  fail {r.get('pair_id')}: {exc!r}")
                for k in usage:
                    usage[k] += int(u.get(k, 0) or 0)
                pred = parse_pred(label) if label is not None else None
                entry = {
                    "pair_id": r.get("pair_id"),
                    "gold": int(r.get("label", 0)),
                    "pred": pred,
                    "label_raw": str(label) if label is not None else "",
                }
                done_map[r.get("pair_id")] = entry
                flog.write(json.dumps(entry) + "\n")
                flog.flush()
                done += 1
                if done % 200 == 0 or done == len(pending):
                    elapsed = time.perf_counter() - t0
                    rate = done / max(elapsed, 1)
                    print(f"  {done}/{len(pending)} pairs, {elapsed:.1f}s, {rate:.1f} pair/s, tokens={usage['total_tokens']}")

    # Build final predictions.csv (in original test row order)
    out_rows = []
    for r in rows:
        pid = r.get("pair_id")
        entry = done_map.get(pid)
        out_rows.append({
            "pair_id": pid,
            "gold": int(r.get("label", 0)),
            "pred": entry["pred"] if entry else None,
            "label_raw": entry["label_raw"] if entry else "",
        })
    df = pd.DataFrame(out_rows)
    df.to_csv(pred_csv, index=False)

    # Metrics
    valid = df[df["pred"].notna()]
    n_valid = len(valid)
    n_parse_fail = len(df) - n_valid
    if n_valid > 0:
        tp = int(((valid["pred"] == 1) & (valid["gold"] == 1)).sum())
        fp = int(((valid["pred"] == 1) & (valid["gold"] == 0)).sum())
        fn = int(((valid["pred"] == 0) & (valid["gold"] == 1)).sum())
        tn = int(((valid["pred"] == 0) & (valid["gold"] == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / n_valid
    else:
        tp = fp = fn = tn = 0
        prec = rec = f1 = acc = 0.0
    summary = {
        "benchmark": benchmark, "labeller": labeller_key, "model": model,
        "n_test_pairs": len(df), "n_predictions": n_valid, "n_parse_fail": n_parse_fail,
        "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
    summary_json.write_text(json.dumps(summary, indent=2))
    print(f"  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}  ({tp}tp/{fp}fp/{fn}fn/{tn}tn, parse_fail={n_parse_fail})")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", default="abt-buy,walmart-amazon,dblp-acm,dblp-scholar,wdc")
    parser.add_argument("--labellers", default="gpt-5.2,qwen,kimi")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    labellers = [l.strip() for l in args.labellers.split(",") if l.strip()]

    summaries = []
    for bm in benchmarks:
        for lab in labellers:
            try:
                s = run_one(bm, lab, args.workers, args.resume)
                summaries.append(s)
            except Exception as e:
                print(f"FAILED {bm} × {lab}: {e!r}")

    # write aggregated summary
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    agg_path = OUT_ROOT / "_summary.csv"
    pd.DataFrame(summaries).to_csv(agg_path, index=False)
    print(f"\nAggregate summary -> {agg_path.relative_to(ROOT)}")
    print(pd.DataFrame(summaries)[["benchmark", "labeller", "f1", "precision", "recall", "n_parse_fail"]].to_string(index=False))


if __name__ == "__main__":
    main()
