#!/usr/bin/env python3
"""Relabel a Ditto train.json.gz file in-place. Each pair is re-labelled with
gpt-5-mini using the agent_precision prompt, and the per-pair original/predicted
labels are recorded so downstream postfilter variants (relabel-drop, closure
combinations) can be built.

Inputs:
  --train-json-gz   path to existing train.json.gz (with pair_id, label, fields)
  --out-dir         where to write outputs
  --prompt-fields   comma-separated fields used in the prompt (e.g. title,description,price)

Outputs (in --out-dir):
  train__relabeled.json.gz       — relabeled version (all pairs kept, labels updated)
  relabel_diff.csv               — per-pair: pair_id, original_label, predicted_label, changed
  relabel_summary.json
  _in_progress.jsonl             — resume checkpoint
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "labeling"))
from active_learning_ml import _label_pair, _make_openai_client  # type: ignore

DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_PROMPT_FILE = (
    ROOT / "scripts" / "archive" / "review_workflows" / "experiments" / "evidence_first_abstain" /
    "prompts" / "agent_precision_system_prompt.txt"
)


def load_jsonl_gz(p):
    rows = []
    with gzip.open(p, "rt") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def split_left_right(row, prompt_fields):
    left, right = {}, {}
    for f in prompt_fields:
        if f"{f}_left" in row:
            left[f] = row[f"{f}_left"]
        if f"{f}_right" in row:
            right[f] = row[f"{f}_right"]
    return left, right


def parse_label(s):
    s = str(s).strip().upper()
    if s == "TRUE":  return 1
    if s == "FALSE": return 0
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-json-gz", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prompt-fields", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--api-base-url", default="")  # default OpenAI
    ap.add_argument("--api-key-env-var", default="OPENAI_API_KEY")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    src = Path(args.train_json_gz)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_fields = [f.strip() for f in args.prompt_fields.split(",") if f.strip()]

    rows = load_jsonl_gz(src)
    print(f"src: {src.relative_to(ROOT)} ({len(rows)} pairs)")
    print(f"out: {out_dir.relative_to(ROOT)}")

    progress_path = out_dir / "_in_progress.jsonl"
    done = {}
    if progress_path.exists():
        with progress_path.open("r") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    done[d["pair_id"]] = d
        print(f"resume: {len(done)} pairs already labelled")

    pending = [r for r in rows if r.get("pair_id") not in done]
    if pending:
        client = _make_openai_client(args.api_base_url, args.api_key_env_var)
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as ex, progress_path.open("a") as flog:
            futs = {}
            for r in pending:
                left, right = split_left_right(r, prompt_fields)
                futs[ex.submit(_label_pair, client, args.model, left, right)] = r
            count = 0
            for fut in as_completed(futs):
                r = futs[fut]
                try:
                    label_str, u = fut.result()
                except Exception as exc:
                    label_str, u = None, {}
                    print(f"  fail {r.get('pair_id')}: {exc!r}")
                for k in usage:
                    usage[k] += int(u.get(k, 0) or 0)
                pred = parse_label(label_str) if label_str is not None else None
                entry = {
                    "pair_id": r.get("pair_id"),
                    "original_label": int(r.get("label", 0)),
                    "predicted_label": pred,
                    "label_raw": str(label_str) if label_str is not None else "",
                }
                done[r.get("pair_id")] = entry
                flog.write(json.dumps(entry) + "\n")
                flog.flush()
                count += 1
                if count % 200 == 0 or count == len(pending):
                    elapsed = time.perf_counter() - t0
                    print(f"  {count}/{len(pending)} pairs, {elapsed:.1f}s, "
                          f"rate={count/max(elapsed,1):.1f}/s, tokens={usage['total_tokens']}")

    # Write outputs
    diff_rows = []
    relabeled_rows = []
    for r in rows:
        pid = r.get("pair_id")
        d = done.get(pid, {})
        pred = d.get("predicted_label")
        orig = int(r.get("label", 0))
        new_label = pred if pred is not None else orig  # parse-fail keeps original
        diff_rows.append({
            "pair_id": pid,
            "original_label": orig,
            "predicted_label": pred,
            "changed": int(pred is not None and pred != orig),
        })
        new_r = dict(r)
        new_r["label"] = int(new_label)
        relabeled_rows.append(new_r)

    relabeled_path = out_dir / "train__relabeled.json.gz"
    with gzip.open(relabeled_path, "wt") as f:
        for r in relabeled_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    diff_path = out_dir / "relabel_diff.csv"
    pd.DataFrame(diff_rows).to_csv(diff_path, index=False)

    n_changed = sum(d["changed"] for d in diff_rows)
    n_parse_fail = sum(1 for d in diff_rows if d["predicted_label"] is None)
    summary = {
        "src": str(src),
        "n_pairs": len(rows),
        "n_changed": int(n_changed),
        "n_parse_fail": int(n_parse_fail),
        "n_kept_original_due_to_fail": int(n_parse_fail),
        "changed_to_match": int(sum(1 for d in diff_rows if d["predicted_label"] == 1 and d["original_label"] == 0)),
        "changed_to_non_match": int(sum(1 for d in diff_rows if d["predicted_label"] == 0 and d["original_label"] == 1)),
        "model": args.model,
        "prompt_fields": prompt_fields,
        "relabeled_json_gz": str(relabeled_path),
        "diff_csv": str(diff_path),
    }
    (out_dir / "relabel_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {relabeled_path.relative_to(ROOT)}")
    print(f"wrote {diff_path.relative_to(ROOT)}")
    print(f"changed={n_changed}/{len(rows)} ({n_changed/len(rows)*100:.1f}%), parse_fail={n_parse_fail}")


if __name__ == "__main__":
    main()
