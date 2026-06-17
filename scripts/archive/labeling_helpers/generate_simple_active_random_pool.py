#!/usr/bin/env python3
"""Generate +1000 random-pool labels for the abt-buy simple_active_learning run.

The AL ML run (output/simple_active_learning_labeling/benchmark_abt-buy_1/) was
produced before random-profile sampling was wired in, so it has no
random_profile_labels.csv. The learning-curve plot needs to reach 6000 examples,
which is 1000 beyond the existing 5000 AL labels, so we draw a 1000-pair random
sample from the FAISS candidate pool (excluding pairs already labelled) and label
them with the same GPT-5.2 prompt the rest of the pipeline uses.

Output mirrors how run_benchmark_labeling.py writes random_profile_* files so the
chronological-subset builder can treat AL ML the same as the other methods.
"""
from __future__ import annotations
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path("/work/aasteine/Automatic-data-labeling")
sys.path.insert(0, str(ROOT / "scripts" / "labeling"))
from active_learning_ml import _label_pair, _make_openai_client  # type: ignore

RUN_DIR = ROOT / "output/simple_active_learning_labeling/benchmark_abt-buy_1"
FAISS_CANDIDATES = RUN_DIR / "faiss_candidates.csv"
EXISTING_LABELS = RUN_DIR / "active_labels_latest.csv"
LEFT_CANONICAL = RUN_DIR / "canonical/left.csv"
RIGHT_CANONICAL = RUN_DIR / "canonical/right.csv"
OUT_CANDIDATES = RUN_DIR / "random_profile_candidates.csv"
OUT_LABELS = RUN_DIR / "random_profile_labels.csv"
OUT_USAGE = RUN_DIR / "random_profile_usage.json"

MODEL = "gpt-5.2"
SAMPLE_N = 1000
SEED = 42
LLM_CONCURRENCY = 10


def main():
    load_dotenv(ROOT / ".env")
    cands = pd.read_csv(FAISS_CANDIDATES)
    labels = pd.read_csv(EXISTING_LABELS)
    labelled = set(zip(labels["id1"].astype(str), labels["id2"].astype(str)))
    keep = ~cands.apply(lambda r: (str(r["id1"]), str(r["id2"])) in labelled, axis=1)
    remaining = cands.loc[keep].reset_index(drop=True)
    print(f"candidates: {len(cands)} total, {len(remaining)} after dedupe-vs-labels")
    take_n = min(SAMPLE_N, len(remaining))
    sampled = remaining.sample(n=take_n, random_state=SEED).reset_index(drop=True)
    # Match the columns sim search uses (id1/id2 are source ids, rid1/rid2 are canonical rids).
    out_cands = pd.DataFrame({
        "id1": sampled["rid1"].astype(str),
        "id2": sampled["rid2"].astype(str),
        "similarity": sampled["similarity"].astype(float),
        "src_id1": sampled["id1"].astype(str),
        "src_id2": sampled["id2"].astype(str),
    })
    out_cands.to_csv(OUT_CANDIDATES, index=False)
    print(f"wrote {OUT_CANDIDATES} ({len(out_cands)} rows)")

    left_df = pd.read_csv(LEFT_CANONICAL)
    right_df = pd.read_csv(RIGHT_CANONICAL)
    left_map = {f"L:{i}": row.to_dict() for i, row in left_df.iterrows()}
    right_map = {f"R:{i}": row.to_dict() for i, row in right_df.iterrows()}

    client = _make_openai_client("", "OPENAI_API_KEY")
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    rows = []
    t0 = time.perf_counter()
    work = [
        {"rid1": str(r["id1"]), "rid2": str(r["id2"]), "src_id1": str(r["src_id1"]), "src_id2": str(r["src_id2"]), "similarity": float(r["similarity"])}
        for _, r in out_cands.iterrows()
    ]
    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as ex:
        fut_to_w = {ex.submit(_label_pair, client, MODEL, left_map[w["rid1"]], right_map[w["rid2"]]): w for w in work}
        done = 0
        for fut in as_completed(fut_to_w):
            w = fut_to_w[fut]
            try:
                label, usage = fut.result()
            except Exception as exc:
                print(f"label failed ({w['src_id1']},{w['src_id2']}): {exc!r}")
                continue
            for k in usage_total:
                usage_total[k] += int(usage.get(k, 0) or 0)
            rows.append({
                "id1": w["src_id1"],
                "id2": w["src_id2"],
                "label": label,
                "similarity": w["similarity"],
                "rid1": w["rid1"],
                "rid2": w["rid2"],
            })
            done += 1
            if done % 100 == 0 or done == len(work):
                print(f"labeled {done}/{len(work)} (tokens={usage_total['total_tokens']}, elapsed={time.perf_counter()-t0:.1f}s)")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_LABELS, index=False)
    OUT_USAGE.write_text(json.dumps(usage_total, indent=2))
    pos = int((out_df["label"].astype(str).str.upper() == "TRUE").sum())
    neg = int((out_df["label"].astype(str).str.upper() == "FALSE").sum())
    print(f"wrote {OUT_LABELS} ({len(out_df)} rows, {pos} pos, {neg} neg)")
    print(f"usage: {usage_total}")


if __name__ == "__main__":
    main()
