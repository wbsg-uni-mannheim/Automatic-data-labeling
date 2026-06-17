#!/usr/bin/env python3
"""Extend walmart-amazon similarity-search random pool by ~600 more random
labels so the total label count lands inside the ±5% band of the benchmark
train-only size (6,144 pairs).
"""
from __future__ import annotations
import json, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path("/work/aasteine/Automatic-data-labeling")
sys.path.insert(0, str(ROOT / "scripts" / "labeling"))
from run_simple_labeling import _label_pair, _make_openai_client  # type: ignore

RUN_DIR = ROOT / "output/seed_round_only_profiles/benchmark_walmart-amazon_20260415_190530"
FAISS_CANDIDATES = RUN_DIR / "faiss_candidates.csv"
ACTIVE = RUN_DIR / "active_labels_latest.csv"
EXISTING_RAND_LABELS = RUN_DIR / "random_profile_labels.csv"
LEFT_CANON = RUN_DIR / "canonical/left.csv"
RIGHT_CANON = RUN_DIR / "canonical/right.csv"

MODEL = "gpt-5.2"
EXTRA_N = 600
SEED = 142
LLM_CONCURRENCY = 10


def main():
    load_dotenv(ROOT / ".env")
    cands = pd.read_csv(FAISS_CANDIDATES)
    al = pd.read_csv(ACTIVE)
    rand = pd.read_csv(EXISTING_RAND_LABELS)
    already = set(zip(al["id1"].astype(str), al["id2"].astype(str))) | set(zip(rand["id1"].astype(str), rand["id2"].astype(str)))
    keep = ~cands.apply(lambda r: (str(r["id1"]), str(r["id2"])) in already, axis=1)
    remaining = cands.loc[keep].reset_index(drop=True)
    sampled = remaining.sample(n=min(EXTRA_N, len(remaining)), random_state=SEED).reset_index(drop=True)
    print(f"sampling {len(sampled)} new pairs (cands left after dedupe: {len(remaining)})")

    left_df = pd.read_csv(LEFT_CANON)
    right_df = pd.read_csv(RIGHT_CANON)
    left_map = {f"left:{i}": row.to_dict() for i, row in left_df.iterrows()}
    right_map = {f"right:{i}": row.to_dict() for i, row in right_df.iterrows()}

    work = [{
        "src_id1": str(r["id1"]), "src_id2": str(r["id2"]),
        "rid1": str(r["rid1"]), "rid2": str(r["rid2"]),
        "similarity": float(r["similarity"]),
    } for _, r in sampled.iterrows()]

    client = _make_openai_client("", "OPENAI_API_KEY")
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    rows = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as ex:
        fut_to_w = {ex.submit(_label_pair, client, MODEL, left_map[w["rid1"]], right_map[w["rid2"]]): w for w in work}
        done = 0
        for fut in as_completed(fut_to_w):
            w = fut_to_w[fut]
            try:
                label, u = fut.result()
            except Exception as exc:
                print(f"fail ({w['src_id1']},{w['src_id2']}): {exc!r}"); continue
            for k in usage: usage[k] += int(u.get(k, 0) or 0)
            rows.append({"id1": w["src_id1"], "id2": w["src_id2"], "label": label,
                         "similarity": w["similarity"], "rid1": w["rid1"], "rid2": w["rid2"]})
            done += 1
            if done % 100 == 0 or done == len(work):
                print(f"labeled {done}/{len(work)} tokens={usage['total_tokens']} elapsed={time.perf_counter()-t0:.1f}s")

    new_df = pd.DataFrame(rows)
    combined = pd.concat([rand, new_df], ignore_index=True).drop_duplicates(subset=["id1", "id2"], keep="first").reset_index(drop=True)
    combined.to_csv(EXISTING_RAND_LABELS, index=False)
    print(f"random_profile_labels.csv: {len(rand)} -> {len(combined)} (added {len(new_df)})")


if __name__ == "__main__":
    main()
