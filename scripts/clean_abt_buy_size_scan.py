#!/usr/bin/env python3
"""Clean abt-buy training sets across all profiles for GPT/Qwen/Kimi.

Mirrors clean_generated_training_sets.py but iterates over every available
profile (small/medium/large/all and their _plus20random variants) instead of
just the canonical one.

Output: output/labeling_cleaned_size_scan/<labeller>/abt-buy/profiles/<profile>/
"""
from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set

ROOT = Path("/work/aasteine/Automatic-data-labeling")
TEST_FILE = ROOT / "data/abt-buy/abt-buy-gs.json.gz"

LABELLER_RUNS = {
    "gpt":  ROOT / "output/three_phase_labeling_ditto_only_v2/benchmark_abt-buy_20260323_202820",
    "qwen": ROOT / "output/ditto_labeling/qwen/benchmark_abt-buy_20260429_173808",
    "kimi": ROOT / "output/ditto_labeling/kimi/benchmark_abt-buy_20260504_181847",
}

PROFILES = ["small", "small_plus20random", "medium", "medium_plus20random",
            "large", "large_plus20random", "all", "all_plus20random"]

OUT_ROOT = ROOT / "output/labeling_cleaned_size_scan"


def load_test_pairs(path: Path) -> Set[frozenset]:
    pairs = set()
    with gzip.open(path, "rt") as f:
        for line in f:
            s = line.strip()
            if s:
                r = json.loads(s)
                pairs.add(frozenset([str(r["id_left"]), str(r["id_right"])]))
    return pairs


def extract_ids(pair_id: str) -> frozenset:
    parts = pair_id.split("__")
    return frozenset([parts[0], parts[1]])


def load_train(path: Path) -> List[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_train(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def clean_one(rows: List[dict], test_pairs: Set[frozenset]):
    survivors: List[dict] = []
    dt_pos = dt_neg = 0
    for r in rows:
        if extract_ids(r["pair_id"]) in test_pairs:
            if r["label"] == 1: dt_pos += 1
            else:               dt_neg += 1
            continue
        survivors.append(r)
    groups: Dict[frozenset, List[dict]] = defaultdict(list)
    for r in survivors:
        groups[extract_ids(r["pair_id"])].append(r)
    kept: List[dict] = []
    dc_pos = dc_neg = dd_pos = dd_neg = 0
    for ids, group in groups.items():
        if len(group) == 1:
            kept.append(group[0]); continue
        labels = {g["label"] for g in group}
        if len(labels) == 1:
            kept.append(group[0])
            for r in group[1:]:
                if r["label"] == 1: dc_pos += 1
                else:               dc_neg += 1
        else:
            for r in group:
                if r["label"] == 1: dd_pos += 1
                else:               dd_neg += 1
    return kept, {
        "source_rows": len(rows),
        "dropped_test_total": dt_pos + dt_neg,
        "dropped_test_pos": dt_pos, "dropped_test_neg": dt_neg,
        "dup_collapsed_pos": dc_pos, "dup_collapsed_neg": dc_neg,
        "dup_disagreement_dropped_pos": dd_pos, "dup_disagreement_dropped_neg": dd_neg,
        "kept_rows": len(kept),
        "kept_pos": sum(1 for r in kept if r["label"] == 1),
        "kept_neg": sum(1 for r in kept if r["label"] == 0),
    }


def main() -> None:
    test_pairs = load_test_pairs(TEST_FILE)
    report: Dict[str, Dict[str, dict]] = {}
    for labeller, run_dir in LABELLER_RUNS.items():
        report[labeller] = {}
        for profile in PROFILES:
            src = run_dir / "profiles" / profile / f"active_labels_latest_abt-buy_{profile}_train.json.gz"
            if not src.exists():
                print(f"[skip] {labeller}/{profile}: source missing")
                continue
            rows = load_train(src)
            kept, stats = clean_one(rows, test_pairs)
            dst = OUT_ROOT / labeller / "abt-buy" / "profiles" / profile / f"active_labels_latest_abt-buy_{profile}_train.json.gz"
            write_train(kept, dst)
            report[labeller][profile] = {"source": str(src), "cleaned": str(dst), **stats}
            print(f"{labeller:>4} | {profile:<22} src={stats['source_rows']:>6} "
                  f"test={stats['dropped_test_total']:>4} ({stats['dropped_test_pos']}p/{stats['dropped_test_neg']}n) "
                  f"kept={stats['kept_rows']:>5} ({stats['kept_pos']}p/{stats['kept_neg']}n)")

    (OUT_ROOT / "cleanup_report_abt-buy.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote {OUT_ROOT / 'cleanup_report_abt-buy.json'}")


if __name__ == "__main__":
    main()
