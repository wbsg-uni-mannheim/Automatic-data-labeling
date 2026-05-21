#!/usr/bin/env python3
"""Clean the 3 additional profiles needed to match the published-table choices.

Adds:
  seed_round    / walmart-amazon / all
  seed_round    / wdc            / medium
  simple_active / wdc            / large
"""
from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set

ROOT = Path("/work/aasteine/Automatic-data-labeling")

TEST_FILES: Dict[str, Path] = {
    "abt-buy":        ROOT / "data/abt-buy/abt-buy-gs.json.gz",
    "dblp-acm":       ROOT / "data/dblp-acm/dblp-acm-gs.json.gz",
    "dblp-scholar":   ROOT / "data/dblp-scholar/dblp-scholar-gs.json.gz",
    "walmart-amazon": ROOT / "data/walmart-amazon/walmart-amazon-gs.json.gz",
    "wdc":            ROOT / "data/wdc/wdcproducts80cc20rnd100un_gs.json.gz",
}

EXTRA = [
    ("seed_round",    "walmart-amazon", "all",    "output/seed_round_only_profiles/benchmark_walmart-amazon_20260415_190530"),
    ("seed_round",    "wdc",            "medium", "output/seed_round_only_profiles/benchmark_wdc_20260415_190530"),
    ("simple_active", "wdc",            "large",  "output/simple_active_learning_labeling/benchmark_wdc_20260413_152105"),
]


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


def clean_one(rows, test_pairs):
    survivors = []
    dt_pos = dt_neg = 0
    for r in rows:
        if extract_ids(r["pair_id"]) in test_pairs:
            if r["label"] == 1: dt_pos += 1
            else:               dt_neg += 1
            continue
        survivors.append(r)
    groups = defaultdict(list)
    for r in survivors:
        groups[extract_ids(r["pair_id"])].append(r)
    kept = []
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
        "dropped_test_pos": dt_pos, "dropped_test_neg": dt_neg,
        "dropped_test_total": dt_pos + dt_neg,
        "dup_collapsed_pos": dc_pos, "dup_collapsed_neg": dc_neg,
        "dup_disagreement_dropped_pos": dd_pos, "dup_disagreement_dropped_neg": dd_neg,
        "kept_rows": len(kept),
        "kept_pos": sum(1 for r in kept if r["label"] == 1),
        "kept_neg": sum(1 for r in kept if r["label"] == 0),
    }


def main():
    test_pairs = {bm: load_test_pairs(p) for bm, p in TEST_FILES.items()}
    for method, bm, profile, run_dir_str in EXTRA:
        run_dir = ROOT / run_dir_str
        src = run_dir / "profiles" / profile / f"active_labels_latest_{bm}_{profile}_train.json.gz"
        rows = load_train(src)
        kept, stats = clean_one(rows, test_pairs[bm])
        dst = ROOT / f"output/labeling_cleaned/{method}/{bm}/profiles/{profile}/active_labels_latest_{bm}_{profile}_train.json.gz"
        write_train(kept, dst)
        print(f"{method:>13} | {bm:<15} {profile:<8} src={stats['source_rows']:>6} "
              f"test={stats['dropped_test_total']:>4} ({stats['dropped_test_pos']}p/{stats['dropped_test_neg']}n) "
              f"dup_coll=({stats['dup_collapsed_pos']}+{stats['dup_collapsed_neg']}) "
              f"dup_disagr=({stats['dup_disagreement_dropped_pos']}+{stats['dup_disagreement_dropped_neg']}) "
              f"kept={stats['kept_rows']} ({stats['kept_pos']}p/{stats['kept_neg']}n)")


if __name__ == "__main__":
    main()
