#!/usr/bin/env python3
"""Drop test-set leaks + dedup for seed_round_only and simple_active training sets.

Same matching/dedup logic as scripts/clean_generated_training_sets.py.
Output goes to:
  output/labeling_cleaned/seed_round/<bm>/profiles/<profile>/...
  output/labeling_cleaned/simple_active/<bm>/profiles/<profile>/...
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

# (method_key, benchmark) -> (run_subdir, profile_to_use)
SOURCES = {
    "seed_round": {
        "abt-buy":        ("benchmark_abt-buy_20260415_190530",        "all_plus20random"),
        "dblp-acm":       ("benchmark_dblp-acm_20260415_190530",       "all_plus20random"),
        "dblp-scholar":   ("benchmark_dblp-scholar_20260415_190530",   "all_plus20random"),
        "walmart-amazon": ("benchmark_walmart-amazon_20260415_190530", "all_plus20random"),
        "wdc":            ("benchmark_wdc_20260415_190530",            "all_plus20random"),
    },
    "simple_active": {
        "abt-buy":        ("benchmark_abt-buy_1",                       "large"),  # no `all`/`_plus20random`
        "dblp-acm":       ("benchmark_dblp-acm_20260302_113733",        "all"),
        "dblp-scholar":   ("benchmark_dblp-scholar_20260302_113733",    "all"),
        "walmart-amazon": ("benchmark_walmart-amazon_20260302_113733",  "all"),
        "wdc":            ("benchmark_wdc_20260413_152105",             "all"),
    },
}

METHOD_ROOT = {
    "seed_round":    ROOT / "output/seed_round_only_profiles",
    "simple_active": ROOT / "output/simple_active_learning_labeling",
}


def load_test_pairs(path: Path) -> Set[frozenset]:
    pairs: Set[frozenset] = set()
    with gzip.open(path, "rt") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
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
        "source_pos": sum(1 for r in rows if r["label"] == 1),
        "source_neg": sum(1 for r in rows if r["label"] == 0),
        "dropped_test_pos": dt_pos, "dropped_test_neg": dt_neg,
        "dropped_test_total": dt_pos + dt_neg,
        "dup_collapsed_pos": dc_pos, "dup_collapsed_neg": dc_neg,
        "dup_disagreement_dropped_pos": dd_pos, "dup_disagreement_dropped_neg": dd_neg,
        "kept_rows": len(kept),
        "kept_pos": sum(1 for r in kept if r["label"] == 1),
        "kept_neg": sum(1 for r in kept if r["label"] == 0),
    }


def main() -> None:
    test_pairs = {bm: load_test_pairs(p) for bm, p in TEST_FILES.items()}

    for method_key, per_bm in SOURCES.items():
        report = {}
        method_dst = ROOT / f"output/labeling_cleaned/{method_key}"
        for bm, (run_subdir, profile) in per_bm.items():
            src = METHOD_ROOT[method_key] / run_subdir / "profiles" / profile / f"active_labels_latest_{bm}_{profile}_train.json.gz"
            if not src.exists():
                print(f"[WARN] missing: {src}")
                continue
            rows = load_train(src)
            kept, stats = clean_one(rows, test_pairs[bm])
            dst = method_dst / bm / "profiles" / profile / f"active_labels_latest_{bm}_{profile}_train.json.gz"
            write_train(kept, dst)
            report[bm] = {"profile": profile, "source": str(src), "cleaned": str(dst), **stats}
            print(f"{method_key:>13} | {bm:<15} {profile:<18} src={stats['source_rows']:>6} "
                  f"test={stats['dropped_test_total']:>4} ({stats['dropped_test_pos']}p/{stats['dropped_test_neg']}n) "
                  f"dup_collapsed=({stats['dup_collapsed_pos']}+{stats['dup_collapsed_neg']}) "
                  f"dup_disagree=({stats['dup_disagreement_dropped_pos']}+{stats['dup_disagreement_dropped_neg']}) "
                  f"kept={stats['kept_rows']} ({stats['kept_pos']}p/{stats['kept_neg']}n)")
        method_dst.mkdir(parents=True, exist_ok=True)
        (method_dst / "cleanup_report.json").write_text(json.dumps(report, indent=2))
        print(f"  wrote {method_dst / 'cleanup_report.json'}\n")


if __name__ == "__main__":
    main()
