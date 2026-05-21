#!/usr/bin/env python3
"""Like clean_generated_training_sets.py but also strip validation-set pairs.

Drops, in order:
  1. rows whose {id_left, id_right} matches a *test* pair
  2. rows whose {id_left, id_right} matches a *validation* pair
  3. label-disagreement duplicates (both rows removed)
  4. label-agreement duplicates (collapsed to one)

Pair_id matching is order-invariant: a training row whose
{id_left, id_right} set matches is dropped.
"""
from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd

ROOT = Path("/work/aasteine/Automatic-data-labeling")

TEST_FILES: Dict[str, Path] = {
    "abt-buy":        ROOT / "data/abt-buy/abt-buy-gs.json.gz",
    "dblp-acm":       ROOT / "data/dblp-acm/dblp-acm-gs.json.gz",
    "dblp-scholar":   ROOT / "data/dblp-scholar/dblp-scholar-gs.json.gz",
    "walmart-amazon": ROOT / "data/walmart-amazon/walmart-amazon-gs.json.gz",
    "wdc":            ROOT / "data/wdc/wdcproducts80cc20rnd100un_gs.json.gz",
}

VALID_FILES: Dict[str, Path] = {
    "abt-buy":        ROOT / "data/abt-buy/abt-buy-valid.csv",          # pair_id only
    "dblp-acm":       ROOT / "data/dblp-acm/dblp-acm-valid.csv",
    "dblp-scholar":   ROOT / "data/dblp-scholar/dblp-scholar-valid.csv",
    "walmart-amazon": ROOT / "data/walmart-amazon/walmart-amazon-valid.csv",
    "wdc":            ROOT / "data/wdc/wdcproducts80cc20rnd000un_valid_medium.json.gz",
}

GPT_RUNS = {
    "abt-buy":        "benchmark_abt-buy_20260323_202820",
    "dblp-acm":       "benchmark_dblp-acm_20260323_202820",
    "dblp-scholar":   "benchmark_dblp-scholar_20260323_202820",
    "walmart-amazon": "benchmark_walmart-amazon_20260323_202820",
    "wdc":            "benchmark_wdc_20260323_202820",
}
QWEN_RUNS = {
    "abt-buy":        "benchmark_abt-buy_20260429_173808",
    "dblp-acm":       "benchmark_dblp-acm_20260430_183630",
    "dblp-scholar":   "benchmark_dblp-scholar_20260430_183630",
    "walmart-amazon": "benchmark_walmart-amazon_20260430_183630",
    "wdc":            "benchmark_wdc_20260430_183630",
}
KIMI_RUNS = {
    "abt-buy":        "benchmark_abt-buy_20260504_181847",
    "dblp-acm":       "benchmark_dblp-acm_20260506_125546",
    "dblp-scholar":   "benchmark_dblp-scholar_20260506_125546",
    "walmart-amazon": "benchmark_walmart-amazon_20260506_125546",
    "wdc":            "benchmark_wdc_20260506_125546",
}

LABELLER_ROOTS = {
    "gpt":  (ROOT / "output/three_phase_labeling_ditto_only_v2", GPT_RUNS),
    "qwen": (ROOT / "output/ditto_labeling/qwen",                QWEN_RUNS),
    "kimi": (ROOT / "output/ditto_labeling/kimi",                KIMI_RUNS),
}

PROFILE = {
    "abt-buy":        "all_plus20random",
    "dblp-acm":       "all_plus20random",
    "dblp-scholar":   "all_plus20random",
    "walmart-amazon": "all_plus20random",
    "wdc":            "all",
}


def load_pairs_from_gs(path: Path) -> Set[frozenset]:
    pairs: Set[frozenset] = set()
    with gzip.open(path, "rt") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            r = json.loads(s)
            pairs.add(frozenset([str(r["id_left"]), str(r["id_right"])]))
    return pairs


def load_pairs_from_valid(path: Path) -> Set[frozenset]:
    """Validation files come in three shapes:
      - .json.gz with id_left/id_right (wdc)
      - .csv with id_left/id_right (dblp-*, walmart-amazon)
      - .csv with pair_id column only (abt-buy) — parse `{left}#{right}`
    """
    pairs: Set[frozenset] = set()
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                r = json.loads(s)
                pairs.add(frozenset([str(r["id_left"]), str(r["id_right"])]))
        return pairs
    df = pd.read_csv(path)
    if "id_left" in df.columns and "id_right" in df.columns:
        for l, r in zip(df["id_left"], df["id_right"]):
            pairs.add(frozenset([str(l), str(r)]))
    elif "pair_id" in df.columns:
        for pid in df["pair_id"]:
            l, r = str(pid).split("#", 1)
            pairs.add(frozenset([l, r]))
    else:
        raise ValueError(f"Cannot parse pair ids from {path}")
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


def clean_one(rows: List[dict], test_pairs: Set[frozenset], valid_pairs: Set[frozenset]):
    survivors: List[dict] = []
    test_pos = test_neg = val_pos = val_neg = 0
    for r in rows:
        ids = extract_ids(r["pair_id"])
        if ids in test_pairs:
            if r["label"] == 1: test_pos += 1
            else:               test_neg += 1
            continue
        if ids in valid_pairs:
            if r["label"] == 1: val_pos += 1
            else:               val_neg += 1
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
        "dropped_test_pos": test_pos, "dropped_test_neg": test_neg,
        "dropped_test_total": test_pos + test_neg,
        "dropped_valid_pos": val_pos, "dropped_valid_neg": val_neg,
        "dropped_valid_total": val_pos + val_neg,
        "dup_collapsed_pos": dc_pos, "dup_collapsed_neg": dc_neg,
        "dup_disagreement_dropped_pos": dd_pos, "dup_disagreement_dropped_neg": dd_neg,
        "kept_rows": len(kept),
        "kept_pos": sum(1 for r in kept if r["label"] == 1),
        "kept_neg": sum(1 for r in kept if r["label"] == 0),
    }


def main() -> None:
    out_root = ROOT / "output/labeling_cleaned_no_val"
    test_pairs = {bm: load_pairs_from_gs(p) for bm, p in TEST_FILES.items()}
    valid_pairs = {bm: load_pairs_from_valid(p) for bm, p in VALID_FILES.items()}

    report: Dict[str, dict] = {}
    for labeller, (root, runs) in LABELLER_ROOTS.items():
        report[labeller] = {}
        for bm, run_dir in runs.items():
            profile = PROFILE[bm]
            src = root / run_dir / "profiles" / profile / f"active_labels_latest_{bm}_{profile}_train.json.gz"
            if not src.exists():
                print(f"[WARN] missing: {src}"); continue
            rows = load_train(src)
            kept, stats = clean_one(rows, test_pairs[bm], valid_pairs[bm])

            dst = out_root / labeller / bm / "profiles" / profile / f"active_labels_latest_{bm}_{profile}_train.json.gz"
            write_train(kept, dst)

            report[labeller][bm] = {"profile": profile, "source": str(src), "cleaned": str(dst), **stats}
            print(
                f"{labeller:>4} | {bm:<15} {profile:<18} src={stats['source_rows']:>6} "
                f"test={stats['dropped_test_total']:>4} ({stats['dropped_test_pos']}p/{stats['dropped_test_neg']}n) "
                f"valid={stats['dropped_valid_total']:>4} ({stats['dropped_valid_pos']}p/{stats['dropped_valid_neg']}n) "
                f"dup_collapsed=({stats['dup_collapsed_pos']}+{stats['dup_collapsed_neg']}) "
                f"dup_disagree=({stats['dup_disagreement_dropped_pos']}+{stats['dup_disagreement_dropped_neg']}) "
                f"kept={stats['kept_rows']} ({stats['kept_pos']}p/{stats['kept_neg']}n)"
            )

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "cleanup_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote report to {out_root / 'cleanup_report.json'}")


if __name__ == "__main__":
    main()
