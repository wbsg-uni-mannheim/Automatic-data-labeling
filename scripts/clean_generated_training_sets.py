#!/usr/bin/env python3
"""Drop test-set pair leaks and deduplicate generated training sets.

For each labeller (gpt-5.2, qwen3.6-plus, kimi-k2.6) and each benchmark
(abt-buy, dblp-acm, dblp-scholar, walmart-amazon, wdc) we use the
profile actually trained on (`all_plus20random`, except `all` for wdc).

Pair_id matching is order-invariant: a training row whose
{id_left, id_right} set matches any test pair is dropped. Duplicate
pairs (same id-set) within a single training set are also collapsed to
one row; if duplicates disagree on label the row is dropped entirely.
"""
from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

ROOT = Path("/work/aasteine/Automatic-data-labeling")

TEST_FILES: Dict[str, Path] = {
    "abt-buy":        ROOT / "data/abt-buy/abt-buy-gs.json.gz",
    "dblp-acm":       ROOT / "data/dblp-acm/dblp-acm-gs.json.gz",
    "dblp-scholar":   ROOT / "data/dblp-scholar/dblp-scholar-gs.json.gz",
    "walmart-amazon": ROOT / "data/walmart-amazon/walmart-amazon-gs.json.gz",
    "wdc":            ROOT / "data/wdc/wdcproducts80cc20rnd100un_gs.json.gz",
}

# (labeller_key, benchmark) -> source training file path
SOURCES: Dict[Tuple[str, str], Tuple[Path, str]] = {}

GPT_ROOT = ROOT / "output/three_phase_labeling_ditto_only_v2"
QWEN_ROOT = ROOT / "output/ditto_labeling/qwen"
KIMI_ROOT = ROOT / "output/ditto_labeling/kimi"

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
    "gpt":  (GPT_ROOT, GPT_RUNS),
    "qwen": (QWEN_ROOT, QWEN_RUNS),
    "kimi": (KIMI_ROOT, KIMI_RUNS),
}

# Profile used per benchmark (wdc has no random pool for any labeller)
PROFILE = {
    "abt-buy":        "all_plus20random",
    "dblp-acm":       "all_plus20random",
    "dblp-scholar":   "all_plus20random",
    "walmart-amazon": "all_plus20random",
    "wdc":            "all",
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
    # Training pair_id is "{left_id}__{right_id}__{suffix}".
    parts = pair_id.split("__")
    return frozenset([parts[0], parts[1]])


def load_train(path: Path) -> List[dict]:
    rows: List[dict] = []
    with gzip.open(path, "rt") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def write_train(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def clean_one(rows: List[dict], test_pairs: Set[frozenset]):
    """Returns (kept_rows, stats)."""
    dropped_test_pos = 0
    dropped_test_neg = 0
    dropped_test_rows: List[dict] = []

    # Pass 1: drop test leaks (track per label)
    survivors: List[dict] = []
    for r in rows:
        ids = extract_ids(r["pair_id"])
        if ids in test_pairs:
            if r["label"] == 1:
                dropped_test_pos += 1
            else:
                dropped_test_neg += 1
            dropped_test_rows.append(r)
            continue
        survivors.append(r)

    # Pass 2: dedup. Group by id-set, count label agreement.
    groups: Dict[frozenset, List[dict]] = defaultdict(list)
    for r in survivors:
        groups[extract_ids(r["pair_id"])].append(r)

    kept: List[dict] = []
    dup_collapsed_pos = 0  # extra rows removed when labels agreed (label == 1)
    dup_collapsed_neg = 0
    dup_dropped_pos = 0  # rows removed when labels disagreed
    dup_dropped_neg = 0
    for ids, group in groups.items():
        if len(group) == 1:
            kept.append(group[0])
            continue
        labels = {g["label"] for g in group}
        if len(labels) == 1:
            # Agreement: keep first, drop rest
            kept.append(group[0])
            removed = group[1:]
            for r in removed:
                if r["label"] == 1:
                    dup_collapsed_pos += 1
                else:
                    dup_collapsed_neg += 1
        else:
            # Disagreement: drop all
            for r in group:
                if r["label"] == 1:
                    dup_dropped_pos += 1
                else:
                    dup_dropped_neg += 1

    stats = {
        "source_rows": len(rows),
        "source_pos": sum(1 for r in rows if r["label"] == 1),
        "source_neg": sum(1 for r in rows if r["label"] == 0),
        "dropped_test_leak_pos": dropped_test_pos,
        "dropped_test_leak_neg": dropped_test_neg,
        "dropped_test_leak_total": dropped_test_pos + dropped_test_neg,
        "dup_collapsed_pos": dup_collapsed_pos,
        "dup_collapsed_neg": dup_collapsed_neg,
        "dup_disagreement_dropped_pos": dup_dropped_pos,
        "dup_disagreement_dropped_neg": dup_dropped_neg,
        "kept_rows": len(kept),
        "kept_pos": sum(1 for r in kept if r["label"] == 1),
        "kept_neg": sum(1 for r in kept if r["label"] == 0),
    }
    return kept, stats


def main() -> None:
    out_root = ROOT / "output/labeling_cleaned"

    # Preload test pair sets
    test_pairs = {bm: load_test_pairs(p) for bm, p in TEST_FILES.items()}

    report: Dict[str, dict] = {}
    for labeller_key, (root, runs) in LABELLER_ROOTS.items():
        report[labeller_key] = {}
        for bm, run_dir in runs.items():
            profile = PROFILE[bm]
            src = root / run_dir / "profiles" / profile / f"active_labels_latest_{bm}_{profile}_train.json.gz"
            if not src.exists():
                print(f"[WARN] missing: {src}")
                continue
            rows = load_train(src)
            kept, stats = clean_one(rows, test_pairs[bm])

            dst_dir = out_root / labeller_key / bm / "profiles" / profile
            dst = dst_dir / f"active_labels_latest_{bm}_{profile}_train.json.gz"
            write_train(kept, dst)

            report[labeller_key][bm] = {
                "profile": profile,
                "source": str(src),
                "cleaned": str(dst),
                **stats,
            }
            print(f"{labeller_key:>4} | {bm:<15} {profile:<18} src={stats['source_rows']:>6} "
                  f"drop_leak={stats['dropped_test_leak_total']:>4} "
                  f"(pos={stats['dropped_test_leak_pos']}, neg={stats['dropped_test_leak_neg']}) "
                  f"dup_collapsed=({stats['dup_collapsed_pos']}+{stats['dup_collapsed_neg']}) "
                  f"dup_disagree=({stats['dup_disagreement_dropped_pos']}+{stats['dup_disagreement_dropped_neg']}) "
                  f"kept={stats['kept_rows']} (pos={stats['kept_pos']}, neg={stats['kept_neg']})")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "cleanup_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote report to {out_root / 'cleanup_report.json'}")


if __name__ == "__main__":
    main()
