#!/usr/bin/env python3
"""Clean abt-buy training sets for similarity_selection and simple_active
across all available profiles (size-scan)."""
from __future__ import annotations
import gzip, json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/work/aasteine/Automatic-data-labeling")
TEST_FILE = ROOT / "data/abt-buy/abt-buy-gs.json.gz"

METHOD_TO_RUN = {
    "seed_round":    ROOT / "output/seed_round_only_profiles/benchmark_abt-buy_20260415_190530",
    "simple_active": ROOT / "output/simple_active_learning_labeling/benchmark_abt-buy_1",
}
METHOD_TO_PROFILES = {
    "seed_round":    ["small","small_plus20random","medium","medium_plus20random",
                      "large","large_plus20random","all","all_plus20random"],
    "simple_active": ["small","medium","large"],
}

def load_test_pairs(path):
    pairs = set()
    with gzip.open(path, "rt") as f:
        for line in f:
            s = line.strip()
            if s:
                r = json.loads(s)
                pairs.add(frozenset([str(r["id_left"]), str(r["id_right"])]))
    return pairs

def extract_ids(pid):
    parts = pid.split("__")
    return frozenset([parts[0], parts[1]])

def load_train(p):
    with gzip.open(p, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]

def write_train(rows, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def clean_one(rows, test_pairs):
    survivors = []
    dt_p = dt_n = 0
    for r in rows:
        if extract_ids(r["pair_id"]) in test_pairs:
            if r["label"] == 1: dt_p += 1
            else: dt_n += 1
            continue
        survivors.append(r)
    groups = defaultdict(list)
    for r in survivors:
        groups[extract_ids(r["pair_id"])].append(r)
    kept = []
    dc_p = dc_n = dd_p = dd_n = 0
    for ids, group in groups.items():
        if len(group) == 1:
            kept.append(group[0]); continue
        labels = {g["label"] for g in group}
        if len(labels) == 1:
            kept.append(group[0])
            for r in group[1:]:
                if r["label"] == 1: dc_p += 1
                else: dc_n += 1
        else:
            for r in group:
                if r["label"] == 1: dd_p += 1
                else: dd_n += 1
    return kept, {"source_rows": len(rows), "dropped_test_total": dt_p+dt_n,
                  "dropped_test_pos": dt_p, "dropped_test_neg": dt_n,
                  "dup_collapsed_pos": dc_p, "dup_collapsed_neg": dc_n,
                  "dup_disagreement_dropped_pos": dd_p, "dup_disagreement_dropped_neg": dd_n,
                  "kept_rows": len(kept),
                  "kept_pos": sum(1 for r in kept if r["label"] == 1),
                  "kept_neg": sum(1 for r in kept if r["label"] == 0)}

def main():
    test_pairs = load_test_pairs(TEST_FILE)
    report = {}
    for method, run_dir in METHOD_TO_RUN.items():
        report[method] = {}
        for profile in METHOD_TO_PROFILES[method]:
            src = run_dir / "profiles" / profile / f"active_labels_latest_abt-buy_{profile}_train.json.gz"
            if not src.exists():
                print(f"[skip] {method}/{profile}: source missing"); continue
            rows = load_train(src)
            kept, stats = clean_one(rows, test_pairs)
            dst = ROOT / "output/labeling_cleaned_size_scan" / method / "abt-buy" / "profiles" / profile / f"active_labels_latest_abt-buy_{profile}_train.json.gz"
            write_train(kept, dst)
            report[method][profile] = {"source": str(src), "cleaned": str(dst), **stats}
            print(f"{method:>14} | {profile:<22} src={stats['source_rows']:>6} test={stats['dropped_test_total']:>4} ({stats['dropped_test_pos']}p/{stats['dropped_test_neg']}n) kept={stats['kept_rows']} ({stats['kept_pos']}p/{stats['kept_neg']}n)")
    (ROOT / "output/labeling_cleaned_size_scan/methods_cleanup_report_abt-buy.json").write_text(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
