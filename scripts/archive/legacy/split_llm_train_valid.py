#!/usr/bin/env python3
"""Take the test-cleaned LLM training sets and hold out 20% as validation.

Source: output/labeling_cleaned/{labeller}/<bm>/profiles/<profile>/...train.json.gz
        (test-leak + dedup already removed; validation rows still present)
Output: output/labeling_cleaned_llmvalid/{labeller}/<bm>/profiles/<profile>/
          {train.json.gz, valid.json.gz}

The split is stratified by label with a fixed seed so the same rows go to
train vs. valid across labellers/benchmarks. Test set is unchanged (we keep
using the official gold-standard test).
"""
from __future__ import annotations

import gzip
import json
import random
from pathlib import Path
from typing import Iterable, List

ROOT = Path("/work/aasteine/Automatic-data-labeling")
SRC_ROOT = ROOT / "output/labeling_cleaned"
DST_ROOT = ROOT / "output/labeling_cleaned_llmvalid"

VALID_FRACTION = 0.20
SPLIT_SEED = 42

LABELLERS = ["gpt", "qwen", "kimi"]
BENCHMARKS = ["abt-buy", "dblp-acm", "dblp-scholar", "walmart-amazon", "wdc"]
PROFILE = {
    "abt-buy":        "all_plus20random",
    "dblp-acm":       "all_plus20random",
    "dblp-scholar":   "all_plus20random",
    "walmart-amazon": "all_plus20random",
    "wdc":            "all",
}


def load(path: Path) -> List[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def write(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def stratified_split(rows: List[dict], frac: float, seed: int):
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_pos_v = int(round(len(pos) * frac))
    n_neg_v = int(round(len(neg) * frac))
    valid = pos[:n_pos_v] + neg[:n_neg_v]
    train = pos[n_pos_v:] + neg[n_neg_v:]
    rng.shuffle(valid)
    rng.shuffle(train)
    return train, valid


def main() -> None:
    report = {}
    for labeller in LABELLERS:
        report[labeller] = {}
        for bm in BENCHMARKS:
            profile = PROFILE[bm]
            src = SRC_ROOT / labeller / bm / "profiles" / profile / f"active_labels_latest_{bm}_{profile}_train.json.gz"
            if not src.exists():
                print(f"[WARN] missing: {src}")
                continue
            rows = load(src)
            train, valid = stratified_split(rows, VALID_FRACTION, SPLIT_SEED)
            tr_path = DST_ROOT / labeller / bm / "profiles" / profile / f"active_labels_latest_{bm}_{profile}_train.json.gz"
            va_path = DST_ROOT / labeller / bm / "profiles" / profile / f"active_labels_latest_{bm}_{profile}_valid.json.gz"
            write(train, tr_path)
            write(valid, va_path)
            stats = {
                "source": str(src),
                "source_rows": len(rows),
                "train_rows": len(train),
                "train_pos": sum(1 for r in train if r["label"] == 1),
                "train_neg": sum(1 for r in train if r["label"] == 0),
                "valid_rows": len(valid),
                "valid_pos": sum(1 for r in valid if r["label"] == 1),
                "valid_neg": sum(1 for r in valid if r["label"] == 0),
                "train_path": str(tr_path),
                "valid_path": str(va_path),
            }
            report[labeller][bm] = stats
            print(f"{labeller:>4} | {bm:<15} {profile:<18} src={stats['source_rows']:>6} "
                  f"-> train={stats['train_rows']:>5} ({stats['train_pos']}p/{stats['train_neg']}n) "
                  f"valid={stats['valid_rows']:>4} ({stats['valid_pos']}p/{stats['valid_neg']}n)")

    DST_ROOT.mkdir(parents=True, exist_ok=True)
    (DST_ROOT / "split_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote split report to {DST_ROOT / 'split_report.json'}")


if __name__ == "__main__":
    main()
