#!/usr/bin/env python3
"""80/20 stratified split on the test-cleaned files for seed_round + simple_active.

Uses the same profile per (method, benchmark) as the published table.
Output: output/labeling_cleaned_llmvalid/{method}/<bm>/profiles/<profile>/
          {train.json.gz, valid.json.gz}
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

# (method, benchmark) -> profile (matches the published-table column for that cell)
SPECS = {
    ("seed_round",    "abt-buy"):        "all_plus20random",
    ("seed_round",    "dblp-acm"):       "all_plus20random",
    ("seed_round",    "dblp-scholar"):   "all_plus20random",
    ("seed_round",    "walmart-amazon"): "all",
    ("seed_round",    "wdc"):            "medium",
    ("simple_active", "abt-buy"):        "large",
    ("simple_active", "dblp-acm"):       "all",
    ("simple_active", "dblp-scholar"):   "all",
    ("simple_active", "walmart-amazon"): "all",
    ("simple_active", "wdc"):            "large",
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
    rng.shuffle(pos); rng.shuffle(neg)
    n_pos_v = int(round(len(pos) * frac))
    n_neg_v = int(round(len(neg) * frac))
    valid = pos[:n_pos_v] + neg[:n_neg_v]
    train = pos[n_pos_v:] + neg[n_neg_v:]
    rng.shuffle(valid); rng.shuffle(train)
    return train, valid


def main() -> None:
    report = {}
    for (method, bm), profile in SPECS.items():
        src = SRC_ROOT / method / bm / "profiles" / profile / f"active_labels_latest_{bm}_{profile}_train.json.gz"
        if not src.exists():
            print(f"[WARN] missing: {src}")
            continue
        rows = load(src)
        train, valid = stratified_split(rows, VALID_FRACTION, SPLIT_SEED)
        tr_path = DST_ROOT / method / bm / "profiles" / profile / f"active_labels_latest_{bm}_{profile}_train.json.gz"
        va_path = DST_ROOT / method / bm / "profiles" / profile / f"active_labels_latest_{bm}_{profile}_valid.json.gz"
        write(train, tr_path)
        write(valid, va_path)
        report.setdefault(method, {})[bm] = {
            "profile": profile, "source": str(src),
            "train_rows": len(train), "train_pos": sum(1 for r in train if r["label"] == 1), "train_neg": sum(1 for r in train if r["label"] == 0),
            "valid_rows": len(valid), "valid_pos": sum(1 for r in valid if r["label"] == 1), "valid_neg": sum(1 for r in valid if r["label"] == 0),
            "train_path": str(tr_path), "valid_path": str(va_path),
        }
        print(f"{method:>13} | {bm:<15} {profile:<18} -> train={len(train):>5} valid={len(valid):>5}")

    DST_ROOT.mkdir(parents=True, exist_ok=True)
    out_report = DST_ROOT / "split_report_methods.json"
    out_report.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_report}")


if __name__ == "__main__":
    main()
