#!/usr/bin/env python3
"""Build only the v_closure_drop variant for all 5 AL Ditto training sets.
Independent of relabel results - uses only original labels + closure graph.
"""
from __future__ import annotations
import gzip
import json
from pathlib import Path

ROOT = Path("/work/aasteine/Automatic-data-labeling")
ORIG_TRAIN = {
    "abt-buy":        ROOT / "output/learning_curve_abtbuy/ditto_active_learning/N6000/abt-buy_N6000_train.json.gz",
    "walmart-amazon": ROOT / "output/three_phase_labeling_ditto_only_v2/benchmark_walmart-amazon_20260323_202820/profiles/all/active_labels_latest_walmart-amazon_all_train.json.gz",
    "dblp-acm":       ROOT / "output/three_phase_labeling_ditto_only_v2/benchmark_dblp-acm_20260323_202820/profiles/all_plus20random/active_labels_latest_dblp-acm_all_plus20random_train.json.gz",
    "dblp-scholar":   ROOT / "output/three_phase_labeling_ditto_only_v2/benchmark_dblp-scholar_20260323_202820/profiles/large/active_labels_latest_dblp-scholar_large_train.json.gz",
    "wdc":            ROOT / "output/benchmark_size_runs/al_ditto/wdc/wdc_train.json.gz",
}
OUT = ROOT / "output/postfilter_variants"


def load_jsonl_gz(p):
    rows = []
    with gzip.open(p, "rt") as f:
        for line in f:
            s = line.strip()
            if s: rows.append(json.loads(s))
    return rows


def write_jsonl_gz(rows, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")


def extract_ids(pair_id):
    parts = str(pair_id).split("__")
    return (parts[0], parts[1]) if len(parts) >= 2 else (None, None)


class UF:
    def __init__(self): self.p = {}
    def find(self, x):
        while self.p.get(x, x) != x:
            self.p[x] = self.p.get(self.p.get(x, x), self.p.get(x, x))
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb


for bm, src in ORIG_TRAIN.items():
    rows = load_jsonl_gz(src)
    uf = UF()
    for r in rows:
        a, b = extract_ids(r["pair_id"])
        if a is None: continue
        uf.p.setdefault(a, a); uf.p.setdefault(b, b)
    for r in rows:
        if int(r["label"]) == 1:
            a, b = extract_ids(r["pair_id"])
            if a is None: continue
            uf.union(a, b)
    conflicts = set()
    for r in rows:
        a, b = extract_ids(r["pair_id"])
        if a is None: continue
        if int(r["label"]) == 0 and uf.find(a) == uf.find(b):
            conflicts.add(r["pair_id"])
    kept = [r for r in rows if str(r["pair_id"]) not in conflicts]
    out_path = OUT / bm / "v_closure_drop" / "train.json.gz"
    write_jsonl_gz(kept, out_path)
    pos = sum(1 for r in kept if int(r["label"]) == 1)
    neg = sum(1 for r in kept if int(r["label"]) == 0)
    print(f"{bm:15s}: src={len(rows)} conflicts={len(conflicts)} kept={len(kept)} ({pos}p/{neg}n) → {out_path.relative_to(ROOT)}")
