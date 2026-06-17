#!/usr/bin/env python3
"""Build 5 postfilter variants from each (benchmark, AL Ditto train.json.gz + relabel diff):
  +Relabel              — full relabel, keep all pairs
  +Relabel drop         — relabel + drop pairs whose label changed
  +Closure drop         — drop pairs where label conflicts with transitive closure
  +Cl AND Rel drop      — drop only if both closure-inconsistent AND relabel-changed
  +Cl OR Rel drop       — drop if either closure-inconsistent OR relabel-changed

Closure logic (entity-cluster closure):
  Treat each pair (id_left, id_right) labelled 1 as an edge in an undirected
  graph; connected components are inferred clusters of "same-entity" records.
  For a pair (a, b) labelled 0, if a and b lie in the same component (i.e. there
  is a positive bridge between them via other pairs), it's a closure conflict.
  For a pair (a, b) labelled 1, if a and b lie in different components, that
  would be inconsistent with one of the labelled edges (also conflict).

Input:
  output/postfilter_variants/<benchmark>/_relabel/{train__relabeled.json.gz,
                                                   relabel_diff.csv,
                                                   relabel_summary.json}

Outputs (under output/postfilter_variants/<benchmark>/<variant>/):
  v_relabel/train.json.gz                — relabeled only
  v_relabel_drop/train.json.gz           — relabeled minus changed
  v_closure_drop/train.json.gz           — original minus closure-conflicts
  v_closure_and_relabel/train.json.gz    — original minus (closure AND relabel-changed)
  v_closure_or_relabel/train.json.gz     — original minus (closure OR relabel-changed)
  _summary.json                          — per-variant rows/pos/neg counts
"""
from __future__ import annotations
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path("/work/aasteine/Automatic-data-labeling")
ROOT_OUT = ROOT / "output/postfilter_variants"

BENCHMARKS = ["abt-buy", "walmart-amazon", "dblp-acm", "dblp-scholar", "wdc"]

# Original AL Ditto train.json.gz per benchmark (same as in benchmark_size_runs.yaml)
ORIG_TRAIN = {
    "abt-buy":        ROOT / "output/learning_curve_abtbuy/ditto_active_learning/N6000/abt-buy_N6000_train.json.gz",
    "walmart-amazon": ROOT / "output/three_phase_labeling_ditto_only_v2/benchmark_walmart-amazon_20260323_202820/profiles/all/active_labels_latest_walmart-amazon_all_train.json.gz",
    "dblp-acm":       ROOT / "output/three_phase_labeling_ditto_only_v2/benchmark_dblp-acm_20260323_202820/profiles/all_plus20random/active_labels_latest_dblp-acm_all_plus20random_train.json.gz",
    "dblp-scholar":   ROOT / "output/three_phase_labeling_ditto_only_v2/benchmark_dblp-scholar_20260323_202820/profiles/large/active_labels_latest_dblp-scholar_large_train.json.gz",
    "wdc":            ROOT / "output/benchmark_size_runs/al_ditto/wdc/wdc_train.json.gz",
}


def load_jsonl_gz(p):
    rows = []
    with gzip.open(p, "rt") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def write_jsonl_gz(rows, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def extract_ids(pair_id):
    """pair_id like 'abt_219__buy_102__679_124_0' or '60641225__74638206__16964_16721_0'
    → returns (id_left, id_right) as strings."""
    parts = str(pair_id).split("__")
    if len(parts) >= 2:
        return parts[0], parts[1]
    # fallback: try '#'
    if "#" in str(pair_id):
        return tuple(str(pair_id).split("#", 1))
    return None, None


class UnionFind:
    def __init__(self):
        self.parent = {}
    def find(self, x):
        while self.parent.get(x, x) != x:
            self.parent[x] = self.parent.get(self.parent.get(x, x), self.parent.get(x, x))
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def closure_conflicts(rows):
    """Identify pair_ids whose label is inconsistent with the transitive closure
    of all positive labels. Returns set of pair_ids to drop."""
    uf = UnionFind()
    for r in rows:
        a, b = extract_ids(r["pair_id"])
        if a is None: continue
        uf.parent.setdefault(a, a); uf.parent.setdefault(b, b)
    # Add positive edges (build clusters)
    for r in rows:
        if int(r["label"]) == 1:
            a, b = extract_ids(r["pair_id"])
            if a is None: continue
            uf.union(a, b)
    # Find conflicts: negative pair with same cluster, OR positive pair with different cluster
    # (the second condition shouldn't happen unless multiple records map to same id pair with different labels)
    conflicts = set()
    for r in rows:
        a, b = extract_ids(r["pair_id"])
        if a is None: continue
        same_cluster = uf.find(a) == uf.find(b)
        label = int(r["label"])
        if label == 0 and same_cluster:
            conflicts.add(r["pair_id"])
        # label == 1 + diff cluster: impossible by construction above
    return conflicts


def main():
    summaries = {}
    for bm in BENCHMARKS:
        bm_dir = ROOT_OUT / bm
        relabel_dir = bm_dir / "_relabel"
        train_relabeled_path = relabel_dir / "train__relabeled.json.gz"
        diff_csv = relabel_dir / "relabel_diff.csv"

        if not train_relabeled_path.exists() or not diff_csv.exists():
            print(f"[skip] {bm}: relabel output missing at {relabel_dir.relative_to(ROOT)}")
            continue

        orig_rows = load_jsonl_gz(ORIG_TRAIN[bm])
        relabeled_rows = load_jsonl_gz(train_relabeled_path)
        diff_df = pd.read_csv(diff_csv)

        # Build sets for quick lookup
        changed_ids = set(diff_df.loc[diff_df["changed"] == 1, "pair_id"].astype(str).tolist())
        conflicts = closure_conflicts(orig_rows)

        # Map pair_id -> relabeled row (so we can swap labels for the +Relabel variant)
        relabel_map = {str(r["pair_id"]): r for r in relabeled_rows}

        variants = {}
        # v_relabel: relabeled rows in original order, keep all
        variants["v_relabel"] = [relabel_map.get(str(r["pair_id"]), r) for r in orig_rows]
        # v_relabel_drop: relabeled minus changed
        variants["v_relabel_drop"] = [relabel_map.get(str(r["pair_id"]), r) for r in orig_rows
                                      if str(r["pair_id"]) not in changed_ids]
        # v_closure_drop: original minus closure conflicts
        variants["v_closure_drop"] = [r for r in orig_rows if str(r["pair_id"]) not in conflicts]
        # v_closure_and_relabel: original minus (conflict AND changed)
        and_drop = conflicts & changed_ids
        variants["v_closure_and_relabel"] = [r for r in orig_rows if str(r["pair_id"]) not in and_drop]
        # v_closure_or_relabel: original minus (conflict OR changed)
        or_drop = conflicts | changed_ids
        variants["v_closure_or_relabel"] = [r for r in orig_rows if str(r["pair_id"]) not in or_drop]

        # Write variants
        for vname, rows in variants.items():
            path = bm_dir / vname / "train.json.gz"
            write_jsonl_gz(rows, path)

        # Build summary
        s = {}
        for vname, rows in variants.items():
            pos = sum(1 for r in rows if int(r["label"]) == 1)
            neg = sum(1 for r in rows if int(r["label"]) == 0)
            s[vname] = {"rows": len(rows), "pos": pos, "neg": neg}
        s["__source__"] = {
            "n_pairs": len(orig_rows),
            "n_relabel_changed": len(changed_ids),
            "n_closure_conflicts": len(conflicts),
            "n_closure_and_relabel_drop": len(and_drop),
            "n_closure_or_relabel_drop": len(or_drop),
        }
        (bm_dir / "_summary.json").write_text(json.dumps(s, indent=2))
        summaries[bm] = s

        print(f"\n=== {bm} ===  source={len(orig_rows)} pairs, relabel_changed={len(changed_ids)}, closure_conflicts={len(conflicts)}")
        for vname in ("v_relabel", "v_relabel_drop", "v_closure_drop", "v_closure_and_relabel", "v_closure_or_relabel"):
            v = s[vname]
            print(f"  {vname:25s}: {v['rows']:>5d} rows ({v['pos']}p/{v['neg']}n)")

    (ROOT_OUT / "_all_summaries.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nAll summaries: {(ROOT_OUT / '_all_summaries.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
