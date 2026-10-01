"""Shared writers for the benchmark preparation scripts.

Output layout (what configs/labeling/benchmarks_active.yaml and
configs/ditto/benchmarks_training.yaml read):

  <out>/<name>-train-left.csv, <name>-train-right.csv   record tables, derived from the
                                                         records that appear in the benchmark
                                                         training pairs (keeps the benchmark's
                                                         record selection, undoes its pair selection)
  <out>/<name>-train.json.gz                             benchmark training pairs, WDC JSONL layout
  <out>/<name>-valid.json.gz                             validation pairs, same layout
  <out>/<name>-valid.csv                                 validation pair ids (pair_id)
  <out>/<name>-gs.json.gz                                gold-standard test pairs, same layout
"""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def clusters_from_pairs(pairs: Iterable[Dict[str, str]]) -> Dict[str, int]:
    """Connected components over positive pairs; keys are the prefixed record ids."""
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in pairs:
        if int(p["label"]) == 1:
            a, b = find(p["id_left"]), find(p["id_right"])
            if a != b:
                parent[a] = b
    out: Dict[str, int] = {}
    nxt: Dict[str, int] = {}
    for k in list(parent):
        r = find(k)
        out[k] = nxt.setdefault(r, len(nxt))
    return out


def write_table(path: Path, records: Sequence[Dict[str, str]], fields: Sequence[str], clusters: Dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id"] + list(fields) + ["cluster_id"])
        for r in records:
            w.writerow([r["id"]] + [r.get(c, "") for c in fields] + [clusters.get(r["id"], "")])


def write_pairs_jsonl(path: Path, pairs: Sequence[Dict[str, str]], left: Dict[str, Dict[str, str]],
                      right: Dict[str, Dict[str, str]], fields: Sequence[str], clusters: Dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for p in pairs:
            l, r = left[p["id_left"]], right[p["id_right"]]
            o: Dict[str, object] = {"id_left": p["id_left"]}
            for c in fields:
                o[f"{c}_left"] = l.get(c, "")
            o["cluster_id_left"] = clusters.get(p["id_left"], "")
            o["id_right"] = p["id_right"]
            for c in fields:
                o[f"{c}_right"] = r.get(c, "")
            o["cluster_id_right"] = clusters.get(p["id_right"], "")
            o["label"] = int(p["label"])
            o["pair_id"] = f"{p['id_left']}#{p['id_right']}"
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


def write_valid_ids(path: Path, pairs: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id"])
        for p in pairs:
            w.writerow([f"{p['id_left']}#{p['id_right']}"])


def emit_all(out: Path, name: str, fields: List[str], left: Dict[str, Dict[str, str]],
             right: Dict[str, Dict[str, str]], train: List[Dict[str, str]], valid: List[Dict[str, str]],
             test: List[Dict[str, str]]) -> None:
    clusters = clusters_from_pairs(train + valid + test)
    lids = sorted({p["id_left"] for p in train}, key=lambda x: (len(x), x))
    rids = sorted({p["id_right"] for p in train}, key=lambda x: (len(x), x))
    write_table(out / f"{name}-train-left.csv", [left[i] for i in lids], fields, clusters)
    write_table(out / f"{name}-train-right.csv", [right[i] for i in rids], fields, clusters)
    write_pairs_jsonl(out / f"{name}-train.json.gz", train, left, right, fields, clusters)
    write_pairs_jsonl(out / f"{name}-valid.json.gz", valid, left, right, fields, clusters)
    write_valid_ids(out / f"{name}-valid.csv", valid)
    write_pairs_jsonl(out / f"{name}-gs.json.gz", test, left, right, fields, clusters)
    npos = lambda ps: sum(int(p["label"]) for ps in [ps] for p in ps)  # noqa: E731
    print(f"{name}: left {len(lids)} / right {len(rids)} records | train {len(train)} ({npos(train)} pos) | "
          f"valid {len(valid)} ({npos(valid)} pos) | test {len(test)} ({npos(test)} pos)")


def yaml_blocks(name: str, out: Path, fields: List[str], profiles: Dict[str, Dict[str, int]] | None = None,
                extra_labeling: Dict[str, object] | None = None) -> str:
    fm = "\n".join(f"      {c}: {c}" for c in fields)
    prof_lines = []
    for pn, pv in (profiles or {}).items():
        prof_lines.append(f"      {pn}: {{target_total: {pv['target_total']}, target_pos: {pv['target_pos']}, "
                          f"target_neg: {pv['target_neg']}, label_budget: {pv['label_budget']}}}")
    prof_lines.append("      all:    {all_examples: true}")
    extra = ""
    if extra_labeling:
        extra = "    labeling_args:\n" + "\n".join(f"      {k}: {v}" for k, v in extra_labeling.items()) + "\n"
    return f"""
# --- configs/labeling/benchmarks_active.yaml ---
  {name}:
    left_csv: {out}/{name}-train-left.csv
    right_csv: {out}/{name}-train-right.csv
    valid_path: {out}/{name}-valid.csv
    test_path: {out}/{name}-gs.json.gz
    id_col: id
    fields:
{fm}
    embeddings:
      dir: {out}/embeddings
      left_file: {name}_left_embeddings.npy
      right_file: {name}_right_embeddings.npy
{extra}    profiles:
{chr(10).join(prof_lines)}

# --- configs/ditto/benchmarks_training.yaml ---
  {name}:
    train: {out}/{name}-train.json.gz
    valid: {out}/{name}-valid.json.gz
    test: {out}/{name}-gs.json.gz
    fields: [{', '.join(fields)}]
    profiles: [{', '.join(list((profiles or {}).keys()) + ['all'])}]
"""
