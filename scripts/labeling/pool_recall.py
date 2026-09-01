"""Report candidate-pool quality for a benchmark under each pool-construction method.

For every method (embedding / bm25 / rrf) builds the pool exactly as the labeling
runners do and reports: pool size, positive-pair recall against the benchmark's
train-split positives (validation pairs excluded), and the share of hard positives
/ hard negatives among the *labeled* train pairs that fall inside the pool, using
the all-field token Jaccard thresholds from the paper when given.

No API calls. Needs the benchmark embeddings on disk.

  python scripts/labeling/pool_recall.py --benchmark dn7-walmart-amazon --methods embedding,bm25,rrf
  python scripts/labeling/pool_recall.py --benchmark wdc --methods embedding,rrf --tau-pos 0.10 --tau-neg 0.17
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "labeling"))
from pool_builders import build_candidates, record_text, tokenize  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


def _read_pairs(path: Path) -> pd.DataFrame:
    p = str(path)
    if p.endswith(".csv"):
        return pd.read_csv(path)
    opener = gzip.open if p.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        first = f.readline()
        f.seek(0)
        try:
            o = json.loads(first)
            # JSONL: one pair per line with scalar values. pandas orient="columns" files
            # (abt-buy-train.json, walmart-amazon-train.json.gz) also carry a top-level
            # id_left key, but its value is a dict of rows.
            if isinstance(o, dict) and "id_left" in o and not isinstance(o["id_left"], dict):
                return pd.DataFrame([json.loads(l) for l in f if l.strip()])
        except Exception:
            pass
        return pd.read_json(f)


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    return len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/labeling/benchmarks_active.yaml")
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--methods", default="embedding,bm25,rrf,union")
    ap.add_argument("--train", default="", help="benchmark train pairs (json/json.gz); default from ditto config")
    ap.add_argument("--ditto-config", default="configs/ditto/benchmarks_training.yaml")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--bottom-k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tau-pos", type=float, default=None, help="hard-positive Jaccard threshold (s <= tau)")
    ap.add_argument("--tau-neg", type=float, default=None, help="hard-negative Jaccard threshold (s >= tau)")
    ap.add_argument("--out", default="", help="optional JSON output path")
    a = ap.parse_args()
    os.chdir(ROOT)

    cfg = yaml.safe_load(Path(a.config).read_text())["benchmarks"][a.benchmark]
    ml = _load("_alml", ROOT / "scripts/labeling/active_learning_ml.py")
    fields_map: Dict[str, str] = cfg.get("fields") or {}
    src_fields = list(fields_map.values())

    ldf = pd.read_csv(cfg["left_csv"]).reset_index(drop=True)
    rdf = pd.read_csv(cfg["right_csv"]).reset_index(drop=True)
    ldf["id"], rdf["id"] = ldf["id"].astype(str), rdf["id"].astype(str)
    emb = cfg["embeddings"]
    lemb = np.load(Path(emb["dir"]) / emb["left_file"]).astype(np.float32)
    remb = np.load(Path(emb["dir"]) / emb["right_file"]).astype(np.float32)
    assert len(ldf) == len(lemb) and len(rdf) == len(remb), "embedding rows != csv rows"
    ltext = [record_text(r, src_fields) for r in ldf.to_dict("records")]
    rtext = [record_text(r, src_fields) for r in rdf.to_dict("records")]
    lrid = np.asarray([f"L:{i}" for i in range(len(ldf))]); rrid = np.asarray([f"R:{i}" for i in range(len(rdf))])
    lrid2id = dict(zip(lrid, ldf["id"])); rrid2id = dict(zip(rrid, rdf["id"]))

    base_name = a.benchmark.split("--")[0]  # pool_sensitivity.yaml keys are <benchmark>--<method>
    train_path = (a.train or cfg.get("train_pairs")
                  or yaml.safe_load(Path(a.ditto_config).read_text())["benchmarks"][base_name]["train"])
    tr = _read_pairs(Path(train_path))
    if "valid_path" in cfg and str(cfg["valid_path"]).endswith(".csv"):
        vp = set(pd.read_csv(cfg["valid_path"])["pair_id"].astype(str))
        if "pair_id" in tr.columns:
            tr = tr[~tr["pair_id"].astype(str).isin(vp)]
    tr = tr.copy()
    tr["id_left"], tr["id_right"] = tr["id_left"].astype(str), tr["id_right"].astype(str)
    tr["label"] = tr["label"].astype(str).str.strip().str.lower().isin({"1", "true", "1.0"}).astype(int)
    keep = tr["id_left"].isin(set(ldf["id"])) & tr["id_right"].isin(set(rdf["id"]))
    tr = tr[keep]
    pos_pairs: Set[Tuple[str, str]] = set(zip(tr.loc[tr.label == 1, "id_left"], tr.loc[tr.label == 1, "id_right"]))
    neg_pairs: Set[Tuple[str, str]] = set(zip(tr.loc[tr.label == 0, "id_left"], tr.loc[tr.label == 0, "id_right"]))

    ltext_by_id = dict(zip(ldf["id"], ltext)); rtext_by_id = dict(zip(rdf["id"], rtext))
    results: Dict[str, Any] = {"benchmark": a.benchmark, "train_positives": len(pos_pairs),
                               "train_negatives": len(neg_pairs), "k": a.k, "bottom_k": a.bottom_k, "methods": {}}
    for method in [m.strip() for m in a.methods.split(",") if m.strip()]:
        c, stats = build_candidates(
            method=method, left_ids=lrid, right_ids=rrid, right_source_ids=rdf["id"].to_numpy(),
            left_emb=lemb, right_emb=remb, left_text=ltext, right_text=rtext, k=a.k, candidate_cap=0,
            bottom_k=a.bottom_k, random_state=a.seed,
            embedding_builder=getattr(ml, "_build_candidates_embedding", ml._build_candidates))
        pool = set(zip(c["id1"].map(lrid2id), c["id2"].map(rrid2id)))
        pos_in = pos_pairs & pool
        neg_in = neg_pairs & pool
        r: Dict[str, Any] = {
            "pool_pairs": len(pool),
            "positive_recall": round(100.0 * len(pos_in) / max(1, len(pos_pairs)), 2),
            "labeled_negatives_in_pool": len(neg_in),
        }
        if a.tau_pos is not None:
            hp = sum(1 for l, rr in pos_in if _jaccard(ltext_by_id[l], rtext_by_id[rr]) <= a.tau_pos)
            r["hard_positives_in_pool"] = hp
        if a.tau_neg is not None:
            hn = sum(1 for l, rr in neg_in if _jaccard(ltext_by_id[l], rtext_by_id[rr]) >= a.tau_neg)
            r["hard_negatives_in_pool"] = hn
        results["methods"][method] = r
        print(f"{a.benchmark:22s} {method:9s} pool={len(pool):>7,}  positive recall={r['positive_recall']:6.2f}%"
              + (f"  hard-pos={r.get('hard_positives_in_pool')}" if a.tau_pos is not None else "")
              + (f"  hard-neg={r.get('hard_negatives_in_pool')}" if a.tau_neg is not None else ""))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
