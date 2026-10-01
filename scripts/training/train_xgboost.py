#!/usr/bin/env python3
"""Train the feature-based students of Table 6 (XGBoost; RandomForest on request) on each
benchmark training set and GPT-5.2 training set, and evaluate them on the benchmark test set.

Features per pair: per-field token-jaccard/exact-match/numeric-diff +
embedding cosine similarity (same feature extraction as AL ML pipeline in
scripts/labeling/active_learning_ml.py:_build_feature_matrix).

Models:
  - XGBoostClassifier
  - RandomForestClassifier (300 trees, class_weight=balanced)

Output:
  output/results_summary/traditional_students.csv
  output/traditional_students/<benchmark>_<method>_seed<seed>.json
"""
from __future__ import annotations
import argparse
import gzip
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
OUT_RESULTS_DIR = ROOT / "output/traditional_students"

# Training sets of Table 6: the released GPT-5.2 sets of the three selection strategies and the
# benchmark training files. billiger.de uses the 5,897-pair sets.
BENCHMARKS = ["abt-buy", "walmart-amazon", "dn7-walmart-amazon", "wdc", "billiger-de", "dblp-acm", "dblp-scholar", "semi-heter"]
METHOD_DIRS = {"sim": "similarity_search", "alml": "active_learning_ml", "alditto": "active_learning_ditto"}
SOURCES = {}
for bm in BENCHMARKS:
    suffix = "_5897" if bm == "billiger-de" else ""
    for method, folder in METHOD_DIRS.items():
        SOURCES[(bm, method)] = ROOT / f"artifacts/training_data/{bm}/gpt-5.2/{folder}{suffix}/{bm}_train.json.gz"

SOURCES[("abt-buy",            "benchmark")] = ROOT / "benchmarks/abt-buy/abt-buy-train.json"
SOURCES[("walmart-amazon",     "benchmark")] = ROOT / "benchmarks/walmart-amazon/walmart-amazon-train.json.gz"
SOURCES[("dn7-walmart-amazon", "benchmark")] = ROOT / "benchmarks/dn7-walmart-amazon/dn7-walmart-amazon-train.json.gz"
SOURCES[("wdc",                "benchmark")] = ROOT / "benchmarks/wdc/wdcproducts80cc20rnd000un_train_large.json.gz"
SOURCES[("billiger-de",        "benchmark")] = ROOT / "benchmarks/billiger-de/billiger-de-train-medium.json.gz"
SOURCES[("dblp-acm",           "benchmark")] = ROOT / "benchmarks/dblp-acm/dblp-acm-train.json.gz"
SOURCES[("dblp-scholar",       "benchmark")] = ROOT / "benchmarks/dblp-scholar/dblp-scholar-train.json.gz"
SOURCES[("semi-heter",         "benchmark")] = ROOT / "benchmarks/semi-heter/semi-heter-train.json.gz"

# Benchmarks where pre-computed embeddings can't be used for test (test entities are
# distinct from train pool, e.g., wdc's "unseen entities" split). For these we drop
# the embedding-cosine column from BOTH train and test so the classifier learns only
# from per-field features.
DROP_EMBEDDING_FEATURE = {"wdc"}

# Per-benchmark: official test file + canonical record sources + embedding dir + fields
BENCHMARK_CONFIGS = {
    "abt-buy": {
        "test":          ROOT / "benchmarks/abt-buy/abt-buy-gs.json.gz",
        "left_csv":      ROOT / "benchmarks/abt-buy/abt-buy-train-left.csv",
        "right_csv":     ROOT / "benchmarks/abt-buy/abt-buy-train-right.csv",
        "left_emb":      ROOT / "benchmarks/abt-buy/embeddings/abt-buy_left_embeddings.npy",
        "right_emb":     ROOT / "benchmarks/abt-buy/embeddings/abt-buy_right_embeddings.npy",
        "fields":        ["title", "description", "price"],
        "field_aliases": {"title": ["name"]},  # test uses 'name' instead of 'title'
    },
    "walmart-amazon": {
        "test":      ROOT / "benchmarks/walmart-amazon/walmart-amazon-gs.json.gz",
        "left_csv":  ROOT / "benchmarks/walmart-amazon/walmart-amazon-train-left.csv",
        "right_csv": ROOT / "benchmarks/walmart-amazon/walmart-amazon-train-right.csv",
        "left_emb":  ROOT / "benchmarks/walmart-amazon/embeddings/walmart-amazon_left_embeddings.npy",
        "right_emb": ROOT / "benchmarks/walmart-amazon/embeddings/walmart-amazon_right_embeddings.npy",
        "fields":    ["title", "category", "brand", "modelno", "price"],
    },
    "dblp-acm": {
        "test":      ROOT / "benchmarks/dblp-acm/dblp-acm-gs.json.gz",
        "left_csv":  ROOT / "benchmarks/dblp-acm/dblp-acm-train-left.csv",
        "right_csv": ROOT / "benchmarks/dblp-acm/dblp-acm-train-right.csv",
        "left_emb":  ROOT / "benchmarks/dblp-acm/embeddings/dblp-acm_left_embeddings.npy",
        "right_emb": ROOT / "benchmarks/dblp-acm/embeddings/dblp-acm_right_embeddings.npy",
        "fields":    ["title", "authors", "venue", "year"],
    },
    "dblp-scholar": {
        "test":      ROOT / "benchmarks/dblp-scholar/dblp-scholar-gs.json.gz",
        "left_csv":  ROOT / "benchmarks/dblp-scholar/dblp-scholar-train-left.csv",
        "right_csv": ROOT / "benchmarks/dblp-scholar/dblp-scholar-train-right.csv",
        "left_emb":  ROOT / "benchmarks/dblp-scholar/embeddings/dblp-scholar_left_embeddings.npy",
        "right_emb": ROOT / "benchmarks/dblp-scholar/embeddings/dblp-scholar_right_embeddings.npy",
        "fields":    ["title", "authors", "venue", "year"],
    },
    "wdc": {
        "test":      ROOT / "benchmarks/wdc/wdcproducts80cc20rnd100un_gs.json.gz",
        "left_csv":  ROOT / "benchmarks/wdc/wdc_train_large_left.csv",
        "right_csv": ROOT / "benchmarks/wdc/wdc_train_large_right.csv",
        "left_emb":  ROOT / "benchmarks/wdc/embeddings/wdc_left_embeddings.npy",
        "right_emb": ROOT / "benchmarks/wdc/embeddings/wdc_right_embeddings.npy",
        "fields":    ["title", "brand", "description", "price", "priceCurrency"],
    },
}


for _bm, _fields in [("dn7-walmart-amazon", ["title", "modelno", "price", "shipweight", "brand", "dimensions"]),
                    ("billiger-de", ["name", "desc", "brand", "price"]),
                    ("semi-heter", ["title", "authors", "publisher", "year", "isbn", "pages", "price"])]:
    BENCHMARK_CONFIGS[_bm] = {
        "test":      ROOT / f"benchmarks/{_bm}/{_bm}-gs.json.gz",
        "left_csv":  ROOT / f"benchmarks/{_bm}/{_bm}-train-left.csv",
        "right_csv": ROOT / f"benchmarks/{_bm}/{_bm}-train-right.csv",
        "left_emb":  ROOT / f"benchmarks/{_bm}/embeddings/{_bm}_left_embeddings.npy",
        "right_emb": ROOT / f"benchmarks/{_bm}/embeddings/{_bm}_right_embeddings.npy",
        "fields":    _fields,
    }


# ─── Feature helpers (copied from active_learning_ml.py for consistency) ──
def _norm_text(v):
    if v is None or (isinstance(v, float) and pd.isna(v)): return ""
    return re.sub(r"\s+", " ", str(v)).strip()

def _tokens(v):
    s = _norm_text(v).lower()
    return set(re.findall(r"[a-z0-9]+", s))

def _jaccard(a, b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return float(len(a & b) / len(a | b)) if (a | b) else 0.0

def _to_price(v):
    if v is None or (isinstance(v, float) and pd.isna(v)): return None
    s = _norm_text(v).replace(",", "")
    if not s: return None
    try: return float(s)
    except Exception: return None

def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_n = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return (a_n * b_n).sum(axis=1)


def _load_canonical_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    df["id"] = df["id"].astype(str)
    # Some canonical CSVs (e.g., wdc) repeat rows per pair → keep first occurrence
    df = df.drop_duplicates(subset="id", keep="first").reset_index(drop=True)
    return df


def _build_features(pair_df: pd.DataFrame, bm_cfg, side_l_df, side_r_df, left_emb, right_emb,
                    left_id_to_idx, right_id_to_idx, field_aliases=None):
    """Build feature matrix matching AL ML.

    pair_df: must have id_left + id_right columns OR pair_id parseable.
    Returns X (n, len(fields)+1) and valid_mask.
    """
    fields = bm_cfg["fields"]
    aliases = bm_cfg.get("field_aliases", {}) or {}
    n = len(pair_df)
    base_dim = len(fields)
    X = np.zeros((n, base_dim + 1), dtype=np.float32)

    # resolve left/right id columns
    if "id_left" in pair_df.columns and "id_right" in pair_df.columns:
        ids_l = pair_df["id_left"].astype(str).tolist()
        ids_r = pair_df["id_right"].astype(str).tolist()
    else:
        # parse from pair_id "abt_X__buy_Y__rid1_rid2_label" or "X#Y"
        pair_ids = pair_df["pair_id"].astype(str).tolist()
        ids_l, ids_r = [], []
        for pid in pair_ids:
            if "__" in pid:
                parts = pid.split("__")
                ids_l.append(parts[0]); ids_r.append(parts[1])
            elif "#" in pid:
                l, r = pid.split("#", 1); ids_l.append(l); ids_r.append(r)
            else:
                ids_l.append(""); ids_r.append("")

    # Index canonical csvs by id
    left_map = side_l_df.set_index("id").to_dict("index")
    right_map = side_r_df.set_index("id").to_dict("index")

    def get_field(rec, f):
        if rec is None: return None
        # try the field, then any aliases
        if f in rec and rec[f] is not None: return rec[f]
        for alias in aliases.get(f, []):
            if alias in rec and rec[alias] is not None: return rec[alias]
        return None

    # Check if pair_df already has joined fields (e.g., wdc test set has title_left/title_right inline)
    joined_present = any(f"{f}_left" in pair_df.columns for f in fields)
    has_aliased_joined = any(f"{a}_left" in pair_df.columns for f, aa in aliases.items() for a in aa)
    joined_fields = joined_present or has_aliased_joined
    pair_dicts = pair_df.to_dict("records") if joined_fields else [None]*n

    valid = np.zeros(n, dtype=bool)
    for i, (lid, rid) in enumerate(zip(ids_l, ids_r)):
        lrow = left_map.get(lid); rrow = right_map.get(rid)
        # Fallback to joined-data row if canonical lookup fails
        if (lrow is None or rrow is None) and joined_fields and pair_dicts[i]:
            row_i = pair_dicts[i]
            lrow = {f: row_i.get(f"{f}_left") for f in fields}
            rrow = {f: row_i.get(f"{f}_right") for f in fields}
            for f, aa in aliases.items():
                for a in aa:
                    if f"{a}_left" in row_i and lrow.get(f) is None: lrow[f] = row_i.get(f"{a}_left")
                    if f"{a}_right" in row_i and rrow.get(f) is None: rrow[f] = row_i.get(f"{a}_right")
        if lrow is None or rrow is None: continue
        valid[i] = True
        for j, f in enumerate(fields):
            v_l = get_field(lrow, f); v_r = get_field(rrow, f)
            p1, p2 = _to_price(v_l), _to_price(v_r)
            t1, t2 = _norm_text(v_l), _norm_text(v_r)
            if p1 is not None and p2 is not None:
                denom = max(abs(p1), abs(p2), 1e-6)
                X[i, j] = max(0.0, 1.0 - abs(p1 - p2) / denom)
            elif t1 and t2 and t1.lower() == t2.lower():
                X[i, j] = 1.0
            elif not t1 and not t2:
                X[i, j] = 0.0
            else:
                X[i, j] = _jaccard(_tokens(v_l), _tokens(v_r))
        li = left_id_to_idx.get(lid); ri = right_id_to_idx.get(rid)
        if li is not None and ri is not None:
            a = left_emb[li:li+1]; b = right_emb[ri:ri+1]
            X[i, base_dim] = float(_cosine_rows(a, b)[0])

    return X, valid


def _drop_embedding_column(X: np.ndarray) -> np.ndarray:
    """Drop the last column (embedding cosine) — used for benchmarks where the
    test entities aren't in the pre-computed embedding pool."""
    return X[:, :-1]


def _load_jsonl_gz(p: Path) -> pd.DataFrame:
    """Handle both JSONL (gzipped or plain) and DataFrame-as-dict formats."""
    p = Path(p)
    opener = gzip.open if str(p).endswith(".gz") else open
    with opener(p, "rt") as f:
        data = f.read().strip()
    # Try DataFrame-as-dict (single JSON object with columns as keys)
    try:
        d = json.loads(data)
        if isinstance(d, dict) and "id_left" in d and isinstance(d.get("id_left"), dict):
            return pd.DataFrame(d)
    except json.JSONDecodeError:
        pass
    # Fall back to JSONL (one row per line)
    rows = [json.loads(l) for l in data.splitlines() if l.strip()]
    return pd.DataFrame(rows)


def prepare_benchmark(benchmark):
    """Load seed-independent inputs and test features once per benchmark."""
    cfg = BENCHMARK_CONFIGS[benchmark]
    side_l = _load_canonical_csv(cfg["left_csv"])
    side_r = _load_canonical_csv(cfg["right_csv"])
    feature_args = (
        cfg, side_l, side_r, np.load(cfg["left_emb"]), np.load(cfg["right_emb"]),
        {str(rid): i for i, rid in enumerate(side_l["id"].tolist())},
        {str(rid): i for i, rid in enumerate(side_r["id"].tolist())},
    )
    # As for WDC: if test records lie outside the embedding pool, drop the cosine feature
    # in both train and test instead of introducing a shift.
    test_frame = _load_jsonl_gz(cfg["test"])
    if "id_left" in test_frame.columns:
        test_ids = zip(test_frame["id_left"].astype(str), test_frame["id_right"].astype(str))
    else:
        test_ids = (re.split(r"#|__", pid)[:2] for pid in test_frame["pair_id"].astype(str))
    if any(a not in feature_args[5] or b not in feature_args[6] for a, b in test_ids):
        DROP_EMBEDDING_FEATURE.add(benchmark)
    test = prepare_features(cfg["test"], feature_args, benchmark)
    return feature_args, test


def prepare_features(path, feature_args, benchmark):
    frame = _load_jsonl_gz(path)
    features, valid = _build_features(frame, *feature_args)
    labels = (frame["label"].astype(int) == 1).astype(int).to_numpy()
    if (~valid).any():
        print(f"  skipped {(~valid).sum()} rows with missing canonical match: {path}")
    features, labels = features[valid], labels[valid]
    if benchmark in DROP_EMBEDDING_FEATURE:
        features = _drop_embedding_column(features)
    return features, labels


def train_eval_one(benchmark, method, train_file, seed, models_to_train, prepared_data=None):
    print(f"\n=== {benchmark} × {method}  seed={seed} ===")
    print(f"  train: {train_file}")
    if prepared_data is None:
        feature_args, test = prepare_benchmark(benchmark)
        prepared_data = (*prepare_features(train_file, feature_args, benchmark), *test)
    X_train, y_train, X_test, y_test_eval = prepared_data
    print(f"  train rows: {len(X_train)}, test rows: {len(X_test)}")

    results = {"benchmark": benchmark, "method": method, "seed": seed, "n_train": int(len(X_train)), "n_test": int(len(X_test)), "models": {}}
    n_pos = int(y_train.sum()); n_neg = int(len(y_train) - n_pos)
    spw = float(n_neg) / max(n_pos, 1)
    results["pos_rate"] = float(n_pos) / max(len(y_train), 1)
    results["scale_pos_weight"] = spw
    for model_name in models_to_train:
        t0 = time.perf_counter()
        if model_name == "xgboost":
            # Per-training-set scale_pos_weight = n_neg/n_pos. Balances minority
            # class proportionally — moderate (3-9) on AL sets, larger (~5-10) on
            # imbalanced official benchmark train sets where it matters most.
            clf = xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                random_state=seed, n_jobs=-1,
                subsample=0.8, colsample_bytree=0.8,  # seed-dependent randomness
                eval_metric="logloss", tree_method="hist",
                scale_pos_weight=spw,
            )
        elif model_name == "random_forest":
            clf = RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced", n_jobs=-1)
        else:
            raise ValueError(model_name)
        clf.fit(X_train, y_train)
        fit_t = time.perf_counter() - t0

        # eval
        t1 = time.perf_counter()
        probs = clf.predict_proba(X_test)[:, 1]
        infer_t = time.perf_counter() - t1
        preds = (probs >= 0.5).astype(int)
        f1 = float(f1_score(y_test_eval, preds, zero_division=0))
        p  = float(precision_score(y_test_eval, preds, zero_division=0))
        r  = float(recall_score(y_test_eval, preds, zero_division=0))
        acc = float(accuracy_score(y_test_eval, preds))
        results["models"][model_name] = {
            "f1": f1, "precision": p, "recall": r, "accuracy": acc,
            "fit_time_s": float(fit_t), "infer_time_s": float(infer_t),
            "tp": int(((preds == 1) & (y_test_eval == 1)).sum()),
            "fp": int(((preds == 1) & (y_test_eval == 0)).sum()),
            "fn": int(((preds == 0) & (y_test_eval == 1)).sum()),
            "tn": int(((preds == 0) & (y_test_eval == 0)).sum()),
        }
        print(f"  {model_name:14s} F1={f1:.4f}  P={p:.4f}  R={r:.4f}  fit={fit_t:.1f}s  infer={infer_t:.2f}s")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--models", default="xgboost", help="Comma-separated models; random_forest is opt-in")
    parser.add_argument("--combos", default="", help="comma-separated 'bm:method' filter (default: all benchmarks and sources)")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.combos:
        wanted = set(tuple(x.split(":")) for x in args.combos.split(","))
        combos = [k for k in SOURCES if k in wanted]
    else:
        combos = list(SOURCES.keys())

    OUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    # Keep only one benchmark's embeddings and one source's train matrix in
    # memory. Every seed reuses exactly these arrays; no persistent stale cache.
    for bm in dict.fromkeys(bm for bm, _ in combos):
        try:
            feature_args, test = prepare_benchmark(bm)
        except Exception as e:
            print(f"  FAILED {bm}: {e!r}")
            continue
        for _, method in (combo for combo in combos if combo[0] == bm):
            train_file = SOURCES[(bm, method)]
            try:
                prepared_data = (*prepare_features(train_file, feature_args, bm), *test)
            except Exception as e:
                print(f"  FAILED {bm}:{method}: {e!r}")
                continue
            for seed in seeds:
                try:
                    res = train_eval_one(bm, method, train_file, seed, models, prepared_data)
                    out_p = OUT_RESULTS_DIR / f"{bm}_{method}_seed{seed}.json"
                    out_p.write_text(json.dumps(res, indent=2))
                    all_results.append(res)
                except Exception as e:
                    print(f"  FAILED: {e!r}")

    # Aggregate to CSV
    rows = []
    for r in all_results:
        for m_name, m_res in r["models"].items():
            rows.append({
                "benchmark": r["benchmark"], "method": r["method"], "model": m_name,
                "seed": r["seed"], "n_train": r["n_train"], "n_test": r["n_test"],
                "f1": m_res["f1"], "precision": m_res["precision"], "recall": m_res["recall"],
                "accuracy": m_res["accuracy"],
                "fit_time_s": m_res["fit_time_s"], "infer_time_s": m_res["infer_time_s"],
            })
    if rows:
        df = pd.DataFrame(rows)
        out_csv = ROOT / "output/results_summary/traditional_students_raw.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"\nWrote raw: {out_csv.relative_to(ROOT)}")
        # mean+std per (benchmark, method, model)
        agg = df.groupby(["benchmark", "method", "model"]).agg(
            n=("f1", "count"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", lambda x: float(x.std()) if len(x) > 1 else 0.0),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
        ).reset_index()
        agg_csv = ROOT / "output/results_summary/traditional_students.csv"
        agg.to_csv(agg_csv, index=False)
        print(f"Wrote summary: {agg_csv.relative_to(ROOT)}")
        print()
        print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
