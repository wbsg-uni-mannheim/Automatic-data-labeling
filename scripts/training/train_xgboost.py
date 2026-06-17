#!/usr/bin/env python3
"""Train traditional ML student models (XGBoost + RandomForest) on each
(benchmark × selection-method) training set at ±5% benchmark size, evaluate
on the official test set.

Features per pair: per-field token-jaccard/exact-match/numeric-diff +
embedding cosine similarity (same feature extraction as AL ML pipeline in
scripts/labeling/active_learning_ml.py:_build_feature_matrix).

Models:
  - XGBoostClassifier
  - RandomForestClassifier (300 trees, class_weight=balanced)

Output:
  output/results_summary/traditional_students_<seed>.csv
  output/traditional_students/<benchmark>_<method>_<model>_seed<seed>.json
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

ROOT = Path("/work/aasteine/Automatic-data-labeling")
OUT_RESULTS_DIR = ROOT / "output/traditional_students"
OUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Train file paths.
#   3 selection methods (sim, alml, alditto) at ±5% benchmark size
#   5 post-filter variants (applied to AL Ditto)
SOURCES = {
    # ─── selection methods ───
    ("abt-buy",        "sim"):     ROOT / "output/learning_curve_abtbuy/similarity_selection/N6000/abt-buy_N6000_train.json.gz",
    ("abt-buy",        "alml"):    ROOT / "output/learning_curve_abtbuy/simple_active_learning/N6000/abt-buy_N6000_train.json.gz",
    ("abt-buy",        "alditto"): ROOT / "output/learning_curve_abtbuy/ditto_active_learning/N6000/abt-buy_N6000_train.json.gz",
    ("walmart-amazon", "sim"):     ROOT / "output/benchmark_size_runs/sim/walmart-amazon/walmart-amazon_train.json.gz",
    ("walmart-amazon", "alml"):    ROOT / "output/simple_active_learning_labeling/benchmark_walmart-amazon_20260302_113733/profiles/large/active_labels_latest_walmart-amazon_large_train.json.gz",
    ("walmart-amazon", "alditto"): ROOT / "output/three_phase_labeling_ditto_only_v2/benchmark_walmart-amazon_20260323_202820/profiles/all/active_labels_latest_walmart-amazon_all_train.json.gz",
    ("dblp-acm",       "sim"):     ROOT / "output/benchmark_size_runs/sim/dblp-acm/dblp-acm_train.json.gz",
    ("dblp-acm",       "alml"):    ROOT / "output/benchmark_size_runs/al_ml/dblp-acm/dblp-acm_train.json.gz",
    ("dblp-acm",       "alditto"): ROOT / "output/three_phase_labeling_ditto_only_v2/benchmark_dblp-acm_20260323_202820/profiles/all_plus20random/active_labels_latest_dblp-acm_all_plus20random_train.json.gz",
    ("dblp-scholar",   "sim"):     ROOT / "output/benchmark_size_runs/sim/dblp-scholar/dblp-scholar_train.json.gz",
    ("dblp-scholar",   "alml"):    ROOT / "output/benchmark_size_runs/al_ml/dblp-scholar/dblp-scholar_train.json.gz",
    ("dblp-scholar",   "alditto"): ROOT / "output/three_phase_labeling_ditto_only_v2/benchmark_dblp-scholar_20260323_202820/profiles/large/active_labels_latest_dblp-scholar_large_train.json.gz",
    ("wdc",            "sim"):     ROOT / "output/seed_round_only_profiles/benchmark_wdc_20260415_190530/profiles/all/active_labels_latest_wdc_all_train.json.gz",
    ("wdc",            "alml"):    ROOT / "output/simple_active_learning_labeling/benchmark_wdc_20260413_152105/profiles/large/active_labels_latest_wdc_large_train.json.gz",
    ("wdc",            "alditto"): ROOT / "output/benchmark_size_runs/al_ditto/wdc/wdc_train.json.gz",
}
# Add 5 post-filter variants per benchmark (all 5 benchmarks)
for bm in ["abt-buy", "walmart-amazon", "dblp-acm", "dblp-scholar", "wdc"]:
    for variant in ["v_relabel", "v_relabel_drop", "v_closure_drop", "v_closure_and_relabel", "v_closure_or_relabel"]:
        SOURCES[(bm, variant)] = ROOT / f"output/postfilter_variants/{bm}/{variant}/train.json.gz"

# Official benchmark train sets (paper baselines) — each released with the benchmark.
SOURCES[("abt-buy",        "benchmark")] = ROOT / "benchmarks/abt-buy/abt-buy-train.json"
SOURCES[("walmart-amazon", "benchmark")] = ROOT / "benchmarks/walmart-amazon/walmart-amazon-train.json.gz"
SOURCES[("dblp-acm",       "benchmark")] = ROOT / "benchmarks/dblp-acm/dblp-acm-train.json.gz"
SOURCES[("dblp-scholar",   "benchmark")] = ROOT / "benchmarks/dblp-scholar/dblp-scholar-train.json.gz"
SOURCES[("wdc",            "benchmark")] = ROOT / "benchmarks/wdc/wdcproducts80cc20rnd000un_train_large.json.gz"

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


def train_eval_one(benchmark, method, train_file, seed, models_to_train):
    print(f"\n=== {benchmark} × {method}  seed={seed} ===")
    print(f"  train: {train_file.relative_to(ROOT)}")
    bm_cfg = BENCHMARK_CONFIGS[benchmark]

    side_l = _load_canonical_csv(bm_cfg["left_csv"])
    side_r = _load_canonical_csv(bm_cfg["right_csv"])
    left_id_to_idx = {str(rid): i for i, rid in enumerate(side_l["id"].tolist())}
    right_id_to_idx = {str(rid): i for i, rid in enumerate(side_r["id"].tolist())}
    left_emb = np.load(bm_cfg["left_emb"])
    right_emb = np.load(bm_cfg["right_emb"])

    # Train data
    train_df = _load_jsonl_gz(train_file)
    print(f"  train rows: {len(train_df)}")
    X_train, m_train = _build_features(train_df, bm_cfg, side_l, side_r, left_emb, right_emb, left_id_to_idx, right_id_to_idx)
    y_train = (train_df["label"].astype(int) == 1).astype(int).to_numpy()
    if (~m_train).any():
        print(f"  ⚠ skipped {(~m_train).sum()} train rows with missing canonical match")
    X_train = X_train[m_train]; y_train = y_train[m_train]

    # Test data
    test_df = _load_jsonl_gz(bm_cfg["test"])
    print(f"  test rows: {len(test_df)}")
    X_test, m_test = _build_features(test_df, bm_cfg, side_l, side_r, left_emb, right_emb, left_id_to_idx, right_id_to_idx)
    y_test = (test_df["label"].astype(int) == 1).astype(int).to_numpy()
    if (~m_test).any():
        print(f"  ⚠ skipped {(~m_test).sum()} test rows with missing canonical match")
    X_test = X_test[m_test]; y_test_eval = y_test[m_test]

    # Drop embedding feature for benchmarks where test entities aren't in pre-computed pool
    if benchmark in DROP_EMBEDDING_FEATURE:
        X_train = _drop_embedding_column(X_train)
        X_test = _drop_embedding_column(X_test)
        print(f"  → dropped embedding feature (using {X_train.shape[1]} per-field features only)")

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
    parser.add_argument("--models", default="xgboost,random_forest")
    parser.add_argument("--combos", default="", help="comma-separated 'bm:method' filter (default all 15)")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.combos:
        wanted = set(tuple(x.split(":")) for x in args.combos.split(","))
        combos = [k for k in SOURCES if k in wanted]
    else:
        combos = list(SOURCES.keys())

    all_results = []
    for (bm, method) in combos:
        for seed in seeds:
            try:
                res = train_eval_one(bm, method, SOURCES[(bm, method)], seed, models)
                # Save per-run JSON
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
