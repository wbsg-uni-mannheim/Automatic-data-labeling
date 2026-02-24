#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:
    XGBClassifier = None  # type: ignore

CANONICAL_SCHEMA_FIELDS: Tuple[str, ...] = (
    "id",
    "title",
    "brand",
    "description",
    "price",
    "priceCurrency",
)


def _norm_text(v: object) -> str:
    if v is None:
        return ""
    return re.sub(r"\s+", " ", str(v)).strip()


def _tokens(v: object) -> set[str]:
    s = _norm_text(v).lower()
    return set(re.findall(r"[a-z0-9]+", s))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter / union) if union else 0.0


def _to_price(v: object) -> float | None:
    if v is None:
        return None
    s = _norm_text(v).replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_n = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return (a_n * b_n).sum(axis=1)


def _json_default(v: object) -> object:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, Path):
        return str(v)
    raise TypeError(f"Object of type {type(v).__name__} is not JSON serializable")


def _save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _parse_schema_map_arg(raw: str | None, side: str) -> Dict[str, str]:
    if raw is None:
        return {}
    raw = str(raw).strip()
    if not raw:
        return {}
    candidate = Path(raw)
    payload = candidate.read_text() if candidate.exists() else raw
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid --{side}-schema-map; expected JSON object or JSON file path. "
            f"Input was: {raw}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"--{side}-schema-map must decode to an object, got {type(parsed).__name__}")
    out: Dict[str, str] = {}
    for k, v in parsed.items():
        key = str(k).strip()
        if key not in CANONICAL_SCHEMA_FIELDS:
            raise ValueError(
                f"--{side}-schema-map contains unsupported canonical field '{key}'. "
                f"Allowed: {list(CANONICAL_SCHEMA_FIELDS)}"
            )
        val = str(v).strip() if v is not None else ""
        if val:
            out[key] = val
    return out


def _load_df(path: Path, side: str, schema_map: Dict[str, str], strict_schema: bool = False) -> pd.DataFrame:
    src = pd.read_csv(path)
    out = pd.DataFrame(index=src.index)
    missing_required: List[str] = []
    for canonical in CANONICAL_SCHEMA_FIELDS:
        source_col = str(schema_map.get(canonical, canonical))
        if source_col in src.columns:
            out[canonical] = src[source_col]
            continue
        if canonical == "id" or strict_schema:
            missing_required.append(f"{canonical}<-{source_col}")
            continue
        out[canonical] = ""
    if missing_required:
        raise ValueError(
            f"Missing required columns in {path}: {missing_required}. "
            f"Provided schema_map={schema_map}"
        )
    out = out.reset_index(drop=True).copy()
    out["id"] = out["id"].astype(str)
    # Use stable row ids internally because source ids can repeat.
    out["__rid"] = [f"{side}:{i}" for i in range(len(out))]
    return out


def _build_candidates(
    left_ids: np.ndarray,
    right_ids: np.ndarray,
    right_source_ids: np.ndarray,
    left_emb: np.ndarray,
    right_emb: np.ndarray,
    k: int,
    candidate_cap: int,
    bottom_k: int,
    random_state: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    k = max(1, int(k))
    bottom_k = max(0, int(bottom_k))
    dim = left_emb.shape[1]
    rng = np.random.RandomState(int(random_state))

    left = np.nan_to_num(left_emb.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    right = np.nan_to_num(right_emb.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)

    # Query LEFT entities against RIGHT index (matches original pipeline direction).
    if candidate_cap > 0:
        max_queries = min(len(left_ids), int(math.ceil(candidate_cap / max(1, k))))
    else:
        max_queries = len(left_ids)
    q_left = left[:max_queries]
    q_left_ids = left_ids[:max_queries]

    fetch_k = int(min(len(right_ids), max(k + 50, len(right_ids))))
    top_k = max(0, k - bottom_k)
    if fetch_k <= 0 or len(q_left_ids) == 0:
        empty = pd.DataFrame(columns=["id1", "id2", "similarity"])
        stats = {
            "faiss_queries": int(max_queries),
            "query_side": "left",
            "neighbor_side": "right",
            "faiss_k": int(k),
            "faiss_top_k": int(top_k),
            "faiss_bottom_k": int(bottom_k),
            "faiss_random_state": int(random_state),
            "raw_pairs": 0,
            "unique_pairs_before_cap": 0,
            "unique_pairs_after_cap": 0,
        }
        return empty, stats

    if faiss is not None:
        right_n = right / np.clip(np.linalg.norm(right, axis=1, keepdims=True), 1e-12, None)
        q_left_n = q_left / np.clip(np.linalg.norm(q_left, axis=1, keepdims=True), 1e-12, None)
        index = faiss.IndexFlatIP(dim)
        index.add(np.ascontiguousarray(right_n))
        sims, idxs = index.search(np.ascontiguousarray(q_left_n), fetch_k)
    else:
        right_n = right / np.clip(np.linalg.norm(right, axis=1, keepdims=True), 1e-12, None)
        q_left_n = q_left / np.clip(np.linalg.norm(q_left, axis=1, keepdims=True), 1e-12, None)
        nn = NearestNeighbors(n_neighbors=fetch_k, metric="cosine", algorithm="auto")
        nn.fit(right_n)
        dists, idxs = nn.kneighbors(q_left_n, n_neighbors=fetch_k, return_distance=True)
        sims = 1.0 - dists

    rows: List[Tuple[str, str, float]] = []
    dedup_dropped = 0
    for l_i in range(idxs.shape[0]):
        l_id = str(q_left_ids[l_i])
        row_idx = idxs[l_i]
        row_sim = sims[l_i]
        valid_mask = row_idx >= 0
        row_idx = row_idx[valid_mask]
        row_sim = row_sim[valid_mask]
        n_valid = len(row_idx)
        if n_valid == 0:
            continue

        selected_indices: List[int] = []
        selected_sims: List[float] = []
        if n_valid >= k and n_valid > top_k + bottom_k:
            if top_k > 0:
                selected_indices.extend(row_idx[:top_k].tolist())
                selected_sims.extend(row_sim[:top_k].astype(float).tolist())

            bottom_half_start = max(top_k, n_valid // 2)
            bottom_pool_idx = row_idx[bottom_half_start:]
            bottom_pool_sim = row_sim[bottom_half_start:]
            if len(bottom_pool_idx) >= bottom_k:
                choice = rng.choice(len(bottom_pool_idx), size=bottom_k, replace=False)
                selected_indices.extend(bottom_pool_idx[choice].tolist())
                selected_sims.extend(bottom_pool_sim[choice].astype(float).tolist())
            else:
                selected_indices.extend(bottom_pool_idx.tolist())
                selected_sims.extend(bottom_pool_sim.astype(float).tolist())
        else:
            actual_k = min(n_valid, k)
            selected_indices.extend(row_idx[:actual_k].tolist())
            selected_sims.extend(row_sim[:actual_k].astype(float).tolist())

        # Always dedupe right entities by source id per query.
        unique_selected: List[Tuple[int, float]] = []
        seen_right_source: set[str] = set()
        for r_i, sim in zip(selected_indices, selected_sims):
            src_id = str(right_source_ids[int(r_i)])
            if src_id in seen_right_source:
                dedup_dropped += 1
                continue
            seen_right_source.add(src_id)
            unique_selected.append((int(r_i), float(sim)))

        # Backfill with next-best neighbors to keep up to k unique right source ids.
        if len(unique_selected) < k:
            for r_i, sim in zip(row_idx.tolist(), row_sim.astype(float).tolist()):
                src_id = str(right_source_ids[int(r_i)])
                if src_id in seen_right_source:
                    continue
                seen_right_source.add(src_id)
                unique_selected.append((int(r_i), float(sim)))
                if len(unique_selected) >= k:
                    break

        for r_i, sim in unique_selected[:k]:
            r_id = str(right_ids[int(r_i)])
            rows.append((l_id, r_id, float(sim)))

    c = pd.DataFrame(rows, columns=["id1", "id2", "similarity"])
    c = c.reset_index(drop=True)
    before_cap = len(c)
    if candidate_cap > 0 and len(c) > candidate_cap:
        c = c.head(candidate_cap).reset_index(drop=True)
    stats = {
        "faiss_queries": int(max_queries),
        "query_side": "left",
        "neighbor_side": "right",
        "faiss_k": int(k),
        "faiss_top_k": int(top_k),
        "faiss_bottom_k": int(bottom_k),
        "faiss_random_state": int(random_state),
        "raw_pairs": int(len(rows)),
        "dedup_dropped_within_query": int(dedup_dropped),
        "unique_pairs_before_cap": int(before_cap),
        "unique_pairs_after_cap": int(len(c)),
    }
    return c, stats


def _label_pair(
    client: OpenAI,
    model: str,
    left: Dict[str, object],
    right: Dict[str, object],
) -> Tuple[str, Dict[str, int]]:
    def _serialize_record(rec: Dict[str, object], max_len: int = 200) -> Dict[str, str]:
        fields = ["title", "brand", "description", "price", "priceCurrency"]
        out: Dict[str, str] = {}
        for f in fields:
            if f in rec and rec[f] is not None:
                val = _norm_text(rec[f])
                if not val:
                    continue
                if len(val) > max_len:
                    val = val[:max_len] + "..."
                out[f] = val
        return out

    def _extract_json(text: str) -> Dict[str, object]:
        raw = text.strip()
        # First try direct parse.
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        # Fallback: extract first JSON object span.
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            frag = raw[start : end + 1]
            parsed = json.loads(frag)
            if isinstance(parsed, dict):
                return parsed
        raise ValueError(f"Could not parse JSON response: {text!r}")

    def _parse_label(text: str) -> str:
        payload = _extract_json(text)
        if "match" not in payload:
            raise ValueError(f"Missing 'match' field in JSON response: {payload}")
        m = payload["match"]
        if isinstance(m, bool):
            return "TRUE" if m else "FALSE"
        if isinstance(m, (int, float)):
            return "TRUE" if int(m) != 0 else "FALSE"
        if isinstance(m, str):
            t = m.strip().upper()
            if t in {"TRUE", "FALSE"}:
                return t
        raise ValueError(f"Unsupported 'match' value in JSON response: {payload}")

    left_json = json.dumps(_serialize_record(left), ensure_ascii=False)
    right_json = json.dumps(_serialize_record(right), ensure_ascii=False)

    system_prompt = (
        "Return only valid JSON with exactly one field: "
        '{"match": true|false}.'
    )
    user_prompt = (
        "Do the two product descriptions refer to the same real-world product? "
        f"Entity 1: '{left_json}'. "
        f"Entity 2: '{right_json}'."
    )

    usage_payload = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(2):
        resp = client.chat.completions.create(model=model, messages=messages)
        usage = getattr(resp, "usage", None)
        usage_payload["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
        usage_payload["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
        usage_payload["total_tokens"] += int(getattr(usage, "total_tokens", 0) or 0)

        text = (resp.choices[0].message.content or "")
        try:
            return _parse_label(text), usage_payload
        except ValueError:
            if attempt == 0:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"{user_prompt}\n\n"
                            "Your previous output was invalid. "
                            'Return valid JSON with exactly one field: {"match": true|false}.'
                        ),
                    },
                ]
            else:
                raise

    raise RuntimeError("Unreachable label parsing state")


def _build_feature_matrix(
    pairs: pd.DataFrame,
    left_map: Dict[str, Dict[str, object]],
    right_map: Dict[str, Dict[str, object]],
    left_idx: Dict[str, int],
    right_idx: Dict[str, int],
    left_emb: np.ndarray,
    right_emb: np.ndarray,
    progress_desc: str | None = None,
) -> np.ndarray:
    n = len(pairs)
    X = np.zeros((n, 6), dtype=np.float32)

    left_rows = [left_map[str(i)] for i in pairs["id1"].tolist()]
    right_rows = [right_map[str(i)] for i in pairs["id2"].tolist()]

    title_l = [_tokens(r.get("title")) for r in left_rows]
    title_r = [_tokens(r.get("title")) for r in right_rows]
    brand_l = [_norm_text(r.get("brand")).lower() for r in left_rows]
    brand_r = [_norm_text(r.get("brand")).lower() for r in right_rows]
    desc_l = [_tokens(r.get("description")) for r in left_rows]
    desc_r = [_tokens(r.get("description")) for r in right_rows]
    price_l = [_to_price(r.get("price")) for r in left_rows]
    price_r = [_to_price(r.get("price")) for r in right_rows]
    curr_l = [_norm_text(r.get("priceCurrency")).upper() for r in left_rows]
    curr_r = [_norm_text(r.get("priceCurrency")).upper() for r in right_rows]

    row_iter: Iterable[int]
    if progress_desc and n >= 1000:
        row_iter = tqdm(range(n), desc=progress_desc, unit="row", leave=False)
    else:
        row_iter = range(n)

    for i in row_iter:
        X[i, 0] = _jaccard(title_l[i], title_r[i])
        X[i, 1] = 1.0 if brand_l[i] and brand_l[i] == brand_r[i] else 0.0
        X[i, 2] = _jaccard(desc_l[i], desc_r[i])
        p1, p2 = price_l[i], price_r[i]
        if p1 is None or p2 is None:
            X[i, 3] = 0.0
        else:
            denom = max(abs(p1), abs(p2), 1e-6)
            X[i, 3] = max(0.0, 1.0 - abs(p1 - p2) / denom)
        X[i, 4] = 1.0 if curr_l[i] and curr_l[i] == curr_r[i] else 0.0

    li = np.array([left_idx[str(x)] for x in pairs["id1"].tolist()], dtype=np.int64)
    ri = np.array([right_idx[str(x)] for x in pairs["id2"].tolist()], dtype=np.int64)
    X[:, 5] = _cosine_rows(left_emb[li], right_emb[ri]).astype(np.float32)
    return X


def _count_labels(df: pd.DataFrame) -> Tuple[int, int]:
    labels = df["label"].astype(str).str.upper().str.strip()
    pos = int((labels == "TRUE").sum())
    neg = int((labels == "FALSE").sum())
    return pos, neg


def _run_old_active_learning(
    *,
    model: str,
    run_dir: Path,
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    labeled: pd.DataFrame,
    seed: pd.DataFrame,
    candidates: pd.DataFrame,
    target_pos: int,
    target_neg: int,
    labels_per_iteration: int,
    active_candidates: int,
    max_iterations: int,
    max_total_labels_override: int | None = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    import sys

    # Prefer local legacy implementation in this repo.
    old_root = Path(__file__).resolve().parents[1] / "old"
    if old_root.exists() and str(old_root) not in sys.path:
        sys.path.insert(0, str(old_root))

    try:
        from langchain_openai import ChatOpenAI  # type: ignore
    except Exception as exc:
        raise RuntimeError("Legacy active learning requires langchain_openai.") from exc

    try:
        from PyDI.pipeline.labeled_set_generation import run_active_learning as run_active_learning_old  # type: ignore
    except Exception as exc:
        raise RuntimeError("Failed to import legacy PyDI active learning implementation.") from exc

    training_set = labeled[["id1", "id2", "label"]].copy()
    training_set["id1"] = training_set["id1"].astype(str)
    training_set["id2"] = training_set["id2"].astype(str)
    training_set["label"] = training_set["label"].astype(str).str.upper().str.strip()

    if seed.empty:
        raise RuntimeError("Legacy active learning requires seed labels to build a validation holdout.")

    seed_work = seed.copy()
    seed_work["id1"] = seed_work["id1"].astype(str)
    seed_work["id2"] = seed_work["id2"].astype(str)
    seed_work["label"] = seed_work["label"].astype(str).str.upper().str.strip()
    if "similarity" not in seed_work.columns:
        seed_work["similarity"] = 0.0
    seed_work = seed_work.drop_duplicates(subset=["id1", "id2"], keep="last").reset_index(drop=True)

    pos_seed = seed_work[seed_work["label"] == "TRUE"]
    neg_seed = seed_work[seed_work["label"] == "FALSE"]
    if pos_seed.empty or neg_seed.empty:
        raise RuntimeError("Seed labels must contain both positive and negative examples.")

    # Legacy optimizer requires a labeled validation set, and it removes train/val overlaps.
    val_n = min(20, len(seed_work) - 2) if len(seed_work) > 2 else 1
    val_n = max(2, val_n)
    pos_ratio = len(pos_seed) / len(seed_work)
    val_pos_n = max(1, min(len(pos_seed) - 1, int(round(val_n * pos_ratio))))
    val_neg_n = max(1, min(len(neg_seed) - 1, val_n - val_pos_n))
    val_pos_n = max(1, min(val_pos_n, val_n - 1))
    val_neg_n = max(1, min(val_neg_n, val_n - val_pos_n))

    validation_set = pd.concat(
        [
            pos_seed.sample(n=val_pos_n, random_state=42),
            neg_seed.sample(n=val_neg_n, random_state=42),
        ],
        ignore_index=True,
    ).sample(frac=1.0, random_state=42).reset_index(drop=True)
    validation_pairs = set(zip(validation_set["id1"].astype(str), validation_set["id2"].astype(str)))
    training_set = training_set[
        ~training_set.apply(lambda r: (str(r["id1"]), str(r["id2"])) in validation_pairs, axis=1)
    ].reset_index(drop=True)

    train_pos = int((training_set["label"] == "TRUE").sum())
    train_neg = int((training_set["label"] == "FALSE").sum())
    if training_set.empty or train_pos == 0 or train_neg == 0:
        raise RuntimeError("Training set became invalid after building validation holdout from seed labels.")

    legacy_candidates = candidates[["id1", "id2", "similarity"]].copy()
    legacy_candidates["id1"] = legacy_candidates["id1"].astype(str)
    legacy_candidates["id2"] = legacy_candidates["id2"].astype(str)

    if max_total_labels_override is None:
        target_total = int(target_pos + target_neg)
        labels_remaining = max(target_total - len(labeled), 0)
        max_total_labels = max(labels_remaining, int(labels_per_iteration))
    else:
        max_total_labels = int(max_total_labels_override)
        if max_total_labels <= 0:
            max_total_labels = int(max(labels_per_iteration, 1))

    chat_model = ChatOpenAI(model=model, temperature=0)
    augmented, _, summary = run_active_learning_old(
        df_left=left_df,
        df_right=right_df,
        left_name="left",
        right_name="right",
        training_set=training_set,
        validation_set=validation_set[["id1", "id2", "label"]].copy(),
        chat_model=chat_model,
        output_dir=run_dir,
        candidates=legacy_candidates,
        id_column="__rid",
        target_positives=int(target_pos),
        target_negatives=int(target_neg) if target_neg > 0 else None,
        max_total_labels=int(max_total_labels),
        labels_per_iteration=int(labels_per_iteration),
        max_candidates=int(active_candidates),
        max_iterations=int(max_iterations),
        label_batch_size=int(min(25, max(1, labels_per_iteration))),
    )

    # Keep holdout labels in the final label pool for parity with this script's original totals.
    if "similarity" not in augmented.columns:
        augmented["similarity"] = 0.0
    holdout = validation_set[["id1", "id2", "label", "similarity"]].copy()
    out = pd.concat([augmented, holdout], ignore_index=True)
    out = out.drop_duplicates(subset=["id1", "id2"], keep="first").reset_index(drop=True)
    out["id1"] = out["id1"].astype(str)
    out["id2"] = out["id2"].astype(str)
    out["label"] = out["label"].astype(str).str.upper().str.strip()
    out["label"] = out["label"].replace({"1": "TRUE", "0": "FALSE"})
    out = out[out["label"].isin(["TRUE", "FALSE"])].copy()
    out["similarity"] = pd.to_numeric(out["similarity"], errors="coerce").fillna(0.0).astype(float)
    out = out.drop_duplicates(subset=["id1", "id2"], keep="last").reset_index(drop=True)
    return out, (summary if isinstance(summary, dict) else {})


def _plan_adaptive_round(
    *,
    current_pos: int,
    current_neg: int,
    target_pos: int,
    target_neg: int,
    round_size: int,
    min_neg_share: float,
    max_neg_share: float,
) -> Dict[str, object]:
    d_pos = max(target_pos - current_pos, 0)
    d_neg = max(target_neg - current_neg, 0)
    deficit_total = d_pos + d_neg
    budget = min(max(int(round_size), 1), deficit_total) if deficit_total > 0 else 0
    if budget <= 0:
        return {
            "d_pos": int(d_pos),
            "d_neg": int(d_neg),
            "budget": 0,
            "raw_neg_share": 0.0,
            "neg_share": 0.0,
            "pos_quota": 0,
            "neg_quota": 0,
            "target_pos": int(current_pos),
            "target_neg": int(current_neg),
        }

    raw_neg_share = float(d_neg / deficit_total) if deficit_total > 0 else 0.5
    neg_share = float(min(max(raw_neg_share, min_neg_share), max_neg_share))

    neg_quota = int(round(budget * neg_share))
    neg_quota = min(max(neg_quota, 0), d_neg, budget)
    pos_quota = budget - neg_quota

    if pos_quota > d_pos:
        shift = pos_quota - d_pos
        pos_quota = d_pos
        neg_quota = min(d_neg, neg_quota + shift)

    if d_pos > 0 and budget >= 2 and pos_quota == 0:
        pos_quota = 1
        neg_quota = min(d_neg, budget - pos_quota)
    if d_neg > 0 and budget >= 2 and neg_quota == 0:
        neg_quota = 1
        pos_quota = min(d_pos, budget - neg_quota)

    used = pos_quota + neg_quota
    if used < budget:
        rem = budget - used
        add_neg = min(rem, max(d_neg - neg_quota, 0))
        neg_quota += add_neg
        rem -= add_neg
        add_pos = min(rem, max(d_pos - pos_quota, 0))
        pos_quota += add_pos

    round_target_pos = min(target_pos, current_pos + int(pos_quota))
    round_target_neg = min(target_neg, current_neg + int(neg_quota))
    return {
        "d_pos": int(d_pos),
        "d_neg": int(d_neg),
        "budget": int(budget),
        "raw_neg_share": float(raw_neg_share),
        "neg_share": float(neg_share),
        "pos_quota": int(pos_quota),
        "neg_quota": int(neg_quota),
        "target_pos": int(round_target_pos),
        "target_neg": int(round_target_neg),
    }


def _materialize_output_ids(
    df: pd.DataFrame,
    left_rid_to_id: Dict[str, str],
    right_rid_to_id: Dict[str, str],
) -> pd.DataFrame:
    out = df.copy()
    out["rid1"] = out["id1"].astype(str)
    out["rid2"] = out["id2"].astype(str)
    out["id1"] = out["rid1"].map(left_rid_to_id)
    out["id2"] = out["rid2"].map(right_rid_to_id)
    return out


def _label_batch_seed(
    client: OpenAI,
    model: str,
    pairs: pd.DataFrame,
    left_map: Dict[str, Dict[str, object]],
    right_map: Dict[str, Dict[str, object]],
    *,
    max_calls: int,
    counters: Dict[str, int],
    usage_stats: Dict[str, int],
    pbar,
) -> Tuple[pd.DataFrame, bool]:
    rows: List[Dict[str, object]] = []
    exhausted = False
    for _, row in pairs.iterrows():
        if counters["calls"] >= max_calls:
            exhausted = True
            break
        id1 = str(row["id1"])
        id2 = str(row["id2"])
        label, usage = _label_pair(client, model, left_map[id1], right_map[id2])
        counters["calls"] += 1
        pbar.update(1)

        usage_stats["prompt_tokens"] += usage["prompt_tokens"]
        usage_stats["completion_tokens"] += usage["completion_tokens"]
        usage_stats["total_tokens"] += usage["total_tokens"]
        usage_stats["seed_prompt_tokens"] += usage["prompt_tokens"]
        usage_stats["seed_completion_tokens"] += usage["completion_tokens"]
        usage_stats["seed_total_tokens"] += usage["total_tokens"]

        rows.append(
            {
                "id1": id1,
                "id2": id2,
                "label": label,
                "similarity": float(row["similarity"]),
            }
        )
    return pd.DataFrame(rows), exhausted


def _label_query_until_satisfied_seed(
    client: OpenAI,
    model: str,
    query_neighbors: pd.DataFrame,
    left_map: Dict[str, Dict[str, object]],
    right_map: Dict[str, Dict[str, object]],
    *,
    target_positives: int = 1,
    target_negatives: int = 4,
    batch_size: int = 5,
    bottom_k: int = 2,
    max_calls: int,
    counters: Dict[str, int],
    usage_stats: Dict[str, int],
    pbar,
) -> Tuple[pd.DataFrame, bool]:
    sorted_neighbors = query_neighbors.sort_values("similarity", ascending=False).reset_index(drop=True)
    candidates = pd.DataFrame(
        {
            "id1": sorted_neighbors["query_id"],
            "id2": sorted_neighbors["neighbor_id"],
            "similarity": sorted_neighbors["similarity"],
        }
    ).reset_index(drop=True)

    labeled_results: List[pd.DataFrame] = []
    labeled_indices: set[int] = set()
    n_positives = 0
    n_negatives_from_top = 0
    exhausted = False

    for i in range(0, len(candidates), batch_size):
        if n_positives >= target_positives and n_negatives_from_top >= target_negatives:
            break
        batch = candidates.iloc[i : i + batch_size]
        batch_indices = set(range(i, min(i + batch_size, len(candidates))))
        labeled, exhausted = _label_batch_seed(
            client,
            model,
            batch,
            left_map,
            right_map,
            max_calls=max_calls,
            counters=counters,
            usage_stats=usage_stats,
            pbar=pbar,
        )
        if not labeled.empty:
            labeled_results.append(labeled)
            labeled_indices.update(batch_indices)
            n_positives += int((labeled["label"] == "TRUE").sum())
            n_negatives_from_top += int((labeled["label"] == "FALSE").sum())
        if exhausted:
            break

    if exhausted:
        if not labeled_results:
            return pd.DataFrame(columns=["id1", "id2", "label", "similarity"]), True
        return pd.concat(labeled_results, ignore_index=True), True

    # Old pipeline rule: if we found 2+ positives, label remaining and keep all.
    if n_positives >= 2:
        unlabeled_indices = [i for i in range(len(candidates)) if i not in labeled_indices]
        if unlabeled_indices:
            remaining = candidates.iloc[unlabeled_indices]
            remaining_labeled, exhausted = _label_batch_seed(
                client,
                model,
                remaining,
                left_map,
                right_map,
                max_calls=max_calls,
                counters=counters,
                usage_stats=usage_stats,
                pbar=pbar,
            )
            if not remaining_labeled.empty:
                labeled_results.append(remaining_labeled)
        if not labeled_results:
            return pd.DataFrame(columns=["id1", "id2", "label", "similarity"]), exhausted
        return pd.concat(labeled_results, ignore_index=True), exhausted

    # Otherwise also label bottom_k and then sample negatives.
    if len(candidates) > bottom_k:
        bottom_indices = list(range(len(candidates) - bottom_k, len(candidates)))
        unlabeled_bottom = [idx for idx in bottom_indices if idx not in labeled_indices]
        if unlabeled_bottom:
            bottom = candidates.iloc[unlabeled_bottom]
            bottom_labeled, exhausted = _label_batch_seed(
                client,
                model,
                bottom,
                left_map,
                right_map,
                max_calls=max_calls,
                counters=counters,
                usage_stats=usage_stats,
                pbar=pbar,
            )
            if not bottom_labeled.empty:
                labeled_results.append(bottom_labeled)

    if not labeled_results:
        return pd.DataFrame(columns=["id1", "id2", "label", "similarity"]), exhausted

    all_labeled = pd.concat(labeled_results, ignore_index=True)
    positives = all_labeled[all_labeled["label"] == "TRUE"]
    negatives = all_labeled[all_labeled["label"] == "FALSE"]
    if len(negatives) > target_negatives:
        sampled_negatives = negatives.sample(n=target_negatives, random_state=42)
    else:
        sampled_negatives = negatives
    result = pd.concat([positives, sampled_negatives], ignore_index=True)
    return result, exhausted


def _select_balanced_set(labeled: pd.DataFrame, target_size: int, target_positives: int) -> pd.DataFrame:
    if labeled.empty:
        return labeled
    positives = labeled[labeled["label"].astype(str).str.upper() == "TRUE"]
    negatives = labeled[labeled["label"].astype(str).str.upper() == "FALSE"]
    n_pos = min(len(positives), target_positives)
    selected_pos = positives.head(n_pos)
    n_neg = min(len(negatives), max(target_size - n_pos, 0))
    selected_neg = negatives.head(n_neg)
    out = pd.concat([selected_pos, selected_neg], ignore_index=True)
    out = out.sample(frac=1, random_state=42).reset_index(drop=True)
    return out


def _model_scores(model: object, X: np.ndarray) -> np.ndarray:
    if callable(model):
        return np.asarray(model(X), dtype=np.float64)
    if hasattr(model, "predict_proba"):
        p = getattr(model, "predict_proba")(X)
        return np.asarray(p[:, 1], dtype=np.float64)
    if hasattr(model, "decision_function"):
        d = np.asarray(getattr(model, "decision_function")(X), dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-d))
    pred = np.asarray(getattr(model, "predict")(X), dtype=np.float64)
    return pred


def _fit_matcher_ensemble(
    labeled: pd.DataFrame,
    left_map: Dict[str, Dict[str, object]],
    right_map: Dict[str, Dict[str, object]],
    left_idx: Dict[str, int],
    right_idx: Dict[str, int],
    left_emb: np.ndarray,
    right_emb: np.ndarray,
    verbose: bool = False,
) -> List[Dict[str, object]]:
    X_train = _build_feature_matrix(
        labeled,
        left_map,
        right_map,
        left_idx,
        right_idx,
        left_emb,
        right_emb,
        progress_desc="AL features train",
    )
    y_train = (labeled["label"].astype(str).str.upper() == "TRUE").astype(int).to_numpy()
    if y_train.sum() == 0 or y_train.sum() == len(y_train):
        return []

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = float(neg / max(pos, 1))

    models: List[Tuple[str, object]] = [
        ("logreg", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1)),
        ("extra_trees", ExtraTreesClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1)),
        ("hist_gbdt", HistGradientBoostingClassifier(random_state=42)),
    ]
    if XGBClassifier is not None:
        models.append(
            (
                "xgboost",
                XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric="logloss",
                    scale_pos_weight=scale_pos_weight,
                    n_jobs=1,
                ),
            )
        )

    fitted: List[Dict[str, object]] = []

    # Rule-style scorers (to mirror old rule+ml ensemble behavior).
    rule_models: List[Tuple[str, object]] = [
        ("rule_title_brand", lambda X: 0.65 * X[:, 0] + 0.35 * X[:, 1]),
        ("rule_text_price", lambda X: 0.45 * X[:, 0] + 0.25 * X[:, 2] + 0.20 * X[:, 3] + 0.10 * X[:, 4]),
        ("rule_embed_text", lambda X: 0.55 * X[:, 5] + 0.30 * X[:, 0] + 0.15 * X[:, 1]),
    ]
    for name, scorer in rule_models:
        try:
            scores = _model_scores(scorer, X_train)
            pred = (scores >= 0.5).astype(int)
            f1 = float(f1_score(y_train, pred, zero_division=0))
            fitted.append({"matcher": name, "model": scorer, "f1": f1, "threshold": 0.5})
        except Exception:
            continue

    for name, model in models:
        try:
            t0 = time.perf_counter()
            if verbose:
                print(f"AL fit: {name}...", flush=True)
            model.fit(X_train, y_train)
            scores = _model_scores(model, X_train)
            pred = (scores >= 0.5).astype(int)
            f1 = float(f1_score(y_train, pred, zero_division=0))
            fitted.append({"matcher": name, "model": model, "f1": f1, "threshold": 0.5})
            if verbose:
                print(f"AL fit: {name} done in {time.perf_counter() - t0:.2f}s (f1={f1:.3f})", flush=True)
        except Exception:
            if verbose:
                print(f"AL fit: {name} failed", flush=True)
            continue
    fitted.sort(key=lambda x: float(x["f1"]), reverse=True)
    return fitted


def _run_matchers_on_pool(
    pool: pd.DataFrame,
    fitted_matchers: List[Dict[str, object]],
    left_map: Dict[str, Dict[str, object]],
    right_map: Dict[str, Dict[str, object]],
    left_idx: Dict[str, int],
    right_idx: Dict[str, int],
    left_emb: np.ndarray,
    right_emb: np.ndarray,
) -> List[Dict[str, object]]:
    if pool.empty or not fitted_matchers:
        return []
    X_pool = _build_feature_matrix(
        pool,
        left_map,
        right_map,
        left_idx,
        right_idx,
        left_emb,
        right_emb,
        progress_desc="AL features pool",
    )
    out: List[Dict[str, object]] = []
    for m in fitted_matchers:
        model = m["model"]
        scores = _model_scores(model, X_pool)
        corr = pd.DataFrame(
            {
                "id1": pool["id1"].astype(str).to_numpy(),
                "id2": pool["id2"].astype(str).to_numpy(),
                "score": scores,
            }
        )
        out.append(
            {
                "matcher": str(m["matcher"]),
                "f1": float(m["f1"]),
                "threshold": float(m["threshold"]),
                "correspondences": corr,
            }
        )
    return out


def _find_matcher_disagreements(
    correspondences_list: List[Dict[str, object]],
    top_n: int = 5,
) -> pd.DataFrame:
    valid = [
        r
        for r in correspondences_list
        if isinstance(r.get("correspondences"), pd.DataFrame) and not r["correspondences"].empty
    ]
    if len(valid) < 2:
        return pd.DataFrame(columns=["id1", "id2", "variance"])

    sorted_results = sorted(valid, key=lambda x: float(x.get("f1", 0.0)), reverse=True)[:top_n]
    pair_scores: Dict[Tuple[str, str], List[float]] = {}
    for result in sorted_results:
        corr = result["correspondences"].copy()
        corr["score"] = pd.to_numeric(corr["score"], errors="coerce")
        for _, row in corr.iterrows():
            score = row["score"]
            if pd.isna(score):
                continue
            key = (str(row["id1"]), str(row["id2"]))
            pair_scores.setdefault(key, []).append(float(score))

    rows: List[Dict[str, object]] = []
    for (id1, id2), scores in pair_scores.items():
        if len(scores) >= 2:
            rows.append({"id1": id1, "id2": id2, "variance": float(np.var(scores))})
    if not rows:
        return pd.DataFrame(columns=["id1", "id2", "variance"])
    return pd.DataFrame(rows).sort_values("variance", ascending=False).reset_index(drop=True)


def _label_iteratively_per_query_seed(
    client: OpenAI,
    model: str,
    all_neighbors: pd.DataFrame,
    left_map: Dict[str, Dict[str, object]],
    right_map: Dict[str, Dict[str, object]],
    *,
    target_positives_per_query: int,
    target_negatives_per_query: int,
    total_target_positives: int,
    total_target_size: int,
    max_calls: int,
    batch_size: int,
    query_order: str,
    bottom_k: int,
    state_path: Path,
    out_csv: Path,
    usage_stats: Dict[str, int],
) -> pd.DataFrame:
    all_labeled: List[pd.DataFrame] = []
    total_positives = 0
    total_negatives = 0
    target_total_negatives = max(int(total_target_size) - int(total_target_positives), 0)
    queries_processed = 0
    queries_with_matches = 0
    counters = {"calls": 0}

    query_groups = {qid: g for qid, g in all_neighbors.groupby("query_id")}
    if query_order == "similarity":
        query_max = all_neighbors.groupby("query_id")["similarity"].max()
        query_ids = query_max.sort_values(ascending=False).index.to_numpy().tolist()
    elif query_order == "random":
        rng = np.random.default_rng(42)
        query_ids = all_neighbors["query_id"].drop_duplicates().tolist()
        rng.shuffle(query_ids)
    else:
        # "left": preserve query order as generated.
        query_ids = all_neighbors["query_id"].drop_duplicates().tolist()

    pbar = tqdm(total=max_calls, desc="Seed labeling", unit="call")
    pbar.set_postfix(
        pos=0,
        neg=0,
        accepted=0,
        calls=f"0/{max_calls}",
        pos_target=total_target_positives,
        neg_target=target_total_negatives,
    )
    exhausted = False

    for qid in query_ids:
        if exhausted:
            break
        if total_positives >= total_target_positives and total_negatives >= target_total_negatives:
            break
        if counters["calls"] >= max_calls:
            break

        query_labeled, exhausted = _label_query_until_satisfied_seed(
            client,
            model,
            query_groups[qid],
            left_map,
            right_map,
            target_positives=target_positives_per_query,
            target_negatives=target_negatives_per_query,
            batch_size=batch_size,
            bottom_k=bottom_k,
            max_calls=max_calls,
            counters=counters,
            usage_stats=usage_stats,
            pbar=pbar,
        )
        queries_processed += 1

        if not query_labeled.empty:
            q_pos = int((query_labeled["label"] == "TRUE").sum())
            q_neg = int((query_labeled["label"] == "FALSE").sum())
            # Keep match-queries always; keep non-match queries until negative target is reached.
            keep_query = (q_pos > 0) or (total_negatives < target_total_negatives and q_neg > 0)
            if keep_query:
                all_labeled.append(query_labeled)
                total_positives += q_pos
                total_negatives += q_neg
                if q_pos > 0:
                    queries_with_matches += 1
        accepted = int(sum(len(x) for x in all_labeled))
        pbar.set_postfix(
            pos=total_positives,
            neg=total_negatives,
            accepted=accepted,
            calls=f"{counters['calls']}/{max_calls}",
            pos_target=total_target_positives,
            neg_target=target_total_negatives,
        )

        if counters["calls"] % 10 == 0:
            current = pd.concat(all_labeled, ignore_index=True) if all_labeled else pd.DataFrame(
                columns=["id1", "id2", "label", "similarity"]
            )
            current.to_csv(out_csv, index=False)
            _save_json(
                state_path,
                {
                    "stage": "seed_labeling",
                    "llm_calls": counters["calls"],
                    "seed_pos": total_positives,
                    "seed_neg": total_negatives,
                    "seed_total": len(current),
                    "seed_target_pos": int(total_target_positives),
                    "seed_target_neg": int(target_total_negatives),
                    "queries_processed": queries_processed,
                    "queries_with_matches": queries_with_matches,
                    "token_usage": usage_stats,
                },
            )

    pbar.close()
    if not all_labeled:
        return pd.DataFrame(columns=["id1", "id2", "label", "similarity"])

    labeled = pd.concat(all_labeled, ignore_index=True)
    labeled = _select_balanced_set(labeled, total_target_size, total_target_positives)
    labeled.to_csv(out_csv, index=False)
    _save_json(
        state_path,
        {
            "stage": "seed_done",
            "llm_calls": counters["calls"],
            "seed_pos": int((labeled["label"] == "TRUE").sum()),
            "seed_neg": int((labeled["label"] == "FALSE").sum()),
            "seed_total": int(len(labeled)),
            "seed_target_pos": int(total_target_positives),
            "seed_target_neg": int(target_total_negatives),
            "queries_processed": queries_processed,
            "queries_with_matches": queries_with_matches,
            "token_usage": usage_stats,
        },
    )
    return labeled


def _build_seed_queue(candidates: pd.DataFrame) -> List[Tuple[str, str, float]]:
    # Use generation order: for each LEFT entity, label its top-k RIGHT neighbors.
    c = candidates.reset_index(drop=True)
    return [(str(r["id1"]), str(r["id2"]), float(r["similarity"])) for _, r in c.iterrows()]


def _trim_exact(df: pd.DataFrame, pos_target: int, neg_target: int) -> pd.DataFrame:
    pos = df[df["label"].astype(str).str.upper() == "TRUE"]
    neg = df[df["label"].astype(str).str.upper() == "FALSE"]
    if len(pos) < pos_target or len(neg) < neg_target:
        return df
    out = pd.concat(
        [
            pos.sample(n=pos_target, random_state=42),
            neg.sample(n=neg_target, random_state=42),
        ],
        ignore_index=True,
    )
    return out.sample(frac=1.0, random_state=42).reset_index(drop=True)


def _build_preview(
    queue: Sequence[Tuple[str, str, float]],
    left_map: Dict[str, Dict[str, object]],
    right_map: Dict[str, Dict[str, object]],
    left_rid_to_id: Dict[str, str],
    right_rid_to_id: Dict[str, str],
    k: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for idx, (rid1, rid2, sim) in enumerate(queue[:k], start=1):
        l = left_map[rid1]
        r = right_map[rid2]
        rows.append(
            {
                "rank": idx,
                "rid1": rid1,
                "rid2": rid2,
                "id1": left_rid_to_id.get(rid1, ""),
                "id2": right_rid_to_id.get(rid2, ""),
                "similarity": float(sim),
                "left_title": _norm_text(l.get("title")),
                "right_title": _norm_text(r.get("title")),
                "left_brand": _norm_text(l.get("brand")),
                "right_brand": _norm_text(r.get("brand")),
                "left_price": _norm_text(l.get("price")),
                "right_price": _norm_text(r.get("price")),
                "left_currency": _norm_text(l.get("priceCurrency")),
                "right_currency": _norm_text(r.get("priceCurrency")),
            }
        )
    return pd.DataFrame(rows)


def _load_resume_labels(run_dir: Path) -> Tuple[pd.DataFrame | None, str | None]:
    candidates = [
        run_dir / "active_labels_latest.csv",
        run_dir / "labels_final.csv",
        run_dir / "seed_labels_internal.csv",
        run_dir / "seed_labels.csv",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty:
            continue
        work = df.copy()
        if {"rid1", "rid2"}.issubset(work.columns):
            work["id1"] = work["rid1"].astype(str)
            work["id2"] = work["rid2"].astype(str)
        if not {"id1", "id2", "label"}.issubset(work.columns):
            continue
        if "similarity" not in work.columns:
            work["similarity"] = 0.0
        keep_cols = ["id1", "id2", "label", "similarity"]
        extra_cols = [c for c in ["iteration", "al_source"] if c in work.columns]
        out = work[keep_cols + extra_cols].copy()
        out["id1"] = out["id1"].astype(str)
        out["id2"] = out["id2"].astype(str)
        out["label"] = out["label"].astype(str).str.upper()
        out["similarity"] = pd.to_numeric(out["similarity"], errors="coerce").fillna(0.0).astype(float)
        out = out.drop_duplicates(subset=["id1", "id2"], keep="last").reset_index(drop=True)
        return out, p.name
    return None, None


def _load_seed_for_summary(run_dir: Path) -> pd.DataFrame:
    for p in [run_dir / "seed_labels_internal.csv", run_dir / "seed_labels.csv"]:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty:
            continue
        work = df.copy()
        if {"rid1", "rid2"}.issubset(work.columns):
            work["id1"] = work["rid1"].astype(str)
            work["id2"] = work["rid2"].astype(str)
        if {"id1", "id2", "label"}.issubset(work.columns):
            if "similarity" not in work.columns:
                work["similarity"] = 0.0
            out = work[["id1", "id2", "label", "similarity"]].copy()
            out["id1"] = out["id1"].astype(str)
            out["id2"] = out["id2"].astype(str)
            out["label"] = out["label"].astype(str).str.upper()
            out["similarity"] = pd.to_numeric(out["similarity"], errors="coerce").fillna(0.0).astype(float)
            out = out.drop_duplicates(subset=["id1", "id2"], keep="last").reset_index(drop=True)
            return out
    return pd.DataFrame(columns=["id1", "id2", "label", "similarity"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal WDC labeling without PyDI.")
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--left-csv", default="data/wdc/wdc_train_large_left.csv")
    parser.add_argument("--right-csv", default="data/wdc/wdc_train_large_right.csv")
    parser.add_argument(
        "--left-schema-map",
        default="",
        help=(
            "JSON object or path to JSON mapping canonical fields "
            "(id,title,brand,description,price,priceCurrency) to source columns for the left CSV."
        ),
    )
    parser.add_argument(
        "--right-schema-map",
        default="",
        help=(
            "JSON object or path to JSON mapping canonical fields "
            "(id,title,brand,description,price,priceCurrency) to source columns for the right CSV."
        ),
    )
    parser.add_argument(
        "--strict-schema",
        action="store_true",
        help="Require all canonical fields to be present after schema mapping. By default only id is required.",
    )
    parser.add_argument("--left-emb", default="wdc_left_embeddings.npy")
    parser.add_argument("--right-emb", default="wdc_right_embeddings.npy")
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--faiss-k", type=int, default=20)
    parser.add_argument("--faiss-random-state", type=int, default=42)
    parser.add_argument("--candidate-cap", type=int, default=0, help="0 means no cap (query all left entities)")

    parser.add_argument("--seed-size", type=int, default=100)
    parser.add_argument("--seed-positives", type=int, default=30)
    parser.add_argument("--seed-max-calls", type=int, default=4000)
    parser.add_argument("--seed-pos-per-query", type=int, default=1)
    parser.add_argument("--seed-neg-per-query", type=int, default=4)
    parser.add_argument("--seed-batch-size", type=int, default=5)
    parser.add_argument("--seed-bottom-k", type=int, default=2)
    parser.add_argument(
        "--seed-query-order",
        type=str,
        default="random",
        choices=["similarity", "random", "left"],
        help="Old pipeline query ordering mode for seed labeling.",
    )

    parser.add_argument("--target-size", type=int, default=2500)
    parser.add_argument("--target-positives", type=int, default=500)
    parser.add_argument("--labels-per-iteration", type=int, default=100)
    parser.add_argument("--active-candidates", type=int, default=5000)
    parser.add_argument("--active-top-matchers", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--adaptive-round-size", type=int, default=0, help="0 uses --labels-per-iteration")
    parser.add_argument("--adaptive-neg-min-share", type=float, default=0.60)
    parser.add_argument("--adaptive-neg-max-share", type=float, default=0.90)
    parser.add_argument("--preview-k", type=int, default=0, help="Show first K comparisons and exit without labeling")

    parser.add_argument("--output-root", default="output/simple_labeling")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints in --run-name folder.")
    parser.add_argument("--no-enforce-exact-final", action="store_true")
    args = parser.parse_args()

    load_dotenv()

    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "run_state.json"
    prev_state = _load_json(state_path) if args.resume else {}
    prev_usage = prev_state.get("token_usage", {}) if isinstance(prev_state, dict) else {}

    left_schema_map = _parse_schema_map_arg(args.left_schema_map, side="left")
    right_schema_map = _parse_schema_map_arg(args.right_schema_map, side="right")
    left_df = _load_df(Path(args.left_csv), side="L", schema_map=left_schema_map, strict_schema=args.strict_schema)
    right_df = _load_df(Path(args.right_csv), side="R", schema_map=right_schema_map, strict_schema=args.strict_schema)

    emb_dir = Path(args.embeddings_dir)
    left_emb = np.load(emb_dir / args.left_emb).astype(np.float32)
    right_emb = np.load(emb_dir / args.right_emb).astype(np.float32)

    if len(left_df) != left_emb.shape[0]:
        raise ValueError(f"Left rows {len(left_df)} != left embedding rows {left_emb.shape[0]}")
    if len(right_df) != right_emb.shape[0]:
        raise ValueError(f"Right rows {len(right_df)} != right embedding rows {right_emb.shape[0]}")
    if left_emb.shape[1] != right_emb.shape[1]:
        raise ValueError("Left/right embedding dimensions differ")

    left_df = left_df.reset_index(drop=True)
    right_df = right_df.reset_index(drop=True)
    left_ids = left_df["__rid"].astype(str).to_numpy()
    right_ids = right_df["__rid"].astype(str).to_numpy()
    left_idx = {str(v): i for i, v in enumerate(left_ids.tolist())}
    right_idx = {str(v): i for i, v in enumerate(right_ids.tolist())}
    left_map = {str(r["__rid"]): r for r in left_df.to_dict("records")}
    right_map = {str(r["__rid"]): r for r in right_df.to_dict("records")}
    left_rid_to_id = {str(r["__rid"]): str(r["id"]) for r in left_df.to_dict("records")}
    right_rid_to_id = {str(r["__rid"]): str(r["id"]) for r in right_df.to_dict("records")}

    left_query_ids = left_df["__rid"].astype(str).to_numpy()
    right_index_ids = right_df["__rid"].astype(str).to_numpy()
    right_source_ids = right_df["id"].astype(str).to_numpy()
    left_query_emb = left_emb
    right_index_emb = right_emb

    _save_json(
        state_path,
        {
            "stage": "init",
            "left_rows": len(left_df),
            "right_rows": len(right_df),
            "left_query_rows": int(len(left_query_ids)),
            "right_index_rows": int(len(right_index_ids)),
            "embedding_dim": int(left_emb.shape[1]),
            "left_schema_map": left_schema_map,
            "right_schema_map": right_schema_map,
            "strict_schema": bool(args.strict_schema),
            "token_usage": {
                "prompt_tokens": int(prev_usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(prev_usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(prev_usage.get("total_tokens", 0) or 0),
                "seed_prompt_tokens": int(prev_usage.get("seed_prompt_tokens", 0) or 0),
                "seed_completion_tokens": int(prev_usage.get("seed_completion_tokens", 0) or 0),
                "seed_total_tokens": int(prev_usage.get("seed_total_tokens", 0) or 0),
                "active_prompt_tokens": int(prev_usage.get("active_prompt_tokens", 0) or 0),
                "active_completion_tokens": int(prev_usage.get("active_completion_tokens", 0) or 0),
                "active_total_tokens": int(prev_usage.get("active_total_tokens", 0) or 0),
            },
        },
    )
    usage_stats: Dict[str, int] = {
        "prompt_tokens": int(prev_usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(prev_usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(prev_usage.get("total_tokens", 0) or 0),
        "seed_prompt_tokens": int(prev_usage.get("seed_prompt_tokens", 0) or 0),
        "seed_completion_tokens": int(prev_usage.get("seed_completion_tokens", 0) or 0),
        "seed_total_tokens": int(prev_usage.get("seed_total_tokens", 0) or 0),
        "active_prompt_tokens": int(prev_usage.get("active_prompt_tokens", 0) or 0),
        "active_completion_tokens": int(prev_usage.get("active_completion_tokens", 0) or 0),
        "active_total_tokens": int(prev_usage.get("active_total_tokens", 0) or 0),
    }

    candidates, faiss_stats = _build_candidates(
        left_ids=left_query_ids,
        right_ids=right_index_ids,
        right_source_ids=right_source_ids,
        left_emb=left_query_emb,
        right_emb=right_index_emb,
        k=args.faiss_k,
        candidate_cap=args.candidate_cap,
        bottom_k=args.seed_bottom_k,
        random_state=args.faiss_random_state,
    )
    # Always dedupe by source-id pair before any labeling.
    candidates_dedup = candidates.copy()
    candidates_dedup["src_id1"] = candidates_dedup["id1"].astype(str).map(left_rid_to_id)
    candidates_dedup["src_id2"] = candidates_dedup["id2"].astype(str).map(right_rid_to_id)
    before_source_pair_dedup = len(candidates_dedup)
    candidates_dedup = candidates_dedup.drop_duplicates(subset=["src_id1", "src_id2"], keep="first")
    after_source_pair_dedup = len(candidates_dedup)
    candidates = candidates_dedup.drop(columns=["src_id1", "src_id2"]).reset_index(drop=True)
    faiss_stats["source_pair_dedup_before"] = int(before_source_pair_dedup)
    faiss_stats["source_pair_dedup_after"] = int(after_source_pair_dedup)
    faiss_stats["source_pair_dedup_dropped"] = int(before_source_pair_dedup - after_source_pair_dedup)
    candidates_with_ids = _materialize_output_ids(candidates, left_rid_to_id, right_rid_to_id)
    same_source_id_pairs = int((candidates_with_ids["id1"].astype(str) == candidates_with_ids["id2"].astype(str)).sum())
    faiss_stats["same_source_id_pairs"] = same_source_id_pairs
    faiss_stats["same_source_id_rate"] = (
        float(same_source_id_pairs / len(candidates_with_ids)) if len(candidates_with_ids) else 0.0
    )
    print(
        "FAISS queries: "
        f"{faiss_stats['faiss_queries']} (k={faiss_stats['faiss_k']}), "
        f"raw_pairs={faiss_stats['raw_pairs']}, "
        f"post_cap_pairs={faiss_stats['unique_pairs_after_cap']}, "
        f"source_pair_dedup={faiss_stats['source_pair_dedup_before']} -> {faiss_stats['source_pair_dedup_after']}, "
        f"left_rows={len(left_query_ids)}, right_rows={len(right_index_ids)}"
    )
    print(
        "Candidate ID overlap: "
        f"{same_source_id_pairs}/{len(candidates_with_ids)} "
        f"({faiss_stats['same_source_id_rate']:.2%}) with identical source ids"
    )
    candidates_with_ids.to_csv(run_dir / "faiss_candidates.csv", index=False)
    _save_json(
        state_path,
        {
            "stage": "candidates_done",
            "candidates": len(candidates),
            "faiss": faiss_stats,
        },
    )

    seed_pos_target = args.seed_positives
    seed_neg_target = args.seed_size - args.seed_positives
    if seed_neg_target < 0:
        raise ValueError("--seed-size must be >= --seed-positives")

    all_neighbors = candidates.rename(columns={"id1": "query_id", "id2": "neighbor_id"}).copy()
    all_neighbors["rank"] = (
        all_neighbors.groupby("query_id")["similarity"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    seed_queue = _build_seed_queue(candidates)
    if args.preview_k and args.preview_k > 0:
        preview = _build_preview(
            queue=seed_queue,
            left_map=left_map,
            right_map=right_map,
            left_rid_to_id=left_rid_to_id,
            right_rid_to_id=right_rid_to_id,
            k=args.preview_k,
        )
        preview_path = run_dir / "preview_first_k.csv"
        preview.to_csv(preview_path, index=False)
        _save_json(
            state_path,
            {
                "stage": "preview_done",
                "preview_k": int(args.preview_k),
                "preview_path": str(preview_path),
                "faiss": faiss_stats,
            },
        )
        print(f"Preview mode: showing first {len(preview)} comparisons (no labeling performed).")
        if not preview.empty:
            print(preview.head(min(len(preview), 20)).to_string(index=False))
        print(f"Saved preview: {preview_path}")
        return

    client = OpenAI()
    resumed_labeled: pd.DataFrame | None = None
    resumed_from: str | None = None
    if args.resume:
        resumed_labeled, resumed_from = _load_resume_labels(run_dir)
        if resumed_labeled is not None:
            rpos, rneg = _count_labels(resumed_labeled)
            print(f"Resume: loaded {len(resumed_labeled)} labels from {resumed_from} ({rpos} pos, {rneg} neg)")
        else:
            print("Resume: no checkpoint found, starting fresh.")

    if resumed_labeled is None:
        seed = _label_iteratively_per_query_seed(
            client=client,
            model=args.model,
            all_neighbors=all_neighbors,
            left_map=left_map,
            right_map=right_map,
            target_positives_per_query=args.seed_pos_per_query,
            target_negatives_per_query=args.seed_neg_per_query,
            total_target_positives=seed_pos_target,
            total_target_size=args.seed_size,
            max_calls=args.seed_max_calls,
            batch_size=args.seed_batch_size,
            query_order=args.seed_query_order,
            bottom_k=args.seed_bottom_k,
            state_path=state_path,
            out_csv=run_dir / "seed_labels_internal.csv",
            usage_stats=usage_stats,
        )
        if seed.empty:
            raise RuntimeError("Seed labeling produced zero rows")
        seed_pos, seed_neg = _count_labels(seed)
        _materialize_output_ids(seed, left_rid_to_id, right_rid_to_id).to_csv(run_dir / "seed_labels.csv", index=False)
        print(f"Seed: {len(seed)} ({seed_pos} pos, {seed_neg} neg)")
        labeled = seed.copy().reset_index(drop=True)
        labeled["label"] = labeled["label"].astype(str).str.upper()
    else:
        seed = _load_seed_for_summary(run_dir)
        seed_pos, seed_neg = _count_labels(seed) if not seed.empty else (0, 0)
        labeled = resumed_labeled.copy().reset_index(drop=True)
        labeled["label"] = labeled["label"].astype(str).str.upper()
        if resumed_from == "labels_final.csv":
            pos_done, neg_done = _count_labels(labeled)
            print(f"Run already finished: {len(labeled)} ({pos_done} pos, {neg_done} neg)")

    target_pos = args.target_positives
    target_neg = args.target_size - args.target_positives
    if target_neg < 0:
        raise ValueError("--target-size must be >= --target-positives")
    if not (0.0 <= args.adaptive_neg_min_share <= 1.0 and 0.0 <= args.adaptive_neg_max_share <= 1.0):
        raise ValueError("--adaptive-neg-min-share and --adaptive-neg-max-share must be in [0,1]")
    if args.adaptive_neg_min_share > args.adaptive_neg_max_share:
        raise ValueError("--adaptive-neg-min-share must be <= --adaptive-neg-max-share")
    adaptive_round_size = args.adaptive_round_size if args.adaptive_round_size > 0 else args.labels_per_iteration
    adaptive_round_size = max(1, int(adaptive_round_size))

    pos, neg = _count_labels(labeled)
    if pos < target_pos or neg < target_neg:
        print(
            f"AL(old): start total={len(labeled)} pos={pos} neg={neg} target={target_pos}/{target_neg}",
            flush=True,
        )
        al_t0 = time.perf_counter()
        legacy_rounds: List[Dict[str, object]] = []
        round_idx = 0
        while pos < target_pos or neg < target_neg:
            plan = _plan_adaptive_round(
                current_pos=pos,
                current_neg=neg,
                target_pos=target_pos,
                target_neg=target_neg,
                round_size=adaptive_round_size,
                min_neg_share=float(args.adaptive_neg_min_share),
                max_neg_share=float(args.adaptive_neg_max_share),
            )
            budget = int(plan["budget"])
            round_target_pos = int(plan["target_pos"])
            round_target_neg = int(plan["target_neg"])
            if budget <= 0 or (round_target_pos <= pos and round_target_neg <= neg):
                break

            round_idx += 1
            print(
                f"AL(old) round {round_idx}: deficits pos={plan['d_pos']} neg={plan['d_neg']}, "
                f"budget={budget}, mix pos={plan['pos_quota']} neg={plan['neg_quota']} "
                f"(neg_share={float(plan['neg_share']):.2f}, raw={float(plan['raw_neg_share']):.2f}), "
                f"targets={round_target_pos}/{round_target_neg}",
                flush=True,
            )

            prev_pos, prev_neg, prev_total = pos, neg, len(labeled)
            labeled, legacy_summary = _run_old_active_learning(
                model=args.model,
                run_dir=run_dir,
                left_df=left_df,
                right_df=right_df,
                labeled=labeled,
                seed=seed,
                candidates=candidates,
                target_pos=round_target_pos,
                target_neg=round_target_neg,
                labels_per_iteration=min(args.labels_per_iteration, budget),
                active_candidates=args.active_candidates,
                max_iterations=args.max_iterations,
                max_total_labels_override=budget,
            )
            pos, neg = _count_labels(labeled)
            round_gain = int(len(labeled) - prev_total)
            legacy_rounds.append(
                {
                    "round": int(round_idx),
                    "start_pos": int(prev_pos),
                    "start_neg": int(prev_neg),
                    "end_pos": int(pos),
                    "end_neg": int(neg),
                    "gain_total": int(round_gain),
                    "budget": int(budget),
                    "plan": plan,
                    "legacy_summary": legacy_summary,
                }
            )
            if pos <= prev_pos and neg <= prev_neg:
                print("AL(old): no class progress in this round, stopping early.", flush=True)
                break

        legacy_summary = {"rounds": legacy_rounds}
        pos, neg = _count_labels(labeled)
        _materialize_output_ids(labeled, left_rid_to_id, right_rid_to_id).to_csv(
            run_dir / "active_labels_latest.csv", index=False
        )
        _save_json(
            state_path,
            {
                "stage": "active_learning",
                "active_learning_impl": "old_pydi",
                "total": len(labeled),
                "pos": pos,
                "neg": neg,
                "target_pos": target_pos,
                "target_neg": target_neg,
                "active_candidates_used": int(min(len(candidates), args.active_candidates))
                if args.active_candidates > 0
                else int(len(candidates)),
                "legacy_summary": legacy_summary,
                "token_usage": usage_stats,
            },
        )
        al_dt = time.perf_counter() - al_t0
        print(f"AL(old): done total={len(labeled)} pos={pos} neg={neg} ({al_dt:.1f}s)", flush=True)

    final_df = labeled.copy()
    if not args.no_enforce_exact_final:
        final_df = _trim_exact(final_df, target_pos, target_neg)

    final_df = final_df.reset_index(drop=True)
    final_out = _materialize_output_ids(final_df, left_rid_to_id, right_rid_to_id)
    final_out.to_csv(run_dir / "labels_final.csv", index=False)
    f_pos, f_neg = _count_labels(final_df)
    _save_json(
        run_dir / "summary.json",
        {
            "seed_total": int(len(seed)),
            "seed_pos": int(seed_pos),
            "seed_neg": int(seed_neg),
            "final_total": int(len(final_out)),
            "final_pos": int(f_pos),
            "final_neg": int(f_neg),
            "target_total": int(args.target_size),
            "target_pos": int(target_pos),
            "target_neg": int(target_neg),
            "faiss": faiss_stats,
            "token_usage": usage_stats,
        },
    )
    _save_json(
        state_path,
        {
            "stage": "done",
            "final_total": len(final_df),
            "final_pos": f_pos,
            "final_neg": f_neg,
            "token_usage": usage_stats,
        },
    )
    print(f"Final: {len(final_out)} ({f_pos} pos, {f_neg} neg)")
    print(
        "Token usage: "
        f"prompt={usage_stats['prompt_tokens']}, "
        f"completion={usage_stats['completion_tokens']}, "
        f"total={usage_stats['total_tokens']}"
    )
    print(f"Output: {run_dir / 'labels_final.csv'}")


if __name__ == "__main__":
    main()
