"""Candidate-pool construction variants for the labeling workflows.

Three pools of identical size per left record (``top_k`` ranked + ``bottom_k``
random from the lower-ranked half), differing only in *which* candidates make
it in:

  embedding  cosine similarity over the stored record embeddings (the pool used
             throughout the paper; delegates to the original FAISS logic)
  bm25       BM25 over the whitespace/alnum tokens of the concatenated record
             fields (token-based blocking with a ranking)
  rrf        the two ranked lists fused by reciprocal rank fusion, truncated to the
             same size (size-controlled: can drop embedding candidates for lexical ones)
  union      embedding top-k plus BM25 top-k, deduplicated, no truncation
             (recall-maximizing: up to ~2k per left record; pool size is reported)

The random ``bottom_k`` are drawn from the lower half of the *full* ranking of unique
right records, as in the original builder. All variants return the same
``(DataFrame[id1, id2, similarity], stats)`` shape
as ``active_learning_ml._build_candidates`` so the downstream seed labeling,
active learning and export code is untouched. ``similarity`` is the ranking
score of the chosen method and is only meaningful *within* a query.

BM25 is implemented here with numpy posting lists (no scipy / rank_bm25 dependency)
and scores one query at a time, so pools over 100k-record tables (WDC) stay in memory.
"""
from __future__ import annotations

import math
import re
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

POOL_METHODS = ("embedding", "bm25", "rrf", "union")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
RRF_K = 60  # standard constant from Cormack et al. 2009


def record_text(row: Dict[str, object], fields: Sequence[str]) -> str:
    """Concatenate the configured fields the same way the embedding input was built."""
    parts: List[str] = []
    for f in fields:
        v = row.get(f, "")
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
            v = int(v)  # CSV round-trip turns 2008 into 2008.0 in columns with blanks
        s = str(v).strip()
        if s and s.lower() != "nan":
            parts.append(s)
    return " ".join(parts)


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(str(text).lower())


class BM25Index:
    """Minimal BM25 (Okapi) scorer over a fixed document collection, one query at a time."""

    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        n_docs = len(docs)
        lengths = np.zeros(n_docs, dtype=np.float32)
        postings: Dict[str, Tuple[List[int], List[float]]] = {}
        for d_i, doc in enumerate(docs):
            toks = tokenize(doc)
            lengths[d_i] = len(toks)
            counts: Dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            for t, c in counts.items():
                pl = postings.setdefault(t, ([], []))
                pl[0].append(d_i)
                pl[1].append(float(c))
        avgdl = float(lengths.mean()) if n_docs else 1.0
        denom = (k1 * (1.0 - b + b * lengths / max(avgdl, 1e-9))).astype(np.float32)
        self.n_docs = n_docs
        self.postings: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for t, (ids, tfs) in postings.items():
            ids_a = np.asarray(ids, dtype=np.int64)
            tf_a = np.asarray(tfs, dtype=np.float32)
            idf = math.log((n_docs - len(ids_a) + 0.5) / (len(ids_a) + 0.5) + 1.0)
            w = tf_a * (k1 + 1.0) / (tf_a + denom[ids_a]) * idf
            self.postings[t] = (ids_a, w.astype(np.float32))

    def score(self, query: str) -> np.ndarray:
        """BM25 scores of every document for one query (length n_docs)."""
        s = np.zeros(self.n_docs, dtype=np.float32)
        for t in set(tokenize(query)):
            pl = self.postings.get(t)
            if pl is not None:
                s[pl[0]] += pl[1]  # doc ids are unique within a posting list
        return s


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def _select_per_query(
    ranked_idx: np.ndarray,
    ranked_score: np.ndarray,
    right_source_ids: np.ndarray,
    k: int,
    bottom_k: int,
    rng: np.random.RandomState,
) -> List[Tuple[int, float]]:
    """Mirror the original selection: top_k ranked + bottom_k random from the lower half,
    dedup by right source id, backfill with next-best to reach k."""
    n_valid = len(ranked_idx)
    top_k = max(0, k - bottom_k)
    sel_idx: List[int] = []
    sel_score: List[float] = []
    if n_valid >= k and n_valid > top_k + bottom_k:
        sel_idx.extend(ranked_idx[:top_k].tolist())
        sel_score.extend(ranked_score[:top_k].tolist())
        start = max(top_k, n_valid // 2)
        pool_idx, pool_score = ranked_idx[start:], ranked_score[start:]
        if len(pool_idx) >= bottom_k > 0:
            ch = rng.choice(len(pool_idx), size=bottom_k, replace=False)
            sel_idx.extend(pool_idx[ch].tolist())
            sel_score.extend(pool_score[ch].tolist())
        else:
            sel_idx.extend(pool_idx.tolist())
            sel_score.extend(pool_score.tolist())
    else:
        a = min(n_valid, k)
        sel_idx.extend(ranked_idx[:a].tolist())
        sel_score.extend(ranked_score[:a].tolist())

    out: List[Tuple[int, float]] = []
    seen: set = set()
    for r_i, s in zip(sel_idx, sel_score):
        src = str(right_source_ids[int(r_i)])
        if src in seen:
            continue
        seen.add(src)
        out.append((int(r_i), float(s)))
    if len(out) < k:
        for r_i, s in zip(ranked_idx.tolist(), ranked_score.tolist()):
            src = str(right_source_ids[int(r_i)])
            if src in seen:
                continue
            seen.add(src)
            out.append((int(r_i), float(s)))
            if len(out) >= k:
                break
    return out[:k]


def build_candidates(
    *,
    method: str,
    left_ids: np.ndarray,
    right_ids: np.ndarray,
    right_source_ids: np.ndarray,
    left_emb: np.ndarray,
    right_emb: np.ndarray,
    left_text: Sequence[str] | None,
    right_text: Sequence[str] | None,
    k: int,
    candidate_cap: int,
    bottom_k: int,
    random_state: int,
    embedding_builder: Callable[..., Tuple[pd.DataFrame, Dict[str, int]]] | None = None,
    fetch_k: int = 200,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    method = str(method or "embedding").strip().lower()
    if method not in POOL_METHODS:
        raise ValueError(f"Unknown pool_method={method!r}; expected one of {POOL_METHODS}")

    if method == "embedding":
        if embedding_builder is None:
            raise ValueError("embedding_builder is required for method='embedding'")
        c, stats = embedding_builder(
            left_ids=left_ids,
            right_ids=right_ids,
            right_source_ids=right_source_ids,
            left_emb=left_emb,
            right_emb=right_emb,
            k=k,
            candidate_cap=candidate_cap,
            bottom_k=bottom_k,
            random_state=random_state,
        )
        stats["pool_method"] = "embedding"
        return c, stats

    if left_text is None or right_text is None:
        raise ValueError(f"method={method} needs left_text and right_text")
    k = max(1, int(k))
    bottom_k = max(0, int(bottom_k))
    rng = np.random.RandomState(int(random_state))
    max_queries = len(left_ids) if candidate_cap <= 0 else min(len(left_ids), int(math.ceil(candidate_cap / k)))
    n_right = len(right_ids)
    fetch = int(min(n_right, max(fetch_k, k + 50)))

    # Rank over unique right *records*: the released tables hold one row per training pair, so
    # the same offer can occupy many rows. The FAISS path fetches every row and dedups afterwards;
    # a bounded fetch would run out of unique records first, so dedup up front instead.
    src = np.asarray(right_source_ids).astype(str)
    _, first_row, inv = np.unique(src, return_index=True, return_inverse=True)
    right_text = list(right_text)
    bm25 = BM25Index([right_text[i] for i in first_row])
    left_text = list(left_text)
    if method in ("rrf", "union"):
        right_n = _unit(right_emb[first_row])
        left_n = _unit(left_emb[:max_queries])

    def _top(scores: np.ndarray) -> np.ndarray:
        if fetch >= len(scores):
            return np.argsort(-scores, kind="stable")
        part = np.argpartition(-scores, fetch - 1)[:fetch]
        return part[np.argsort(-scores[part], kind="stable")]

    def _full_rank(scores: np.ndarray) -> np.ndarray:
        return np.argsort(-scores, kind="stable")

    rows: List[Tuple[str, str, float]] = []
    for l_i in range(max_queries):
        b = bm25.score(left_text[l_i])
        if method == "bm25":
            order = _full_rank(b)
            ranked_idx, ranked_score = order, b[order]
        else:  # rrf / union both need the cosine ranking over unique records
            with np.errstate(all="ignore"):
                e = np.nan_to_num(right_n @ left_n[l_i], nan=-1.0, posinf=1.0, neginf=-1.0)
            b_order = _full_rank(b)
            e_order = _full_rank(e)
            if method == "rrf":
                fused: Dict[int, float] = {}
                for rank, d in enumerate(e_order[:fetch].tolist()):
                    fused[d] = fused.get(d, 0.0) + 1.0 / (RRF_K + rank + 1)
                for rank, d in enumerate(b_order[:fetch].tolist()):
                    fused[d] = fused.get(d, 0.0) + 1.0 / (RRF_K + rank + 1)
                head = sorted(fused.items(), key=lambda kv: -kv[1])
                head_ids = np.asarray([d for d, _ in head], dtype=np.int64)
                head_score = np.asarray([v for _, v in head], dtype=np.float32)
                # beyond the fused head, keep the embedding order so the lower half is well defined
                tail = e_order[~np.isin(e_order, head_ids)]
                ranked_idx = np.concatenate([head_ids, tail])
                ranked_score = np.concatenate([head_score, np.full(len(tail), -1.0, dtype=np.float32)])
            else:  # union: embedding top-k and bm25 top-k, plus bottom_k random from the embedding lower half
                top_k = max(0, k - bottom_k)
                picked: Dict[int, float] = {}
                for d in e_order[:top_k].tolist():
                    picked.setdefault(int(d), float(e[d]))
                for d in b_order[:top_k].tolist():
                    picked.setdefault(int(d), float(e[d]))
                if bottom_k > 0:
                    start_i = max(top_k, len(e_order) // 2)
                    lower = e_order[start_i:]
                    if len(lower) >= bottom_k:
                        ch = rng.choice(len(lower), size=bottom_k, replace=False)
                        for d in lower[ch].tolist():
                            picked.setdefault(int(d), float(e[d]))
                l_id = str(left_ids[l_i])
                for d, score in picked.items():
                    rows.append((l_id, str(right_ids[first_row[d]]), score))
                continue
        ranked_rows = first_row[ranked_idx]
        chosen = _select_per_query(ranked_rows, ranked_score, right_source_ids, k, bottom_k, rng)
        l_id = str(left_ids[l_i])
        for r_i, score in chosen:
            rows.append((l_id, str(right_ids[r_i]), float(score)))

    c = pd.DataFrame(rows, columns=["id1", "id2", "similarity"]).reset_index(drop=True)
    before_cap = len(c)
    if candidate_cap > 0 and len(c) > candidate_cap:
        c = c.head(candidate_cap).reset_index(drop=True)
    stats: Dict[str, int] = {
        "pool_method": method,
        "faiss_queries": int(max_queries),
        "query_side": "left",
        "neighbor_side": "right",
        "faiss_k": int(k),
        "faiss_top_k": int(max(0, k - bottom_k)),
        "faiss_bottom_k": int(bottom_k),
        "faiss_random_state": int(random_state),
        "rrf_fusion_depth": int(fetch),
        "unique_right_records": int(len(first_row)),
        "raw_pairs": int(len(rows)),
        "unique_pairs_before_cap": int(before_cap),
        "unique_pairs_after_cap": int(len(c)),
    }
    return c, stats
