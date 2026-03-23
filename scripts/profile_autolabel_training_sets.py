#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yaml
from scipy.linalg import sqrtm


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "labeling" / "benchmarks_active.yaml"
DEFAULT_REPORT_MD = ROOT / "reports" / "autolabel_training_profile.md"
DEFAULT_REPORT_JSON = ROOT / "reports" / "autolabel_training_profile.json"
DEFAULT_REPORT_XLSX = ROOT / "reports" / "autolabel_training_profile.xlsx"
GENERATED_GLOB = "output/simple_labeling/**/*_train.json.gz"
AUTO_LABEL_V1_ROOT = ROOT / "data" / "auto_label_v1"
EMBED_SAMPLE_MAX = 1500
EMBED_MMD_MAX = 1000
EMBED_K = 5

POS_LABELS = {"1", "TRUE", "T", "YES", "Y"}
NEG_LABELS = {"0", "FALSE", "F", "NO", "N"}
SKIP_BASE_FIELDS = {"id", "pair_id", "label", "cluster_id", "is_hard_negative", "__rid", "rid1", "rid2"}
EMBED_CACHE: Dict[str, Dict[str, Any]] = {}


@dataclass(frozen=True)
class GeneratedSet:
    benchmark: str
    label: str
    profile: str
    path: Path
    summary_path: Optional[Path]


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return payload


def _open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _read_json_table(path: Path) -> pd.DataFrame:
    first: Optional[str] = None
    second: Optional[str] = None
    with _open_text(path) as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if first is None:
                first = stripped
                continue
            second = stripped
            break

    if first is None:
        return pd.DataFrame()

    if second is not None:
        try:
            obj1 = json.loads(first)
            obj2 = json.loads(second)
            if isinstance(obj1, dict) and isinstance(obj2, dict):
                rows: List[Dict[str, Any]] = []
                with _open_text(path) as handle:
                    for line in handle:
                        stripped = line.strip()
                        if not stripped:
                            continue
                        rows.append(json.loads(stripped))
                return pd.DataFrame(rows)
        except Exception:
            pass

    with _open_text(path) as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        return pd.DataFrame(payload)
    raise ValueError(f"Unsupported JSON structure in {path}")


def _read_table(path: Path) -> pd.DataFrame:
    lower = path.name.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    if lower.endswith(".json") or lower.endswith(".json.gz"):
        return _read_json_table(path)
    raise ValueError(f"Unsupported input extension for {path}")


def _normalize_label(value: Any) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        iv = int(value)
        if iv in {0, 1}:
            return iv
    text = str(value).strip().upper()
    if text in POS_LABELS:
        return 1
    if text in NEG_LABELS:
        return 0
    return None


def _normalize_pair_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {}
    if "id1" in out.columns and "id_left" not in out.columns:
        rename_map["id1"] = "id_left"
    if "id2" in out.columns and "id_right" not in out.columns:
        rename_map["id2"] = "id_right"
    if rename_map:
        out = out.rename(columns=rename_map)

    if "pair_id" not in out.columns:
        if {"id_left", "id_right"}.issubset(out.columns):
            out["pair_id"] = out["id_left"].astype(str).str.strip() + "#" + out["id_right"].astype(str).str.strip()
        else:
            raise ValueError("Could not construct pair_id; missing pair_id and id_left/id_right")
    elif not {"id_left", "id_right"}.issubset(out.columns):
        pair_parts = out["pair_id"].astype(str).str.split("#", n=1, expand=True)
        if pair_parts.shape[1] == 2:
            out["id_left"] = pair_parts[0]
            out["id_right"] = pair_parts[1]

    if "label" in out.columns:
        out["label"] = out["label"].map(_normalize_label)
    else:
        out["label"] = None

    if "id_left" in out.columns:
        out["id_left"] = out["id_left"].astype(str).str.strip()
    if "id_right" in out.columns:
        out["id_right"] = out["id_right"].astype(str).str.strip()
    out["pair_id"] = out["pair_id"].astype(str).str.strip()
    if {"id_left", "id_right"}.issubset(out.columns):
        out["pair_key"] = out["id_left"] + "#" + out["id_right"]
    else:
        out["pair_key"] = out["pair_id"]
    return out.reset_index(drop=True)


def _classify_split(path: Path) -> str:
    name = path.name.lower()
    if "valid" in name or "validation" in name:
        return "valid"
    if "gs" in name or "test" in name:
        return "test"
    return "train"


def _discover_official_split_paths(benchmark: str) -> List[Path]:
    bench_dir = ROOT / "data" / benchmark
    out: List[Path] = []
    for path in sorted(bench_dir.iterdir()):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.endswith((".pkl", ".jsonl", ".map.csv", "_left.csv", "_right.csv", "-left.csv", "-right.csv")):
            continue
        if "batch_embed" in name or "predictions" in name or "manifest" in name or "sampled_gs" in name:
            continue
        if not any(token in name for token in ("train", "valid", "validation", "gs", "test")):
            continue
        if not name.endswith((".csv", ".json", ".json.gz")):
            continue
        out.append(path)
    return out


def _load_official_splits(benchmark: str) -> Dict[str, pd.DataFrame]:
    splits: Dict[str, pd.DataFrame] = {}
    for path in _discover_official_split_paths(benchmark):
        key = path.name
        frame = _normalize_pair_frame(_read_table(path))
        splits[key] = frame
    return splits


def _infer_profile(path: Path, benchmark: str, summary_payload: Dict[str, Any], config_profiles: Dict[str, Any]) -> str:
    parts = list(path.parts)
    if "profiles" in parts:
        idx = parts.index("profiles")
        if idx + 1 < len(parts):
            return parts[idx + 1]

    match = re.search(rf"active_labels_latest_{re.escape(benchmark)}_(small|medium|large|all)_train\.json\.gz$", path.name)
    if match:
        return match.group(1)

    target_total = int(summary_payload.get("target_total", -1) or -1)
    target_pos = int(summary_payload.get("target_pos", -1) or -1)
    target_neg = int(summary_payload.get("target_neg", -1) or -1)
    for profile_name, profile_cfg in config_profiles.items():
        if bool(profile_cfg.get("all_examples")):
            continue
        total = int(profile_cfg.get("target_total", -1) or -1)
        pos = int(profile_cfg.get("target_pos", -1) or -1)
        neg = int(profile_cfg.get("target_neg", total - pos) or -1)
        if (target_total, target_pos, target_neg) == (total, pos, neg):
            return str(profile_name)
    return "unknown"


def _discover_generated_sets(config: Dict[str, Any]) -> List[GeneratedSet]:
    config_benchmarks = config.get("benchmarks") or {}
    manifest_sets: List[GeneratedSet] = []
    if AUTO_LABEL_V1_ROOT.exists():
        for manifest_path in sorted(AUTO_LABEL_V1_ROOT.glob("*/profile_manifest.json")):
            try:
                manifest = json.loads(manifest_path.read_text())
            except Exception:
                continue
            if not isinstance(manifest, dict):
                continue
            benchmark = str(manifest.get("benchmark", "")).strip()
            if benchmark not in config_benchmarks:
                continue
            profiles = manifest.get("profiles") or {}
            if not isinstance(profiles, dict):
                continue
            run_dir = manifest_path.parent
            summary_path = run_dir / "summary.json"
            for profile_name in sorted(profiles.keys()):
                profile_cfg = profiles.get(profile_name) or {}
                if not isinstance(profile_cfg, dict):
                    continue
                ditto_name = Path(str(profile_cfg.get("ditto_train_json_gz", ""))).name
                if not ditto_name:
                    continue
                generated_path = run_dir / "profiles" / str(profile_name) / ditto_name
                if not generated_path.exists():
                    continue
                manifest_sets.append(
                    GeneratedSet(
                        benchmark=benchmark,
                        label=str(generated_path.relative_to(ROOT)),
                        profile=str(profile_name),
                        path=generated_path,
                        summary_path=summary_path if summary_path.exists() else None,
                    )
                )
    if manifest_sets:
        return manifest_sets

    out: List[GeneratedSet] = []
    for path in sorted(ROOT.glob(GENERATED_GLOB)):
        match = re.search(r"active_labels_latest_([a-z0-9-]+)(?:_(small|medium|large|all))?_train\.json\.gz$", path.name)
        if not match:
            continue
        benchmark = match.group(1)
        if benchmark not in config_benchmarks:
            continue
        summary_path = path.parent / "summary.json"
        summary_payload = {}
        if summary_path.exists():
            try:
                summary_payload = json.loads(summary_path.read_text())
            except Exception:
                summary_payload = {}
        profile = _infer_profile(
            path=path,
            benchmark=benchmark,
            summary_payload=summary_payload,
            config_profiles=((config_benchmarks.get(benchmark) or {}).get("profiles") or {}),
        )
        label = str(path.relative_to(ROOT))
        out.append(
            GeneratedSet(
                benchmark=benchmark,
                label=label,
                profile=profile,
                path=path,
                summary_path=summary_path if summary_path.exists() else None,
            )
        )
    return out


def _normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return " ".join(str(value).strip().lower().split())


def _tokenize(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _normalize_text(value)))


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    lset = set(left)
    rset = set(right)
    if not lset and not rset:
        return 1.0
    if not lset or not rset:
        return 0.0
    return len(lset & rset) / len(lset | rset)


def _enrich_clusters(df: pd.DataFrame, left_source: pd.DataFrame, right_source: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "cluster_id_left" not in out.columns:
        left_clusters = left_source[["id", "cluster_id"]].rename(columns={"id": "id_left", "cluster_id": "cluster_id_left"})
        out = out.merge(left_clusters, on="id_left", how="left")
    if "cluster_id_right" not in out.columns:
        right_clusters = right_source[["id", "cluster_id"]].rename(columns={"id": "id_right", "cluster_id": "cluster_id_right"})
        out = out.merge(right_clusters, on="id_right", how="left")
    return out


def _base_fields(df: pd.DataFrame) -> List[str]:
    left_fields = {col[:-5] for col in df.columns if col.endswith("_left")}
    right_fields = {col[:-6] for col in df.columns if col.endswith("_right")}
    fields = sorted(left_fields & right_fields)
    return [field for field in fields if field not in SKIP_BASE_FIELDS]


def _annotate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    fields = _base_fields(out)
    left_lengths: List[int] = []
    right_lengths: List[int] = []
    all_jaccards: List[float] = []
    title_jaccards: List[float] = []
    cluster_match: List[Optional[bool]] = []

    has_title = "title_left" in out.columns and "title_right" in out.columns
    for _, row in out.iterrows():
        left_tokens: set[str] = set()
        right_tokens: set[str] = set()
        for field in fields:
            left_tokens |= _tokenize(row.get(f"{field}_left"))
            right_tokens |= _tokenize(row.get(f"{field}_right"))
        left_lengths.append(len(left_tokens))
        right_lengths.append(len(right_tokens))
        all_jaccards.append(_jaccard(left_tokens, right_tokens))
        if has_title:
            title_jaccards.append(_jaccard(_tokenize(row.get("title_left")), _tokenize(row.get("title_right"))))
        else:
            title_jaccards.append(math.nan)

        left_cluster = row.get("cluster_id_left")
        right_cluster = row.get("cluster_id_right")
        if pd.isna(left_cluster) or pd.isna(right_cluster):
            cluster_match.append(None)
        else:
            cluster_match.append(str(left_cluster).strip() == str(right_cluster).strip())

    out["left_token_len"] = left_lengths
    out["right_token_len"] = right_lengths
    out["all_text_jaccard"] = all_jaccards
    out["title_jaccard"] = title_jaccards
    out["cluster_match"] = cluster_match
    return out


def _materialize_from_labels_csv(
    labels_csv: Path,
    *,
    left_source: pd.DataFrame,
    right_source: pd.DataFrame,
    field_map: Dict[str, str],
) -> pd.DataFrame:
    labels_df = _normalize_pair_frame(pd.read_csv(labels_csv))
    left_cols = ["id", "cluster_id"] + [src for src in field_map.values() if src in left_source.columns]
    right_cols = ["id", "cluster_id"] + [src for src in field_map.values() if src in right_source.columns]
    left_lookup = left_source[left_cols].drop_duplicates(subset=["id"], keep="first").copy()
    right_lookup = right_source[right_cols].drop_duplicates(subset=["id"], keep="first").copy()
    left_lookup = left_lookup.rename(columns={"id": "id_left", "cluster_id": "cluster_id_left"})
    right_lookup = right_lookup.rename(columns={"id": "id_right", "cluster_id": "cluster_id_right"})
    left_lookup["id_left"] = left_lookup["id_left"].astype(str).str.strip()
    right_lookup["id_right"] = right_lookup["id_right"].astype(str).str.strip()
    for out_name, src_name in field_map.items():
        if src_name in left_lookup.columns:
            left_lookup = left_lookup.rename(columns={src_name: f"{out_name}_left"})
        if src_name in right_lookup.columns:
            right_lookup = right_lookup.rename(columns={src_name: f"{out_name}_right"})

    merged = labels_df.merge(left_lookup, on="id_left", how="left").merge(right_lookup, on="id_right", how="left")
    return _normalize_pair_frame(merged)


def _stable_seed(text: str) -> int:
    return 17 + sum(ord(ch) for ch in text) % 10_000


def _safe_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return left @ right


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def _load_embedding_resources(benchmark: str, benchmark_cfg: Dict[str, Any], left_source: pd.DataFrame, right_source: pd.DataFrame) -> Dict[str, Any]:
    cached = EMBED_CACHE.get(benchmark)
    if cached is not None:
        return cached

    embeddings_cfg = benchmark_cfg.get("embeddings") or {}
    emb_dir = ROOT / str(embeddings_cfg.get("dir", ""))
    left_path = emb_dir / str(embeddings_cfg.get("left_file", ""))
    right_path = emb_dir / str(embeddings_cfg.get("right_file", ""))
    left_emb = np.load(left_path).astype(np.float32, copy=False)
    right_emb = np.load(right_path).astype(np.float32, copy=False)

    left_ids = left_source["id"].astype(str).str.strip().tolist()
    right_ids = right_source["id"].astype(str).str.strip().tolist()
    bundle = {
        "left_emb": _l2_normalize(left_emb),
        "right_emb": _l2_normalize(right_emb),
        "left_id_to_idx": {rid: idx for idx, rid in enumerate(left_ids)},
        "right_id_to_idx": {rid: idx for idx, rid in enumerate(right_ids)},
    }
    EMBED_CACHE[benchmark] = bundle
    return bundle


def _build_pair_embeddings(df: pd.DataFrame, *, benchmark: str, benchmark_cfg: Dict[str, Any], left_source: pd.DataFrame, right_source: pd.DataFrame) -> np.ndarray:
    resources = _load_embedding_resources(benchmark, benchmark_cfg, left_source=left_source, right_source=right_source)
    left_id_to_idx = resources["left_id_to_idx"]
    right_id_to_idx = resources["right_id_to_idx"]
    left_emb = resources["left_emb"]
    right_emb = resources["right_emb"]

    left_indices: List[int] = []
    right_indices: List[int] = []
    for left_id, right_id in zip(df["id_left"].astype(str), df["id_right"].astype(str)):
        lidx = left_id_to_idx.get(str(left_id).strip())
        ridx = right_id_to_idx.get(str(right_id).strip())
        if lidx is None or ridx is None:
            continue
        left_indices.append(lidx)
        right_indices.append(ridx)

    if not left_indices:
        return np.zeros((0, 0), dtype=np.float32)

    pair_emb = np.concatenate([left_emb[left_indices], right_emb[right_indices]], axis=1)
    return _l2_normalize(pair_emb.astype(np.float64, copy=False))


def _sample_embeddings(matrix: np.ndarray, *, max_points: int, seed_key: str) -> np.ndarray:
    if matrix.shape[0] <= max_points:
        return matrix
    rng = np.random.RandomState(_stable_seed(seed_key))
    idx = rng.choice(matrix.shape[0], size=max_points, replace=False)
    return matrix[idx]


def _cosine_nn_similarities(query: np.ndarray, ref: np.ndarray, *, self_compare: bool = False, batch_size: int = 256) -> np.ndarray:
    if query.size == 0 or ref.size == 0:
        return np.array([], dtype=np.float32)
    out: List[np.ndarray] = []
    for start in range(0, query.shape[0], batch_size):
        end = min(start + batch_size, query.shape[0])
        sims = _safe_matmul(query[start:end], ref.T)
        if self_compare:
            local_rows = np.arange(end - start)
            global_cols = start + local_rows
            sims[local_rows, global_cols] = -np.inf
        out.append(np.max(sims, axis=1))
    return np.concatenate(out, axis=0)


def _nn_cosine_stats(query: np.ndarray, ref: np.ndarray, *, self_compare: bool = False) -> Dict[str, Optional[float]]:
    sims = _cosine_nn_similarities(query, ref, self_compare=self_compare)
    if sims.size == 0:
        return {"mean_similarity": None, "median_similarity": None, "p90_similarity": None, "mean_distance": None}
    sims = sims[np.isfinite(sims)]
    if sims.size == 0:
        return {"mean_similarity": None, "median_similarity": None, "p90_similarity": None, "mean_distance": None}
    return {
        "mean_similarity": float(np.mean(sims)),
        "median_similarity": float(np.median(sims)),
        "p90_similarity": float(np.quantile(sims, 0.90)),
        "mean_distance": float(np.mean(1.0 - sims)),
    }


def _centroid_cosine(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.size == 0 or b.size == 0:
        return None
    a_mean = a.mean(axis=0, keepdims=True)
    b_mean = b.mean(axis=0, keepdims=True)
    a_mean = _l2_normalize(a_mean)
    b_mean = _l2_normalize(b_mean)
    return float(_safe_matmul(a_mean, b_mean.T)[0, 0])


def _frechet_distance(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.shape[0] < 2 or b.shape[0] < 2:
        return None
    mu_a = a.mean(axis=0)
    mu_b = b.mean(axis=0)
    cov_a = np.cov(a, rowvar=False)
    cov_b = np.cov(b, rowvar=False)
    cov_prod = sqrtm(_safe_matmul(cov_a.astype(np.float64, copy=False), cov_b.astype(np.float64, copy=False)))
    if np.iscomplexobj(cov_prod):
        cov_prod = cov_prod.real
    diff = mu_a - mu_b
    value = diff.dot(diff) + np.trace(cov_a + cov_b - 2.0 * cov_prod)
    return float(np.real(value))


def _rbf_mmd(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.shape[0] < 2 or b.shape[0] < 2:
        return None

    combined = np.vstack([a, b])
    sample = _sample_embeddings(combined, max_points=min(combined.shape[0], 512), seed_key="mmd_sigma")
    sq = np.sum(sample * sample, axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2.0 * _safe_matmul(sample, sample.T), 0.0)
    upper = d2[np.triu_indices_from(d2, k=1)]
    sigma2 = float(np.median(upper[upper > 0])) if np.any(upper > 0) else 1.0
    if sigma2 <= 0:
        sigma2 = 1.0

    def _kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_sq = np.sum(x * x, axis=1, keepdims=True)
        y_sq = np.sum(y * y, axis=1, keepdims=True).T
        dist2 = np.maximum(x_sq + y_sq - 2.0 * _safe_matmul(x, y.T), 0.0)
        return np.exp(-dist2 / (2.0 * sigma2))

    kaa = _kernel(a, a)
    kbb = _kernel(b, b)
    kab = _kernel(a, b)
    np.fill_diagonal(kaa, 0.0)
    np.fill_diagonal(kbb, 0.0)
    term_aa = kaa.sum() / (a.shape[0] * (a.shape[0] - 1))
    term_bb = kbb.sum() / (b.shape[0] * (b.shape[0] - 1))
    term_ab = kab.mean()
    return float(term_aa + term_bb - 2.0 * term_ab)


def _knn_squared_radius(x: np.ndarray, *, k: int, batch_size: int = 256) -> np.ndarray:
    if x.shape[0] <= 1:
        return np.zeros((x.shape[0],), dtype=np.float32)
    radii = np.zeros((x.shape[0],), dtype=np.float32)
    kth_index = min(max(k - 1, 0), x.shape[0] - 2)
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        sims = _safe_matmul(x[start:end], x.T)
        local_rows = np.arange(end - start)
        global_cols = start + local_rows
        sims[local_rows, global_cols] = -np.inf
        dist2 = np.maximum(2.0 - 2.0 * sims, 0.0)
        radii[start:end] = np.partition(dist2, kth_index, axis=1)[:, kth_index]
    return radii


def _precision_coverage(gen: np.ndarray, ref: np.ndarray, *, k: int = EMBED_K, batch_size: int = 256) -> Dict[str, Optional[float]]:
    if gen.shape[0] <= 1 or ref.shape[0] <= 1:
        return {"precision": None, "coverage": None}

    ref_r2 = _knn_squared_radius(ref, k=k, batch_size=batch_size)

    precise = 0
    for start in range(0, gen.shape[0], batch_size):
        end = min(start + batch_size, gen.shape[0])
        sims = _safe_matmul(gen[start:end], ref.T)
        dist2 = np.maximum(2.0 - 2.0 * sims, 0.0)
        precise += int(np.any(dist2 <= ref_r2[None, :], axis=1).sum())

    covered = 0
    for start in range(0, ref.shape[0], batch_size):
        end = min(start + batch_size, ref.shape[0])
        sims = _safe_matmul(ref[start:end], gen.T)
        dist2 = np.maximum(2.0 - 2.0 * sims, 0.0)
        min_dist2 = np.min(dist2, axis=1)
        covered += int((min_dist2 <= ref_r2[start:end]).sum())

    return {
        "precision": float(precise / gen.shape[0]),
        "coverage": float(covered / ref.shape[0]),
    }


def _embedding_summary(
    *,
    benchmark: str,
    benchmark_cfg: Dict[str, Any],
    left_source: pd.DataFrame,
    right_source: pd.DataFrame,
    generated_all_df: pd.DataFrame,
    generated_novel_df: pd.DataFrame,
    official_train_df: pd.DataFrame,
    official_valid_df: pd.DataFrame,
    official_test_df: pd.DataFrame,
) -> Dict[str, Any]:
    generated_all = _sample_embeddings(
        _build_pair_embeddings(generated_all_df, benchmark=benchmark, benchmark_cfg=benchmark_cfg, left_source=left_source, right_source=right_source),
        max_points=EMBED_SAMPLE_MAX,
        seed_key=f"{benchmark}_generated_all",
    )
    generated_novel = _sample_embeddings(
        _build_pair_embeddings(generated_novel_df, benchmark=benchmark, benchmark_cfg=benchmark_cfg, left_source=left_source, right_source=right_source),
        max_points=EMBED_SAMPLE_MAX,
        seed_key=f"{benchmark}_generated_novel",
    )
    official_train = _sample_embeddings(
        _build_pair_embeddings(official_train_df, benchmark=benchmark, benchmark_cfg=benchmark_cfg, left_source=left_source, right_source=right_source),
        max_points=EMBED_SAMPLE_MAX,
        seed_key=f"{benchmark}_official_train",
    )
    official_valid = _sample_embeddings(
        _build_pair_embeddings(official_valid_df, benchmark=benchmark, benchmark_cfg=benchmark_cfg, left_source=left_source, right_source=right_source),
        max_points=EMBED_SAMPLE_MAX,
        seed_key=f"{benchmark}_official_valid",
    )
    official_test = _sample_embeddings(
        _build_pair_embeddings(official_test_df, benchmark=benchmark, benchmark_cfg=benchmark_cfg, left_source=left_source, right_source=right_source),
        max_points=EMBED_SAMPLE_MAX,
        seed_key=f"{benchmark}_official_test",
    )

    all_for_mmd = _sample_embeddings(generated_all, max_points=EMBED_MMD_MAX, seed_key=f"{benchmark}_all_mmd")
    novel_for_mmd = _sample_embeddings(generated_novel, max_points=EMBED_MMD_MAX, seed_key=f"{benchmark}_novel_mmd")
    train_for_mmd = _sample_embeddings(official_train, max_points=EMBED_MMD_MAX, seed_key=f"{benchmark}_train_mmd")

    return {
        "sampling": {
            "max_points": EMBED_SAMPLE_MAX,
            "mmd_max_points": EMBED_MMD_MAX,
            "k": EMBED_K,
            "generated_all_points": int(generated_all.shape[0]),
            "generated_novel_points": int(generated_novel.shape[0]),
            "official_train_points": int(official_train.shape[0]),
            "official_valid_points": int(official_valid.shape[0]),
            "official_test_points": int(official_test.shape[0]),
        },
        "diversity": {
            "generated_all": _nn_cosine_stats(generated_all, generated_all, self_compare=True),
            "generated_novel": _nn_cosine_stats(generated_novel, generated_novel, self_compare=True),
            "official_train": _nn_cosine_stats(official_train, official_train, self_compare=True),
        },
        "comparisons": {
            "generated_all_vs_train": {
                "centroid_cosine": _centroid_cosine(generated_all, official_train),
                "frechet_distance": _frechet_distance(generated_all, official_train),
                "mmd_rbf": _rbf_mmd(all_for_mmd, train_for_mmd),
                "nn_to_train": _nn_cosine_stats(generated_all, official_train),
                **_precision_coverage(generated_all, official_train),
            },
            "generated_novel_vs_train": {
                "centroid_cosine": _centroid_cosine(generated_novel, official_train),
                "frechet_distance": _frechet_distance(generated_novel, official_train),
                "mmd_rbf": _rbf_mmd(novel_for_mmd, train_for_mmd),
                "nn_to_train": _nn_cosine_stats(generated_novel, official_train),
                **_precision_coverage(generated_novel, official_train),
            },
            "generated_all_vs_valid": {
                "centroid_cosine": _centroid_cosine(generated_all, official_valid),
                "nn_to_valid": _nn_cosine_stats(generated_all, official_valid),
            },
            "generated_all_vs_test": {
                "centroid_cosine": _centroid_cosine(generated_all, official_test),
                "nn_to_test": _nn_cosine_stats(generated_all, official_test),
            },
        },
    }


def _concat_split_group(official_splits: Dict[str, pd.DataFrame], group: str) -> pd.DataFrame:
    frames = [df for split_name, df in official_splits.items() if _classify_split(Path(split_name)) == group]
    if not frames:
        return pd.DataFrame(columns=["pair_key", "id_left", "id_right", "label"])
    combined = pd.concat(frames, ignore_index=True)
    if "pair_key" in combined.columns:
        combined = combined.drop_duplicates(subset=["pair_key"], keep="first").reset_index(drop=True)
    return combined


def _pct(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _series_stats(values: pd.Series) -> Dict[str, Optional[float]]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {"mean": None, "median": None, "p10": None, "p90": None}
    return {
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "p10": float(clean.quantile(0.10)),
        "p90": float(clean.quantile(0.90)),
    }


def _fmt_float(value: Optional[float], digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "-"
    return f"{value:.{digits}f}"


def _subset_summary(df: pd.DataFrame, *, total_left_entities: int, total_right_entities: int) -> Dict[str, Any]:
    rows = int(len(df))
    pos = int((df["label"] == 1).sum()) if "label" in df.columns else 0
    neg = int((df["label"] == 0).sum()) if "label" in df.columns else 0
    unique_left = int(df["id_left"].nunique()) if "id_left" in df.columns else 0
    unique_right = int(df["id_right"].nunique()) if "id_right" in df.columns else 0
    left_counts = df["id_left"].value_counts() if "id_left" in df.columns else pd.Series(dtype="int64")
    right_counts = df["id_right"].value_counts() if "id_right" in df.columns else pd.Series(dtype="int64")

    cluster_known = df["cluster_match"].dropna() if "cluster_match" in df.columns else pd.Series(dtype="bool")
    false_pos = int(((df["label"] == 1) & (df["cluster_match"] == False)).sum()) if "cluster_match" in df.columns else 0
    false_neg = int(((df["label"] == 0) & (df["cluster_match"] == True)).sum()) if "cluster_match" in df.columns else 0

    return {
        "rows": rows,
        "pos": pos,
        "neg": neg,
        "pos_rate": _pct(pos, rows),
        "unique_left_entities": unique_left,
        "unique_right_entities": unique_right,
        "left_entity_coverage": _pct(unique_left, total_left_entities),
        "right_entity_coverage": _pct(unique_right, total_right_entities),
        "pairs_per_left_avg": float(left_counts.mean()) if not left_counts.empty else 0.0,
        "pairs_per_left_median": float(left_counts.median()) if not left_counts.empty else 0.0,
        "pairs_per_left_max": int(left_counts.max()) if not left_counts.empty else 0,
        "pairs_per_right_avg": float(right_counts.mean()) if not right_counts.empty else 0.0,
        "pairs_per_right_median": float(right_counts.median()) if not right_counts.empty else 0.0,
        "pairs_per_right_max": int(right_counts.max()) if not right_counts.empty else 0,
        "left_token_len": _series_stats(df["left_token_len"]) if "left_token_len" in df.columns else _series_stats(pd.Series(dtype="float64")),
        "right_token_len": _series_stats(df["right_token_len"]) if "right_token_len" in df.columns else _series_stats(pd.Series(dtype="float64")),
        "all_text_jaccard": _series_stats(df["all_text_jaccard"]) if "all_text_jaccard" in df.columns else _series_stats(pd.Series(dtype="float64")),
        "title_jaccard": _series_stats(df["title_jaccard"]) if "title_jaccard" in df.columns else _series_stats(pd.Series(dtype="float64")),
        "cluster_check_pairs": int(len(cluster_known)),
        "cluster_false_positive": false_pos,
        "cluster_false_negative": false_neg,
        "cluster_inconsistency_rate": _pct(false_pos + false_neg, int(len(cluster_known))),
    }


def _summary_to_rows(name: str, summary: Dict[str, Any]) -> List[List[str]]:
    return [
        [name, str(summary["rows"]), str(summary["pos"]), str(summary["neg"]), f'{summary["pos_rate"]:.3f}', f'{summary["left_entity_coverage"]:.3f}', f'{summary["right_entity_coverage"]:.3f}', f'{summary["all_text_jaccard"]["mean"] or 0.0:.3f}', f'{summary["cluster_inconsistency_rate"]:.3f}'],
    ]


def _format_table(headers: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _load_json_if_exists(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def profile_generated_set(generated: GeneratedSet, config: Dict[str, Any]) -> Dict[str, Any]:
    benchmark_cfg = (config.get("benchmarks") or {}).get(generated.benchmark) or {}
    left_source = pd.read_csv(ROOT / benchmark_cfg["left_csv"])
    right_source = pd.read_csv(ROOT / benchmark_cfg["right_csv"])
    total_left_entities = int(left_source["id"].nunique())
    total_right_entities = int(right_source["id"].nunique())
    field_map = {str(k): str(v) for k, v in (benchmark_cfg.get("fields") or {}).items()}
    summary_payload = _load_json_if_exists(generated.summary_path)

    labels_csv = generated.path.parent / "labels_final.csv"
    generated_source = str(generated.path.relative_to(ROOT))
    if labels_csv.exists():
        generated_df = _materialize_from_labels_csv(
            labels_csv,
            left_source=left_source,
            right_source=right_source,
            field_map=field_map,
        )
        generated_source = str(labels_csv.relative_to(ROOT))
    else:
        generated_df = _normalize_pair_frame(_read_table(generated.path))
    generated_df = _enrich_clusters(generated_df, left_source=left_source, right_source=right_source)
    generated_df = _annotate_metrics(generated_df)

    official_splits = _load_official_splits(generated.benchmark)
    official_summaries: Dict[str, Any] = {}
    train_ids: set[str] = set()
    valid_ids: set[str] = set()
    test_ids: set[str] = set()
    split_overlap_rows: List[Dict[str, Any]] = []

    for split_name, split_df in official_splits.items():
        split_df = _enrich_clusters(split_df, left_source=left_source, right_source=right_source)
        split_df = _annotate_metrics(split_df)
        official_summaries[split_name] = _subset_summary(
            split_df,
            total_left_entities=total_left_entities,
            total_right_entities=total_right_entities,
        )
        split_group = _classify_split(Path(split_name))
        pair_ids = set(split_df["pair_key"].astype(str))
        if split_group == "train":
            train_ids |= pair_ids
        elif split_group == "valid":
            valid_ids |= pair_ids
        else:
            test_ids |= pair_ids

        overlap = generated_df[generated_df["pair_key"].isin(pair_ids)].copy()
        label_agree = 0
        label_conflict = 0
        if not overlap.empty and "label" in split_df.columns and split_df["label"].notna().any():
            label_lookup = split_df.set_index("pair_key")["label"].to_dict()
            overlap["official_label"] = overlap["pair_key"].map(label_lookup)
            label_agree = int((overlap["label"] == overlap["official_label"]).sum())
            label_conflict = int((overlap["label"] != overlap["official_label"]).sum())
        split_overlap_rows.append(
            {
                "split_name": split_name,
                "split_group": split_group,
                "overlap_pairs": int(len(overlap)),
                "overlap_rate_generated": _pct(int(len(overlap)), int(len(generated_df))),
                "overlap_rate_split": _pct(int(len(overlap)), int(len(split_df))),
                "label_agreement": label_agree,
                "label_conflict": label_conflict,
            }
        )

    all_official_ids = train_ids | valid_ids | test_ids
    generated_df["in_official_train"] = generated_df["pair_key"].isin(train_ids)
    generated_df["in_official_valid"] = generated_df["pair_key"].isin(valid_ids)
    generated_df["in_official_test"] = generated_df["pair_key"].isin(test_ids)
    generated_df["is_novel"] = ~generated_df["pair_key"].isin(all_official_ids)

    subset_stats = {
        "generated_all": _subset_summary(generated_df, total_left_entities=total_left_entities, total_right_entities=total_right_entities),
        "generated_overlap_train": _subset_summary(generated_df[generated_df["in_official_train"]], total_left_entities=total_left_entities, total_right_entities=total_right_entities),
        "generated_overlap_valid": _subset_summary(generated_df[generated_df["in_official_valid"]], total_left_entities=total_left_entities, total_right_entities=total_right_entities),
        "generated_overlap_test": _subset_summary(generated_df[generated_df["in_official_test"]], total_left_entities=total_left_entities, total_right_entities=total_right_entities),
        "generated_novel": _subset_summary(generated_df[generated_df["is_novel"]], total_left_entities=total_left_entities, total_right_entities=total_right_entities),
    }

    official_train_df = _concat_split_group(official_splits, "train")
    official_valid_df = _concat_split_group(official_splits, "valid")
    official_test_df = _concat_split_group(official_splits, "test")
    embedding_metrics: Dict[str, Any] = {}
    try:
        embedding_metrics = _embedding_summary(
            benchmark=generated.benchmark,
            benchmark_cfg=benchmark_cfg,
            left_source=left_source,
            right_source=right_source,
            generated_all_df=generated_df,
            generated_novel_df=generated_df[generated_df["is_novel"]].copy(),
            official_train_df=official_train_df,
            official_valid_df=official_valid_df,
            official_test_df=official_test_df,
        )
    except Exception as exc:
        embedding_metrics = {"error": str(exc)}

    return {
        "benchmark": generated.benchmark,
        "profile": generated.profile,
        "generated_file": str(generated.path.relative_to(ROOT)),
        "analysis_source": generated_source,
        "summary_file": str(generated.summary_path.relative_to(ROOT)) if generated.summary_path else "",
        "official_split_files": sorted(official_splits.keys()),
        "run_summary": summary_payload,
        "subset_stats": subset_stats,
        "official_summaries": official_summaries,
        "split_overlaps": split_overlap_rows,
        "embedding_metrics": embedding_metrics,
    }


def _headline_finding(profile: Dict[str, Any]) -> List[str]:
    stats = profile["subset_stats"]
    all_rows = stats["generated_all"]["rows"]
    train_overlap = stats["generated_overlap_train"]["rows"]
    valid_overlap = stats["generated_overlap_valid"]["rows"]
    test_overlap = stats["generated_overlap_test"]["rows"]
    novel_rows = stats["generated_novel"]["rows"]
    novel_pos = stats["generated_novel"]["pos_rate"]
    train_pos = stats["generated_overlap_train"]["pos_rate"]
    novel_sim = stats["generated_novel"]["all_text_jaccard"]["mean"] or 0.0
    train_sim = stats["generated_overlap_train"]["all_text_jaccard"]["mean"] or 0.0
    incons = stats["generated_all"]["cluster_inconsistency_rate"]
    embed_comp = ((profile.get("embedding_metrics") or {}).get("comparisons") or {}).get("generated_novel_vs_train") or {}
    embed_nn = embed_comp.get("nn_to_train") or {}
    embed_cov = embed_comp.get("coverage")
    embed_prec = embed_comp.get("precision")

    findings = [
        f"- Exakte Ueberlappung mit offiziellen Trainingspaaren: {train_overlap}/{all_rows} ({_pct(train_overlap, all_rows):.2%}).",
        f"- Leakage in offizielle Validierungs-/Test-Splits: {valid_overlap}/{all_rows} bzw. {test_overlap}/{all_rows}.",
        f"- Wirklich neue Paare: {novel_rows}/{all_rows} ({_pct(novel_rows, all_rows):.2%}).",
        f"- Positive Rate neu vs. train-overlap: {novel_pos:.3f} vs. {train_pos:.3f}.",
        f"- Mittlere Text-Jaccard neu vs. train-overlap: {novel_sim:.3f} vs. {train_sim:.3f}.",
        f"- Cluster-basierte Inkonsistenzrate im generierten Set: {incons:.3%}.",
    ]
    if embed_nn:
        findings.append(
            f"- Embedding-Raum: mean NN cosine `novel -> official train` {_fmt_float(embed_nn.get('mean_similarity'))}, "
            f"precision {_fmt_float(embed_prec)}, coverage {_fmt_float(embed_cov)}."
        )
    return findings


def render_markdown(results: List[Dict[str, Any]]) -> str:
    lines = [
        "# Auto-Label Training Set Profile",
        "",
        f"Stand: {date.today().isoformat()}",
        "",
        "## Scope",
        "",
        "- Profilierung nur fuer Auto-Label-Rohdaten, die aktuell wirklich im Repository liegen.",
        "- Die Analyse nutzt exakte Ueberlappung auf dem kanonischen Paar-Key `id_left#id_right` sowie einen Cluster-basierten Konsistenzcheck (`cluster_id_left == cluster_id_right`).",
        "- Manuelle Fehlerannotation nach Ralphs Schema ist damit noch nicht ersetzt; sie bleibt ein naechster Schritt.",
        "",
        "## Verfuegbare generierte Sets",
        "",
    ]

    available_rows = [
        [row["benchmark"], row["profile"], row["generated_file"], row["analysis_source"], row["summary_file"] or "-"]
        for row in results
    ]
    lines.append(_format_table(["Benchmark", "Profil", "Generated File", "Analysis Source", "Summary File"], available_rows))
    lines.append("")

    lines.append("## Wichtigste Befunde")
    lines.append("")
    for result in results:
        stats = result["subset_stats"]
        embedding_metrics = result.get("embedding_metrics") or {}
        novel_train = (embedding_metrics.get("comparisons") or {}).get("generated_novel_vs_train") or {}
        novel_train_nn = novel_train.get("nn_to_train") or {}
        all_rows = stats["generated_all"]["rows"]
        train_rows = stats["generated_overlap_train"]["rows"]
        valid_rows = stats["generated_overlap_valid"]["rows"]
        test_rows = stats["generated_overlap_test"]["rows"]
        novel_rows = stats["generated_novel"]["rows"]
        novel_pos = stats["generated_novel"]["pos_rate"]
        train_pos = stats["generated_overlap_train"]["pos_rate"]
        err_rate = stats["generated_all"]["cluster_inconsistency_rate"]
        lines.append(
            f"- `{result['benchmark']}/{result['profile']}`: "
            f"{_pct(train_rows, all_rows):.2%} Train-Overlap, "
            f"{_pct(valid_rows, all_rows):.2%} Valid-Leakage, "
            f"{_pct(test_rows, all_rows):.2%} Test-Leakage, "
            f"{_pct(novel_rows, all_rows):.2%} neue Paare, "
            f"Positive Rate neu vs. Train-Overlap {novel_pos:.3f} vs. {train_pos:.3f}, "
            f"Cluster-Fehlerrate {err_rate:.3%}, "
            f"mean NN cosine novel->train {_fmt_float(novel_train_nn.get('mean_similarity'))}, "
            f"coverage {_fmt_float(novel_train.get('coverage'))}."
        )
    lines.append("")

    for result in results:
        lines.append(f"## {result['benchmark']} / {result['profile']}")
        lines.append("")
        lines.extend(_headline_finding(result))
        lines.append("")

        subset_rows: List[List[str]] = []
        for key in [
            "generated_all",
            "generated_overlap_train",
            "generated_overlap_valid",
            "generated_overlap_test",
            "generated_novel",
        ]:
            summary = result["subset_stats"][key]
            subset_rows.append(
                [
                    key,
                    str(summary["rows"]),
                    str(summary["pos"]),
                    str(summary["neg"]),
                    f'{summary["pos_rate"]:.3f}',
                    f'{summary["left_entity_coverage"]:.3f}',
                    f'{summary["right_entity_coverage"]:.3f}',
                    f'{(summary["all_text_jaccard"]["mean"] or 0.0):.3f}',
                    f'{summary["cluster_inconsistency_rate"]:.3f}',
                ]
            )
        lines.append(_format_table(
            ["Subset", "Rows", "Pos", "Neg", "Pos Rate", "Left Cov", "Right Cov", "Mean Jaccard", "Cluster Err Rate"],
            subset_rows,
        ))
        lines.append("")

        overlap_rows = [
            [
                row["split_name"],
                row["split_group"],
                str(row["overlap_pairs"]),
                f'{row["overlap_rate_generated"]:.3f}',
                f'{row["overlap_rate_split"]:.3f}',
                str(row["label_agreement"]),
                str(row["label_conflict"]),
            ]
            for row in result["split_overlaps"]
        ]
        lines.append(_format_table(
            ["Official Split", "Group", "Overlap Pairs", "Rate vs Generated", "Rate vs Split", "Label Agree", "Label Conflict"],
            overlap_rows,
        ))
        lines.append("")

        official_rows = []
        for split_name, summary in sorted(result["official_summaries"].items()):
            official_rows.append(
                [
                    split_name,
                    str(summary["rows"]),
                    str(summary["pos"]),
                    str(summary["neg"]),
                    f'{summary["pos_rate"]:.3f}',
                    f'{(summary["all_text_jaccard"]["mean"] or 0.0):.3f}',
                    f'{summary["cluster_inconsistency_rate"]:.3f}',
                ]
            )
        lines.append(_format_table(
            ["Official Split", "Rows", "Pos", "Neg", "Pos Rate", "Mean Jaccard", "Cluster Err Rate"],
            official_rows,
        ))
        lines.append("")

        embedding_metrics = result.get("embedding_metrics") or {}
        if embedding_metrics.get("error"):
            lines.append(f"Embedding metrics error: `{embedding_metrics['error']}`")
            lines.append("")
        elif embedding_metrics:
            sampling = embedding_metrics.get("sampling") or {}
            lines.append(
                "Embedding sample sizes: "
                f"generated_all={sampling.get('generated_all_points', 0)}, "
                f"generated_novel={sampling.get('generated_novel_points', 0)}, "
                f"official_train={sampling.get('official_train_points', 0)}, "
                f"official_valid={sampling.get('official_valid_points', 0)}, "
                f"official_test={sampling.get('official_test_points', 0)}, "
                f"k={sampling.get('k', EMBED_K)}."
            )
            lines.append("")

            diversity_rows = []
            for subset_name in ["generated_all", "generated_novel", "official_train"]:
                metrics = (embedding_metrics.get("diversity") or {}).get(subset_name) or {}
                diversity_rows.append(
                    [
                        subset_name,
                        _fmt_float(metrics.get("mean_similarity")),
                        _fmt_float(metrics.get("median_similarity")),
                        _fmt_float(metrics.get("p90_similarity")),
                        _fmt_float(metrics.get("mean_distance")),
                    ]
                )
            lines.append(_format_table(
                ["Embedding Subset", "Mean NN Cosine", "Median NN Cosine", "P90 NN Cosine", "Mean NN Distance"],
                diversity_rows,
            ))
            lines.append("")

            comparison_rows = []
            comparison_specs = [
                ("generated_all_vs_train", "generated_all_vs_train", "nn_to_train"),
                ("generated_novel_vs_train", "generated_novel_vs_train", "nn_to_train"),
                ("generated_all_vs_valid", "generated_all_vs_valid", "nn_to_valid"),
                ("generated_all_vs_test", "generated_all_vs_test", "nn_to_test"),
            ]
            for label, key, nn_key in comparison_specs:
                metrics = (embedding_metrics.get("comparisons") or {}).get(key) or {}
                nn_stats = metrics.get(nn_key) or {}
                comparison_rows.append(
                    [
                        label,
                        _fmt_float(metrics.get("centroid_cosine")),
                        _fmt_float(nn_stats.get("mean_similarity")),
                        _fmt_float(nn_stats.get("mean_distance")),
                        _fmt_float(metrics.get("frechet_distance")),
                        _fmt_float(metrics.get("mmd_rbf")),
                        _fmt_float(metrics.get("precision")),
                        _fmt_float(metrics.get("coverage")),
                    ]
                )
            lines.append(_format_table(
                ["Embedding Comparison", "Centroid Cosine", "Mean NN Cosine", "Mean NN Distance", "Frechet", "MMD-RBF", "Precision", "Coverage"],
                comparison_rows,
            ))
            lines.append("")

        run_summary = result.get("run_summary") or {}
        if run_summary:
            lines.append("Run summary:")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(run_summary, indent=2, sort_keys=True))
            lines.append("```")
            lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Wenn die exakte Trainings-Ueberlappung sehr klein ist, erzeugt Auto-Labeling tatsaechlich neue Paarmengen statt nur offizielle Train-Paare wiederzuverwenden.")
    lines.append("- Wenn `generated_novel` deutlich andere Positive-Raten oder Aehnlichkeiten als `generated_overlap_train` hat, dann veraendert Auto-Labeling die Trainingsverteilung substanziell.")
    lines.append("- Embedding-Metriken ergaenzen die exakte Paar-Ueberlappung: hohe `coverage` bei gleichzeitig niedriger Exakt-Ueberlappung bedeutet semantisch aehnliche, aber nicht identische Trainingspaare.")
    lines.append("- Hohe mean NN cosine zu `valid` oder `test` ist ein Soft-Leakage-Signal, auch wenn kein exakter Paar-Key ueberschneidet.")
    lines.append("- Cluster-basierte Inkonsistenzen sind ein erster harter Fehlerindikator. Fuer das Paper sollte darauf noch eine manuelle Typisierung folgen.")
    lines.append("")
    return "\n".join(lines)


def _flatten_mapping(payload: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_mapping(value, prefix=full_key))
        else:
            flat[full_key] = value
    return flat


def _build_summary_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result in results:
        stats = result["subset_stats"]
        embedding_metrics = result.get("embedding_metrics") or {}
        novel_train = (embedding_metrics.get("comparisons") or {}).get("generated_novel_vs_train") or {}
        novel_train_nn = novel_train.get("nn_to_train") or {}
        all_valid = (embedding_metrics.get("comparisons") or {}).get("generated_all_vs_valid") or {}
        all_test = (embedding_metrics.get("comparisons") or {}).get("generated_all_vs_test") or {}
        all_valid_nn = all_valid.get("nn_to_valid") or {}
        all_test_nn = all_test.get("nn_to_test") or {}
        rows.append(
            {
                "benchmark": result["benchmark"],
                "profile": result["profile"],
                "generated_file": result["generated_file"],
                "analysis_source": result["analysis_source"],
                "summary_file": result["summary_file"],
                "rows_generated": stats["generated_all"]["rows"],
                "rows_train_overlap": stats["generated_overlap_train"]["rows"],
                "rows_valid_overlap": stats["generated_overlap_valid"]["rows"],
                "rows_test_overlap": stats["generated_overlap_test"]["rows"],
                "rows_novel": stats["generated_novel"]["rows"],
                "train_overlap_rate": _pct(stats["generated_overlap_train"]["rows"], stats["generated_all"]["rows"]),
                "valid_overlap_rate": _pct(stats["generated_overlap_valid"]["rows"], stats["generated_all"]["rows"]),
                "test_overlap_rate": _pct(stats["generated_overlap_test"]["rows"], stats["generated_all"]["rows"]),
                "novel_rate": _pct(stats["generated_novel"]["rows"], stats["generated_all"]["rows"]),
                "pos_rate_generated": stats["generated_all"]["pos_rate"],
                "pos_rate_train_overlap": stats["generated_overlap_train"]["pos_rate"],
                "pos_rate_novel": stats["generated_novel"]["pos_rate"],
                "mean_jaccard_generated": stats["generated_all"]["all_text_jaccard"]["mean"],
                "mean_jaccard_train_overlap": stats["generated_overlap_train"]["all_text_jaccard"]["mean"],
                "mean_jaccard_novel": stats["generated_novel"]["all_text_jaccard"]["mean"],
                "cluster_error_generated": stats["generated_all"]["cluster_inconsistency_rate"],
                "embed_novel_train_centroid_cosine": novel_train.get("centroid_cosine"),
                "embed_novel_train_mean_nn_cosine": novel_train_nn.get("mean_similarity"),
                "embed_novel_train_mean_nn_distance": novel_train_nn.get("mean_distance"),
                "embed_novel_train_frechet": novel_train.get("frechet_distance"),
                "embed_novel_train_mmd_rbf": novel_train.get("mmd_rbf"),
                "embed_novel_train_precision": novel_train.get("precision"),
                "embed_novel_train_coverage": novel_train.get("coverage"),
                "embed_generated_valid_mean_nn_cosine": all_valid_nn.get("mean_similarity"),
                "embed_generated_test_mean_nn_cosine": all_test_nn.get("mean_similarity"),
            }
        )
    return rows


def write_excel_report(results: List[Dict[str, Any]], output_path: Path) -> None:
    summary_rows = _build_summary_rows(results)
    subset_rows: List[Dict[str, Any]] = []
    overlap_rows: List[Dict[str, Any]] = []
    official_rows: List[Dict[str, Any]] = []
    embedding_diversity_rows: List[Dict[str, Any]] = []
    embedding_comparison_rows: List[Dict[str, Any]] = []
    run_summary_rows: List[Dict[str, Any]] = []

    for result in results:
        base = {
            "benchmark": result["benchmark"],
            "profile": result["profile"],
            "generated_file": result["generated_file"],
            "analysis_source": result["analysis_source"],
        }

        for subset_name, summary in (result.get("subset_stats") or {}).items():
            row = dict(base)
            row["subset"] = subset_name
            row.update(_flatten_mapping(summary))
            subset_rows.append(row)

        for overlap in result.get("split_overlaps") or []:
            row = dict(base)
            row.update(overlap)
            overlap_rows.append(row)

        for split_name, summary in sorted((result.get("official_summaries") or {}).items()):
            row = dict(base)
            row["official_split"] = split_name
            row.update(_flatten_mapping(summary))
            official_rows.append(row)

        embedding_metrics = result.get("embedding_metrics") or {}
        for subset_name, summary in sorted((embedding_metrics.get("diversity") or {}).items()):
            row = dict(base)
            row["embedding_subset"] = subset_name
            row.update(summary)
            embedding_diversity_rows.append(row)

        for comparison_name, summary in sorted((embedding_metrics.get("comparisons") or {}).items()):
            row = dict(base)
            row["embedding_comparison"] = comparison_name
            row.update(_flatten_mapping(summary))
            embedding_comparison_rows.append(row)

        run_summary = result.get("run_summary") or {}
        if run_summary:
            row = dict(base)
            row.update(_flatten_mapping(run_summary))
            run_summary_rows.append(row)

    with pd.ExcelWriter(output_path) as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(subset_rows).to_excel(writer, sheet_name="subset_stats", index=False)
        pd.DataFrame(overlap_rows).to_excel(writer, sheet_name="split_overlaps", index=False)
        pd.DataFrame(official_rows).to_excel(writer, sheet_name="official_summaries", index=False)
        pd.DataFrame(embedding_diversity_rows).to_excel(writer, sheet_name="embed_diversity", index=False)
        pd.DataFrame(embedding_comparison_rows).to_excel(writer, sheet_name="embed_comparisons", index=False)
        pd.DataFrame(run_summary_rows).to_excel(writer, sheet_name="run_summary", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile available auto-labeled training sets against official splits.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--output-md", default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--output-json", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--output-xlsx", default=str(DEFAULT_REPORT_XLSX))
    args = parser.parse_args()

    config = _load_yaml(Path(args.config))
    generated_sets = _discover_generated_sets(config)
    if not generated_sets:
        raise RuntimeError("No generated training json.gz files found under output/simple_labeling.")

    results = [profile_generated_set(generated, config=config) for generated in generated_sets]

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_xlsx = Path(args.output_xlsx)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"generated_profiles": results}, indent=2))
    output_md.write_text(render_markdown(results))
    write_excel_report(results, output_xlsx)

    print(output_md)
    print(output_json)
    print(output_xlsx)


if __name__ == "__main__":
    main()
