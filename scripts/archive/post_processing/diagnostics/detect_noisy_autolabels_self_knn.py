#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "labeling" / "benchmarks_active.yaml"
AUTO_LABEL_V1_ROOT = ROOT / "data" / "auto_label_v1"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "noisy_autolabels_self_knn"
DEFAULT_TOP_RATES = "0.01,0.02,0.05,0.10"
DEFAULT_K = 25
DEFAULT_BATCH_SIZE = 512

POS_LABELS = {"1", "TRUE", "T", "YES", "Y"}
NEG_LABELS = {"0", "FALSE", "F", "NO", "N"}
BENCHMARK_CACHE: Dict[str, Dict[str, Any]] = {}


@dataclass(frozen=True)
class GeneratedSet:
    benchmark: str
    profile: str
    labels_csv: Path


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return payload


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
    if "pair_id" not in out.columns and {"id_left", "id_right"}.issubset(out.columns):
        out["pair_id"] = out["id_left"].astype(str).str.strip() + "#" + out["id_right"].astype(str).str.strip()
    elif "pair_id" in out.columns and not {"id_left", "id_right"}.issubset(out.columns):
        pair_parts = out["pair_id"].astype(str).str.split("#", n=1, expand=True)
        if pair_parts.shape[1] == 2:
            out["id_left"] = pair_parts[0]
            out["id_right"] = pair_parts[1]
    out["label"] = out["label"].map(_normalize_label)
    out["id_left"] = out["id_left"].astype(str).str.strip()
    out["id_right"] = out["id_right"].astype(str).str.strip()
    out["pair_id"] = out["pair_id"].astype(str).str.strip()
    out["pair_key"] = out["id_left"] + "#" + out["id_right"]
    return out.reset_index(drop=True)


def _discover_generated_sets(config: Dict[str, Any]) -> List[GeneratedSet]:
    out: List[GeneratedSet] = []
    config_benchmarks = config.get("benchmarks") or {}
    for manifest_path in sorted(AUTO_LABEL_V1_ROOT.glob("*/profile_manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        benchmark = str(manifest.get("benchmark", "")).strip()
        if benchmark not in config_benchmarks:
            continue
        profiles = manifest.get("profiles") or {}
        if not isinstance(profiles, dict):
            continue
        run_dir = manifest_path.parent
        for profile_name in sorted(profiles.keys()):
            labels_csv = run_dir / "profiles" / profile_name / "labels_final.csv"
            if labels_csv.exists():
                out.append(GeneratedSet(benchmark=benchmark, profile=profile_name, labels_csv=labels_csv))
    return out


def _safe_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return left @ right


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float64, copy=False)
    arr = matrix.astype(np.float64, copy=False)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-12, None)


def _load_benchmark_bundle(benchmark: str, benchmark_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cached = BENCHMARK_CACHE.get(benchmark)
    if cached is not None:
        return cached

    left_source = pd.read_csv(ROOT / benchmark_cfg["left_csv"])
    right_source = pd.read_csv(ROOT / benchmark_cfg["right_csv"])
    left_source["id"] = left_source["id"].astype(str).str.strip()
    right_source["id"] = right_source["id"].astype(str).str.strip()
    embeddings_cfg = benchmark_cfg.get("embeddings") or {}
    emb_dir = ROOT / str(embeddings_cfg.get("dir", ""))
    left_emb = np.load(emb_dir / str(embeddings_cfg.get("left_file", ""))).astype(np.float32, copy=False)
    right_emb = np.load(emb_dir / str(embeddings_cfg.get("right_file", ""))).astype(np.float32, copy=False)
    bundle = {
        "left_source": left_source,
        "right_source": right_source,
        "left_id_to_idx": {rid: idx for idx, rid in enumerate(left_source["id"].tolist())},
        "right_id_to_idx": {rid: idx for idx, rid in enumerate(right_source["id"].tolist())},
        "left_emb": _l2_normalize(left_emb),
        "right_emb": _l2_normalize(right_emb),
    }
    BENCHMARK_CACHE[benchmark] = bundle
    return bundle


def _pair_embeddings(df: pd.DataFrame, *, bundle: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    left_lookup = bundle["left_id_to_idx"]
    right_lookup = bundle["right_id_to_idx"]
    left_emb = bundle["left_emb"]
    right_emb = bundle["right_emb"]
    keep_mask = np.zeros(len(df), dtype=bool)
    left_indices: List[int] = []
    right_indices: List[int] = []
    for idx, (left_id, right_id) in enumerate(zip(df["id_left"].astype(str), df["id_right"].astype(str))):
        lidx = left_lookup.get(left_id.strip())
        ridx = right_lookup.get(right_id.strip())
        if lidx is None or ridx is None:
            continue
        left_indices.append(lidx)
        right_indices.append(ridx)
        keep_mask[idx] = True
    if not left_indices:
        return np.zeros((0, 0), dtype=np.float64), keep_mask
    pair_emb = np.concatenate([left_emb[left_indices], right_emb[right_indices]], axis=1)
    return _l2_normalize(pair_emb), keep_mask


def _prepare_generated_df(labels_csv: Path, bundle: Dict[str, Any]) -> pd.DataFrame:
    df = _normalize_pair_frame(pd.read_csv(labels_csv))
    left_clusters = bundle["left_source"][["id", "cluster_id"]].rename(columns={"id": "id_left", "cluster_id": "cluster_id_left"})
    right_clusters = bundle["right_source"][["id", "cluster_id"]].rename(columns={"id": "id_right", "cluster_id": "cluster_id_right"})
    out = df.merge(left_clusters, on="id_left", how="left").merge(right_clusters, on="id_right", how="left")
    out["cluster_known"] = out["cluster_id_left"].notna() & out["cluster_id_right"].notna()
    out["cluster_match"] = False
    known = out["cluster_known"]
    out.loc[known, "cluster_match"] = (
        out.loc[known, "cluster_id_left"].astype(str).str.strip() == out.loc[known, "cluster_id_right"].astype(str).str.strip()
    )
    out["cluster_proxy_fp"] = ((out["label"] == 1) & (~out["cluster_match"]) & out["cluster_known"])
    out["cluster_proxy_fn"] = ((out["label"] == 0) & (out["cluster_match"]) & out["cluster_known"])
    out["cluster_proxy_error"] = out["cluster_proxy_fp"] | out["cluster_proxy_fn"]
    return out


def _weighted_neighbor_positive_rate(top_sims: np.ndarray, top_labels: np.ndarray) -> np.ndarray:
    weights = np.clip((top_sims + 1.0) / 2.0, 1e-6, None)
    return (weights * top_labels).sum(axis=1) / weights.sum(axis=1)


def _score_self_knn(emb: np.ndarray, labels: np.ndarray, *, k: int, batch_size: int) -> Dict[str, np.ndarray]:
    k_eff = min(k, max(1, emb.shape[0] - 1))
    weighted_pos_rate_parts: List[np.ndarray] = []
    mean_similarity_parts: List[np.ndarray] = []
    max_similarity_parts: List[np.ndarray] = []
    neighbor_pos_rate_parts: List[np.ndarray] = []

    for start in range(0, emb.shape[0], batch_size):
        end = min(start + batch_size, emb.shape[0])
        sims = _safe_matmul(emb[start:end], emb.T)
        local_rows = np.arange(end - start)
        global_cols = start + local_rows
        sims[local_rows, global_cols] = -np.inf
        top_idx = np.argpartition(sims, kth=sims.shape[1] - k_eff, axis=1)[:, -k_eff:]
        top_sims = np.take_along_axis(sims, top_idx, axis=1)
        order = np.argsort(top_sims, axis=1)[:, ::-1]
        top_idx = np.take_along_axis(top_idx, order, axis=1)
        top_sims = np.take_along_axis(top_sims, order, axis=1)
        top_labels = labels[top_idx]
        weighted_pos_rate_parts.append(_weighted_neighbor_positive_rate(top_sims, top_labels))
        neighbor_pos_rate_parts.append(top_labels.mean(axis=1))
        mean_similarity_parts.append(top_sims.mean(axis=1))
        max_similarity_parts.append(top_sims[:, 0])

    return {
        "weighted_pos_rate": np.concatenate(weighted_pos_rate_parts, axis=0),
        "neighbor_pos_rate": np.concatenate(neighbor_pos_rate_parts, axis=0),
        "mean_similarity": np.concatenate(mean_similarity_parts, axis=0),
        "max_similarity": np.concatenate(max_similarity_parts, axis=0),
    }


def _compute_budget_rows(scored_df: pd.DataFrame, top_rates: List[float]) -> List[Dict[str, Any]]:
    proxy_total = int(scored_df["cluster_proxy_error"].sum())
    proxy_fp_total = int(scored_df["cluster_proxy_fp"].sum())
    proxy_fn_total = int(scored_df["cluster_proxy_fn"].sum())
    ranked = scored_df.sort_values(["suspicion_score", "self_mean_similarity"], ascending=[False, False]).reset_index(drop=True)
    rows: List[Dict[str, Any]] = []
    for rate in top_rates:
        top_n = max(1, int(math.ceil(rate * len(ranked))))
        flagged = ranked.head(top_n)
        found = int(flagged["cluster_proxy_error"].sum())
        fp_found = int(flagged["cluster_proxy_fp"].sum())
        fn_found = int(flagged["cluster_proxy_fn"].sum())
        rows.append(
            {
                "budget_value": rate,
                "flagged_pairs": top_n,
                "proxy_errors_found": found,
                "proxy_false_positive_found": fp_found,
                "proxy_false_negative_found": fn_found,
                "proxy_precision": float(found / top_n) if top_n else None,
                "proxy_recall": float(found / proxy_total) if proxy_total else None,
                "proxy_fp_recall": float(fp_found / proxy_fp_total) if proxy_fp_total else None,
                "proxy_fn_recall": float(fn_found / proxy_fn_total) if proxy_fn_total else None,
            }
        )
    return rows


def _parse_rates(text: str) -> List[float]:
    values: List[float] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = float(chunk)
        if value <= 0 or value >= 1:
            raise ValueError(f"Top rate must be between 0 and 1, got {value}")
        values.append(value)
    if not values:
        raise ValueError("Need at least one top-rate")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect suspicious auto-labeled pairs via self-kNN label inconsistency in embedding space.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--top-rates", default=DEFAULT_TOP_RATES)
    args = parser.parse_args()

    top_rates = _parse_rates(args.top_rates)
    config = _load_yaml(Path(args.config))
    generated_sets = _discover_generated_sets(config)
    if not generated_sets:
        raise RuntimeError(f"No generated labels found under {AUTO_LABEL_V1_ROOT}")

    output_dir = Path(args.output_dir)
    candidates_dir = output_dir / "candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    budget_rows: List[Dict[str, Any]] = []
    top_candidates: List[pd.DataFrame] = []

    for generated in generated_sets:
        benchmark_cfg = (config.get("benchmarks") or {}).get(generated.benchmark) or {}
        bundle = _load_benchmark_bundle(generated.benchmark, benchmark_cfg)
        generated_df = _prepare_generated_df(generated.labels_csv, bundle)
        emb, keep_mask = _pair_embeddings(generated_df, bundle=bundle)
        scored = generated_df.loc[keep_mask].reset_index(drop=True).copy()
        labels = scored["label"].astype(int).to_numpy()
        scores = _score_self_knn(emb, labels, k=args.k, batch_size=args.batch_size)

        scored["self_neighbor_pos_rate"] = scores["neighbor_pos_rate"]
        scored["self_weighted_pos_rate"] = scores["weighted_pos_rate"]
        scored["self_mean_similarity"] = scores["mean_similarity"]
        scored["self_max_similarity"] = scores["max_similarity"]
        scored["self_pred_label"] = (scored["self_weighted_pos_rate"] >= 0.5).astype(int)
        scored["disagreement"] = np.where(
            scored["label"] == 1,
            1.0 - scored["self_weighted_pos_rate"],
            scored["self_weighted_pos_rate"],
        )
        scored["suspicion_score"] = scored["disagreement"] * np.clip(scored["self_mean_similarity"], 0.0, 1.0)
        scored["rank"] = scored["suspicion_score"].rank(method="first", ascending=False).astype(int)

        proxy_total = int(scored["cluster_proxy_error"].sum())
        proxy_fp_total = int(scored["cluster_proxy_fp"].sum())
        proxy_fn_total = int(scored["cluster_proxy_fn"].sum())
        summary_rows.append(
            {
                "benchmark": generated.benchmark,
                "profile": generated.profile,
                "labels_csv": str(generated.labels_csv.relative_to(ROOT)),
                "rows_scored": int(len(scored)),
                "cluster_proxy_errors": proxy_total,
                "cluster_proxy_false_positive": proxy_fp_total,
                "cluster_proxy_false_negative": proxy_fn_total,
                "mean_suspicion_score": float(scored["suspicion_score"].mean()),
                "mean_disagreement": float(scored["disagreement"].mean()),
                "mean_similarity": float(scored["self_mean_similarity"].mean()),
                "k": args.k,
            }
        )

        budgets = _compute_budget_rows(scored, top_rates)
        for row in budgets:
            row["benchmark"] = generated.benchmark
            row["profile"] = generated.profile
        budget_rows.extend(budgets)

        scored = scored.sort_values(["suspicion_score", "self_mean_similarity"], ascending=[False, False]).reset_index(drop=True)
        out_path = candidates_dir / f"{generated.benchmark}_{generated.profile}_self_knn_candidates.csv.gz"
        scored.to_csv(out_path, index=False, compression="gzip")
        top_candidates.append(scored.head(200).assign(source_file=out_path.name))

    summary_json = output_dir / "summary.json"
    summary_xlsx = output_dir / "summary.xlsx"
    summary_json.write_text(json.dumps({"method": "self_knn_generated_only", "summary": summary_rows, "budgets": budget_rows}, indent=2))
    top_candidates_df = pd.concat(top_candidates, ignore_index=True) if top_candidates else pd.DataFrame()
    with pd.ExcelWriter(summary_xlsx) as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(budget_rows).to_excel(writer, sheet_name="budget_metrics", index=False)
        top_candidates_df.to_excel(writer, sheet_name="top_candidates", index=False)

    print(summary_xlsx)
    print(summary_json)
    print(candidates_dir)


if __name__ == "__main__":
    main()
