#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "labeling" / "benchmarks_active.yaml"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark_module = _load_module(
    "_run_benchmark_labeling_module",
    ROOT / "scripts" / "archive" / "labeling_helpers" / "run_benchmark_labeling.py",
)
simple_module = _load_module(
    "_run_simple_labeling_module",
    Path(__file__).resolve().with_name("active_learning_ml.py"),
)

_build_candidates = simple_module._build_candidates
_count_labels = simple_module._count_labels
_label_query_until_satisfied_seed = simple_module._label_query_until_satisfied_seed
_label_iteratively_per_query_seed = simple_module._label_iteratively_per_query_seed
_load_resume_labels = simple_module._load_resume_labels
_materialize_output_ids = simple_module._materialize_output_ids
_save_json = simple_module._save_json
_select_balanced_set = simple_module._select_balanced_set

_build_random_profile_name = benchmark_module._build_random_profile_name
_build_random_profile_labels = benchmark_module._build_random_profile_labels
_coerce_field_list = benchmark_module._coerce_field_list
_coerce_mapping = benchmark_module._coerce_mapping
_coerce_str_mapping = benchmark_module._coerce_str_mapping
_estimate_usage_costs = benchmark_module._estimate_usage_costs
_load_df = benchmark_module._load_df
_load_yaml = benchmark_module._load_yaml
_materialize_source_csv = benchmark_module._materialize_source_csv
_merge_base_with_random_labels = benchmark_module._merge_base_with_random_labels
_normalize_benchmark_config = benchmark_module._normalize_benchmark_config
_normalize_field_mapping = benchmark_module._normalize_field_mapping
_normalize_label = benchmark_module._normalize_label
_read_json_if_exists = benchmark_module._read_json_if_exists
_resolve_random_profile_settings = benchmark_module._resolve_random_profile_settings
_resolve_train_fields = benchmark_module._resolve_train_fields
_select_profiles = benchmark_module._select_profiles
_subset_from_master = benchmark_module._subset_from_master
_write_profile_outputs = benchmark_module._write_profile_outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build benchmark profiles using only the simple-active-learning seed-round strategy. "
            "No active-learning loop and no Ditto phase are run."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Benchmark labeling config.")
    parser.add_argument(
        "--benchmarks",
        default="",
        help="Comma-separated benchmark keys. Default: all benchmarks in config.",
    )
    parser.add_argument(
        "--profiles",
        default="",
        help="Comma-separated profile names. Default: all configured profiles.",
    )
    parser.add_argument(
        "--output-root",
        default="output/seed_round_only_profiles",
        help="Output root for generated runs.",
    )
    parser.add_argument(
        "--existing-run-dir",
        default="",
        help=(
            "Resume or rebuild an existing run directory in-place instead of creating a new "
            "timestamped run directory. Intended for crashed runs."
        ),
    )
    parser.add_argument(
        "--run-name-suffix",
        default="",
        help="Optional suffix appended after benchmark_<name> in the run directory name.",
    )
    parser.add_argument("--model", default="", help="Override labeling model from config.")
    parser.add_argument("--resume", action="store_true", help="Reuse an existing run directory if seed labels exist.")
    parser.add_argument("--no-export-ditto", action="store_true", help="Skip Ditto train json.gz export.")
    parser.add_argument(
        "--include-random-profiles",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Random profiles are now built by default when "
            "random_profile_enabled is set in the config."
        ),
    )
    parser.add_argument(
        "--skip-random-profiles",
        action="store_true",
        help=(
            "Do not build *_plus20random profiles. By default they are built when "
            "random_profile_enabled is set in the config."
        ),
    )
    parser.add_argument(
        "--seed-max-calls",
        type=int,
        default=0,
        help=(
            "Override the seed-round max LLM calls. Default: max(config seed_max_calls, "
            "largest profile label_budget if present, 2 * largest profile size)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved plans without labeling.",
    )
    return parser.parse_args()


def _resolve_repo_path(raw: Any) -> Path:
    path = Path(str(raw)).expanduser()
    return path if path.is_absolute() else ROOT / path


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _count_binary_labels(df: pd.DataFrame) -> tuple[int, int, int]:
    labels = df["label"].astype(str).str.upper()
    pos = int((labels == "TRUE").sum())
    neg = int((labels == "FALSE").sum())
    return int(len(df)), pos, neg


def _selected_benchmarks(config: Dict[str, Any], raw: str) -> List[str]:
    benchmarks = _coerce_mapping(config.get("benchmarks"), "benchmarks")
    requested = [x.strip() for x in raw.replace(",", " ").split() if x.strip()]
    if not requested:
        return list(benchmarks.keys())
    missing = [name for name in requested if name not in benchmarks]
    if missing:
        raise KeyError(f"Unknown benchmark(s): {missing}. Available: {sorted(benchmarks)}")
    return requested


def _infer_benchmark_from_run_dir(run_dir: Path) -> str | None:
    manifest_path = run_dir / "profile_manifest.json"
    if manifest_path.exists():
        payload = _read_json_if_exists(manifest_path)
        benchmark = str(payload.get("benchmark", "")).strip()
        if benchmark:
            return benchmark

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        payload = _read_json_if_exists(summary_path)
        benchmark = str(payload.get("benchmark", "")).strip()
        if benchmark:
            return benchmark

    name = run_dir.name
    if name.startswith("benchmark_"):
        suffix = name[len("benchmark_") :]
        parts = suffix.split("_")
        if len(parts) >= 3:
            return "_".join(parts[:-2]).replace("_", "-")
    return None


def _profile_budget(raw_profile_cfg: Dict[str, Any]) -> int | None:
    for key in ("label_budget", "seed_label_budget", "max_calls"):
        value = raw_profile_cfg.get(key)
        if value is not None:
            return int(value)
    return None


def _prepare_canonical(
    *,
    run_dir: Path,
    benchmark_cfg: Dict[str, Any],
    left_fields: Dict[str, str],
    right_fields: Dict[str, str],
    train_fields: Sequence[str],
) -> tuple[Path, Path]:
    canonical_dir = run_dir / "canonical"
    left_canonical = canonical_dir / "left.csv"
    right_canonical = canonical_dir / "right.csv"
    _materialize_source_csv(
        _resolve_repo_path(benchmark_cfg["left_csv"]),
        str(benchmark_cfg.get("left_id_col", "id")),
        left_fields,
        left_canonical,
        ensure_fields=train_fields,
    )
    _materialize_source_csv(
        _resolve_repo_path(benchmark_cfg["right_csv"]),
        str(benchmark_cfg.get("right_id_col", "id")),
        right_fields,
        right_canonical,
        ensure_fields=train_fields,
    )
    return left_canonical, right_canonical


def _continue_seed_labeling_from_existing(
    *,
    client: OpenAI,
    model: str,
    all_neighbors: pd.DataFrame,
    left_map: Dict[str, Dict[str, object]],
    right_map: Dict[str, Dict[str, object]],
    resumed_labeled: pd.DataFrame,
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
    current = resumed_labeled.copy().reset_index(drop=True)
    current["id1"] = current["id1"].astype(str)
    current["id2"] = current["id2"].astype(str)
    current["label"] = current["label"].astype(str).str.upper()
    current["similarity"] = pd.to_numeric(current["similarity"], errors="coerce").fillna(0.0).astype(float)
    current = current.drop_duplicates(subset=["id1", "id2"], keep="last").reset_index(drop=True)

    total_positives, total_negatives = _count_labels(current)
    target_total_negatives = max(int(total_target_size) - int(total_target_positives), 0)
    if total_positives >= total_target_positives and total_negatives >= target_total_negatives:
        labeled = _select_balanced_set(current, total_target_size, total_target_positives)
        labeled.to_csv(out_csv, index=False)
        return labeled

    completed_queries = set(current["id1"].astype(str).unique().tolist())
    query_groups = {qid: g for qid, g in all_neighbors.groupby("query_id")}
    if query_order == "similarity":
        query_max = all_neighbors.groupby("query_id")["similarity"].max()
        query_ids = query_max.sort_values(ascending=False).index.to_numpy().tolist()
    elif query_order == "random":
        rng = np.random.default_rng(42)
        query_ids = all_neighbors["query_id"].drop_duplicates().tolist()
        rng.shuffle(query_ids)
    else:
        query_ids = all_neighbors["query_id"].drop_duplicates().tolist()

    prior_state = _read_json_if_exists(state_path)
    start_calls = int(prior_state.get("llm_calls", len(current)) or len(current))
    counters = {"calls": int(min(start_calls, max_calls))}
    queries_processed = len(completed_queries)
    queries_with_matches = int(current.loc[current["label"] == "TRUE", "id1"].nunique())
    all_labeled: List[pd.DataFrame] = [current]

    pbar = tqdm(total=max_calls, initial=counters["calls"], desc="Seed labeling", unit="call")
    pbar.set_postfix(
        pos=total_positives,
        neg=total_negatives,
        accepted=int(len(current)),
        calls=f"{counters['calls']}/{max_calls}",
        pos_target=total_target_positives,
        neg_target=target_total_negatives,
    )

    exhausted = counters["calls"] >= max_calls
    for qid in query_ids:
        if exhausted:
            break
        if total_positives >= total_target_positives and total_negatives >= target_total_negatives:
            break
        if str(qid) in completed_queries:
            continue

        query_labeled, exhausted = _label_query_until_satisfied_seed(
            client,
            model,
            query_groups[qid],
            left_map,
            right_map,
            target_positives=int(target_positives_per_query),
            target_negatives=int(target_negatives_per_query),
            batch_size=int(batch_size),
            bottom_k=int(bottom_k),
            max_calls=int(max_calls),
            counters=counters,
            usage_stats=usage_stats,
            pbar=pbar,
        )
        queries_processed += 1
        completed_queries.add(str(qid))

        if not query_labeled.empty:
            q_pos = int((query_labeled["label"] == "TRUE").sum())
            q_neg = int((query_labeled["label"] == "FALSE").sum())
            keep_query = (q_pos > 0) or (total_negatives < target_total_negatives and q_neg > 0)
            if keep_query:
                all_labeled.append(query_labeled)
                total_positives += q_pos
                total_negatives += q_neg
                if q_pos > 0:
                    queries_with_matches += 1

        accepted = int(sum(len(frame) for frame in all_labeled))
        pbar.set_postfix(
            pos=total_positives,
            neg=total_negatives,
            accepted=accepted,
            calls=f"{counters['calls']}/{max_calls}",
            pos_target=total_target_positives,
            neg_target=target_total_negatives,
        )

        if counters["calls"] % 10 == 0:
            snapshot = pd.concat(all_labeled, ignore_index=True) if all_labeled else pd.DataFrame(
                columns=["id1", "id2", "label", "similarity"]
            )
            snapshot.to_csv(out_csv, index=False)
            _save_json(
                state_path,
                {
                    "stage": "seed_labeling",
                    "llm_calls": counters["calls"],
                    "seed_pos": total_positives,
                    "seed_neg": total_negatives,
                    "seed_total": len(snapshot),
                    "seed_target_pos": int(total_target_positives),
                    "seed_target_neg": int(target_total_negatives),
                    "queries_processed": queries_processed,
                    "queries_with_matches": queries_with_matches,
                    "token_usage": usage_stats,
                },
            )

    pbar.close()
    labeled = pd.concat(all_labeled, ignore_index=True) if all_labeled else pd.DataFrame(
        columns=["id1", "id2", "label", "similarity"]
    )
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


def _build_seed_master(
    *,
    benchmark: str,
    benchmark_cfg: Dict[str, Any],
    labeling_args: Dict[str, Any],
    run_dir: Path,
    left_canonical: Path,
    right_canonical: Path,
    target_size: int,
    target_positives: int,
    max_calls: int,
    model: str,
    resume: bool,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    state_path = run_dir / "run_state.json"
    existing_usage = _read_json_if_exists(state_path).get("token_usage") or {}
    usage_stats: Dict[str, int] = {
        "prompt_tokens": int(existing_usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(existing_usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(existing_usage.get("total_tokens", 0) or 0),
        "seed_prompt_tokens": int(existing_usage.get("seed_prompt_tokens", 0) or 0),
        "seed_completion_tokens": int(existing_usage.get("seed_completion_tokens", 0) or 0),
        "seed_total_tokens": int(existing_usage.get("seed_total_tokens", 0) or 0),
        "active_prompt_tokens": 0,
        "active_completion_tokens": 0,
        "active_total_tokens": 0,
    }

    left_df = _load_df(left_canonical, side="left", schema_map={})
    right_df = _load_df(right_canonical, side="right", schema_map={})
    left_rid_to_id = dict(zip(left_df["__rid"].astype(str), left_df["id"].astype(str)))
    right_rid_to_id = dict(zip(right_df["__rid"].astype(str), right_df["id"].astype(str)))
    left_map = {str(row["__rid"]): row.to_dict() for _, row in left_df.iterrows()}
    right_map = {str(row["__rid"]): row.to_dict() for _, row in right_df.iterrows()}

    embeddings_dir = _resolve_repo_path(benchmark_cfg["embeddings_dir"])
    left_emb = np.load(embeddings_dir / str(benchmark_cfg["left_emb"]))
    right_emb = np.load(embeddings_dir / str(benchmark_cfg["right_emb"]))

    candidates, faiss_stats = _build_candidates(
        left_ids=left_df["__rid"].astype(str).to_numpy(),
        right_ids=right_df["__rid"].astype(str).to_numpy(),
        right_source_ids=right_df["id"].astype(str).to_numpy(),
        left_emb=left_emb,
        right_emb=right_emb,
        k=int(labeling_args.get("faiss_k", 20) or 20),
        candidate_cap=int(labeling_args.get("candidate_cap", 0) or 0),
        bottom_k=int(labeling_args.get("seed_bottom_k", 2) or 2),
        random_state=int(labeling_args.get("faiss_random_state", 42) or 42),
    )
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
    _materialize_output_ids(candidates, left_rid_to_id, right_rid_to_id).to_csv(
        run_dir / "faiss_candidates.csv",
        index=False,
    )

    resumed_labeled: pd.DataFrame | None = None
    resumed_from: str | None = None
    if resume:
        resumed_labeled, resumed_from = _load_resume_labels(run_dir)
        if resumed_labeled is not None:
            pos, neg = _count_labels(resumed_labeled)
            if pos >= target_positives and len(resumed_labeled) >= target_size:
                seed_internal = resumed_labeled.copy().reset_index(drop=True)
                seed_internal = seed_internal.head(target_size)
                seed_output = _materialize_output_ids(seed_internal, left_rid_to_id, right_rid_to_id)
                seed_output.to_csv(run_dir / "seed_labels.csv", index=False)
                seed_output.to_csv(run_dir / "active_labels_latest.csv", index=False)
                seed_output.to_csv(run_dir / "labels_final.csv", index=False)
                return seed_output, {
                    "resumed_from": resumed_from,
                    "target_size": int(target_size),
                    "target_positives": int(target_positives),
                    "max_calls": int(max_calls),
                    "faiss": faiss_stats,
                    "token_usage": usage_stats,
                }
            print(
                f"[{benchmark}] Resume labels exist in {resumed_from}, but they are incomplete "
                f"for target size={target_size}, positives={target_positives}. Continuing from checkpoint.",
                flush=True,
            )
            all_neighbors = candidates.rename(columns={"id1": "query_id", "id2": "neighbor_id"}).copy()
            all_neighbors["rank"] = (
                all_neighbors.groupby("query_id")["similarity"]
                .rank(method="first", ascending=False)
                .astype(int)
            )
            load_dotenv()
            client = OpenAI()
            seed_internal = _continue_seed_labeling_from_existing(
                client=client,
                model=model,
                all_neighbors=all_neighbors,
                left_map=left_map,
                right_map=right_map,
                resumed_labeled=resumed_labeled,
                target_positives_per_query=int(labeling_args.get("seed_pos_per_query", 1) or 1),
                target_negatives_per_query=int(labeling_args.get("seed_neg_per_query", 4) or 4),
                total_target_positives=int(target_positives),
                total_target_size=int(target_size),
                max_calls=int(max_calls),
                batch_size=int(labeling_args.get("seed_batch_size", 5) or 5),
                query_order=str(labeling_args.get("seed_query_order", "random") or "random"),
                bottom_k=int(labeling_args.get("seed_bottom_k", 2) or 2),
                state_path=state_path,
                out_csv=run_dir / "seed_labels_internal.csv",
                usage_stats=usage_stats,
            )
            if seed_internal.empty:
                raise RuntimeError(f"[{benchmark}] Seed labeling continuation produced zero rows")

            seed_output = _materialize_output_ids(seed_internal, left_rid_to_id, right_rid_to_id)
            seed_output["label"] = seed_output["label"].apply(_normalize_label)
            seed_output.to_csv(run_dir / "seed_labels.csv", index=False)
            seed_output.to_csv(run_dir / "active_labels_latest.csv", index=False)
            seed_output.to_csv(run_dir / "labels_final.csv", index=False)
            pos, neg = _count_labels(seed_output)
            print(f"[{benchmark}] Seed-only master: total={len(seed_output)} pos={pos} neg={neg}")
            return seed_output, {
                "resumed_from": resumed_from,
                "target_size": int(target_size),
                "target_positives": int(target_positives),
                "max_calls": int(max_calls),
                "faiss": faiss_stats,
                "token_usage": usage_stats,
            }

    all_neighbors = candidates.rename(columns={"id1": "query_id", "id2": "neighbor_id"}).copy()
    all_neighbors["rank"] = (
        all_neighbors.groupby("query_id")["similarity"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    load_dotenv()
    client = OpenAI()
    seed_internal = _label_iteratively_per_query_seed(
        client=client,
        model=model,
        all_neighbors=all_neighbors,
        left_map=left_map,
        right_map=right_map,
        target_positives_per_query=int(labeling_args.get("seed_pos_per_query", 1) or 1),
        target_negatives_per_query=int(labeling_args.get("seed_neg_per_query", 4) or 4),
        total_target_positives=int(target_positives),
        total_target_size=int(target_size),
        max_calls=int(max_calls),
        batch_size=int(labeling_args.get("seed_batch_size", 5) or 5),
        query_order=str(labeling_args.get("seed_query_order", "random") or "random"),
        bottom_k=int(labeling_args.get("seed_bottom_k", 2) or 2),
        state_path=state_path,
        out_csv=run_dir / "seed_labels_internal.csv",
        usage_stats=usage_stats,
    )
    if seed_internal.empty:
        raise RuntimeError(f"[{benchmark}] Seed labeling produced zero rows")

    seed_output = _materialize_output_ids(seed_internal, left_rid_to_id, right_rid_to_id)
    seed_output["label"] = seed_output["label"].apply(_normalize_label)
    seed_output.to_csv(run_dir / "seed_labels.csv", index=False)
    seed_output.to_csv(run_dir / "active_labels_latest.csv", index=False)
    seed_output.to_csv(run_dir / "labels_final.csv", index=False)
    pos, neg = _count_labels(seed_output)
    print(f"[{benchmark}] Seed-only master: total={len(seed_output)} pos={pos} neg={neg}")

    return seed_output, {
        "target_size": int(target_size),
        "target_positives": int(target_positives),
        "max_calls": int(max_calls),
        "faiss": faiss_stats,
        "token_usage": usage_stats,
    }


def _run_benchmark(
    *,
    benchmark: str,
    cfg: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    defaults = _coerce_mapping(cfg.get("defaults"), "defaults")
    benchmarks = _coerce_mapping(cfg.get("benchmarks"), "benchmarks")
    profiles_cfg = _coerce_mapping(cfg.get("profiles"), "profiles")
    benchmark_cfg = _normalize_benchmark_config(
        _coerce_mapping(benchmarks[benchmark], f"benchmarks.{benchmark}")
    )

    left_fields = _normalize_field_mapping(_coerce_str_mapping(benchmark_cfg.get("left_fields"), "left_fields"))
    right_fields = _normalize_field_mapping(_coerce_str_mapping(benchmark_cfg.get("right_fields"), "right_fields"))
    train_fields_raw = _coerce_field_list(benchmark_cfg.get("train_fields"), "train_fields")
    train_fields = _resolve_train_fields(train_fields_raw, left_fields=left_fields, right_fields=right_fields)

    benchmark_profile_overrides = _coerce_mapping(benchmark_cfg.get("profiles"), "benchmarks.*.profiles")
    effective_profiles_cfg = dict(profiles_cfg)
    effective_profiles_cfg.update(benchmark_profile_overrides)
    requested_profiles = [x.strip() for x in args.profiles.split(",") if x.strip()]
    profiles = _select_profiles(effective_profiles_cfg, requested_profiles)
    numeric_profiles = [profile for profile in profiles if not profile.all_examples]
    if not numeric_profiles:
        raise ValueError(f"[{benchmark}] At least one numeric profile is required for seed-only labeling")
    largest = numeric_profiles[-1]

    raw_largest_profile_cfg = _coerce_mapping(
        effective_profiles_cfg.get(largest.name),
        f"profiles.{largest.name}",
    )
    profile_budget = _profile_budget(raw_largest_profile_cfg)

    labeling_defaults = _coerce_mapping(defaults.get("labeling_args"), "defaults.labeling_args")
    labeling_override = _coerce_mapping(benchmark_cfg.get("labeling_args"), "benchmarks.*.labeling_args")
    labeling_args = dict(labeling_defaults)
    labeling_args.update(labeling_override)
    model = args.model.strip() or str(labeling_args.get("model", "gpt-5.2") or "gpt-5.2")
    max_calls = int(
        args.seed_max_calls
        or max(
            int(labeling_args.get("seed_max_calls", 0) or 0),
            int(profile_budget or 0),
            int(largest.target_size * 2),
        )
    )

    if str(args.existing_run_dir).strip():
        run_dir = Path(str(args.existing_run_dir)).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        output_root = run_dir.parent
    else:
        suffix = str(args.run_name_suffix).strip()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"benchmark_{benchmark}_{suffix or timestamp}"
        output_root = Path(args.output_root).expanduser()
        run_dir = (output_root / run_name).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[{benchmark}] seed-only target profile={largest.name} "
        f"target={largest.target_size} pos={largest.target_pos} max_calls={max_calls}"
    )
    if args.dry_run:
        return {
            "benchmark": benchmark,
            "run_dir": str(run_dir),
            "largest_profile": largest.name,
            "target_size": int(largest.target_size),
            "target_positives": int(largest.target_pos),
            "max_calls": int(max_calls),
            "random_profiles": bool(not args.skip_random_profiles),
            "dry_run": True,
        }

    left_canonical, right_canonical = _prepare_canonical(
        run_dir=run_dir,
        benchmark_cfg=benchmark_cfg,
        left_fields=left_fields,
        right_fields=right_fields,
        train_fields=train_fields,
    )

    master, seed_summary = _build_seed_master(
        benchmark=benchmark,
        benchmark_cfg=benchmark_cfg,
        labeling_args=labeling_args,
        run_dir=run_dir,
        left_canonical=left_canonical,
        right_canonical=right_canonical,
        target_size=largest.target_size,
        target_positives=largest.target_pos,
        max_calls=max_calls,
        model=model,
        resume=bool(args.resume or str(args.existing_run_dir).strip()),
    )
    master["label"] = master["label"].apply(_normalize_label)

    random_profile_settings = _resolve_random_profile_settings(defaults, benchmark_cfg)
    build_random_profiles = bool(not args.skip_random_profiles)
    if not build_random_profiles:
        random_profile_settings = {**random_profile_settings, "enabled": False}

    labeling_cost = _estimate_usage_costs(model, seed_summary["token_usage"])
    manifest: Dict[str, Any] = {
        "benchmark": benchmark,
        "run_dir": str(run_dir),
        "runner_script": "scripts/labeling/similarity_search.py",
        "run_summary_json": str(run_dir / "summary.json"),
        "labeling_cost": labeling_cost,
        "source_train_file": str(run_dir / "labels_final.csv"),
        "source_all_examples_file": str(run_dir / "active_labels_latest.csv"),
        "canonical_left_csv": str(left_canonical),
        "canonical_right_csv": str(right_canonical),
        "left_fields": left_fields,
        "right_fields": right_fields,
        "ditto_fields": train_fields,
        "seed_round_only": {
            "applied": True,
            "master_profile": largest.name,
            "target_total": int(largest.target_size),
            "target_pos": int(largest.target_pos),
            "max_calls": int(max_calls),
            "model": model,
            "uses_active_learning_loop": False,
            "uses_ditto_phase": False,
        },
        "random_profile_settings": random_profile_settings,
        "profiles": {},
    }

    export_ditto = not args.no_export_ditto
    profiles_root = run_dir / "profiles"
    profiles_root.mkdir(parents=True, exist_ok=True)
    built_profiles: List[str] = []
    skipped_profiles: List[Dict[str, str]] = []
    random_labels_df = pd.DataFrame()
    random_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    random_meta: Dict[str, Any] = {"candidate_count": 0, "labeled_count": 0}

    if build_random_profiles and random_profile_settings.get("enabled"):
        random_profile_counts = {
            spec.name: max(1, int(round(spec.target_size * float(random_profile_settings["fraction"]))))
            for spec in profiles
            if not spec.all_examples
        }
        max_random_additions = max(random_profile_counts.values()) if random_profile_counts else 0
        if max_random_additions > 0:
            random_labels_df, random_usage, random_meta = _build_random_profile_labels(
                run_dir=run_dir,
                benchmark_cfg=benchmark_cfg,
                labeling_args=labeling_args,
                left_canonical=left_canonical,
                right_canonical=right_canonical,
                active_master_path=run_dir / "active_labels_latest.csv",
                master=master,
                sample_n=max_random_additions,
                model=str(random_profile_settings["model"]),
                seed=int(random_profile_settings["seed"]),
            )
            manifest["random_profile_usage"] = random_usage
            manifest["random_profile_cost"] = _estimate_usage_costs(str(random_profile_settings["model"]), random_usage)
            manifest["random_profile_meta"] = random_meta
            if manifest.get("labeling_cost") and manifest.get("random_profile_cost", {}).get("available"):
                manifest["combined_labeling_cost_usd"] = float(manifest["labeling_cost"]["total_cost_usd"]) + float(
                    manifest["random_profile_cost"]["total_cost_usd"]
                )
            print(
                f"[{benchmark}] Random profile pool: labeled {len(random_labels_df)}/{max_random_additions}",
                flush=True,
            )

    for spec in profiles:
        try:
            if spec.all_examples:
                subset = master.copy().reset_index(drop=True)
                target_total, target_pos, target_neg = _count_binary_labels(subset)
            else:
                subset = _subset_from_master(master, target_pos=spec.target_pos, target_neg=spec.target_neg)
                target_total = int(spec.target_size)
                target_pos = int(spec.target_pos)
                target_neg = int(spec.target_neg)
        except RuntimeError as exc:
            skipped_profiles.append({"profile": spec.name, "reason": str(exc)})
            print(f"[{benchmark}] Skipping profile {spec.name}: {exc}", flush=True)
            continue

        output_meta = _write_profile_outputs(
            benchmark=benchmark,
            profile_name=spec.name,
            subset=subset,
            profile_dir=profiles_root / spec.name,
            left_canonical=left_canonical,
            right_canonical=right_canonical,
            train_fields=train_fields,
            export_ditto=export_ditto,
        )
        profile_meta: Dict[str, Any] = {
            "all_examples": bool(spec.all_examples),
            "target_total": target_total,
            "target_pos": target_pos,
            "target_neg": target_neg,
        }
        profile_meta.update(output_meta)
        manifest["profiles"][spec.name] = profile_meta
        built_profiles.append(spec.name)
        print(
            f"[{benchmark}] Profile {spec.name}: total={profile_meta['actual_total']} "
            f"pos={profile_meta['actual_pos']} neg={profile_meta['actual_neg']}"
        )

        if (
            not spec.all_examples
            and build_random_profiles
            and random_profile_settings.get("enabled")
            and not random_labels_df.empty
        ):
            augmented_name = _build_random_profile_name(spec.name, float(random_profile_settings["fraction"]))
            extra_n = max(1, int(round(spec.target_size * float(random_profile_settings["fraction"]))))
            augmented_subset = _merge_base_with_random_labels(subset, random_labels_df, extra_n=extra_n)
            augmented_output_meta = _write_profile_outputs(
                benchmark=benchmark,
                profile_name=augmented_name,
                subset=augmented_subset,
                profile_dir=profiles_root / augmented_name,
                left_canonical=left_canonical,
                right_canonical=right_canonical,
                train_fields=train_fields,
                export_ditto=export_ditto,
            )
            augmented_meta: Dict[str, Any] = {
                "all_examples": False,
                "base_profile": spec.name,
                "random_fraction": float(random_profile_settings["fraction"]),
                "random_additions": int(extra_n),
                "target_total": int(len(augmented_subset)),
                "shared_random_pool": True,
                "shared_random_model": str(random_profile_settings["model"]),
            }
            augmented_meta["target_pos"] = int(augmented_output_meta["actual_pos"])
            augmented_meta["target_neg"] = int(augmented_output_meta["actual_neg"])
            augmented_meta.update(augmented_output_meta)
            manifest["profiles"][augmented_name] = augmented_meta
            built_profiles.append(augmented_name)
            print(
                f"[{benchmark}] Profile {augmented_name}: total={augmented_meta['actual_total']} "
                f"pos={augmented_meta['actual_pos']} neg={augmented_meta['actual_neg']} "
                f"(base={spec.name}, random_additions={extra_n})"
            )

    if build_random_profiles and random_profile_settings.get("enabled") and not random_labels_df.empty:
        all_augmented_name = _build_random_profile_name("all", float(random_profile_settings["fraction"]))
        all_augmented_subset = _merge_base_with_random_labels(
            master,
            random_labels_df,
            extra_n=int(len(random_labels_df)),
        )
        all_augmented_output_meta = _write_profile_outputs(
            benchmark=benchmark,
            profile_name=all_augmented_name,
            subset=all_augmented_subset,
            profile_dir=profiles_root / all_augmented_name,
            left_canonical=left_canonical,
            right_canonical=right_canonical,
            train_fields=train_fields,
            export_ditto=export_ditto,
        )
        all_augmented_meta: Dict[str, Any] = {
            "all_examples": True,
            "base_profile": "all",
            "random_fraction": float(random_profile_settings["fraction"]),
            "random_additions": int(len(random_labels_df)),
            "target_total": int(len(all_augmented_subset)),
            "target_pos": int(all_augmented_output_meta["actual_pos"]),
            "target_neg": int(all_augmented_output_meta["actual_neg"]),
            "shared_random_pool": True,
            "shared_random_model": str(random_profile_settings["model"]),
        }
        all_augmented_meta.update(all_augmented_output_meta)
        manifest["profiles"][all_augmented_name] = all_augmented_meta
        built_profiles.append(all_augmented_name)
        print(
            f"[{benchmark}] Profile {all_augmented_name}: total={all_augmented_meta['actual_total']} "
            f"pos={all_augmented_meta['actual_pos']} neg={all_augmented_meta['actual_neg']} "
            f"(base=all, random_additions={len(random_labels_df)})"
        )

    manifest_path = run_dir / "profile_manifest.json"
    _write_json(manifest_path, manifest)
    summary_payload = {
        "benchmark": benchmark,
        "run_dir": str(run_dir),
        "config": str(args.config),
        "seed_summary": seed_summary,
        "labeling_cost": labeling_cost,
        "built_profiles": built_profiles,
        "skipped_profiles": skipped_profiles,
        "export_ditto": bool(export_ditto),
    }
    _write_json(run_dir / "summary.json", summary_payload)
    print(f"[{benchmark}] Saved manifest: {manifest_path}")
    return summary_payload


def main() -> None:
    os.chdir(ROOT)
    args = _parse_args()
    config_path = _resolve_repo_path(args.config)
    cfg = _load_yaml(config_path)
    benchmark_names = _selected_benchmarks(cfg, args.benchmarks)

    summaries: List[Dict[str, Any]] = []
    for benchmark in benchmark_names:
        summaries.append(_run_benchmark(benchmark=benchmark, cfg=cfg, args=args))

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_root / "seed_round_only_run_summary.json",
        {
            "config": str(config_path),
            "benchmarks": benchmark_names,
            "runs": summaries,
        },
    )
    print(output_root)


if __name__ == "__main__":
    main()
