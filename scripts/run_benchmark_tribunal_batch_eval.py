#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

DEFAULT_CONFIG_PATH = Path("configs/labeling/benchmarks_active.yaml")
DEFAULT_DATA_ROOT = Path("data")
DEFAULT_OUTPUT_DIR = Path("output/batch_benchmark_tribunal_eval")
DEFAULT_MODEL = "gpt-5.2"
DEFAULT_COMPLETION_WINDOW = "24h"
DEFAULT_ENDPOINT = "/v1/chat/completions"
DEFAULT_MAX_FIELD_LENGTH = 200

FIELD_ALIASES: Dict[str, tuple[str, ...]] = {
    "title": ("name",),
}

ACTIVE_LEARNING_USER_TEMPLATE = (
    "Do the two entity descriptions refer to the same real-world entity? "
    "Entity 1: '{left_json}'. "
    "Entity 2: '{right_json}'."
)

AGENTS: Dict[str, Dict[str, str]] = {
    "balanced": {
        "name": "Balanced Resolver",
        "system_prompt": (
            "You are an expert entity matcher. Decide if two records refer to the same real-world entity. "
            "Balance precision and recall. Be tolerant of wording and formatting differences, but reject clear "
            "identifier, model, version, size, or variant conflicts. Return only valid JSON with exactly two fields: "
            '{"match": true|false, "confidence": 50-100}. '
            "confidence is your certainty in that exact decision, where 50 means borderline/uncertain and 100 means near-certain."
        ),
    },
    "precision": {
        "name": "Precision Sentinel",
        "system_prompt": (
            "You are an expert entity matcher. Be conservative: predict match=true only when the evidence strongly supports "
            "that both records are the same real-world entity and there is no meaningful contradiction. "
            "Treat conflicting model numbers, editions, capacities, sizes, colors, venues, years, or variants as strong evidence against a match. "
            "Return only valid JSON with exactly two fields: {\"match\": true|false, \"confidence\": 50-100}."
        ),
    },
    "recall": {
        "name": "Recall Scout",
        "system_prompt": (
            "You are an expert entity matcher. Be permissive when records are noisy, incomplete, abbreviated, or differently formatted. "
            "If the evidence points to the same underlying entity and there is no explicit contradiction, lean toward match=true. "
            "Return only valid JSON with exactly two fields: {\"match\": true|false, \"confidence\": 50-100}."
        ),
    },
    "variant_skeptic": {
        "name": "Variant Skeptic",
        "system_prompt": (
            "You are an expert entity matcher. Focus on whether the two records might be near-duplicates or variants rather than the exact same entity. "
            "Be especially alert to subtle differences in model number, edition, version, quantity, size, color, year, venue, or author list. "
            "Return only valid JSON with exactly two fields: {\"match\": true|false, \"confidence\": 50-100}."
        ),
    },
    "contextualist": {
        "name": "Contextual Synthesizer",
        "system_prompt": (
            "You are an expert entity matcher. Judge the total evidence holistically, explicitly tolerating missing fields and asymmetric detail. "
            "Use all attributes together rather than relying on any single field, but still reject hard contradictions. "
            "Return only valid JSON with exactly two fields: {\"match\": true|false, \"confidence\": 50-100}."
        ),
    },
}


@dataclass(frozen=True)
class TestSetSpec:
    benchmark: str
    dataset_name: str
    data_path: str
    output_slug: str
    prompt_fields: tuple[str, ...]


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping at {path}")
    return payload


def _coerce_mapping(value: Any, name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def _normalize_gold_label(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not pd.isna(value):
        return int(value) != 0
    text = str(value).strip().upper()
    if text in {"TRUE", "1", "YES", "Y", "T"}:
        return True
    if text in {"FALSE", "0", "NO", "N", "F"}:
        return False
    raise ValueError(f"Unsupported label value: {value!r}")


def _strip_json_gz_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".json.gz"):
        return name[: -len(".json.gz")]
    return path.stem


def _slugify_dataset_name(name: str) -> str:
    out = []
    prev_dash = False
    for ch in str(name):
        if ch.isalnum():
            out.append(ch.lower())
            prev_dash = False
            continue
        if not prev_dash:
            out.append("-")
        prev_dash = True
    slug = "".join(out).strip("-")
    return slug or "dataset"


def _resolve_prompt_fields(benchmark_cfg: Dict[str, Any]) -> tuple[str, ...]:
    ordered: List[str] = []

    fields = _coerce_mapping(benchmark_cfg.get("fields"), "benchmarks.*.fields")
    if fields:
        ordered.extend(str(key).strip() for key in fields.keys() if str(key).strip())

    for side_key in ("left_fields", "right_fields"):
        side_fields = _coerce_mapping(benchmark_cfg.get(side_key), f"benchmarks.*.{side_key}")
        for key in side_fields.keys():
            field = str(key).strip()
            if field and field not in ordered:
                ordered.append(field)

    if not ordered:
        raise ValueError("Benchmark config must define prompt fields via fields/left_fields/right_fields")
    return tuple(ordered)


def discover_test_sets(
    config_path: Optional[Path] = None,
    data_root: Optional[Path] = None,
    benchmarks: Optional[Iterable[str]] = None,
) -> List[TestSetSpec]:
    config_path = config_path or DEFAULT_CONFIG_PATH
    data_root = data_root or DEFAULT_DATA_ROOT
    config = _load_yaml(config_path)
    benchmark_cfg = config.get("benchmarks") or {}
    if not isinstance(benchmark_cfg, dict) or not benchmark_cfg:
        raise ValueError(f"No benchmarks found in {config_path}")

    requested = set(benchmarks or [])
    missing = sorted(requested.difference(benchmark_cfg.keys()))
    if missing:
        raise ValueError(f"Requested benchmarks missing from config: {', '.join(missing)}")

    specs: List[TestSetSpec] = []
    for benchmark, raw_cfg in benchmark_cfg.items():
        if requested and benchmark not in requested:
            continue
        benchmark_fields = _resolve_prompt_fields(_coerce_mapping(raw_cfg, f"benchmarks.{benchmark}"))
        benchmark_dir = data_root / benchmark
        matches = sorted(benchmark_dir.glob("*gs.json.gz"))
        if not matches:
            raise FileNotFoundError(f"No test sets found for benchmark '{benchmark}' in {benchmark_dir}")
        for data_path in matches:
            dataset_name = _strip_json_gz_suffix(data_path)
            specs.append(
                TestSetSpec(
                    benchmark=benchmark,
                    dataset_name=dataset_name,
                    data_path=str(data_path),
                    output_slug=f"{benchmark}__{_slugify_dataset_name(dataset_name)}",
                    prompt_fields=benchmark_fields,
                )
            )
    return specs


def _read_jsonl_gz(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def _extract_entity_payload(
    record: Dict[str, Any],
    suffix: str,
    max_field_length: int,
    prompt_fields: Iterable[str],
) -> Dict[str, str]:
    suffix_token = f"_{suffix}"
    out: Dict[str, str] = {}
    for field in prompt_fields:
        field_name = str(field).strip()
        if not field_name:
            continue
        candidate_fields = (field_name,) + FIELD_ALIASES.get(field_name, ())
        for candidate in candidate_fields:
            key = f"{candidate}{suffix_token}"
            value = _normalize_text(record.get(key))
            if not value:
                continue
            if len(value) > max_field_length:
                value = value[:max_field_length] + "..."
            out[field_name] = value
            break
    return out


def build_user_prompt(
    record: Dict[str, Any],
    *,
    prompt_fields: Iterable[str],
    max_field_length: int = DEFAULT_MAX_FIELD_LENGTH,
) -> str:
    left_json = json.dumps(
        _extract_entity_payload(record, "left", max_field_length, prompt_fields),
        ensure_ascii=False,
    )
    right_json = json.dumps(
        _extract_entity_payload(record, "right", max_field_length, prompt_fields),
        ensure_ascii=False,
    )
    return ACTIVE_LEARNING_USER_TEMPLATE.format(left_json=left_json, right_json=right_json)


def _resolve_output_dir(output_dir: Path, *, create: bool) -> Path:
    if create:
        output_dir.mkdir(parents=True, exist_ok=True)
    elif not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    return output_dir


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _dataset_dir(run_dir: Path, spec_or_slug: TestSetSpec | str) -> Path:
    slug = spec_or_slug.output_slug if isinstance(spec_or_slug, TestSetSpec) else str(spec_or_slug)
    return run_dir / slug


def _custom_id(pair_index: int, agent_key: str) -> str:
    return f"pair-{pair_index}__agent-{agent_key}"


def _parse_custom_id(custom_id: str) -> Optional[tuple[int, str]]:
    try:
        left, right = custom_id.split("__", 1)
        idx = int(left.replace("pair-", ""))
        agent = right.replace("agent-", "")
        if agent not in AGENTS:
            return None
        return idx, agent
    except Exception:
        return None


def _prepare_single_dataset(
    *,
    run_dir: Path,
    spec: TestSetSpec,
    model: str,
    max_field_length: int,
) -> Dict[str, Any]:
    dataset_dir = _dataset_dir(run_dir, spec)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    records = _read_jsonl_gz(Path(spec.data_path))
    metadata_rows: List[Dict[str, Any]] = []
    batch_input_path = dataset_dir / "batch_input.jsonl"

    with batch_input_path.open("w", encoding="utf-8") as handle:
        for idx, record in enumerate(records):
            user_prompt = build_user_prompt(
                record,
                prompt_fields=spec.prompt_fields,
                max_field_length=max_field_length,
            )
            for agent_key, agent_cfg in AGENTS.items():
                request = {
                    "custom_id": _custom_id(idx, agent_key),
                    "method": "POST",
                    "url": DEFAULT_ENDPOINT,
                    "body": {
                        "model": model,
                        #"temperature": 0,
                        "messages": [
                            {"role": "system", "content": agent_cfg["system_prompt"]},
                            {"role": "user", "content": user_prompt},
                        ],
                    },
                }
                handle.write(json.dumps(request, ensure_ascii=False) + "\n")

            gold_bool = _normalize_gold_label(record.get("label"))
            metadata_rows.append(
                {
                    "pair_index": idx,
                    "pair_id": _normalize_text(record.get("pair_id")),
                    "id_left": _normalize_text(record.get("id_left")),
                    "id_right": _normalize_text(record.get("id_right")),
                    "gold_label": "TRUE" if gold_bool else "FALSE",
                    "gold_match": gold_bool,
                }
            )

    pd.DataFrame(metadata_rows).to_csv(dataset_dir / "metadata.csv", index=False)

    manifest = {
        "benchmark": spec.benchmark,
        "dataset_name": spec.dataset_name,
        "output_slug": spec.output_slug,
        "data_path": spec.data_path,
        "pair_count": len(metadata_rows),
        "request_count": len(metadata_rows) * len(AGENTS),
        "model": model,
        "prompt_fields": list(spec.prompt_fields),
        "agents": {key: cfg["name"] for key, cfg in AGENTS.items()},
        "batch_input_path": str(batch_input_path),
    }
    _write_json(dataset_dir / "dataset_manifest.json", manifest)
    return manifest


def _split_csv_arg(value: str) -> List[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def prepare_run(*, output_dir: Path, model: str, benchmarks: Optional[Iterable[str]] = None) -> Path:
    resolved_run_dir = _resolve_output_dir(output_dir, create=True)
    specs = discover_test_sets(benchmarks=benchmarks)
    dataset_manifests = [
        _prepare_single_dataset(
            run_dir=resolved_run_dir,
            spec=spec,
            model=model,
            max_field_length=DEFAULT_MAX_FIELD_LENGTH,
        )
        for spec in specs
    ]
    run_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(DEFAULT_CONFIG_PATH),
        "data_root": str(DEFAULT_DATA_ROOT),
        "model": model,
        "benchmarks": list(benchmarks or []),
        "max_field_length": DEFAULT_MAX_FIELD_LENGTH,
        "agents": {key: cfg["name"] for key, cfg in AGENTS.items()},
        "datasets": dataset_manifests,
    }
    _write_json(resolved_run_dir / "run_manifest.json", run_manifest)
    return resolved_run_dir


def submit_run(run_dir: Path) -> None:
    from openai import OpenAI

    client = OpenAI()
    run_manifest = _load_json(run_dir / "run_manifest.json")
    for dataset in run_manifest.get("datasets") or []:
        dataset_dir = _dataset_dir(run_dir, str(dataset["output_slug"]))
        batch_info_path = dataset_dir / "batch_info.json"
        if batch_info_path.exists():
            batch_info = _load_json(batch_info_path)
            status = str(batch_info.get("status", "")).lower()
            if status not in {"failed", "expired", "cancelled"}:
                continue

        batch_input_path = dataset_dir / "batch_input.jsonl"
        with batch_input_path.open("rb") as handle:
            file_obj = client.files.create(file=handle, purpose="batch")

        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint=DEFAULT_ENDPOINT,
            completion_window=DEFAULT_COMPLETION_WINDOW,
            metadata={
                "benchmark": str(dataset["benchmark"]),
                "dataset_name": str(dataset["dataset_name"]),
                "output_slug": str(dataset["output_slug"]),
                "agents": str(len(AGENTS)),
            },
        )
        _write_json(
            batch_info_path,
            {
                "submitted_at": datetime.now().isoformat(timespec="seconds"),
                "status": batch.status,
                "file_id": file_obj.id,
                "batch_id": batch.id,
                "output_file_id": getattr(batch, "output_file_id", None),
                "error_file_id": getattr(batch, "error_file_id", None),
            },
        )


def _download_file_if_missing(client: Any, file_id: Optional[str], target_path: Path) -> None:
    if not file_id or target_path.exists():
        return
    content = client.files.content(file_id)
    with target_path.open("wb") as handle:
        handle.write(content.read())


def refresh_status(run_dir: Path) -> List[Dict[str, Any]]:
    from openai import OpenAI

    client = OpenAI()
    run_manifest = _load_json(run_dir / "run_manifest.json")
    rows: List[Dict[str, Any]] = []
    for dataset in run_manifest.get("datasets") or []:
        dataset_dir = _dataset_dir(run_dir, str(dataset["output_slug"]))
        batch_info_path = dataset_dir / "batch_info.json"
        if not batch_info_path.exists():
            rows.append(
                {
                    "benchmark": dataset["benchmark"],
                    "dataset_name": dataset["dataset_name"],
                    "status": "not_submitted",
                    "completed": 0,
                    "failed": 0,
                    "total": int(dataset["request_count"]),
                }
            )
            continue

        batch_info = _load_json(batch_info_path)
        batch = client.batches.retrieve(str(batch_info["batch_id"]))
        counts = getattr(batch, "request_counts", None)
        updated = {
            **batch_info,
            "checked_at": datetime.now().isoformat(timespec="seconds"),
            "status": batch.status,
            "output_file_id": getattr(batch, "output_file_id", None),
            "error_file_id": getattr(batch, "error_file_id", None),
            "request_counts": {
                "completed": int(getattr(counts, "completed", 0) or 0),
                "failed": int(getattr(counts, "failed", 0) or 0),
                "total": int(getattr(counts, "total", 0) or 0),
            },
        }
        _write_json(batch_info_path, updated)
        _download_file_if_missing(client, updated.get("output_file_id"), dataset_dir / "batch_output.jsonl")
        _download_file_if_missing(client, updated.get("error_file_id"), dataset_dir / "batch_error.jsonl")
        rows.append(
            {
                "benchmark": dataset["benchmark"],
                "dataset_name": dataset["dataset_name"],
                "status": updated["status"],
                "completed": updated["request_counts"]["completed"],
                "failed": updated["request_counts"]["failed"],
                "total": updated["request_counts"]["total"],
            }
        )
    return rows


def _extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        parts = raw.split("\n", 1)
        raw = parts[1] if len(parts) == 2 else raw
        raw = raw.rsplit("```", 1)[0].strip()
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(raw[start : end + 1])
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def parse_agent_response(content: str) -> Dict[str, Any]:
    payload = _extract_json_payload(content)
    if not payload or "match" not in payload or "confidence" not in payload:
        return {
            "match": None,
            "confidence": None,
            "p_match": None,
            "error": "invalid_json_payload",
            "response_text": content,
        }

    match_value = payload["match"]
    if isinstance(match_value, bool):
        match_bool = match_value
    elif isinstance(match_value, (int, float)):
        match_bool = bool(int(match_value))
    elif isinstance(match_value, str):
        text = match_value.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            match_bool = True
        elif text in {"false", "0", "no", "n"}:
            match_bool = False
        else:
            match_bool = None
    else:
        match_bool = None

    try:
        confidence_raw = float(payload["confidence"])
    except Exception:
        confidence_raw = math.nan

    if match_bool is None or math.isnan(confidence_raw):
        return {
            "match": None,
            "confidence": None,
            "p_match": None,
            "error": "invalid_fields",
            "response_text": content,
        }

    if confidence_raw > 1.0:
        confidence = max(min(confidence_raw / 100.0, 1.0), 0.5)
    else:
        confidence = max(min(confidence_raw, 1.0), 0.5)

    p_match = confidence if match_bool else (1.0 - confidence)
    return {
        "match": bool(match_bool),
        "confidence": float(confidence),
        "p_match": float(p_match),
        "error": "",
        "response_text": content,
    }


def _parse_batch_output(output_path: Path) -> Dict[tuple[int, str], Dict[str, Any]]:
    parsed: Dict[tuple[int, str], Dict[str, Any]] = {}
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            custom_id = str(row.get("custom_id", ""))
            parsed_id = _parse_custom_id(custom_id)
            if parsed_id is None:
                continue
            pair_index, agent_key = parsed_id
            if row.get("error"):
                parsed[(pair_index, agent_key)] = {
                    "match": None,
                    "confidence": None,
                    "p_match": None,
                    "error": json.dumps(row["error"]),
                    "response_text": "",
                }
                continue
            try:
                content = row["response"]["body"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                parsed[(pair_index, agent_key)] = {
                    "match": None,
                    "confidence": None,
                    "p_match": None,
                    "error": "missing_response_content",
                    "response_text": "",
                }
                continue
            parsed[(pair_index, agent_key)] = parse_agent_response(content)
    return parsed


def compute_binary_metrics(gold: Iterable[bool], pred: Iterable[bool]) -> Dict[str, Any]:
    tp = tn = fp = fn = 0
    for gold_value, pred_value in zip(gold, pred):
        if gold_value and pred_value:
            tp += 1
        elif gold_value and not pred_value:
            fn += 1
        elif not gold_value and pred_value:
            fp += 1
        else:
            tn += 1
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "pairs_scored": int(total),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _majority_vote(row: pd.Series) -> bool:
    votes = []
    for agent_key in AGENTS:
        value = row[f"{agent_key}_match"]
        if pd.isna(value):
            continue
        votes.append(bool(value))
    if not votes:
        return False
    pos = sum(1 for vote in votes if vote)
    neg = len(votes) - pos
    if pos > neg:
        return True
    if neg > pos:
        return False
    balanced_value = row.get("balanced_match")
    return bool(balanced_value) if pd.notna(balanced_value) else False


def _confidence_weighted_vote(row: pd.Series) -> tuple[bool, float]:
    probs: List[float] = []
    weights: List[float] = []
    for agent_key in AGENTS:
        p_match = row.get(f"{agent_key}_p_match")
        confidence = row.get(f"{agent_key}_confidence")
        if pd.isna(p_match) or pd.isna(confidence):
            continue
        weight = max(2.0 * (float(confidence) - 0.5), 1e-3)
        probs.append(float(p_match))
        weights.append(weight)
    if not probs:
        return False, 0.5
    score = float(np.average(np.asarray(probs, dtype=float), weights=np.asarray(weights, dtype=float)))
    return score >= 0.5, score


def _fit_confidence_weighted_dawid_skene(
    labels: np.ndarray,
    strengths: np.ndarray,
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
    smoothing: float = 1e-3,
) -> tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    n_items, n_agents = labels.shape
    mask = strengths > 0

    if n_items == 0:
        return np.zeros(0, dtype=float), {}

    weighted_votes = np.zeros(n_items, dtype=float)
    weight_sums = np.zeros(n_items, dtype=float)
    for agent_idx in range(n_agents):
        observed = mask[:, agent_idx]
        weighted_votes[observed] += strengths[observed, agent_idx] * labels[observed, agent_idx]
        weight_sums[observed] += strengths[observed, agent_idx]
    posterior = np.divide(
        weighted_votes,
        weight_sums,
        out=np.full(n_items, 0.5, dtype=float),
        where=weight_sums > 0,
    )
    posterior = np.clip(posterior, 1e-4, 1.0 - 1e-4)

    confusion = np.full((n_agents, 2, 2), 0.5, dtype=float)
    for _ in range(max_iter):
        old_posterior = posterior.copy()
        prior_pos = float(np.clip(posterior.mean(), 1e-4, 1.0 - 1e-4))
        prior_neg = 1.0 - prior_pos

        for agent_idx in range(n_agents):
            observed = mask[:, agent_idx]
            if not observed.any():
                confusion[agent_idx] = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=float)
                continue
            for true_label in (0, 1):
                q_true = posterior if true_label == 1 else (1.0 - posterior)
                denom = float((q_true[observed] * strengths[observed, agent_idx]).sum())
                if denom <= 0:
                    confusion[agent_idx, true_label] = np.array([0.5, 0.5], dtype=float)
                    continue
                for observed_label in (0, 1):
                    obs_mask = observed & (labels[:, agent_idx] == observed_label)
                    numer = float((q_true[obs_mask] * strengths[obs_mask, agent_idx]).sum())
                    confusion[agent_idx, true_label, observed_label] = numer
                confusion[agent_idx, true_label, :] += smoothing
                confusion[agent_idx, true_label, :] /= confusion[agent_idx, true_label, :].sum()

        for item_idx in range(n_items):
            log_pos = math.log(prior_pos)
            log_neg = math.log(prior_neg)
            for agent_idx in range(n_agents):
                if not mask[item_idx, agent_idx]:
                    continue
                observed_label = int(labels[item_idx, agent_idx])
                strength = float(strengths[item_idx, agent_idx])
                log_pos += strength * math.log(max(confusion[agent_idx, 1, observed_label], 1e-9))
                log_neg += strength * math.log(max(confusion[agent_idx, 0, observed_label], 1e-9))
            max_log = max(log_pos, log_neg)
            pos_prob = math.exp(log_pos - max_log)
            neg_prob = math.exp(log_neg - max_log)
            posterior[item_idx] = pos_prob / (pos_prob + neg_prob)
        if float(np.max(np.abs(posterior - old_posterior))) < tol:
            break

    agent_stats: Dict[str, Dict[str, float]] = {}
    for agent_idx, agent_key in enumerate(AGENTS.keys()):
        agent_stats[agent_key] = {
            "specificity": float(confusion[agent_idx, 0, 0]),
            "false_positive_rate": float(confusion[agent_idx, 0, 1]),
            "false_negative_rate": float(confusion[agent_idx, 1, 0]),
            "sensitivity": float(confusion[agent_idx, 1, 1]),
        }
    return posterior, agent_stats


def _tribunal_predict(eval_df: pd.DataFrame) -> tuple[pd.Series, pd.Series, Dict[str, Dict[str, float]]]:
    agent_keys = list(AGENTS.keys())
    labels = np.zeros((len(eval_df), len(agent_keys)), dtype=int)
    strengths = np.zeros((len(eval_df), len(agent_keys)), dtype=float)
    for agent_idx, agent_key in enumerate(agent_keys):
        match_col = eval_df[f"{agent_key}_match"]
        conf_col = eval_df[f"{agent_key}_confidence"]
        valid = match_col.notna() & conf_col.notna()
        labels[valid.to_numpy(), agent_idx] = match_col[valid].astype(bool).astype(int).to_numpy()
        strengths[valid.to_numpy(), agent_idx] = np.maximum(
            2.0 * (conf_col[valid].astype(float).to_numpy() - 0.5),
            1e-3,
        )
    posterior, agent_stats = _fit_confidence_weighted_dawid_skene(labels, strengths)
    return pd.Series(posterior >= 0.5, index=eval_df.index), pd.Series(posterior, index=eval_df.index), agent_stats


def _safe_bool_series(values: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(values.notna(), values.astype(bool), False),
        index=values.index,
        dtype=bool,
    )


def evaluate_run(run_dir: Path) -> pd.DataFrame:
    run_manifest = _load_json(run_dir / "run_manifest.json")
    summary_rows: List[Dict[str, Any]] = []

    for dataset in run_manifest.get("datasets") or []:
        slug = str(dataset["output_slug"])
        dataset_dir = _dataset_dir(run_dir, slug)
        metadata_path = dataset_dir / "metadata.csv"
        output_path = dataset_dir / "batch_output.jsonl"
        metadata = pd.read_csv(metadata_path)

        parsed: Dict[tuple[int, str], Dict[str, Any]] = {}
        if output_path.exists():
            parsed = _parse_batch_output(output_path)

        eval_df = metadata.copy()
        parse_failures = 0
        any_agent_valid = pd.Series(False, index=eval_df.index)
        for agent_key in AGENTS:
            matches: List[Optional[bool]] = []
            confidences: List[Optional[float]] = []
            probs: List[Optional[float]] = []
            errors: List[str] = []
            responses: List[str] = []
            for pair_index in eval_df["pair_index"].astype(int):
                payload = parsed.get((int(pair_index), agent_key), {})
                matches.append(payload.get("match"))
                confidences.append(payload.get("confidence"))
                probs.append(payload.get("p_match"))
                errors.append(str(payload.get("error", "")))
                responses.append(str(payload.get("response_text", "")))
            eval_df[f"{agent_key}_match"] = matches
            eval_df[f"{agent_key}_confidence"] = confidences
            eval_df[f"{agent_key}_p_match"] = probs
            eval_df[f"{agent_key}_error"] = errors
            eval_df[f"{agent_key}_response_text"] = responses
            valid_agent = eval_df[f"{agent_key}_match"].notna()
            any_agent_valid = any_agent_valid | valid_agent
            parse_failures += int((~valid_agent).sum())

        eval_df["majority_match"] = eval_df.apply(_majority_vote, axis=1)
        weighted_results = eval_df.apply(_confidence_weighted_vote, axis=1)
        eval_df["weighted_vote_match"] = weighted_results.map(lambda x: x[0])
        eval_df["weighted_vote_score"] = weighted_results.map(lambda x: x[1])
        tribunal_match, tribunal_score, tribunal_agent_stats = _tribunal_predict(eval_df)
        eval_df["tribunal_match"] = tribunal_match
        eval_df["tribunal_score"] = tribunal_score

        eval_df.to_csv(dataset_dir / "predictions.csv", index=False)
        _write_json(dataset_dir / "tribunal_agent_stats.json", tribunal_agent_stats)

        gold = eval_df["gold_match"].astype(bool)
        methods: Dict[str, pd.Series] = {
            "tribunal": _safe_bool_series(eval_df["tribunal_match"]),
            "weighted_vote": _safe_bool_series(eval_df["weighted_vote_match"]),
            "majority_vote": _safe_bool_series(eval_df["majority_match"]),
            "balanced_single": _safe_bool_series(eval_df["balanced_match"]),
        }
        valid_masks: Dict[str, pd.Series] = {
            "tribunal": any_agent_valid,
            "weighted_vote": any_agent_valid,
            "majority_vote": any_agent_valid,
            "balanced_single": eval_df["balanced_match"].notna(),
        }
        for agent_key in AGENTS:
            methods[f"agent_{agent_key}"] = _safe_bool_series(eval_df[f"{agent_key}_match"])
            valid_masks[f"agent_{agent_key}"] = eval_df[f"{agent_key}_match"].notna()

        for method, predictions in methods.items():
            valid_mask = valid_masks[method]
            if not bool(valid_mask.any()):
                metrics = {
                    "pairs_scored": 0,
                    "tp": 0,
                    "tn": 0,
                    "fp": 0,
                    "fn": 0,
                    "accuracy": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                }
            else:
                metrics = compute_binary_metrics(
                    gold[valid_mask].tolist(),
                    predictions[valid_mask].tolist(),
                )
            summary_rows.append(
                {
                    "benchmark": dataset["benchmark"],
                    "dataset_name": dataset["dataset_name"],
                    "method": method,
                    "pair_count": int(len(eval_df)),
                    "parse_failures": int(parse_failures),
                    **metrics,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(run_dir / "summary.csv", index=False)

    comparison_rows: List[Dict[str, Any]] = []
    if not summary_df.empty:
        for (benchmark, dataset_name), group in summary_df.groupby(["benchmark", "dataset_name"], sort=False):
            lookup = {row["method"]: row for _, row in group.iterrows()}
            tribunal_row = lookup.get("tribunal")
            balanced_row = lookup.get("balanced_single")
            if (
                tribunal_row is not None
                and balanced_row is not None
                and pd.notna(tribunal_row["f1"])
                and pd.notna(balanced_row["f1"])
                and pd.notna(tribunal_row["accuracy"])
                and pd.notna(balanced_row["accuracy"])
            ):
                comparison_rows.append(
                    {
                        "benchmark": benchmark,
                        "dataset_name": dataset_name,
                        "tribunal_f1": float(tribunal_row["f1"]),
                        "balanced_f1": float(balanced_row["f1"]),
                        "tribunal_minus_balanced_f1": float(tribunal_row["f1"] - balanced_row["f1"]),
                        "tribunal_accuracy": float(tribunal_row["accuracy"]),
                        "balanced_accuracy": float(balanced_row["accuracy"]),
                        "tribunal_minus_balanced_accuracy": float(tribunal_row["accuracy"] - balanced_row["accuracy"]),
                    }
                )
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(run_dir / "comparison.csv", index=False)
    return summary_df


def print_status_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No datasets found.")
        return
    print(pd.DataFrame(rows).to_string(index=False))


def print_summary_tables(summary_df: pd.DataFrame, run_dir: Path) -> None:
    if summary_df.empty:
        print("No evaluation results available.")
        return
    tribunal = summary_df[summary_df["method"] == "tribunal"][
        ["benchmark", "dataset_name", "accuracy", "precision", "recall", "f1"]
    ]
    print("Tribunal")
    print(tribunal.to_string(index=False))
    comparison_path = run_dir / "comparison.csv"
    if comparison_path.exists() and comparison_path.read_text(encoding="utf-8").strip():
        comparison_df = pd.read_csv(comparison_path)
        if not comparison_df.empty:
            print("\nVs balanced_single")
            print(
                comparison_df[
                    [
                        "benchmark",
                        "dataset_name",
                        "tribunal_f1",
                        "balanced_f1",
                        "tribunal_minus_balanced_f1",
                    ]
                ].to_string(index=False)
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Five-personality batch evaluation with confidence-aware tribunal aggregation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create one batch_input.jsonl per test set with five personalities.")
    prepare.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    prepare.add_argument("--model", default=DEFAULT_MODEL)
    prepare.add_argument("--benchmarks", default="", help="Comma-separated benchmark keys, e.g. abt-buy")

    submit = subparsers.add_parser("submit", help="Submit one batch per test set.")
    submit.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    status = subparsers.add_parser("status", help="Refresh per-test-set batch status and download outputs.")
    status.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    evaluate = subparsers.add_parser("evaluate", help="Evaluate all agents, baselines, and tribunal.")
    evaluate.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    return parser


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare":
        run_dir = prepare_run(
            output_dir=Path(args.output_dir),
            model=args.model,
            benchmarks=_split_csv_arg(args.benchmarks),
        )
        print(run_dir)
        return

    run_dir = _resolve_output_dir(Path(args.output_dir), create=False)
    if args.command == "submit":
        submit_run(run_dir)
        print(run_dir)
        return
    if args.command == "status":
        rows = refresh_status(run_dir)
        print_status_table(rows)
        return
    if args.command == "evaluate":
        summary_df = evaluate_run(run_dir)
        print_summary_tables(summary_df, run_dir)
        return
    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
