#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "labeling" / "benchmarks_active.yaml"
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
PROMPT_PRESETS: Dict[str, str] = {
    "active_learning": "active_learning_system_prompt.txt",
    "agent_precision": "agent_precision_system_prompt.txt",
    "agent_balanced": "agent_balanced_system_prompt.txt",
    "agent_variant_skeptic": "agent_variant_skeptic_system_prompt.txt",
    "agent_recall": "agent_recall_system_prompt.txt",
    "agent_recall_toned": "agent_recall_toned_system_prompt.txt",
}
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "experiments" / "dense_binary_dual_review"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_SPLIT = "test"
DEFAULT_FIRST_PASS_PROMPT = "agent_precision"
DEFAULT_POSITIVE_REVIEW_PROMPT = "agent_variant_skeptic"
DEFAULT_NEGATIVE_REVIEW_PROMPT = "agent_recall"
DEFAULT_WORKERS = 8
DEFAULT_CHECKPOINT_EVERY = 50
ACTIVE_LEARNING_USER_TEMPLATE = (
    "Do the two entity descriptions refer to the same real-world entity? "
    "Entity 1: '{left_json}'. "
    "Entity 2: '{right_json}'."
)

_thread_local = threading.local()


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark: str
    split: str
    data_path: Path
    prompt_fields: tuple[str, ...]
    left_csv: Path
    right_csv: Path
    left_id_col: str
    right_id_col: str
    left_field_map: Dict[str, str]
    right_field_map: Dict[str, str]


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


def _coerce_str_mapping(value: Any, name: str) -> Dict[str, str]:
    raw = _coerce_mapping(value, name)
    out: Dict[str, str] = {}
    for key, val in raw.items():
        k = str(key).strip()
        v = str(val).strip() if val is not None else ""
        if k and v:
            out[k] = v
    return out


def _normalize_field_mapping(field_map: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, val in field_map.items():
        normalized_key = "priceCurrency" if str(key).strip() == "currency" else str(key).strip()
        normalized_val = str(val).strip()
        if normalized_key and normalized_val:
            out[normalized_key] = normalized_val
    return out


def _load_benchmark_spec(benchmark: str, split: str) -> BenchmarkSpec:
    cfg = _load_yaml(CONFIG_PATH)
    benchmarks = cfg.get("benchmarks") or {}
    if not isinstance(benchmarks, dict) or benchmark not in benchmarks:
        raise KeyError(f"Unknown benchmark '{benchmark}'. Available: {sorted(benchmarks.keys())}")
    raw = benchmarks[benchmark]
    if not isinstance(raw, dict):
        raise ValueError(f"Benchmark config for '{benchmark}' must be a mapping")
    path_key = f"{split}_path"
    data_path = raw.get(path_key)
    if not data_path:
        raise ValueError(f"Benchmark '{benchmark}' has no '{path_key}' entry in {CONFIG_PATH}")

    left_csv = raw.get("left_csv")
    right_csv = raw.get("right_csv")
    if not left_csv or not right_csv:
        raise ValueError(f"Benchmark '{benchmark}' must define left_csv and right_csv in {CONFIG_PATH}")

    id_col = str(raw.get("id_col", "id")).strip() or "id"
    left_id_col = str(raw.get("left_id_col", id_col)).strip() or id_col
    right_id_col = str(raw.get("right_id_col", id_col)).strip() or id_col

    fields_cfg = _normalize_field_mapping(_coerce_str_mapping(raw.get("fields"), f"benchmarks.{benchmark}.fields"))
    left_fields_cfg = _normalize_field_mapping(
        _coerce_str_mapping(raw.get("left_fields"), f"benchmarks.{benchmark}.left_fields")
    )
    right_fields_cfg = _normalize_field_mapping(
        _coerce_str_mapping(raw.get("right_fields"), f"benchmarks.{benchmark}.right_fields")
    )
    if not left_fields_cfg and fields_cfg:
        left_fields_cfg = dict(fields_cfg)
    if not right_fields_cfg and fields_cfg:
        right_fields_cfg = dict(fields_cfg)
    if not left_fields_cfg or not right_fields_cfg:
        raise ValueError(
            f"Benchmark '{benchmark}' must define canonical prompt fields via fields/left_fields/right_fields"
        )

    ordered_fields: List[str] = []
    for name in left_fields_cfg.keys():
        field = str(name).strip()
        if field and field not in ordered_fields:
            ordered_fields.append(field)
    for name in right_fields_cfg.keys():
        field = str(name).strip()
        if field and field not in ordered_fields:
            ordered_fields.append(field)
    if not ordered_fields:
        raise ValueError(f"Benchmark '{benchmark}' has no prompt fields in {CONFIG_PATH}")
    prompt_fields = tuple(ordered_fields)
    resolved = ROOT / str(data_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Configured dataset path does not exist: {resolved}")
    resolved_left_csv = ROOT / str(left_csv)
    resolved_right_csv = ROOT / str(right_csv)
    if not resolved_left_csv.exists():
        raise FileNotFoundError(f"Configured left_csv does not exist: {resolved_left_csv}")
    if not resolved_right_csv.exists():
        raise FileNotFoundError(f"Configured right_csv does not exist: {resolved_right_csv}")
    return BenchmarkSpec(
        benchmark=benchmark,
        split=split,
        data_path=resolved,
        prompt_fields=prompt_fields,
        left_csv=resolved_left_csv,
        right_csv=resolved_right_csv,
        left_id_col=left_id_col,
        right_id_col=right_id_col,
        left_field_map=left_fields_cfg,
        right_field_map=right_fields_cfg,
    )


def _resolve_prompt_path(prompt_template: str) -> Path:
    try:
        filename = PROMPT_PRESETS[prompt_template]
    except KeyError as exc:
        raise ValueError(f"Unsupported prompt template: {prompt_template}") from exc
    path = PROMPT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found for template '{prompt_template}': {path}")
    return path


def _model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _normalize_gold_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and not pd.isna(value):
        return int(value != 0)
    text = str(value).strip().upper()
    if text in {"TRUE", "1", "YES", "Y", "T"}:
        return 1
    if text in {"FALSE", "0", "NO", "N", "F"}:
        return 0
    raise ValueError(f"Unsupported label value: {value!r}")


def _load_records(path: Path) -> List[Dict[str, Any]]:
    if path.name.endswith(".json.gz"):
        records: List[Dict[str, Any]] = []
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                records.append(json.loads(line))
        return records
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    raise ValueError(f"Unsupported dataset format: {path}")


def _record_pair_id(record: Dict[str, Any], pair_index: int) -> str:
    pair_id = _normalize_text(record.get("pair_id"))
    if pair_id:
        return pair_id
    left_id = _normalize_text(record.get("id_left")) or f"left-{pair_index}"
    right_id = _normalize_text(record.get("id_right")) or f"right-{pair_index}"
    return f"{left_id}#{right_id}"


def _load_source_index(csv_path: Path, *, id_col: str, field_map: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    required_cols = [id_col] + [source_col for source_col in field_map.values()]
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, usecols=lambda col: col in set(required_cols))
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    index: Dict[str, Dict[str, str]] = {}
    for row in df.to_dict(orient="records"):
        row_id = _normalize_text(row.get(id_col))
        if row_id and row_id not in index:
            index[row_id] = {str(k): _normalize_text(v) for k, v in row.items()}
    return index


def _extract_entity_payload(
    record: Dict[str, Any],
    *,
    suffix: str,
    prompt_fields: Iterable[str],
    field_map: Dict[str, str],
    source_index: Dict[str, Dict[str, str]],
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    row_id = _normalize_text(record.get(f"id_{suffix}"))
    source_row = source_index.get(row_id, {})
    for field in prompt_fields:
        field_name = str(field).strip()
        if not field_name:
            continue
        source_col = field_map.get(field_name, field_name)
        value = _normalize_text(source_row.get(source_col))
        if not value:
            value = _normalize_text(record.get(f"{field_name}_{suffix}"))
        if not value and source_col != field_name:
            value = _normalize_text(record.get(f"{source_col}_{suffix}"))
        if not value:
            continue
        out[field_name] = value
    return out


def _build_messages(
    record: Dict[str, Any],
    *,
    system_prompt: str,
    spec: BenchmarkSpec,
    left_index: Dict[str, Dict[str, str]],
    right_index: Dict[str, Dict[str, str]],
) -> List[Dict[str, str]]:
    left_json = json.dumps(
        _extract_entity_payload(
            record,
            suffix="left",
            prompt_fields=spec.prompt_fields,
            field_map=spec.left_field_map,
            source_index=left_index,
        ),
        ensure_ascii=False,
    )
    right_json = json.dumps(
        _extract_entity_payload(
            record,
            suffix="right",
            prompt_fields=spec.prompt_fields,
            field_map=spec.right_field_map,
            source_index=right_index,
        ),
        ensure_ascii=False,
    )
    user_prompt = ACTIVE_LEARNING_USER_TEMPLATE.format(left_json=left_json, right_json=right_json)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _get_client() -> OpenAI:
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = OpenAI()
        _thread_local.client = client
    return client


def _extract_usage(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = text.strip()
    if not raw:
        raise ValueError("Empty model response")
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text!r}")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError(f"Parsed JSON is not an object: {payload!r}")
    return payload


def _parse_match(payload: Dict[str, Any]) -> Dict[str, Any]:
    value = payload.get("match")
    if isinstance(value, bool):
        match = int(value)
    elif isinstance(value, (int, float)):
        match = int(value != 0)
    else:
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y"}:
            match = 1
        elif text in {"false", "0", "no", "n"}:
            match = 0
        else:
            raise ValueError(f"Unsupported 'match' value: {payload!r}")
    confidence_value = payload.get("confidence")
    confidence: Optional[float] = None
    if confidence_value is not None:
        try:
            conf = float(confidence_value)
            if conf > 1.0:
                conf = max(min(conf / 100.0, 1.0), 0.5)
            else:
                conf = max(min(conf, 1.0), 0.5)
            confidence = float(conf)
        except Exception:
            confidence = None
    return {"match": match, "confidence": confidence}


def _call_model(messages: List[Dict[str, str]], model: str, max_retries: int) -> tuple[Dict[str, Any], str, Dict[str, int]]:
    client = _get_client()
    aggregate_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=messages,
            )
            usage = _extract_usage(response)
            for key, value in usage.items():
                aggregate_usage[key] += int(value)
            content = response.choices[0].message.content or ""
            payload = _extract_json_object(content)
            return _parse_match(payload), content, aggregate_usage
        except Exception as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Model call failed after retries: {last_error}")


def _run_binary_review(
    record: Dict[str, Any],
    *,
    system_prompt: str,
    model: str,
    spec: BenchmarkSpec,
    left_index: Dict[str, Dict[str, str]],
    right_index: Dict[str, Dict[str, str]],
    max_retries: int,
) -> tuple[int, Optional[float], str, Dict[str, int]]:
    messages = _build_messages(
        record,
        system_prompt=system_prompt,
        spec=spec,
        left_index=left_index,
        right_index=right_index,
    )
    review_payload, review_text, review_usage = _call_model(messages, model, max_retries)
    return int(review_payload["match"]), review_payload.get("confidence"), review_text, review_usage


def _score_record(
    record: Dict[str, Any],
    *,
    spec: BenchmarkSpec,
    model: str,
    first_prompt_name: str,
    first_system_prompt: str,
    positive_review_prompt_name: str,
    positive_review_system_prompt: str,
    negative_review_prompt_name: Optional[str],
    negative_review_system_prompt: Optional[str],
    first_pass_only: bool,
    review_non_matches: bool,
    left_index: Dict[str, Dict[str, str]],
    right_index: Dict[str, Dict[str, str]],
    max_retries: int,
    pair_index: int,
    seed_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    pair_id = _record_pair_id(record, pair_index)
    gold_label = _normalize_gold_label(record.get("label"))
    base_result: Dict[str, Any] = {
        "benchmark": spec.benchmark,
        "split": spec.split,
        "pair_index": int(pair_index),
        "pair_id": pair_id,
        "id_left": str(record.get("id_left", "")),
        "id_right": str(record.get("id_right", "")),
        "gold_label": int(gold_label),
        "status": "error",
        "first_pass_decision": None,
        "first_pass_confidence": None,
        "first_pass_pred_label": None,
        "post_recall_decision": None,
        "post_recall_pred_label": None,
        "post_precision_decision": None,
        "post_precision_pred_label": None,
        "positive_review_match": None,
        "positive_review_confidence": None,
        "negative_review_match": None,
        "negative_review_confidence": None,
        "contention": False,
        "contention_reason": None,
        "final_decision": None,
        "pred_label": None,
        "combine_reason": None,
        "first_pass_response": None,
        "positive_review_response": None,
        "negative_review_response": None,
        "error": None,
        "first_pass_prompt_tokens": 0,
        "first_pass_completion_tokens": 0,
        "first_pass_total_tokens": 0,
        "positive_review_prompt_tokens": 0,
        "positive_review_completion_tokens": 0,
        "positive_review_total_tokens": 0,
        "negative_review_prompt_tokens": 0,
        "negative_review_completion_tokens": 0,
        "negative_review_total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    result: Dict[str, Any] = dict(base_result)
    if seed_result:
        result.update(seed_result)
        result.update(
            {
                "benchmark": spec.benchmark,
                "split": spec.split,
                "pair_index": int(pair_index),
                "pair_id": pair_id,
                "id_left": str(record.get("id_left", "")),
                "id_right": str(record.get("id_right", "")),
                "gold_label": int(gold_label),
                "negative_review_match": None,
                "negative_review_confidence": None,
                "negative_review_response": None,
                "negative_review_prompt_tokens": 0,
                "negative_review_completion_tokens": 0,
                "negative_review_total_tokens": 0,
                "contention": False,
                "contention_reason": None,
                "error": None,
            }
        )
    try:
        if seed_result:
            first_match = int(result["first_pass_pred_label"])
            post_precision_decision = str(result.get("post_precision_decision") or ("match" if int(result.get("post_precision_pred_label", 0)) == 1 else "non_match"))
            post_precision_pred_label = int(result["post_precision_pred_label"])
            final_decision = post_precision_decision
            pred_label = post_precision_pred_label
            combine_reason = str(result.get("combine_reason") or "seed_post_precision")
        else:
            first_messages = _build_messages(
                record,
                system_prompt=first_system_prompt,
                spec=spec,
                left_index=left_index,
                right_index=right_index,
            )
            first_payload, first_text, first_usage = _call_model(first_messages, model, max_retries)
            first_match = int(first_payload["match"])
            final_decision = "match" if first_match == 1 else "non_match"
            pred_label = first_match
            combine_reason = "first_pass"

            result.update(
                {
                    "first_pass_decision": final_decision,
                    "first_pass_confidence": first_payload.get("confidence"),
                    "first_pass_pred_label": first_match,
                    "first_pass_response": first_text,
                    "first_pass_prompt_tokens": first_usage["prompt_tokens"],
                    "first_pass_completion_tokens": first_usage["completion_tokens"],
                    "first_pass_total_tokens": first_usage["total_tokens"],
                }
            )

            post_precision_decision = final_decision
            post_precision_pred_label = pred_label

        if first_pass_only:
            result.update(
                {
                    "status": "ok",
                    "post_precision_decision": post_precision_decision,
                    "post_precision_pred_label": post_precision_pred_label,
                    "post_recall_decision": post_precision_decision,
                    "post_recall_pred_label": post_precision_pred_label,
                    "final_decision": final_decision,
                    "pred_label": pred_label,
                    "combine_reason": "first_pass_only",
                }
            )
            result["prompt_tokens"] = result["first_pass_prompt_tokens"]
            result["completion_tokens"] = result["first_pass_completion_tokens"]
            result["total_tokens"] = result["first_pass_total_tokens"]
            return result

        if first_match == 1:
            review_match, review_confidence, review_text, review_usage = _run_binary_review(
                record,
                system_prompt=positive_review_system_prompt,
                model=model,
                spec=spec,
                left_index=left_index,
                right_index=right_index,
                max_retries=max_retries,
            )
            result.update(
                {
                    "positive_review_match": review_match,
                    "positive_review_confidence": review_confidence,
                    "positive_review_response": review_text,
                    "positive_review_prompt_tokens": review_usage["prompt_tokens"],
                    "positive_review_completion_tokens": review_usage["completion_tokens"],
                    "positive_review_total_tokens": review_usage["total_tokens"],
                }
            )
            if review_match == 1:
                post_precision_decision = "match"
                post_precision_pred_label = 1
                combine_reason = "positive_review_confirmed_match"
            else:
                post_precision_decision = "non_match"
                post_precision_pred_label = 0
                combine_reason = "positive_review_rejected_match"
        else:
            post_precision_decision = "non_match"
            post_precision_pred_label = 0

        final_decision = post_precision_decision
        pred_label = post_precision_pred_label
        post_recall_decision = post_precision_decision
        post_recall_pred_label = post_precision_pred_label

        if review_non_matches and negative_review_system_prompt is not None and post_precision_pred_label == 0:
            review_match, review_confidence, review_text, review_usage = _run_binary_review(
                record,
                system_prompt=negative_review_system_prompt,
                model=model,
                spec=spec,
                left_index=left_index,
                right_index=right_index,
                max_retries=max_retries,
            )
            result.update(
                {
                    "negative_review_match": review_match,
                    "negative_review_confidence": review_confidence,
                    "negative_review_response": review_text,
                    "negative_review_prompt_tokens": review_usage["prompt_tokens"],
                    "negative_review_completion_tokens": review_usage["completion_tokens"],
                    "negative_review_total_tokens": review_usage["total_tokens"],
                }
            )
            if review_match == 1:
                post_recall_decision = "match"
                post_recall_pred_label = 1
                pos_match, pos_confidence, pos_text, pos_usage = _run_binary_review(
                    record,
                    system_prompt=positive_review_system_prompt,
                    model=model,
                    spec=spec,
                    left_index=left_index,
                    right_index=right_index,
                    max_retries=max_retries,
                )
                result.update(
                    {
                        "positive_review_match": pos_match,
                        "positive_review_confidence": pos_confidence,
                        "positive_review_response": pos_text,
                        "positive_review_prompt_tokens": pos_usage["prompt_tokens"],
                        "positive_review_completion_tokens": pos_usage["completion_tokens"],
                        "positive_review_total_tokens": pos_usage["total_tokens"],
                    }
                )
                if pos_match == 1:
                    final_decision = "match"
                    pred_label = 1
                    combine_reason = "negative_review_promoted_and_positive_review_confirmed"
                else:
                    final_decision = "non_match"
                    pred_label = 0
                    combine_reason = "negative_review_promoted_but_positive_review_rejected"
                    result["contention"] = True
                    result["contention_reason"] = "recall_promoted_then_skeptic_rejected"
            else:
                post_recall_decision = "non_match"
                post_recall_pred_label = 0
                final_decision = "non_match"
                pred_label = 0
                combine_reason = "negative_review_kept_non_match"

        result.update(
            {
                "status": "ok",
                "post_recall_decision": post_recall_decision,
                "post_recall_pred_label": post_recall_pred_label,
                "post_precision_decision": post_precision_decision,
                "post_precision_pred_label": post_precision_pred_label,
                "final_decision": final_decision,
                "pred_label": pred_label,
                "combine_reason": combine_reason,
            }
        )
        result["prompt_tokens"] = (
            result["first_pass_prompt_tokens"]
            + result["positive_review_prompt_tokens"]
            + result["negative_review_prompt_tokens"]
        )
        result["completion_tokens"] = (
            result["first_pass_completion_tokens"]
            + result["positive_review_completion_tokens"]
            + result["negative_review_completion_tokens"]
        )
        result["total_tokens"] = (
            result["first_pass_total_tokens"]
            + result["positive_review_total_tokens"]
            + result["negative_review_total_tokens"]
        )
        return result
    except Exception as exc:
        result["error"] = str(exc)
        return result


def _compute_stage_metrics(results: List[Dict[str, Any]], pred_key: str) -> Dict[str, Any]:
    pair_count = len(results)
    parse_failures = int(sum(1 for row in results if row.get("status") != "ok"))
    ok_rows = [row for row in results if row.get("status") == "ok" and row.get(pred_key) is not None]

    tp = tn = fp = fn = 0
    for row in ok_rows:
        gold = int(row["gold_label"])
        pred = int(row[pred_key])
        if gold == 1 and pred == 1:
            tp += 1
        elif gold == 0 and pred == 0:
            tn += 1
        elif gold == 0 and pred == 1:
            fp += 1
        elif gold == 1 and pred == 0:
            fn += 1

    pairs_scored = len(ok_rows)
    accuracy = float((tp + tn) / pairs_scored) if pairs_scored else None
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0
    return {
        "pair_count": pair_count,
        "pairs_scored": pairs_scored,
        "coverage": float(pairs_scored / pair_count) if pair_count else 0.0,
        "parse_failures": parse_failures,
        "abstentions": 0,
        "abstain_rate": 0.0,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "all_pair_accuracy": accuracy,
        "wrong_scored_decisions": int(fp + fn),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    pair_count = len(results)
    prompt_tokens = int(sum(int(row.get("prompt_tokens", 0) or 0) for row in results))
    completion_tokens = int(sum(int(row.get("completion_tokens", 0) or 0) for row in results))
    total_tokens = int(sum(int(row.get("total_tokens", 0) or 0) for row in results))
    first_pass_total_tokens = int(sum(int(row.get("first_pass_total_tokens", 0) or 0) for row in results))
    positive_review_total_tokens = int(sum(int(row.get("positive_review_total_tokens", 0) or 0) for row in results))
    negative_review_total_tokens = int(sum(int(row.get("negative_review_total_tokens", 0) or 0) for row in results))

    first_pass_metrics = _compute_stage_metrics(results, "first_pass_pred_label")
    post_precision_metrics = _compute_stage_metrics(results, "post_precision_pred_label")
    final_metrics = _compute_stage_metrics(results, "post_recall_pred_label")

    contention_rows = [row for row in results if bool(row.get("contention"))]
    contention_positive_count = int(sum(1 for row in contention_rows if int(row["gold_label"]) == 1))
    contention_negative_count = int(sum(1 for row in contention_rows if int(row["gold_label"]) == 0))

    perfect_tp = perfect_tn = perfect_fp = perfect_fn = 0
    ok_rows = [row for row in results if row.get("status") == "ok" and row.get("post_recall_pred_label") is not None]
    for row in ok_rows:
        gold = int(row["gold_label"])
        pred = int(row["post_recall_pred_label"])
        if bool(row.get("contention")):
            pred = gold
        if gold == 1 and pred == 1:
            perfect_tp += 1
        elif gold == 0 and pred == 0:
            perfect_tn += 1
        elif gold == 0 and pred == 1:
            perfect_fp += 1
        elif gold == 1 and pred == 0:
            perfect_fn += 1

    perfect_precision = float(perfect_tp / (perfect_tp + perfect_fp)) if (perfect_tp + perfect_fp) else 0.0
    perfect_recall = float(perfect_tp / (perfect_tp + perfect_fn)) if (perfect_tp + perfect_fn) else 0.0
    perfect_f1 = (
        float((2 * perfect_precision * perfect_recall) / (perfect_precision + perfect_recall))
        if (perfect_precision + perfect_recall)
        else 0.0
    )
    perfect_accuracy = float((perfect_tp + perfect_tn) / pair_count) if pair_count else 0.0

    summary = {
        **final_metrics,
        "contention_count": int(len(contention_rows)),
        "contention_positive_count": contention_positive_count,
        "contention_negative_count": contention_negative_count,
        "perfect_contention_resolution_f1": perfect_f1,
        "perfect_contention_resolution_accuracy": perfect_accuracy,
        "first_pass_metrics": first_pass_metrics,
        "post_precision_metrics": post_precision_metrics,
        "post_recall_metrics": final_metrics,
        "positive_reviewed": int(sum(1 for row in results if row.get("positive_review_match") is not None)),
        "positive_review_confirmed": int(
            sum(
                1
                for row in results
                if row.get("combine_reason") in {
                    "positive_review_confirmed_match",
                    "negative_review_promoted_and_positive_review_confirmed",
                }
            )
        ),
        "positive_review_rejected": int(
            sum(
                1
                for row in results
                if row.get("combine_reason") in {
                    "positive_review_rejected_match",
                    "negative_review_promoted_but_positive_review_rejected",
                }
            )
        ),
        "negative_reviewed": int(sum(1 for row in results if row.get("negative_review_match") is not None)),
        "negative_review_promoted": int(
            sum(
                1
                for row in results
                if row.get("combine_reason") in {
                    "negative_review_promoted_and_positive_review_confirmed",
                    "negative_review_promoted_but_positive_review_rejected",
                }
            )
        ),
        "negative_review_kept": int(sum(1 for row in results if row.get("combine_reason") == "negative_review_kept_non_match")),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "first_pass_total_tokens": first_pass_total_tokens,
        "positive_review_total_tokens": positive_review_total_tokens,
        "negative_review_total_tokens": negative_review_total_tokens,
    }
    return summary


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_existing_results(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            pair_id = str(payload.get("pair_id", ""))
            if pair_id:
                out[pair_id] = payload
    return out


def _write_predictions_csv(path: Path, results: List[Dict[str, Any]]) -> None:
    rows = []
    for row in sorted(results, key=lambda item: int(item.get("pair_index", 0))):
        rows.append(
            {
                "pair_index": row.get("pair_index"),
                "pair_id": row.get("pair_id"),
                "id_left": row.get("id_left"),
                "id_right": row.get("id_right"),
                "gold_label": row.get("gold_label"),
                "first_pass_decision": row.get("first_pass_decision"),
                "first_pass_confidence": row.get("first_pass_confidence"),
                "first_pass_pred_label": row.get("first_pass_pred_label"),
                "post_recall_decision": row.get("post_recall_decision"),
                "post_recall_pred_label": row.get("post_recall_pred_label"),
                "positive_review_match": row.get("positive_review_match"),
                "positive_review_confidence": row.get("positive_review_confidence"),
                "negative_review_match": row.get("negative_review_match"),
                "negative_review_confidence": row.get("negative_review_confidence"),
                "contention": row.get("contention"),
                "contention_reason": row.get("contention_reason"),
                "post_precision_decision": row.get("post_precision_decision"),
                "post_precision_pred_label": row.get("post_precision_pred_label"),
                "final_decision": row.get("final_decision"),
                "pred_label": row.get("pred_label"),
                "combine_reason": row.get("combine_reason"),
                "status": row.get("status"),
                "prompt_tokens": row.get("prompt_tokens"),
                "completion_tokens": row.get("completion_tokens"),
                "total_tokens": row.get("total_tokens"),
                "error": row.get("error"),
            }
        )
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_summary(
    output_dir: Path,
    *,
    benchmark: str,
    split: str,
    model: str,
    first_pass_prompt: str,
    positive_review_prompt: Optional[str],
    negative_review_prompt: Optional[str],
    first_pass_only: bool,
    review_non_matches: bool,
    data_path: Path,
    prompt_fields: Iterable[str],
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    summary = _compute_metrics(results)
    summary.update(
        {
            "benchmark": benchmark,
            "split": split,
            "model": model,
            "first_pass_prompt": first_pass_prompt,
            "positive_review_prompt": positive_review_prompt,
            "negative_review_prompt": negative_review_prompt,
            "first_pass_only": bool(first_pass_only),
            "review_non_matches": bool(review_non_matches),
            "data_path": str(data_path),
            "output_dir": str(output_dir),
            "fields": list(prompt_fields),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    _write_json(output_dir / "summary.json", summary)
    _write_predictions_csv(output_dir / "predictions.csv", results)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense binary matcher with first-pass precision, skeptical positive review, and optional non-match recovery review.")
    parser.add_argument("--benchmark", default="abt-buy", help="Benchmark key from configs/labeling/benchmarks_active.yaml")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split key from the benchmark config")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name to use for all passes")
    parser.add_argument("--first-pass-prompt", default=DEFAULT_FIRST_PASS_PROMPT, choices=sorted(PROMPT_PRESETS.keys()))
    parser.add_argument("--positive-review-prompt", default=DEFAULT_POSITIVE_REVIEW_PROMPT, choices=sorted(PROMPT_PRESETS.keys()))
    parser.add_argument("--negative-review-prompt", default=DEFAULT_NEGATIVE_REVIEW_PROMPT, choices=sorted(PROMPT_PRESETS.keys()))
    parser.add_argument("--first-pass-only", action="store_true", help="Run only the first-pass binary prompt with no review stages")
    parser.add_argument("--review-non-matches", action="store_true", help="Also review first-pass non-matches with a high-recall prompt")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed-results-dir", default="", help="Existing dense run directory to reuse first-pass/post-precision results from")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    spec = _load_benchmark_spec(args.benchmark, args.split)
    left_index = _load_source_index(
        spec.left_csv,
        id_col=spec.left_id_col,
        field_map=spec.left_field_map,
    )
    right_index = _load_source_index(
        spec.right_csv,
        id_col=spec.right_id_col,
        field_map=spec.right_field_map,
    )
    first_prompt_path = _resolve_prompt_path(args.first_pass_prompt)
    positive_prompt_path = _resolve_prompt_path(args.positive_review_prompt)
    negative_prompt_path = _resolve_prompt_path(args.negative_review_prompt) if args.review_non_matches else None
    first_system_prompt = first_prompt_path.read_text(encoding="utf-8").strip()
    positive_system_prompt = positive_prompt_path.read_text(encoding="utf-8").strip()
    negative_system_prompt = negative_prompt_path.read_text(encoding="utf-8").strip() if negative_prompt_path else None

    output_dir = (
        Path(args.output_root)
        / args.benchmark
        / args.split
        / args.first_pass_prompt
        / ("first-pass-only" if args.first_pass_only else f"pos-{args.positive_review_prompt}")
        / (
            "neg-none"
            if args.first_pass_only or not args.review_non_matches
            else f"neg-{args.negative_review_prompt}"
        )
        / _model_slug(args.model)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    seed_results: Dict[str, Dict[str, Any]] = {}
    seed_results_dir = Path(args.seed_results_dir) if str(args.seed_results_dir).strip() else None
    if seed_results_dir is not None:
        seed_path = seed_results_dir / "results.jsonl"
        if not seed_path.exists():
            raise FileNotFoundError(f"Seed results not found: {seed_path}")
        seed_results = _load_existing_results(seed_path)

    records = _load_records(spec.data_path)
    if args.limit and args.limit > 0:
        records = records[: int(args.limit)]

    run_manifest = {
        "benchmark": args.benchmark,
        "split": args.split,
        "model": args.model,
        "first_pass_prompt": args.first_pass_prompt,
        "positive_review_prompt": None if args.first_pass_only else args.positive_review_prompt,
        "negative_review_prompt": (
            None if args.first_pass_only or not args.review_non_matches else args.negative_review_prompt
        ),
        "first_pass_only": bool(args.first_pass_only),
        "review_non_matches": bool(args.review_non_matches and not args.first_pass_only),
        "data_path": str(spec.data_path),
        "prompt_fields": list(spec.prompt_fields),
        "workers": int(args.workers),
        "checkpoint_every": int(args.checkpoint_every),
        "max_retries": int(args.max_retries),
        "pair_count": int(len(records)),
        "seed_results_dir": str(seed_results_dir) if seed_results_dir is not None else None,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(output_dir / "run_manifest.json", run_manifest)

    existing = _load_existing_results(results_path) if args.resume else {}
    if results_path.exists() and not args.resume and not args.dry_run:
        results_path.unlink()
    pending_records = [
        record
        for idx, record in enumerate(records)
        if _record_pair_id(record, idx) not in existing
    ]

    print(f"Benchmark: {args.benchmark}")
    print(f"Split: {args.split}")
    print(f"Model: {args.model}")
    print(f"First pass prompt: {args.first_pass_prompt}")
    print(f"Positive review prompt: {'disabled' if args.first_pass_only else args.positive_review_prompt}")
    print(
        f"Negative review prompt: "
        f"{'disabled' if args.first_pass_only or not args.review_non_matches else args.negative_review_prompt}"
    )
    print(f"Data path: {spec.data_path}")
    print(f"Output dir: {output_dir}")
    print(f"Pairs total: {len(records)}")
    print(f"Pairs already completed: {len(existing)}")
    print(f"Pairs pending: {len(pending_records)}")
    if seed_results_dir is not None:
        print(f"Seed results dir: {seed_results_dir}")
        print(f"Seed rows loaded: {len(seed_results)}")
    if args.dry_run:
        return

    results_by_pair = dict(existing)
    if pending_records:
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            futures = {
                executor.submit(
                    _score_record,
                    record,
                    spec=spec,
                    model=args.model,
                    first_prompt_name=args.first_pass_prompt,
                    first_system_prompt=first_system_prompt,
                    positive_review_prompt_name=args.positive_review_prompt,
                    positive_review_system_prompt=positive_system_prompt,
                    negative_review_prompt_name=(args.negative_review_prompt if args.review_non_matches else None),
                    negative_review_system_prompt=negative_system_prompt,
                    first_pass_only=bool(args.first_pass_only),
                    review_non_matches=bool(args.review_non_matches and not args.first_pass_only),
                    left_index=left_index,
                    right_index=right_index,
                    max_retries=int(args.max_retries),
                    pair_index=idx,
                    seed_result=seed_results.get(_record_pair_id(record, idx)),
                ): record
                for idx, record in enumerate(records)
                if _record_pair_id(record, idx) not in existing
            }
            progress = tqdm(total=len(futures), desc="Dense review", unit="pair")
            completed_since_checkpoint = 0
            for future in as_completed(futures):
                result = future.result()
                results_by_pair[str(result["pair_id"])] = result
                _append_jsonl(results_path, result)
                progress.update(1)
                completed_since_checkpoint += 1
                if completed_since_checkpoint >= int(args.checkpoint_every):
                    _save_summary(
                        output_dir,
                        benchmark=args.benchmark,
                        split=args.split,
                        model=args.model,
                        first_pass_prompt=args.first_pass_prompt,
                        positive_review_prompt=(None if args.first_pass_only else args.positive_review_prompt),
                        negative_review_prompt=(
                            None if args.first_pass_only or not args.review_non_matches else args.negative_review_prompt
                        ),
                        first_pass_only=bool(args.first_pass_only),
                        review_non_matches=bool(args.review_non_matches and not args.first_pass_only),
                        data_path=spec.data_path,
                        prompt_fields=spec.prompt_fields,
                        results=list(results_by_pair.values()),
                    )
                    completed_since_checkpoint = 0
            progress.close()

    summary = _save_summary(
        output_dir,
        benchmark=args.benchmark,
        split=args.split,
        model=args.model,
        first_pass_prompt=args.first_pass_prompt,
        positive_review_prompt=(None if args.first_pass_only else args.positive_review_prompt),
        negative_review_prompt=(
            None if args.first_pass_only or not args.review_non_matches else args.negative_review_prompt
        ),
        first_pass_only=bool(args.first_pass_only),
        review_non_matches=bool(args.review_non_matches and not args.first_pass_only),
        data_path=spec.data_path,
        prompt_fields=spec.prompt_fields,
        results=list(results_by_pair.values()),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
