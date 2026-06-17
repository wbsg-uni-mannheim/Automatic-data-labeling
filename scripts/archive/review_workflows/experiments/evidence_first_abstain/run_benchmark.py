#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
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


ROOT = Path(__file__).resolve().parents[5]
CONFIG_PATH = ROOT / "configs" / "labeling" / "benchmarks_active.yaml"
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "experiments" / "evidence_first_abstain"
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_SPLIT = "test"
DEFAULT_MAX_FIELD_LENGTH = 200
DEFAULT_WORKERS = 8
DEFAULT_CHECKPOINT_EVERY = 50
FIELD_ALIASES: Dict[str, tuple[str, ...]] = {
    "title": ("name",),
}
DECISION_ORDER = {"match": 1, "non_match": 2, "abstain": 3}

STAGE1_SYSTEM_PROMPT = """You are an expert entity matcher working in evidence extraction mode.
Your task is to extract structured evidence about whether two records refer to the same real-world product.

Do not make the final match decision.
Do not guess hidden facts.
Copy concrete evidence from the records when possible.

Return only valid JSON with exactly these fields:
{
  "shared_identifiers": ["..."],
  "left_only_identifiers": ["..."],
  "right_only_identifiers": ["..."],
  "shared_attributes": ["..."],
  "conflicting_attributes": ["..."],
  "same_brand": "yes|no|unclear",
  "same_product_family": "yes|no|unclear",
  "price_relation": "compatible|incompatible|missing|unclear",
  "positive_evidence_strength": "none|weak|moderate|strong",
  "negative_evidence_strength": "none|weak|moderate|strong",
  "critical_missing_information": ["..."],
  "notes": "short evidence summary"
}

Use shared_identifiers for model numbers, SKU-like strings, exact product names, or other identity-bearing tokens.
Use conflicting_attributes for concrete contradictions such as different brand, different model number, incompatible size/capacity, or different product family.
Use critical_missing_information for fields that would be necessary to resolve an otherwise plausible match.
Keep notes short and factual."""

STAGE2_SYSTEM_PROMPT = """You are an expert entity matcher making the final decision from extracted evidence.

You must return one of:
- "match"
- "non_match"
- "abstain"

Use "abstain" only when the case is genuinely unresolved because the evidence is mixed or incomplete.
Do not use "abstain" when there is decisive conflicting evidence.
Do not use "abstain" when there is decisive positive identity evidence and no meaningful contradiction.

Return only valid JSON with exactly these fields:
{
  "decision": "match|non_match|abstain",
  "reason_code": "strong_positive_identifier|hard_conflict|mixed_evidence|insufficient_identifier|missing_critical_field|other",
  "reason": "short explanation"
}"""

_thread_local = threading.local()


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark: str
    split: str
    data_path: Path
    prompt_fields: tuple[str, ...]


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping at {path}")
    return payload


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
    field_cfg = raw.get("fields") or raw.get("left_fields") or {}
    if not isinstance(field_cfg, dict) or not field_cfg:
        raise ValueError(f"Benchmark '{benchmark}' has no prompt fields in {CONFIG_PATH}")
    prompt_fields = tuple(str(name).strip() for name in field_cfg.keys() if str(name).strip())
    resolved = ROOT / str(data_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Configured dataset path does not exist: {resolved}")
    return BenchmarkSpec(
        benchmark=benchmark,
        split=split,
        data_path=resolved,
        prompt_fields=prompt_fields,
    )


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


def _extract_entity_payload(
    record: Dict[str, Any],
    suffix: str,
    prompt_fields: Iterable[str],
    max_field_length: int,
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    suffix_token = f"_{suffix}"
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


def _build_stage1_messages(record: Dict[str, Any], prompt_fields: Iterable[str], max_field_length: int) -> List[Dict[str, str]]:
    left = json.dumps(
        _extract_entity_payload(record, "left", prompt_fields, max_field_length),
        ensure_ascii=False,
    )
    right = json.dumps(
        _extract_entity_payload(record, "right", prompt_fields, max_field_length),
        ensure_ascii=False,
    )
    user_prompt = (
        "Extract structured matching evidence for these two records.\n"
        f"Entity 1: {left}\n"
        f"Entity 2: {right}"
    )
    return [
        {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _build_stage2_messages(
    record: Dict[str, Any],
    evidence: Dict[str, Any],
    prompt_fields: Iterable[str],
    max_field_length: int,
) -> List[Dict[str, str]]:
    left = json.dumps(
        _extract_entity_payload(record, "left", prompt_fields, max_field_length),
        ensure_ascii=False,
    )
    right = json.dumps(
        _extract_entity_payload(record, "right", prompt_fields, max_field_length),
        ensure_ascii=False,
    )
    evidence_json = json.dumps(evidence, ensure_ascii=False)
    user_prompt = (
        "Make the final entity-matching decision from the records and extracted evidence.\n"
        f"Entity 1: {left}\n"
        f"Entity 2: {right}\n"
        f"Extracted evidence: {evidence_json}"
    )
    return [
        {"role": "system", "content": STAGE2_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _get_client() -> OpenAI:
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = OpenAI()
        _thread_local.client = client
    return client


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


def _extract_usage(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _call_json_model(messages: List[Dict[str, str]], model: str, max_retries: int) -> tuple[Dict[str, Any], str, Dict[str, int]]:
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
            return _extract_json_object(content), content, aggregate_usage
        except Exception as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Model call failed after retries: {last_error}")


def _normalize_strength(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"none", "weak", "moderate", "strong"}:
        return text
    return "none"


def _normalize_ternary(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"yes", "no", "unclear"}:
        return text
    return "unclear"


def _normalize_price_relation(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"compatible", "incompatible", "missing", "unclear"}:
        return text
    return "unclear"


def _normalize_str_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _normalize_stage1(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "shared_identifiers": _normalize_str_list(payload.get("shared_identifiers")),
        "left_only_identifiers": _normalize_str_list(payload.get("left_only_identifiers")),
        "right_only_identifiers": _normalize_str_list(payload.get("right_only_identifiers")),
        "shared_attributes": _normalize_str_list(payload.get("shared_attributes")),
        "conflicting_attributes": _normalize_str_list(payload.get("conflicting_attributes")),
        "same_brand": _normalize_ternary(payload.get("same_brand")),
        "same_product_family": _normalize_ternary(payload.get("same_product_family")),
        "price_relation": _normalize_price_relation(payload.get("price_relation")),
        "positive_evidence_strength": _normalize_strength(payload.get("positive_evidence_strength")),
        "negative_evidence_strength": _normalize_strength(payload.get("negative_evidence_strength")),
        "critical_missing_information": _normalize_str_list(payload.get("critical_missing_information")),
        "notes": str(payload.get("notes", "")).strip(),
    }


def _normalize_stage2(payload: Dict[str, Any]) -> Dict[str, str]:
    decision = str(payload.get("decision", "")).strip().lower()
    if decision not in {"match", "non_match", "abstain"}:
        raise ValueError(f"Unsupported decision value: {payload!r}")
    reason_code = str(payload.get("reason_code", "other")).strip().lower()
    if not reason_code:
        reason_code = "other"
    return {
        "decision": decision,
        "reason_code": reason_code,
        "reason": str(payload.get("reason", "")).strip(),
    }


def _has_model_like_identifier(values: Iterable[str]) -> bool:
    pattern = re.compile(r"[a-z]+\d+[a-z\d-]*|\d+[a-z]+[a-z\d-]*", re.IGNORECASE)
    for value in values:
        if pattern.search(str(value)):
            return True
    return False


def _should_override_match_to_abstain(evidence: Dict[str, Any], decision_payload: Dict[str, str]) -> Optional[Dict[str, str]]:
    if decision_payload.get("decision") != "match":
        return None

    shared_identifiers = evidence.get("shared_identifiers") or []
    critical_missing = " ".join(evidence.get("critical_missing_information") or []).lower()
    positive_strength = _normalize_strength(evidence.get("positive_evidence_strength"))
    negative_strength = _normalize_strength(evidence.get("negative_evidence_strength"))

    missing_exact_identifier = any(
        token in critical_missing
        for token in (
            "model number",
            "exact model",
            "full model",
            "sku",
            "upc",
            "part number",
        )
    )
    has_model_identifier = _has_model_like_identifier(shared_identifiers)

    if (
        missing_exact_identifier
        and not has_model_identifier
        and positive_strength in {"moderate", "strong"}
        and negative_strength in {"none", "weak"}
    ):
        return {
            "decision": "abstain",
            "reason_code": "insufficient_identifier",
            "reason": (
                "Downgraded from match to abstain because the evidence is only family-level "
                "and the exact model/SKU identifier is missing."
            ),
        }

    return None


def _decision_to_pred_label(decision: str) -> Optional[int]:
    if decision == "match":
        return 1
    if decision == "non_match":
        return 0
    return None


def _score_record(
    record: Dict[str, Any],
    *,
    benchmark: str,
    split: str,
    model: str,
    prompt_fields: Iterable[str],
    max_field_length: int,
    max_retries: int,
    pair_index: int,
) -> Dict[str, Any]:
    pair_id = _record_pair_id(record, pair_index)
    gold_label = _normalize_gold_label(record.get("label"))
    result: Dict[str, Any] = {
        "benchmark": benchmark,
        "split": split,
        "pair_index": int(pair_index),
        "pair_id": pair_id,
        "id_left": str(record.get("id_left", "")),
        "id_right": str(record.get("id_right", "")),
        "gold_label": int(gold_label),
        "status": "error",
        "decision": None,
        "pred_label": None,
        "reason_code": None,
        "reason": None,
        "stage1_response": None,
        "stage2_response": None,
        "stage1_evidence": None,
        "error": None,
        "stage1_prompt_tokens": 0,
        "stage1_completion_tokens": 0,
        "stage1_total_tokens": 0,
        "stage2_prompt_tokens": 0,
        "stage2_completion_tokens": 0,
        "stage2_total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    try:
        stage1_messages = _build_stage1_messages(record, prompt_fields, max_field_length)
        stage1_payload, stage1_text, stage1_usage = _call_json_model(stage1_messages, model, max_retries)
        evidence = _normalize_stage1(stage1_payload)

        stage2_messages = _build_stage2_messages(record, evidence, prompt_fields, max_field_length)
        stage2_payload, stage2_text, stage2_usage = _call_json_model(stage2_messages, model, max_retries)
        decision_payload = _normalize_stage2(stage2_payload)
        override_payload = _should_override_match_to_abstain(evidence, decision_payload)
        if override_payload is not None:
            decision_payload = override_payload

        result.update(
            {
                "status": "ok",
                "decision": decision_payload["decision"],
                "pred_label": _decision_to_pred_label(decision_payload["decision"]),
                "reason_code": decision_payload["reason_code"],
                "reason": decision_payload["reason"],
                "stage1_response": stage1_text,
                "stage2_response": stage2_text,
                "stage1_evidence": evidence,
                "stage1_prompt_tokens": stage1_usage["prompt_tokens"],
                "stage1_completion_tokens": stage1_usage["completion_tokens"],
                "stage1_total_tokens": stage1_usage["total_tokens"],
                "stage2_prompt_tokens": stage2_usage["prompt_tokens"],
                "stage2_completion_tokens": stage2_usage["completion_tokens"],
                "stage2_total_tokens": stage2_usage["total_tokens"],
            }
        )
        result["prompt_tokens"] = result["stage1_prompt_tokens"] + result["stage2_prompt_tokens"]
        result["completion_tokens"] = result["stage1_completion_tokens"] + result["stage2_completion_tokens"]
        result["total_tokens"] = result["stage1_total_tokens"] + result["stage2_total_tokens"]
        return result
    except Exception as exc:
        result["error"] = str(exc)
        return result


def _compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    pair_count = len(results)
    parse_failures = int(sum(1 for row in results if row.get("status") != "ok"))
    abstentions = int(sum(1 for row in results if row.get("status") == "ok" and row.get("pred_label") is None))
    ok_rows = [row for row in results if row.get("status") == "ok" and row.get("pred_label") is not None]

    prompt_tokens = int(sum(int(row.get("prompt_tokens", 0) or 0) for row in results))
    completion_tokens = int(sum(int(row.get("completion_tokens", 0) or 0) for row in results))
    total_tokens = int(sum(int(row.get("total_tokens", 0) or 0) for row in results))
    stage1_prompt_tokens = int(sum(int(row.get("stage1_prompt_tokens", 0) or 0) for row in results))
    stage1_completion_tokens = int(sum(int(row.get("stage1_completion_tokens", 0) or 0) for row in results))
    stage1_total_tokens = int(sum(int(row.get("stage1_total_tokens", 0) or 0) for row in results))
    stage2_prompt_tokens = int(sum(int(row.get("stage2_prompt_tokens", 0) or 0) for row in results))
    stage2_completion_tokens = int(sum(int(row.get("stage2_completion_tokens", 0) or 0) for row in results))
    stage2_total_tokens = int(sum(int(row.get("stage2_total_tokens", 0) or 0) for row in results))

    if not ok_rows:
        return {
            "pair_count": pair_count,
            "pairs_scored": 0,
            "coverage": 0.0,
            "parse_failures": parse_failures,
            "abstentions": abstentions,
            "abstain_rate": (abstentions / pair_count) if pair_count else 0.0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "all_pair_accuracy": 0.0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "stage1_prompt_tokens": stage1_prompt_tokens,
            "stage1_completion_tokens": stage1_completion_tokens,
            "stage1_total_tokens": stage1_total_tokens,
            "stage2_prompt_tokens": stage2_prompt_tokens,
            "stage2_completion_tokens": stage2_completion_tokens,
            "stage2_total_tokens": stage2_total_tokens,
        }

    tp = tn = fp = fn = 0
    for row in ok_rows:
        gold = int(row["gold_label"])
        pred = int(row["pred_label"])
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
    all_pair_accuracy = float((tp + tn) / pair_count) if pair_count else 0.0
    return {
        "pair_count": pair_count,
        "pairs_scored": pairs_scored,
        "coverage": float(pairs_scored / pair_count) if pair_count else 0.0,
        "parse_failures": parse_failures,
        "abstentions": abstentions,
        "abstain_rate": (abstentions / pair_count) if pair_count else 0.0,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "all_pair_accuracy": all_pair_accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "stage1_prompt_tokens": stage1_prompt_tokens,
        "stage1_completion_tokens": stage1_completion_tokens,
        "stage1_total_tokens": stage1_total_tokens,
        "stage2_prompt_tokens": stage2_prompt_tokens,
        "stage2_completion_tokens": stage2_completion_tokens,
        "stage2_total_tokens": stage2_total_tokens,
    }


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


def _result_rows_for_csv(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in sorted(results, key=lambda item: int(item.get("pair_index", 0))):
        evidence = row.get("stage1_evidence") or {}
        rows.append(
            {
                "pair_index": row.get("pair_index"),
                "pair_id": row.get("pair_id"),
                "id_left": row.get("id_left"),
                "id_right": row.get("id_right"),
                "gold_label": row.get("gold_label"),
                "decision": row.get("decision"),
                "pred_label": row.get("pred_label"),
                "status": row.get("status"),
                "reason_code": row.get("reason_code"),
                "reason": row.get("reason"),
                "positive_evidence_strength": evidence.get("positive_evidence_strength"),
                "negative_evidence_strength": evidence.get("negative_evidence_strength"),
                "same_brand": evidence.get("same_brand"),
                "same_product_family": evidence.get("same_product_family"),
                "price_relation": evidence.get("price_relation"),
                "shared_identifiers": " | ".join(evidence.get("shared_identifiers", [])),
                "conflicting_attributes": " | ".join(evidence.get("conflicting_attributes", [])),
                "critical_missing_information": " | ".join(evidence.get("critical_missing_information", [])),
                "prompt_tokens": row.get("prompt_tokens"),
                "completion_tokens": row.get("completion_tokens"),
                "total_tokens": row.get("total_tokens"),
                "error": row.get("error"),
            }
        )
    return rows


def _write_predictions_csv(path: Path, results: List[Dict[str, Any]]) -> None:
    rows = _result_rows_for_csv(results)
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
    parser = argparse.ArgumentParser(description="Evidence-first benchmark runner with abstain support.")
    parser.add_argument("--benchmark", default="abt-buy", help="Benchmark key from configs/labeling/benchmarks_active.yaml")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split key from the benchmark config, e.g. test or valid")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name to use, e.g. gpt-5-mini")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Root directory for experiment outputs")
    parser.add_argument("--max-field-length", type=int, default=DEFAULT_MAX_FIELD_LENGTH)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for quick test runs")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing results.jsonl in the output directory")
    parser.add_argument("--dry-run", action="store_true", help="Resolve config and output path without making API calls")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    spec = _load_benchmark_spec(args.benchmark, args.split)
    output_dir = Path(args.output_root) / args.benchmark / args.split / _model_slug(args.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    manifest_path = output_dir / "run_manifest.json"

    records = _load_records(spec.data_path)
    if args.limit and args.limit > 0:
        records = records[: int(args.limit)]

    run_manifest = {
        "benchmark": args.benchmark,
        "split": args.split,
        "model": args.model,
        "data_path": str(spec.data_path),
        "prompt_fields": list(spec.prompt_fields),
        "max_field_length": int(args.max_field_length),
        "workers": int(args.workers),
        "checkpoint_every": int(args.checkpoint_every),
        "max_retries": int(args.max_retries),
        "pair_count": int(len(records)),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(manifest_path, run_manifest)

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
    print(f"Data path: {spec.data_path}")
    print(f"Output dir: {output_dir}")
    print(f"Pairs total: {len(records)}")
    print(f"Pairs already completed: {len(existing)}")
    print(f"Pairs pending: {len(pending_records)}")
    if args.dry_run:
        return

    results_by_pair = dict(existing)

    if pending_records:
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            futures = {
                executor.submit(
                    _score_record,
                    record,
                    benchmark=args.benchmark,
                    split=args.split,
                    model=args.model,
                    prompt_fields=spec.prompt_fields,
                    max_field_length=int(args.max_field_length),
                    max_retries=int(args.max_retries),
                    pair_index=idx,
                ): record
                for idx, record in enumerate(records)
                if _record_pair_id(record, idx) not in existing
            }
            progress = tqdm(total=len(futures), desc="Scoring pairs", unit="pair")
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
        data_path=spec.data_path,
        prompt_fields=spec.prompt_fields,
        results=list(results_by_pair.values()),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
