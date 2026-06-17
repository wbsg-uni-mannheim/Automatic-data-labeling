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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[5]
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
PROMPT_PRESETS: Dict[str, str] = {
    "agent_variant_skeptic": "agent_variant_skeptic_abstain_system_prompt.txt",
    "agent_precision": "agent_precision_abstain_system_prompt.txt",
    "agent_balanced": "agent_balanced_abstain_system_prompt.txt",
}
DEFAULT_WORKERS = 8
DEFAULT_CHECKPOINT_EVERY = 50
DEFAULT_MAX_FIELD_LENGTH = 350
FIELD_ALIASES: Dict[str, tuple[str, ...]] = {
    "title": ("name",),
}
COLOR_SYNONYMS: Dict[str, str] = {
    "grey": "gray",
    "silver": "silver",
    "black": "black",
    "white": "white",
    "blue": "blue",
    "red": "red",
    "green": "green",
    "pink": "pink",
    "orange": "orange",
    "yellow": "yellow",
    "brown": "brown",
    "gold": "gold",
    "purple": "purple",
    "violet": "purple",
    "beige": "beige",
    "bronze": "bronze",
    "platinum": "silver",
    "charcoal": "gray",
}
MODEL_TOKEN_STOPLIST = {
    "1080P",
    "720P",
    "2160P",
    "60HZ",
    "120HZ",
    "240HZ",
    "80211N",
    "80211AC",
    "80211AX",
}
ACTIVE_LEARNING_USER_TEMPLATE = (
    "Do the two entity descriptions refer to the same real-world entity? "
    "Entity 1: '{left_json}'. "
    "Entity 2: '{right_json}'."
)

_thread_local = threading.local()


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


def _record_pair_id(record: Dict[str, Any], pair_index: int) -> str:
    pair_id = _normalize_text(record.get("pair_id"))
    if pair_id:
        return pair_id
    left_id = _normalize_text(record.get("id_left")) or f"left-{pair_index}"
    right_id = _normalize_text(record.get("id_right")) or f"right-{pair_index}"
    return f"{left_id}#{right_id}"


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


def _resolve_prompt_path(prompt_template: str) -> Path:
    try:
        filename = PROMPT_PRESETS[prompt_template]
    except KeyError as exc:
        raise ValueError(f"Unsupported prompt template: {prompt_template}") from exc
    path = PROMPT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found for template '{prompt_template}': {path}")
    return path


def _build_messages(
    record: Dict[str, Any],
    *,
    system_prompt: str,
    prompt_fields: Iterable[str],
    max_field_length: int,
) -> List[Dict[str, str]]:
    left_json = json.dumps(
        _extract_entity_payload(record, "left", prompt_fields, max_field_length),
        ensure_ascii=False,
    )
    right_json = json.dumps(
        _extract_entity_payload(record, "right", prompt_fields, max_field_length),
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


def _parse_review_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    decision = str(payload.get("decision", "")).strip().lower()
    if decision not in {"match", "non_match", "abstain"}:
        raise ValueError(f"Unsupported decision value: {payload!r}")
    try:
        confidence_raw = float(payload.get("confidence"))
    except Exception as exc:
        raise ValueError(f"Unsupported confidence value: {payload!r}") from exc
    if confidence_raw > 1.0:
        confidence = max(min(confidence_raw / 100.0, 1.0), 0.5)
    else:
        confidence = max(min(confidence_raw, 1.0), 0.5)
    return {
        "decision": decision,
        "confidence": float(confidence),
        "reason_code": str(payload.get("reason_code", "other")).strip().lower() or "other",
        "reason": str(payload.get("reason", "")).strip(),
    }


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
            return _parse_review_payload(payload), content, aggregate_usage
        except Exception as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Model call failed after retries: {last_error}")


def _normalize_model_token(token: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]", "", token.upper())
    return normalized


def _extract_model_tokens(text: str) -> List[str]:
    candidates = re.findall(r"\b[A-Z0-9/-]{4,}\b", text.upper())
    out: List[str] = []
    for candidate in candidates:
        normalized = _normalize_model_token(candidate)
        if not normalized or normalized in MODEL_TOKEN_STOPLIST:
            continue
        letter_count = sum(1 for ch in normalized if ch.isalpha())
        digit_count = sum(1 for ch in normalized if ch.isdigit())
        if letter_count < 2 or digit_count < 2:
            continue
        if normalized not in out:
            out.append(normalized)
    return out


def _extract_colors(text: str) -> List[str]:
    found: List[str] = []
    for raw, canonical in COLOR_SYNONYMS.items():
        if re.search(rf"\b{re.escape(raw)}\b", text.lower()):
            if canonical not in found:
                found.append(canonical)
    return found


def _extract_storage_values(text: str) -> List[str]:
    values: List[str] = []
    for amount, unit in re.findall(r"\b(\d+(?:\.\d+)?)\s*(tb|gb|mb)\b", text.lower()):
        normalized = f"{amount}{unit}"
        if normalized not in values:
            values.append(normalized)
    return values


def _text_for_veto(record: Dict[str, Any], suffix: str, prompt_fields: Iterable[str]) -> str:
    pieces: List[str] = []
    suffix_token = f"_{suffix}"
    for field in prompt_fields:
        field_name = str(field).strip()
        if not field_name or field_name == "price":
            continue
        candidate_fields = (field_name,) + FIELD_ALIASES.get(field_name, ())
        for candidate in candidate_fields:
            value = _normalize_text(record.get(f"{candidate}{suffix_token}"))
            if value:
                pieces.append(value)
                break
    return " ".join(pieces)


def _deterministic_veto(record: Dict[str, Any], prompt_fields: Iterable[str]) -> Optional[Dict[str, Any]]:
    left_text = _text_for_veto(record, "left", prompt_fields)
    right_text = _text_for_veto(record, "right", prompt_fields)

    left_models = _extract_model_tokens(left_text)
    right_models = _extract_model_tokens(right_text)
    if left_models and right_models and not (set(left_models) & set(right_models)):
        return {
            "veto_rule": "model_token_conflict",
            "veto_details": {
                "left_model_tokens": left_models,
                "right_model_tokens": right_models,
            },
        }

    left_colors = _extract_colors(left_text)
    right_colors = _extract_colors(right_text)
    if left_colors and right_colors and not (set(left_colors) & set(right_colors)):
        return {
            "veto_rule": "color_conflict",
            "veto_details": {
                "left_colors": left_colors,
                "right_colors": right_colors,
            },
        }

    left_storage = _extract_storage_values(left_text)
    right_storage = _extract_storage_values(right_text)
    if left_storage and right_storage and not (set(left_storage) & set(right_storage)):
        return {
            "veto_rule": "storage_conflict",
            "veto_details": {
                "left_storage": left_storage,
                "right_storage": right_storage,
            },
        }

    return None


def _base_output_row(source_row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pair_index": int(source_row.get("pair_index", 0) or 0),
        "pair_id": str(source_row.get("pair_id", "")),
        "id_left": source_row.get("id_left"),
        "id_right": source_row.get("id_right"),
        "gold_label": int(source_row.get("gold_label")),
        "first_decision": source_row.get("decision"),
        "first_pred_label": source_row.get("pred_label"),
        "final_decision": source_row.get("decision"),
        "final_pred_label": source_row.get("pred_label"),
        "status": source_row.get("status", "ok"),
        "veto_rule": None,
        "veto_details": None,
        "review_decision": None,
        "review_confidence": None,
        "review_reason_code": None,
        "review_reason": None,
        "review_response": None,
        "combine_reason": "kept_first_pass",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "error": source_row.get("error"),
    }


def _review_positive(
    source_row: Dict[str, Any],
    record: Dict[str, Any],
    *,
    review_model: str,
    system_prompt: str,
    prompt_fields: Iterable[str],
    max_field_length: int,
    max_retries: int,
) -> Dict[str, Any]:
    out = _base_output_row(source_row)
    if str(source_row.get("status", "")).strip().lower() != "ok":
        out["combine_reason"] = "source_error"
        return out

    first_decision = str(source_row.get("decision", "")).strip().lower()
    if first_decision != "match":
        if first_decision == "non_match":
            out["combine_reason"] = "kept_first_pass_non_match"
        elif first_decision == "abstain":
            out["final_pred_label"] = None
            out["combine_reason"] = "kept_first_pass_abstain"
        else:
            out["final_decision"] = "abstain"
            out["final_pred_label"] = None
            out["combine_reason"] = "invalid_first_decision"
        return out

    veto = _deterministic_veto(record, prompt_fields)
    if veto is not None:
        out.update(
            {
                "final_decision": "abstain",
                "final_pred_label": None,
                "veto_rule": veto["veto_rule"],
                "veto_details": veto["veto_details"],
                "combine_reason": "deterministic_veto",
            }
        )
        return out

    try:
        messages = _build_messages(
            record,
            system_prompt=system_prompt,
            prompt_fields=prompt_fields,
            max_field_length=max_field_length,
        )
        review_payload, review_text, usage = _call_model(messages, review_model, max_retries)
        final_decision = "match" if review_payload["decision"] == "match" else "abstain"
        out.update(
            {
                "final_decision": final_decision,
                "final_pred_label": 1 if final_decision == "match" else None,
                "review_decision": review_payload["decision"],
                "review_confidence": review_payload["confidence"],
                "review_reason_code": review_payload["reason_code"],
                "review_reason": review_payload["reason"],
                "review_response": review_text,
                "combine_reason": (
                    "confirmed_by_skeptical_review"
                    if final_decision == "match"
                    else f"skeptical_review_{review_payload['decision']}"
                ),
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
            }
        )
        return out
    except Exception as exc:
        out.update(
            {
                "status": "error",
                "final_decision": "abstain",
                "final_pred_label": None,
                "combine_reason": "review_error",
                "error": str(exc),
            }
        )
        return out


def _compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    pair_count = len(results)
    parse_failures = int(sum(1 for row in results if row.get("status") != "ok"))
    abstentions = int(sum(1 for row in results if row.get("status") == "ok" and row.get("final_pred_label") is None))
    abstained_rows = [row for row in results if row.get("status") == "ok" and row.get("final_pred_label") is None]
    ok_rows = [row for row in results if row.get("status") == "ok" and row.get("final_pred_label") is not None]

    prompt_tokens = int(sum(int(row.get("prompt_tokens", 0) or 0) for row in results))
    completion_tokens = int(sum(int(row.get("completion_tokens", 0) or 0) for row in results))
    total_tokens = int(sum(int(row.get("total_tokens", 0) or 0) for row in results))

    positive_candidates = int(sum(1 for row in results if str(row.get("first_decision", "")).strip().lower() == "match"))
    deterministic_vetoes = int(sum(1 for row in results if row.get("combine_reason") == "deterministic_veto"))
    reviewed_positives = int(sum(1 for row in results if row.get("review_decision") is not None))
    confirmed_matches = int(sum(1 for row in results if row.get("combine_reason") == "confirmed_by_skeptical_review"))
    abstained_after_review = int(
        sum(
            1
            for row in results
            if str(row.get("combine_reason", "")).startswith("skeptical_review_")
            and row.get("final_pred_label") is None
        )
    )

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
            "wrong_scored_decisions": 0,
            "abstain_positive_count": int(sum(1 for row in abstained_rows if int(row["gold_label"]) == 1)),
            "abstain_negative_count": int(sum(1 for row in abstained_rows if int(row["gold_label"]) == 0)),
            "perfect_abstain_resolution_f1": None,
            "perfect_abstain_resolution_accuracy": None,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "positive_candidates": positive_candidates,
            "deterministic_vetoes": deterministic_vetoes,
            "reviewed_positives": reviewed_positives,
            "confirmed_matches": confirmed_matches,
            "abstained_after_review": abstained_after_review,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    tp = tn = fp = fn = 0
    for row in ok_rows:
        gold = int(row["gold_label"])
        pred = int(row["final_pred_label"])
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
    wrong_scored_decisions = int(fp + fn)
    abstain_positive_count = int(sum(1 for row in abstained_rows if int(row["gold_label"]) == 1))
    abstain_negative_count = int(sum(1 for row in abstained_rows if int(row["gold_label"]) == 0))
    perfect_tp = int(tp + abstain_positive_count)
    perfect_tn = int(tn + abstain_negative_count)
    perfect_precision = float(perfect_tp / (perfect_tp + fp)) if (perfect_tp + fp) else 0.0
    perfect_recall = float(perfect_tp / (perfect_tp + fn)) if (perfect_tp + fn) else 0.0
    perfect_f1 = (
        float((2 * perfect_precision * perfect_recall) / (perfect_precision + perfect_recall))
        if (perfect_precision + perfect_recall)
        else 0.0
    )
    perfect_accuracy = float((perfect_tp + perfect_tn) / pair_count) if pair_count else 0.0

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
        "wrong_scored_decisions": wrong_scored_decisions,
        "abstain_positive_count": abstain_positive_count,
        "abstain_negative_count": abstain_negative_count,
        "perfect_abstain_resolution_f1": perfect_f1,
        "perfect_abstain_resolution_accuracy": perfect_accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "positive_candidates": positive_candidates,
        "deterministic_vetoes": deterministic_vetoes,
        "reviewed_positives": reviewed_positives,
        "confirmed_matches": confirmed_matches,
        "abstained_after_review": abstained_after_review,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
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
                "first_decision": row.get("first_decision"),
                "final_decision": row.get("final_decision"),
                "final_pred_label": row.get("final_pred_label"),
                "combine_reason": row.get("combine_reason"),
                "veto_rule": row.get("veto_rule"),
                "veto_details": json.dumps(row.get("veto_details"), ensure_ascii=False) if row.get("veto_details") else "",
                "review_decision": row.get("review_decision"),
                "review_confidence": row.get("review_confidence"),
                "review_reason_code": row.get("review_reason_code"),
                "review_reason": row.get("review_reason"),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Review first-pass positive matches with deterministic vetoes and a skeptical second-pass model.")
    parser.add_argument("--input-dir", required=True, help="Existing first-pass experiment output dir with results.jsonl and summary.json")
    parser.add_argument("--review-model", required=True, help="Model to use for the skeptical positive-only review")
    parser.add_argument(
        "--prompt-template",
        default="agent_variant_skeptic",
        choices=sorted(PROMPT_PRESETS.keys()),
        help="Skeptical review system prompt to use",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--max-field-length", type=int, default=DEFAULT_MAX_FIELD_LENGTH)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    input_dir = Path(args.input_dir)
    summary_path = input_dir / "summary.json"
    results_path = input_dir / "results.jsonl"
    if not summary_path.exists() or not results_path.exists():
        raise FileNotFoundError(f"Expected summary.json and results.jsonl under {input_dir}")

    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_results = [json.loads(line) for line in results_path.open("r", encoding="utf-8") if line.strip()]
    data_path = Path(str(source_summary["data_path"]))
    if not data_path.exists():
        raise FileNotFoundError(f"Source dataset not found: {data_path}")
    prompt_fields = list(source_summary.get("fields") or [])
    prompt_path = _resolve_prompt_path(args.prompt_template)
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()

    records = _load_records(data_path)
    record_by_pair = {_record_pair_id(record, idx): record for idx, record in enumerate(records)}
    selected = list(source_results)
    if args.limit and args.limit > 0:
        selected = selected[: int(args.limit)]

    output_dir = input_dir / "positive_only_skeptical_review" / args.prompt_template / _model_slug(args.review_model)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_results_path = output_dir / "results.jsonl"
    if out_results_path.exists() and not args.resume and not args.dry_run:
        out_results_path.unlink()
    existing = _load_existing_results(out_results_path) if args.resume else {}
    pending = [row for row in selected if str(row.get("pair_id")) not in existing]

    positive_candidates = sum(1 for row in selected if str(row.get("decision", "")).strip().lower() == "match")
    print(f"Input dir: {input_dir}")
    print(f"Review model: {args.review_model}")
    print(f"Prompt template: {args.prompt_template}")
    print(f"Pairs total: {len(selected)}")
    print(f"Positive candidates: {positive_candidates}")
    print(f"Already completed: {len(existing)}")
    print(f"Pending pairs: {len(pending)}")
    print(f"Output dir: {output_dir}")
    if args.dry_run:
        return

    results_by_pair = dict(existing)
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            futures = {
                executor.submit(
                    _review_positive,
                    source_row,
                    record_by_pair[str(source_row["pair_id"])],
                    review_model=args.review_model,
                    system_prompt=system_prompt,
                    prompt_fields=prompt_fields,
                    max_field_length=int(args.max_field_length),
                    max_retries=int(args.max_retries),
                ): source_row
                for source_row in pending
            }
            progress = tqdm(total=len(futures), desc="Positive review", unit="pair")
            completed_since_checkpoint = 0
            for future in as_completed(futures):
                result = future.result()
                results_by_pair[str(result["pair_id"])] = result
                _append_jsonl(out_results_path, result)
                progress.update(1)
                completed_since_checkpoint += 1
                if completed_since_checkpoint >= int(args.checkpoint_every):
                    partial_summary = _compute_metrics(list(results_by_pair.values()))
                    partial_summary.update(
                        {
                            "source_input_dir": str(input_dir),
                            "source_model": source_summary.get("model"),
                            "review_model": args.review_model,
                            "prompt_template": args.prompt_template,
                            "system_prompt_path": str(prompt_path),
                            "data_path": str(data_path),
                            "fields": prompt_fields,
                            "created_at": datetime.now().isoformat(timespec="seconds"),
                        }
                    )
                    _write_json(output_dir / "summary.json", partial_summary)
                    completed_since_checkpoint = 0
            progress.close()

    final_results = list(results_by_pair.values())
    metrics = _compute_metrics(final_results)
    metrics.update(
        {
            "source_input_dir": str(input_dir),
            "source_model": source_summary.get("model"),
            "review_model": args.review_model,
            "prompt_template": args.prompt_template,
            "system_prompt_path": str(prompt_path),
            "data_path": str(data_path),
            "output_dir": str(output_dir),
            "fields": prompt_fields,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    _write_json(output_dir / "summary.json", metrics)
    _write_predictions_csv(output_dir / "predictions.csv", final_results)
    _write_json(
        output_dir / "run_manifest.json",
        {
            "input_dir": str(input_dir),
            "source_model": source_summary.get("model"),
            "review_model": args.review_model,
            "prompt_template": args.prompt_template,
            "system_prompt_path": str(prompt_path),
            "workers": int(args.workers),
            "max_field_length": int(args.max_field_length),
            "max_retries": int(args.max_retries),
            "prompt_fields": prompt_fields,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
