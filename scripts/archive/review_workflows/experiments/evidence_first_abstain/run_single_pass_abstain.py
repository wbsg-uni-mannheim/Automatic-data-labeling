#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
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
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
PROMPT_PRESETS: Dict[str, str] = {
    "agent_precision": "agent_precision_abstain_system_prompt.txt",
    "agent_balanced": "agent_balanced_abstain_system_prompt.txt",
    "agent_variant_skeptic": "agent_variant_skeptic_abstain_system_prompt.txt",
}
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "experiments" / "single_pass_abstain"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_SPLIT = "test"
DEFAULT_PROMPT_TEMPLATE = "agent_precision"
DECISION_POLICY = "model_decides_abstain"
DEFAULT_MAX_FIELD_LENGTH = 200
DEFAULT_WORKERS = 8
DEFAULT_CHECKPOINT_EVERY = 50
FIELD_ALIASES: Dict[str, tuple[str, ...]] = {
    "title": ("name",),
}

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


def _parse_response_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "decision" not in payload or "confidence" not in payload:
        raise ValueError(f"Missing required fields: {payload!r}")
    decision = str(payload.get("decision", "")).strip().lower()
    if decision not in {"match", "non_match", "abstain"}:
        raise ValueError(f"Unsupported decision value: {payload!r}")

    try:
        confidence_raw = float(payload["confidence"])
    except Exception as exc:
        raise ValueError(f"Unsupported confidence value: {payload!r}") from exc

    if math.isnan(confidence_raw):
        raise ValueError(f"Unsupported confidence value: {payload!r}")

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
            return _parse_response_payload(payload), content, aggregate_usage
        except Exception as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Model call failed after retries: {last_error}")


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
    system_prompt: str,
    prompt_template: str,
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
        "raw_decision": None,
        "raw_confidence": None,
        "response_text": None,
        "prompt_template": prompt_template,
        "error": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    try:
        messages = _build_messages(
            record,
            system_prompt=system_prompt,
            prompt_fields=prompt_fields,
            max_field_length=max_field_length,
        )
        parsed, raw_response, usage = _call_model(messages, model, max_retries)
        decision = str(parsed["decision"])
        pred_label = _decision_to_pred_label(decision)
        reason_code = str(parsed["reason_code"])
        result.update(
            {
                "status": "ok",
                "decision": decision,
                "pred_label": pred_label,
                "reason_code": reason_code,
                "reason": parsed["reason"] or f"Model chose {decision} with confidence={parsed['confidence']:.3f}",
                "raw_decision": decision,
                "raw_confidence": float(parsed["confidence"]),
                "response_text": raw_response,
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
            }
        )
        return result
    except Exception as exc:
        result["error"] = str(exc)
        return result


def _compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    pair_count = len(results)
    parse_failures = int(sum(1 for row in results if row.get("status") != "ok"))
    abstentions = int(sum(1 for row in results if row.get("status") == "ok" and row.get("pred_label") is None))
    abstained_rows = [row for row in results if row.get("status") == "ok" and row.get("pred_label") is None]
    ok_rows = [row for row in results if row.get("status") == "ok" and row.get("pred_label") is not None]

    prompt_tokens = int(sum(int(row.get("prompt_tokens", 0) or 0) for row in results))
    completion_tokens = int(sum(int(row.get("completion_tokens", 0) or 0) for row in results))
    total_tokens = int(sum(int(row.get("total_tokens", 0) or 0) for row in results))

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
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "mean_raw_confidence": None,
            "mean_scored_confidence": None,
            "mean_abstained_confidence": None,
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

    raw_confidences = [
        float(row["raw_confidence"])
        for row in results
        if row.get("status") == "ok" and row.get("raw_confidence") is not None
    ]
    scored_confidences = [
        float(row["raw_confidence"])
        for row in ok_rows
        if row.get("raw_confidence") is not None
    ]
    abstained_confidences = [
        float(row["raw_confidence"])
        for row in results
        if row.get("status") == "ok" and row.get("pred_label") is None and row.get("raw_confidence") is not None
    ]

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
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "mean_raw_confidence": float(sum(raw_confidences) / len(raw_confidences)) if raw_confidences else None,
        "mean_scored_confidence": float(sum(scored_confidences) / len(scored_confidences)) if scored_confidences else None,
        "mean_abstained_confidence": float(sum(abstained_confidences) / len(abstained_confidences)) if abstained_confidences else None,
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
                "decision": row.get("decision"),
                "pred_label": row.get("pred_label"),
                "raw_decision": row.get("raw_decision"),
                "raw_confidence": row.get("raw_confidence"),
                "reason_code": row.get("reason_code"),
                "reason": row.get("reason"),
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
    prompt_template: str,
    prompt_path: Path,
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
            "decision_policy": DECISION_POLICY,
            "prompt_template": prompt_template,
            "system_prompt_path": str(prompt_path),
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
    parser = argparse.ArgumentParser(description="Single-pass benchmark runner where the model directly chooses match, non_match, or abstain.")
    parser.add_argument("--benchmark", default="abt-buy", help="Benchmark key from configs/labeling/benchmarks_active.yaml")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split key from the benchmark config, e.g. test or valid")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name to use, e.g. gpt-5.4-mini")
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        choices=sorted(PROMPT_PRESETS.keys()),
        help="System prompt to use for the first pass",
    )
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
    prompt_path = _resolve_prompt_path(args.prompt_template)
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    output_dir = (
        Path(args.output_root)
        / args.benchmark
        / args.split
        / args.prompt_template
        / DECISION_POLICY
        / _model_slug(args.model)
    )
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
        "decision_policy": DECISION_POLICY,
        "prompt_template": args.prompt_template,
        "system_prompt_path": str(prompt_path),
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
    print(f"Prompt template: {args.prompt_template}")
    print(f"Decision policy: {DECISION_POLICY}")
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
                    system_prompt=system_prompt,
                    prompt_template=args.prompt_template,
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
                        prompt_template=args.prompt_template,
                        prompt_path=prompt_path,
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
        prompt_template=args.prompt_template,
        prompt_path=prompt_path,
        data_path=spec.data_path,
        prompt_fields=spec.prompt_fields,
        results=list(results_by_pair.values()),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
