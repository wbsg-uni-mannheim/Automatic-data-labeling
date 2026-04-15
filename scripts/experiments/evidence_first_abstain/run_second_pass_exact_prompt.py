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


ROOT = Path(__file__).resolve().parents[3]
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
PROMPT_PRESETS: Dict[str, str] = {
    "agent_precision": "agent_precision_system_prompt.txt",
    "agent_balanced": "agent_balanced_system_prompt.txt",
    "agent_variant_skeptic": "agent_variant_skeptic_system_prompt.txt",
    "exact_batch": "batch_exact_system_prompt.txt",
}
DEFAULT_WORKERS = 8
DEFAULT_CHECKPOINT_EVERY = 50
DEFAULT_MAX_FIELD_LENGTH = 350
FIELD_ALIASES: Dict[str, tuple[str, ...]] = {
    "title": ("name",),
}

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


def _build_messages(
    record: Dict[str, Any],
    *,
    system_prompt: str,
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
    user_prompt = (
        f"Listing A: {left}\n"
        f"Listing B: {right}\n"
        "Do these refer to the same underlying product model?"
    )
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


def _resolve_prompt_path(prompt_template: str) -> Path:
    try:
        filename = PROMPT_PRESETS[prompt_template]
    except KeyError as exc:
        raise ValueError(f"Unsupported prompt template: {prompt_template}") from exc
    path = PROMPT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found for template '{prompt_template}': {path}")
    return path


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


def _parse_match(payload: Dict[str, Any]) -> int:
    value = payload.get("match")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value != 0)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return 1
    if text in {"false", "0", "no", "n"}:
        return 0
    raise ValueError(f"Unsupported 'match' value: {payload!r}")


def _call_model(messages: List[Dict[str, str]], model: str, max_retries: int) -> tuple[int, str, Dict[str, int]]:
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


def _combine_decisions(first_decision: Optional[str], second_pass_match: int, resolve_abstains: bool) -> tuple[str, Optional[int], str]:
    first = str(first_decision or "").strip().lower()
    second_decision = "match" if second_pass_match == 1 else "non_match"
    if first == "abstain":
        if resolve_abstains:
            return second_decision, second_pass_match, "resolved_from_abstain"
        return "abstain", None, "kept_existing_abstain"
    if first not in {"match", "non_match"}:
        return "abstain", None, "invalid_first_decision"
    if first == second_decision:
        return first, second_pass_match, "agreement"
    return "abstain", None, "first_second_disagreement"


def _compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    pair_count = len(results)
    parse_failures = int(sum(1 for row in results if row.get("status") != "ok"))
    abstentions = int(sum(1 for row in results if row.get("status") == "ok" and row.get("final_pred_label") is None))
    ok_rows = [row for row in results if row.get("status") == "ok" and row.get("final_pred_label") is not None]

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
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
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
                "second_pass_pred_label": row.get("second_pass_pred_label"),
                "final_decision": row.get("final_decision"),
                "final_pred_label": row.get("final_pred_label"),
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


def _selection_matches(decision: str, selection: str) -> bool:
    decision = str(decision or "").strip().lower()
    if selection == "all":
        return True
    if selection == "scored":
        return decision in {"match", "non_match"}
    if selection == "match":
        return decision == "match"
    if selection == "non_match":
        return decision == "non_match"
    if selection == "abstain":
        return decision == "abstain"
    raise ValueError(f"Unsupported selection: {selection}")


def _score_pair(
    pair_result: Dict[str, Any],
    record: Dict[str, Any],
    *,
    model: str,
    system_prompt: str,
    prompt_fields: Iterable[str],
    max_field_length: int,
    max_retries: int,
    resolve_abstains: bool,
) -> Dict[str, Any]:
    pair_id = str(pair_result.get("pair_id"))
    gold_label = int(pair_result.get("gold_label"))
    out: Dict[str, Any] = {
        "pair_index": int(pair_result.get("pair_index", 0) or 0),
        "pair_id": pair_id,
        "id_left": pair_result.get("id_left"),
        "id_right": pair_result.get("id_right"),
        "gold_label": gold_label,
        "first_decision": pair_result.get("decision"),
        "first_pred_label": pair_result.get("pred_label"),
        "status": "error",
        "second_pass_pred_label": None,
        "second_pass_response": None,
        "final_decision": None,
        "final_pred_label": None,
        "combine_reason": None,
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
        pred_label, raw_response, usage = _call_model(messages, model, max_retries)
        final_decision, final_pred_label, combine_reason = _combine_decisions(
            str(pair_result.get("decision", "")),
            pred_label,
            resolve_abstains=resolve_abstains,
        )
        out.update(
            {
                "status": "ok",
                "second_pass_pred_label": pred_label,
                "second_pass_response": raw_response,
                "final_decision": final_decision,
                "final_pred_label": final_pred_label,
                "combine_reason": combine_reason,
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
            }
        )
        return out
    except Exception as exc:
        out["error"] = str(exc)
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a second-pass reviewer with a selectable system prompt.")
    parser.add_argument("--input-dir", required=True, help="Existing experiment output dir with results.jsonl and summary.json")
    parser.add_argument("--review-model", required=True, help="Model to use for the second pass")
    parser.add_argument(
        "--prompt-template",
        default="agent_precision",
        choices=sorted(PROMPT_PRESETS.keys()),
        help="System prompt to use for the second pass",
    )
    parser.add_argument("--selection", default="scored", choices=["all", "scored", "match", "non_match", "abstain"])
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--max-field-length", type=int, default=DEFAULT_MAX_FIELD_LENGTH)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resolve-abstains", action="store_true", help="Let second-pass labels resolve existing abstains instead of keeping them abstained")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    input_dir = Path(args.input_dir)
    summary_path = input_dir / "summary.json"
    results_path = input_dir / "results.jsonl"
    if not summary_path.exists() or not results_path.exists():
        raise FileNotFoundError(f"Expected summary.json and results.jsonl under {input_dir}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_results = [json.loads(line) for line in results_path.open("r", encoding="utf-8") if line.strip()]
    data_path = Path(str(summary["data_path"]))
    if not data_path.exists():
        raise FileNotFoundError(f"Source dataset not found: {data_path}")
    prompt_fields = list(summary.get("fields") or [])
    system_prompt_path = _resolve_prompt_path(args.prompt_template)
    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()

    records = _load_records(data_path)
    record_by_pair = {_record_pair_id(record, idx): record for idx, record in enumerate(records)}
    selected = [row for row in source_results if _selection_matches(str(row.get("decision", "")), args.selection)]
    if args.limit and args.limit > 0:
        selected = selected[: int(args.limit)]

    output_dir = input_dir / "second_pass_review" / args.prompt_template / _model_slug(args.review_model)
    output_dir.mkdir(parents=True, exist_ok=True)
    second_results_path = output_dir / "results.jsonl"
    if second_results_path.exists() and not args.resume and not args.dry_run:
        second_results_path.unlink()
    existing = _load_existing_results(second_results_path) if args.resume else {}
    pending = [row for row in selected if str(row.get("pair_id")) not in existing]

    print(f"Input dir: {input_dir}")
    print(f"Review model: {args.review_model}")
    print(f"Prompt template: {args.prompt_template}")
    print(f"Selection: {args.selection}")
    print(f"Selected pairs: {len(selected)}")
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
                    _score_pair,
                    pair_result,
                    record_by_pair[str(pair_result["pair_id"])],
                    model=args.review_model,
                    system_prompt=system_prompt,
                    prompt_fields=prompt_fields,
                    max_field_length=int(args.max_field_length),
                    max_retries=int(args.max_retries),
                    resolve_abstains=bool(args.resolve_abstains),
                ): pair_result
                for pair_result in pending
            }
            progress = tqdm(total=len(futures), desc="Second pass", unit="pair")
            completed_since_checkpoint = 0
            for future in as_completed(futures):
                result = future.result()
                results_by_pair[str(result["pair_id"])] = result
                _append_jsonl(second_results_path, result)
                progress.update(1)
                completed_since_checkpoint += 1
                if completed_since_checkpoint >= int(args.checkpoint_every):
                    partial_summary = _compute_metrics(list(results_by_pair.values()))
                    partial_summary.update(
                        {
                            "source_input_dir": str(input_dir),
                            "review_model": args.review_model,
                            "prompt_template": args.prompt_template,
                            "selection": args.selection,
                            "resolve_abstains": bool(args.resolve_abstains),
                            "data_path": str(data_path),
                            "fields": prompt_fields,
                            "system_prompt_path": str(system_prompt_path),
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
            "review_model": args.review_model,
            "prompt_template": args.prompt_template,
            "selection": args.selection,
            "resolve_abstains": bool(args.resolve_abstains),
            "data_path": str(data_path),
            "fields": prompt_fields,
            "system_prompt_path": str(system_prompt_path),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    _write_json(output_dir / "summary.json", metrics)
    _write_predictions_csv(output_dir / "predictions.csv", final_results)
    _write_json(
        output_dir / "run_manifest.json",
        {
            "input_dir": str(input_dir),
            "review_model": args.review_model,
            "prompt_template": args.prompt_template,
            "selection": args.selection,
            "resolve_abstains": bool(args.resolve_abstains),
            "workers": int(args.workers),
            "max_field_length": int(args.max_field_length),
            "max_retries": int(args.max_retries),
            "prompt_fields": prompt_fields,
            "system_prompt_path": str(system_prompt_path),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
