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

FIELD_ALIASES: Dict[str, tuple[str, ...]] = {
    "title": ("name",),
}
PROMPT_FILE_DEFAULT = Path("scripts/experiments/evidence_first_abstain/prompts/agent_variant_skeptic_system_prompt.txt")
DEFAULT_WORKERS = 10
DEFAULT_CHECKPOINT_EVERY = 25
DEFAULT_MAX_RETRIES = 3

_thread_local = threading.local()


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


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


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


def _build_messages(
    record: Dict[str, Any],
    *,
    system_prompt: str,
    prompt_fields: Iterable[str],
    max_field_length: int,
) -> List[Dict[str, str]]:
    left_json = json.dumps(
        _extract_entity_payload(record, "left", max_field_length, prompt_fields),
        ensure_ascii=False,
    )
    right_json = json.dumps(
        _extract_entity_payload(record, "right", max_field_length, prompt_fields),
        ensure_ascii=False,
    )
    user_prompt = (
        "Do the two entity descriptions refer to the same real-world entity? "
        f"Entity 1: '{left_json}'. "
        f"Entity 2: '{right_json}'."
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


def _extract_usage(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def parse_match_from_content(content: str) -> Optional[bool]:
    text = str(content or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        parts = text.split("\n", 1)
        text = parts[1] if len(parts) == 2 else text
        text = text.rsplit("```", 1)[0].strip()

    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and "match" in payload:
            return bool(payload["match"])
    except json.JSONDecodeError:
        pass

    lower = text.lower()
    if '"match": true' in lower or '"match":true' in lower:
        return True
    if '"match": false' in lower or '"match":false' in lower:
        return False
    return None


def _call_model(messages: List[Dict[str, str]], model: str, max_retries: int) -> tuple[Optional[bool], str, Dict[str, int], str]:
    client = _get_client()
    aggregate_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            usage = _extract_usage(response)
            for key, value in usage.items():
                aggregate_usage[key] += int(value)
            content = response.choices[0].message.content or ""
            return parse_match_from_content(content), content, aggregate_usage, ""
        except Exception as exc:
            last_error = str(exc)
            time.sleep(min(2 ** attempt, 8))
    return None, "", aggregate_usage, last_error or "request_failed"


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


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


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


def _score_review(
    source_row: Dict[str, Any],
    record: Dict[str, Any],
    *,
    review_model: str,
    system_prompt: str,
    prompt_fields: Iterable[str],
    max_field_length: int,
    max_retries: int,
) -> Dict[str, Any]:
    pair_id = str(source_row.get("pair_id", ""))
    gold_bool = _normalize_gold_label(source_row.get("gold_label"))
    first_pred = bool(source_row.get("predicted_label"))
    result = {
        "pair_index": int(source_row.get("pair_index", 0) or 0),
        "pair_id": pair_id,
        "id_left": source_row.get("id_left"),
        "id_right": source_row.get("id_right"),
        "gold_label": bool(gold_bool),
        "first_pred_label": first_pred,
        "review_pred_label": None,
        "final_pred_label": first_pred,
        "status": "ok",
        "combine_reason": "kept_non_match" if not first_pred else "pending_review",
        "review_response": "",
        "review_error": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    if not first_pred:
        return result

    messages = _build_messages(
        record,
        system_prompt=system_prompt,
        prompt_fields=prompt_fields,
        max_field_length=max_field_length,
    )
    review_match, response_text, usage, error = _call_model(messages, review_model, max_retries)
    result.update(
        {
            "review_pred_label": review_match,
            "review_response": response_text,
            "review_error": error,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
        }
    )
    if review_match is None:
        result["status"] = "error"
        result["final_pred_label"] = first_pred
        result["combine_reason"] = "review_parse_error_kept_first_pass"
        return result

    result["final_pred_label"] = bool(review_match)
    result["combine_reason"] = "skeptic_confirmed_match" if review_match else "skeptic_rejected_match"
    return result


def _write_predictions_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "pair_index",
        "pair_id",
        "id_left",
        "id_right",
        "gold_label",
        "first_pred_label",
        "review_pred_label",
        "final_pred_label",
        "combine_reason",
        "status",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "review_error",
        "review_response",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: int(item.get("pair_index", 0))):
            writer.writerow({key: row.get(key) for key in fieldnames})


def _compute_summary(rows: List[Dict[str, Any]], source_predictions: pd.DataFrame, *, input_dir: Path, review_model: str, prompt_file: Path, max_field_length: int) -> Dict[str, Any]:
    before_gold = source_predictions["gold_label"].astype(bool).tolist()
    before_pred = source_predictions["predicted_label"].astype(bool).tolist()
    before_metrics = compute_binary_metrics(before_gold, before_pred)

    final_gold = [bool(row["gold_label"]) for row in rows]
    final_pred = [bool(row["final_pred_label"]) for row in rows]
    final_metrics = compute_binary_metrics(final_gold, final_pred)

    summary = {
        **final_metrics,
        "pair_count": len(rows),
        "pairs_scored": len(rows),
        "coverage": 1.0,
        "parse_failures": int(sum(1 for row in rows if row.get("status") != "ok")),
        "first_pass_metrics": before_metrics,
        "post_skeptic_metrics": final_metrics,
        "reviewed_matches": int(sum(1 for row in rows if row.get("first_pred_label") is True)),
        "skeptic_confirmed_matches": int(sum(1 for row in rows if row.get("combine_reason") == "skeptic_confirmed_match")),
        "skeptic_rejected_matches": int(sum(1 for row in rows if row.get("combine_reason") == "skeptic_rejected_match")),
        "prompt_tokens": int(sum(int(row.get("prompt_tokens", 0) or 0) for row in rows)),
        "completion_tokens": int(sum(int(row.get("completion_tokens", 0) or 0) for row in rows)),
        "total_tokens": int(sum(int(row.get("total_tokens", 0) or 0) for row in rows)),
        "source_input_dir": str(input_dir),
        "review_model": review_model,
        "prompt_file": str(prompt_file),
        "max_field_length": int(max_field_length),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt-only skeptical review of first-pass positive matches from realtime benchmark outputs.")
    parser.add_argument("--input-dir", required=True, help="Per-dataset realtime eval directory containing predictions.csv and dataset_manifest.json")
    parser.add_argument("--review-model", required=True)
    parser.add_argument("--prompt-file", default=str(PROMPT_FILE_DEFAULT))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    load_dotenv()

    input_dir = Path(args.input_dir)
    dataset_manifest = _load_json(input_dir / "dataset_manifest.json")
    predictions = pd.read_csv(input_dir / "predictions.csv")
    data_path = Path(dataset_manifest["data_path"])
    records = _read_jsonl_gz(data_path)
    record_by_pair = {str(record.get("pair_id", "")): record for record in records}
    prompt_fields = list(dataset_manifest.get("prompt_fields") or [])
    max_field_length = int(dataset_manifest.get("max_field_length", 200) or 200)
    prompt_file = Path(args.prompt_file)
    system_prompt = prompt_file.read_text(encoding="utf-8").strip()

    output_dir = input_dir / "positive_only_skeptical_review_realtime" / prompt_file.stem / re.sub(r"[^a-z0-9]+", "-", args.review_model.lower()).strip("-")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    existing = _load_existing_results(results_path) if args.resume else {}
    if results_path.exists() and not args.resume:
        results_path.unlink()
        existing = {}

    source_rows = predictions.to_dict(orient="records")
    pending = [row for row in source_rows if str(row.get("pair_id", "")) not in existing]

    print(f"Input dir: {input_dir}")
    print(f"Review model: {args.review_model}")
    print(f"Prompt file: {prompt_file}")
    print(f"Pairs total: {len(source_rows)}")
    print(f"Pairs already completed: {len(existing)}")
    print(f"Pairs pending: {len(pending)}")

    results_by_pair = dict(existing)
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            futures = {
                executor.submit(
                    _score_review,
                    row,
                    record_by_pair[str(row.get("pair_id", ""))],
                    review_model=args.review_model,
                    system_prompt=system_prompt,
                    prompt_fields=prompt_fields,
                    max_field_length=max_field_length,
                    max_retries=int(args.max_retries),
                ): row
                for row in pending
            }
            progress = tqdm(total=len(futures), desc="Positive skeptical review", unit="pair")
            completed_since_checkpoint = 0
            for future in as_completed(futures):
                result = future.result()
                results_by_pair[str(result["pair_id"])] = result
                _append_jsonl(results_path, result)
                progress.update(1)
                completed_since_checkpoint += 1
                if completed_since_checkpoint >= int(args.checkpoint_every):
                    _write_predictions_csv(output_dir / "predictions.csv", list(results_by_pair.values()))
                    completed_since_checkpoint = 0
            progress.close()

    final_rows = list(results_by_pair.values())
    _write_predictions_csv(output_dir / "predictions.csv", final_rows)
    summary = _compute_summary(
        final_rows,
        predictions,
        input_dir=input_dir,
        review_model=args.review_model,
        prompt_file=prompt_file,
        max_field_length=max_field_length,
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
