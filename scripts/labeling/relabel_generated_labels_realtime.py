#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import run_benchmark_realtime_eval as realtime  # noqa: E402


DEFAULT_INPUT_DIR = ROOT / "generated_labels" / "abt_ditto_active_labelling" / "all_plus20random"
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_PROMPT_FILE = ROOT / "scripts" / "experiments" / "evidence_first_abstain" / "prompts" / "agent_precision_system_prompt.txt"
DEFAULT_WORKERS = 10
DEFAULT_MAX_FIELD_LENGTH = 100000
DEFAULT_PROMPT_FIELDS = ("title", "description", "price")
RESULTS_BASENAME = "relabel_results"


def _slugify(value: str) -> str:
    out: List[str] = []
    prev_dash = False
    for ch in str(value):
        if ch.isalnum():
            out.append(ch.lower())
            prev_dash = False
        else:
            if not prev_dash:
                out.append("-")
            prev_dash = True
    return "".join(out).strip("-") or "run"


def _load_jsonl_gz(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_existing_results(path: Path) -> Dict[str, Dict[str, Any]]:
    existing: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return existing
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            pair_id = str(payload.get("pair_id", ""))
            if pair_id:
                existing[pair_id] = payload
    return existing


def _bool_to_csv_label(value: bool) -> bool:
    return bool(value)


def _bool_to_json_label(value: bool) -> int:
    return 1 if value else 0


def _write_results_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "pair_index",
        "pair_id",
        "predicted_label",
        "predicted_match",
        "original_label",
        "changed",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "error",
        "response_text",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _score_record(
    *,
    record: Dict[str, Any],
    pair_index: int,
    model: str,
    system_prompt: str,
    max_field_length: int,
    prompt_fields: List[str],
    max_retries: int,
) -> Dict[str, Any]:
    messages = realtime.build_messages(
        record,
        prompt_fields=prompt_fields,
        system_prompt=system_prompt,
        max_field_length=max_field_length,
    )
    predicted_match, response_text, usage, error = realtime._call_model(messages, model, max_retries)
    original_match = realtime._normalize_gold_label(record.get("label"))
    return {
        "pair_index": int(pair_index),
        "pair_id": str(record.get("pair_id", "")),
        "predicted_match": predicted_match,
        "predicted_label": "TRUE" if predicted_match else "FALSE" if predicted_match is False else "",
        "original_label": "TRUE" if original_match else "FALSE",
        "changed": bool(predicted_match is not None and predicted_match != original_match),
        "prompt_tokens": int(usage["prompt_tokens"]),
        "completion_tokens": int(usage["completion_tokens"]),
        "total_tokens": int(usage["total_tokens"]),
        "error": error,
        "response_text": response_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Relabel a generated label set using the realtime entity-matching prompt.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_FILE))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--max-field-length", type=int, default=DEFAULT_MAX_FIELD_LENGTH)
    parser.add_argument("--prompt-fields", default=",".join(DEFAULT_PROMPT_FIELDS))
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    input_dir = Path(args.input_dir)
    prompt_file = Path(args.prompt_file)
    prompt_fields = [part.strip() for part in str(args.prompt_fields).split(",") if part.strip()]
    if not prompt_fields:
        raise ValueError("At least one prompt field is required")

    active_csv_path = input_dir / "active_labels_latest.csv"
    final_csv_path = input_dir / "labels_final.csv"
    train_gz_path = next(input_dir.glob("*train.json.gz"))

    active_df = pd.read_csv(active_csv_path)
    final_df = pd.read_csv(final_csv_path)
    records = _load_jsonl_gz(train_gz_path)

    if len(active_df) != len(records) or len(final_df) != len(records):
        raise ValueError(
            f"Row count mismatch: active={len(active_df)} final={len(final_df)} train={len(records)}"
        )

    prompt_slug = _slugify(prompt_file.stem)
    model_slug = _slugify(args.model)
    run_slug = f"{model_slug}__{prompt_slug}"

    results_jsonl_path = input_dir / f"{RESULTS_BASENAME}__{run_slug}.jsonl"
    results_csv_path = input_dir / f"{RESULTS_BASENAME}__{run_slug}.csv"
    summary_path = input_dir / f"relabel_summary__{run_slug}.json"
    active_out_path = input_dir / f"active_labels_latest__relabeled__{run_slug}.csv"
    final_out_path = input_dir / f"labels_final__relabeled__{run_slug}.csv"
    train_out_path = input_dir / f"{train_gz_path.stem.replace('.json', '')}__relabeled__{run_slug}.json.gz"

    system_prompt = prompt_file.read_text(encoding="utf-8").strip()
    existing = _load_existing_results(results_jsonl_path) if args.resume else {}
    pending_records = [
        (idx, record)
        for idx, record in enumerate(records)
        if str(record.get("pair_id", "")) not in existing
    ]

    print(f"Input dir: {input_dir}")
    print(f"Model: {args.model}")
    print(f"Prompt file: {prompt_file}")
    print(f"Prompt fields: {prompt_fields}")
    print(f"Pairs total: {len(records)}")
    print(f"Pairs already completed: {len(existing)}")
    print(f"Pairs pending: {len(pending_records)}")

    with tqdm(total=len(records), initial=len(existing), desc="Relabeling", unit="pair") as progress:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(
                    _score_record,
                    record=record,
                    pair_index=idx,
                    model=args.model,
                    system_prompt=system_prompt,
                    max_field_length=args.max_field_length,
                    prompt_fields=prompt_fields,
                    max_retries=args.max_retries,
                ): record
                for idx, record in pending_records
            }
            for future in as_completed(futures):
                result = future.result()
                _append_jsonl(results_jsonl_path, result)
                progress.update(1)

    all_results = _load_existing_results(results_jsonl_path)
    ordered_results = [all_results[str(record.get("pair_id", ""))] for record in records]
    _write_results_csv(results_csv_path, ordered_results)

    new_labels: List[bool] = []
    parse_failures = 0
    for result, record in zip(ordered_results, records):
        predicted = result.get("predicted_match")
        if predicted is None:
            parse_failures += 1
            predicted = realtime._normalize_gold_label(record.get("label"))
        new_labels.append(bool(predicted))

    active_new = active_df.copy()
    final_new = final_df.copy()
    active_new["label"] = [_bool_to_csv_label(value) for value in new_labels]
    final_new["label"] = [_bool_to_csv_label(value) for value in new_labels]
    active_new.to_csv(active_out_path, index=False)
    final_new.to_csv(final_out_path, index=False)

    with gzip.open(train_out_path, "wt", encoding="utf-8") as handle:
        for record, new_label in zip(records, new_labels):
            updated = dict(record)
            updated["label"] = _bool_to_json_label(new_label)
            handle.write(json.dumps(updated, ensure_ascii=False) + "\n")

    original_labels = [bool(value) for value in active_df["label"].tolist()]
    changed_count = int(sum(int(old != new) for old, new in zip(original_labels, new_labels)))
    changed_to_match = int(sum(int((not old) and new) for old, new in zip(original_labels, new_labels)))
    changed_to_non_match = int(sum(int(old and (not new)) for old, new in zip(original_labels, new_labels)))

    total_prompt_tokens = int(sum(int(row.get("prompt_tokens", 0) or 0) for row in ordered_results))
    total_completion_tokens = int(sum(int(row.get("completion_tokens", 0) or 0) for row in ordered_results))
    total_tokens = int(sum(int(row.get("total_tokens", 0) or 0) for row in ordered_results))

    summary = {
        "input_dir": str(input_dir),
        "pair_count": int(len(records)),
        "model": args.model,
        "prompt_file": str(prompt_file),
        "prompt_fields": prompt_fields,
        "max_field_length": int(args.max_field_length),
        "parse_failures": int(parse_failures),
        "changed_labels": changed_count,
        "changed_to_match": changed_to_match,
        "changed_to_non_match": changed_to_non_match,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "results_jsonl": str(results_jsonl_path),
        "results_csv": str(results_csv_path),
        "active_labels_output": str(active_out_path),
        "labels_final_output": str(final_out_path),
        "train_output": str(train_out_path),
    }
    _write_json(summary_path, summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
