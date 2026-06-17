#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import run_benchmark_realtime_eval as realtime  # noqa: E402


DEFAULT_INPUT_DIR = ROOT / "generated_labels" / "abt_ditto_active_labelling_rebuilt_gpt-5-mini_agent_precision" / "all_plus20random"
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_RECALL_PROMPT = ROOT / "scripts" / "experiments" / "evidence_first_abstain" / "prompts" / "agent_recall_toned_system_prompt.txt"
DEFAULT_SKEPTIC_PROMPT = ROOT / "scripts" / "experiments" / "evidence_first_abstain" / "prompts" / "agent_variant_skeptic_system_prompt.txt"
DEFAULT_WORKERS = 10
DEFAULT_MAX_FIELD_LENGTH = 100000
DEFAULT_PROMPT_FIELDS = ("title", "description", "price")
RUN_SLUG = "recall-then-skeptic"


def _load_jsonl_gz(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_existing_results(path: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            out[int(payload["row_index"])] = payload
    return out


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_results_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "row_index",
        "pair_id",
        "original_label",
        "recall_pred",
        "skeptic_pred",
        "final_action",
        "final_label",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "recall_response",
        "skeptic_response",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _call_prompt(
    record: Dict[str, Any],
    *,
    model: str,
    system_prompt: str,
    prompt_fields: List[str],
    max_field_length: int,
    max_retries: int,
) -> Tuple[Any, str, Dict[str, int], str]:
    messages = realtime.build_messages(
        record,
        prompt_fields=prompt_fields,
        system_prompt=system_prompt,
        max_field_length=max_field_length,
    )
    return realtime._call_model(messages, model, max_retries)


def _score_row(
    *,
    row_index: int,
    record: Dict[str, Any],
    model: str,
    recall_prompt: str,
    skeptic_prompt: str,
    prompt_fields: List[str],
    max_field_length: int,
    max_retries: int,
) -> Dict[str, Any]:
    original_label = bool(realtime._normalize_gold_label(record.get("label")))
    result = {
        "row_index": int(row_index),
        "pair_id": str(record.get("pair_id", "")),
        "original_label": original_label,
        "recall_pred": None,
        "skeptic_pred": None,
        "final_action": "kept_match" if original_label else "pending_recall",
        "final_label": original_label,
        "recall_response": "",
        "skeptic_response": "",
        "error": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    if original_label:
        return result

    recall_pred, recall_text, recall_usage, recall_error = _call_prompt(
        record,
        model=model,
        system_prompt=recall_prompt,
        prompt_fields=prompt_fields,
        max_field_length=max_field_length,
        max_retries=max_retries,
    )
    result["recall_pred"] = recall_pred
    result["recall_response"] = recall_text
    result["error"] = recall_error
    for key, value in recall_usage.items():
        result[key] += int(value)

    if recall_pred is not True:
        result["final_action"] = "kept_non_match"
        result["final_label"] = False
        return result

    skeptic_pred, skeptic_text, skeptic_usage, skeptic_error = _call_prompt(
        record,
        model=model,
        system_prompt=skeptic_prompt,
        prompt_fields=prompt_fields,
        max_field_length=max_field_length,
        max_retries=max_retries,
    )
    result["skeptic_pred"] = skeptic_pred
    result["skeptic_response"] = skeptic_text
    if skeptic_error:
        result["error"] = (result["error"] + " | " + skeptic_error).strip(" |")
    for key, value in skeptic_usage.items():
        result[key] += int(value)

    if skeptic_pred is True:
        result["final_action"] = "promoted_to_match"
        result["final_label"] = True
    else:
        result["final_action"] = "excluded_after_skeptic"
        result["final_label"] = False
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply recall review to non-matches, then gate promotions through a skeptic prompt.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--recall-prompt-file", default=str(DEFAULT_RECALL_PROMPT))
    parser.add_argument("--skeptic-prompt-file", default=str(DEFAULT_SKEPTIC_PROMPT))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--max-field-length", type=int, default=DEFAULT_MAX_FIELD_LENGTH)
    parser.add_argument("--prompt-fields", default=",".join(DEFAULT_PROMPT_FIELDS))
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--run-slug", default=RUN_SLUG)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    input_dir = Path(args.input_dir)
    prompt_fields = [part.strip() for part in str(args.prompt_fields).split(",") if part.strip()]
    recall_prompt = Path(args.recall_prompt_file).read_text(encoding="utf-8").strip()
    skeptic_prompt = Path(args.skeptic_prompt_file).read_text(encoding="utf-8").strip()

    active_csv_path = input_dir / "active_labels_latest.csv"
    final_csv_path = input_dir / "labels_final.csv"
    train_gz_path = next(input_dir.glob("*train.json.gz"))

    active_df = pd.read_csv(active_csv_path)
    final_df = pd.read_csv(final_csv_path)
    records = _load_jsonl_gz(train_gz_path)

    if len(active_df) != len(final_df) or len(active_df) != len(records):
        raise ValueError("Row counts must match across active labels, final labels, and train json.gz")

    run_slug = str(args.run_slug).strip() or RUN_SLUG
    results_jsonl_path = input_dir / f"{run_slug}__results__{args.model}.jsonl"
    results_csv_path = input_dir / f"{run_slug}__results__{args.model}.csv"
    summary_path = input_dir / f"{run_slug}__summary__{args.model}.json"
    active_out_path = input_dir / f"active_labels_latest__{run_slug}__{args.model}.csv"
    final_out_path = input_dir / f"labels_final__{run_slug}__{args.model}.csv"
    train_out_path = input_dir / f"{train_gz_path.stem.replace('.json', '')}__{run_slug}__{args.model}.json.gz"

    existing = _load_existing_results(results_jsonl_path) if args.resume else {}
    pending_indices = [idx for idx in range(len(records)) if idx not in existing]

    print(f"Input dir: {input_dir}")
    print(f"Model: {args.model}")
    print(f"Pairs total: {len(records)}")
    print(f"Pairs already completed: {len(existing)}")
    print(f"Pairs pending: {len(pending_indices)}")

    with tqdm(total=len(records), initial=len(existing), desc="Recall->Skeptic relabel", unit="pair") as progress:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(
                    _score_row,
                    row_index=idx,
                    record=records[idx],
                    model=args.model,
                    recall_prompt=recall_prompt,
                    skeptic_prompt=skeptic_prompt,
                    prompt_fields=prompt_fields,
                    max_field_length=args.max_field_length,
                    max_retries=args.max_retries,
                ): idx
                for idx in pending_indices
            }
            for future in as_completed(futures):
                result = future.result()
                _append_jsonl(results_jsonl_path, result)
                progress.update(1)

    all_results_map = _load_existing_results(results_jsonl_path)
    ordered_results = [all_results_map[idx] for idx in range(len(records))]
    _write_results_csv(results_csv_path, ordered_results)

    keep_mask: List[bool] = []
    new_labels: List[bool] = []
    for result in ordered_results:
        action = str(result["final_action"])
        final_label = bool(result["final_label"])
        keep = action != "excluded_after_skeptic"
        keep_mask.append(keep)
        new_labels.append(final_label)

    kept_active = active_df.loc[keep_mask].copy().reset_index(drop=True)
    kept_final = final_df.loc[keep_mask].copy().reset_index(drop=True)
    kept_active["label"] = [label for label, keep in zip(new_labels, keep_mask) if keep]
    kept_final["label"] = [label for label, keep in zip(new_labels, keep_mask) if keep]
    kept_active.to_csv(active_out_path, index=False)
    kept_final.to_csv(final_out_path, index=False)

    kept_records = []
    for record, label, keep in zip(records, new_labels, keep_mask):
        if not keep:
            continue
        updated = dict(record)
        updated["label"] = 1 if label else 0
        kept_records.append(updated)
    with gzip.open(train_out_path, "wt", encoding="utf-8") as handle:
        for row in kept_records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    non_match_candidates = int((~active_df["label"].astype(bool)).sum())
    recall_promoted = int(sum(1 for row in ordered_results if row["recall_pred"] is True))
    skeptic_confirmed = int(sum(1 for row in ordered_results if row["final_action"] == "promoted_to_match"))
    excluded = int(sum(1 for row in ordered_results if row["final_action"] == "excluded_after_skeptic"))
    labels_changed_to_match = int(
        sum(1 for original, final, keep in zip(active_df["label"].astype(bool).tolist(), new_labels, keep_mask) if keep and (not original) and final)
    )
    total_prompt_tokens = int(sum(int(row.get("prompt_tokens", 0) or 0) for row in ordered_results))
    total_completion_tokens = int(sum(int(row.get("completion_tokens", 0) or 0) for row in ordered_results))
    total_tokens = int(sum(int(row.get("total_tokens", 0) or 0) for row in ordered_results))

    summary = {
        "input_dir": str(input_dir),
        "pair_count_before": int(len(records)),
        "pair_count_after": int(len(kept_records)),
        "excluded_rows": excluded,
        "model": args.model,
        "recall_prompt_file": str(args.recall_prompt_file),
        "skeptic_prompt_file": str(args.skeptic_prompt_file),
        "non_match_candidates_reviewed": non_match_candidates,
        "recall_promoted_to_match": recall_promoted,
        "skeptic_confirmed_matches": skeptic_confirmed,
        "labels_changed_to_match": labels_changed_to_match,
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
