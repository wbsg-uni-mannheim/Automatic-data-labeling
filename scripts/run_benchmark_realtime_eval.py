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

DEFAULT_CONFIG_PATH = Path("configs/labeling/benchmarks_active.yaml")
DEFAULT_DATA_ROOT = Path("data")
DEFAULT_OUTPUT_DIR = Path("output/realtime_benchmark_eval")
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_MAX_FIELD_LENGTH = 200
DEFAULT_WORKERS = 8
DEFAULT_CHECKPOINT_EVERY = 50
DEFAULT_MAX_RETRIES = 3

SKIP_EXACT_FIELDS = {"id", "label", "pair_id", "explanation"}
FIELD_ALIASES: Dict[str, tuple[str, ...]] = {
    "title": ("name",),
}

ACTIVE_LEARNING_SYSTEM_PROMPT = (
    "You are an expert entity matcher. "
    "Decide if two records refer to the same real-world entity. "
    "Return only valid JSON with exactly one field: "
    '{"match": true|false}.'
)


@dataclass(frozen=True)
class TestSetSpec:
    benchmark: str
    dataset_name: str
    data_path: str
    output_slug: str
    prompt_fields: tuple[str, ...]


_thread_local = threading.local()


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping at {path}")
    return payload


def _split_csv_arg(value: str) -> List[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _coerce_mapping(value: Any, name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


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


def discover_test_sets(
    config_path: Path = DEFAULT_CONFIG_PATH,
    data_root: Path = DEFAULT_DATA_ROOT,
    benchmarks: Optional[Iterable[str]] = None,
) -> List[TestSetSpec]:
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


def build_messages(
    record: Dict[str, Any],
    *,
    prompt_fields: Iterable[str],
    system_prompt: str = ACTIVE_LEARNING_SYSTEM_PROMPT,
    max_field_length: int = DEFAULT_MAX_FIELD_LENGTH,
) -> List[Dict[str, str]]:
    left_json = json.dumps(
        _extract_entity_payload(
            record,
            "left",
            max_field_length=max_field_length,
            prompt_fields=prompt_fields,
        ),
        ensure_ascii=False,
    )
    right_json = json.dumps(
        _extract_entity_payload(
            record,
            "right",
            max_field_length=max_field_length,
            prompt_fields=prompt_fields,
        ),
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


def _load_prompt_text(prompt_file: Optional[Path]) -> str:
    if prompt_file is None:
        return ACTIVE_LEARNING_SYSTEM_PROMPT
    return prompt_file.read_text(encoding="utf-8").strip()


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


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _dataset_dir(run_dir: Path, spec_or_slug: TestSetSpec | str) -> Path:
    slug = spec_or_slug.output_slug if isinstance(spec_or_slug, TestSetSpec) else str(spec_or_slug)
    return run_dir / slug


def _get_client() -> OpenAI:
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = OpenAI()
        _thread_local.client = client
    return client


def _record_custom_id(spec: TestSetSpec, idx: int) -> str:
    return f"{spec.output_slug}__req_{idx}"


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
            custom_id = str(payload.get("custom_id", ""))
            if custom_id:
                out[custom_id] = payload
    return out


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


def _score_one(
    *,
    spec: TestSetSpec,
    record: Dict[str, Any],
    pair_index: int,
    model: str,
    system_prompt: str,
    max_field_length: int,
    max_retries: int,
) -> Dict[str, Any]:
    custom_id = _record_custom_id(spec, pair_index)
    messages = build_messages(
        record,
        prompt_fields=spec.prompt_fields,
        system_prompt=system_prompt,
        max_field_length=max_field_length,
    )
    predicted_match, response_text, usage, error = _call_model(messages, model, max_retries)
    gold_bool = _normalize_gold_label(record.get("label"))
    return {
        "custom_id": custom_id,
        "benchmark": spec.benchmark,
        "dataset_name": spec.dataset_name,
        "pair_index": int(pair_index),
        "pair_id": _normalize_text(record.get("pair_id")),
        "id_left": _normalize_text(record.get("id_left")),
        "id_right": _normalize_text(record.get("id_right")),
        "gold_match": bool(gold_bool),
        "gold_label": "TRUE" if gold_bool else "FALSE",
        "predicted_match": predicted_match,
        "predicted_label": "TRUE" if predicted_match is True else ("FALSE" if predicted_match is False else ""),
        "response_text": response_text,
        "error": error,
        "messages": messages,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
    }


def _write_predictions_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "custom_id",
        "pair_index",
        "pair_id",
        "id_left",
        "id_right",
        "gold_label",
        "predicted_label",
        "error",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "response_text",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: int(item.get("pair_index", 0))):
            writer.writerow({key: row.get(key) for key in fieldnames})


def _evaluate_dataset(dataset_dir: Path, manifest: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    eval_df = pd.DataFrame(rows)
    if eval_df.empty:
        summary = {
            "benchmark": manifest["benchmark"],
            "dataset_name": manifest["dataset_name"],
            "output_slug": manifest["output_slug"],
            "model": manifest["model"],
            "pair_count": 0,
            "pairs_scored": 0,
            "coverage": 0.0,
            "parse_failures": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        _write_json(dataset_dir / "summary.json", summary)
        return summary

    _write_predictions_csv(dataset_dir / "predictions.csv", rows)
    scored_df = eval_df[eval_df["predicted_match"].notna()].copy()
    gold = scored_df["gold_match"].astype(bool).tolist()
    pred = scored_df["predicted_match"].astype(bool).tolist()
    metrics = compute_binary_metrics(gold, pred)
    total_pairs = int(len(eval_df))
    parsed_pairs = int(len(scored_df))
    parse_failures = int(total_pairs - parsed_pairs)
    coverage = (parsed_pairs / total_pairs) if total_pairs else 0.0
    summary = {
        "benchmark": manifest["benchmark"],
        "dataset_name": manifest["dataset_name"],
        "output_slug": manifest["output_slug"],
        "model": manifest["model"],
        "pair_count": total_pairs,
        "pairs_scored": metrics["pairs_scored"],
        "coverage": coverage,
        "parse_failures": parse_failures,
        "accuracy": metrics["accuracy"] if parsed_pairs else None,
        "precision": metrics["precision"] if parsed_pairs else None,
        "recall": metrics["recall"] if parsed_pairs else None,
        "f1": metrics["f1"] if parsed_pairs else None,
        "tp": metrics["tp"],
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "prompt_tokens": int(eval_df["prompt_tokens"].fillna(0).astype(int).sum()),
        "completion_tokens": int(eval_df["completion_tokens"].fillna(0).astype(int).sum()),
        "total_tokens": int(eval_df["total_tokens"].fillna(0).astype(int).sum()),
    }
    _write_json(dataset_dir / "summary.json", summary)
    return summary


def print_summary_table(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        print("No evaluation results available.")
        return
    display_cols = [
        "benchmark",
        "dataset_name",
        "pair_count",
        "pairs_scored",
        "coverage",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "parse_failures",
    ]
    print(summary_df[display_cols].to_string(index=False))


def run_realtime_eval(
    *,
    output_dir: Path,
    model: str,
    benchmarks: Optional[Iterable[str]],
    system_prompt: str,
    max_field_length: int,
    workers: int,
    checkpoint_every: int,
    max_retries: int,
    resume: bool,
) -> pd.DataFrame:
    resolved_run_dir = _resolve_output_dir(output_dir, create=True)
    specs = discover_test_sets(
        config_path=DEFAULT_CONFIG_PATH,
        data_root=DEFAULT_DATA_ROOT,
        benchmarks=benchmarks,
    )
    run_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(DEFAULT_CONFIG_PATH),
        "data_root": str(DEFAULT_DATA_ROOT),
        "model": model,
        "benchmarks": list(benchmarks or []),
        "system_prompt_source": "active_learning_default",
        "system_prompt": system_prompt,
        "max_field_length": int(max_field_length),
        "workers": int(workers),
        "max_retries": int(max_retries),
        "datasets": [],
    }
    _write_json(resolved_run_dir / "run_manifest.json", run_manifest)

    summary_rows: List[Dict[str, Any]] = []
    for spec in specs:
        dataset_dir = _dataset_dir(resolved_run_dir, spec)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        records = _read_jsonl_gz(Path(spec.data_path))
        manifest = {
            "benchmark": spec.benchmark,
            "dataset_name": spec.dataset_name,
            "data_path": spec.data_path,
            "output_slug": spec.output_slug,
            "pair_count": len(records),
            "model": model,
            "system_prompt": system_prompt,
            "max_field_length": int(max_field_length),
            "prompt_fields": list(spec.prompt_fields),
        }
        _write_json(dataset_dir / "dataset_manifest.json", manifest)
        run_manifest["datasets"].append(manifest)
        _write_json(resolved_run_dir / "run_manifest.json", run_manifest)

        results_path = dataset_dir / "results.jsonl"
        existing = _load_existing_results(results_path) if resume else {}
        if results_path.exists() and not resume:
            results_path.unlink()
            existing = {}

        pending = [
            (idx, record)
            for idx, record in enumerate(records)
            if _record_custom_id(spec, idx) not in existing
        ]

        print(f"Dataset: {spec.output_slug}")
        print(f"Pairs total: {len(records)}")
        print(f"Pairs already completed: {len(existing)}")
        print(f"Pairs pending: {len(pending)}")

        results_by_id = dict(existing)
        if pending:
            with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
                futures = {
                    executor.submit(
                        _score_one,
                        spec=spec,
                        record=record,
                        pair_index=idx,
                        model=model,
                        system_prompt=system_prompt,
                        max_field_length=max_field_length,
                        max_retries=max_retries,
                    ): idx
                    for idx, record in pending
                }
                progress = tqdm(total=len(futures), desc=spec.output_slug, unit="pair")
                completed_since_checkpoint = 0
                for future in as_completed(futures):
                    result = future.result()
                    results_by_id[str(result["custom_id"])] = result
                    _append_jsonl(results_path, result)
                    progress.update(1)
                    completed_since_checkpoint += 1
                    if completed_since_checkpoint >= int(checkpoint_every):
                        _evaluate_dataset(dataset_dir, manifest, list(results_by_id.values()))
                        completed_since_checkpoint = 0
                progress.close()

        summary_rows.append(_evaluate_dataset(dataset_dir, manifest, list(results_by_id.values())))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(resolved_run_dir / "summary.csv", index=False)
    _write_json(
        resolved_run_dir / "summary.json",
        {
            "run_dir": str(resolved_run_dir),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "results": summary_rows,
        },
    )
    return summary_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Realtime benchmark evaluation using the same active-learning prompt construction as the batch runner."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--benchmarks", default="", help="Comma-separated benchmark keys, e.g. abt-buy")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--max-field-length", type=int, default=DEFAULT_MAX_FIELD_LENGTH)
    parser.add_argument("--prompt-file", default="", help="Optional system prompt file")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    prompt_file = Path(args.prompt_file) if args.prompt_file else None
    system_prompt = _load_prompt_text(prompt_file)
    summary_df = run_realtime_eval(
        output_dir=Path(args.output_dir),
        model=args.model,
        benchmarks=_split_csv_arg(args.benchmarks),
        system_prompt=system_prompt,
        max_field_length=int(args.max_field_length),
        workers=int(args.workers),
        checkpoint_every=int(args.checkpoint_every),
        max_retries=int(args.max_retries),
        resume=bool(args.resume),
    )
    print_summary_table(summary_df)


if __name__ == "__main__":
    main()
