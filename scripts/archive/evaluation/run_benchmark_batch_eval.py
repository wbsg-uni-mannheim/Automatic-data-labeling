#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import yaml
from dotenv import load_dotenv

DEFAULT_CONFIG_PATH = Path("configs/labeling/benchmarks_active.yaml")
DEFAULT_DATA_ROOT = Path("data")
DEFAULT_OUTPUT_DIR = Path("output/batch_benchmark_eval")
DEFAULT_MODEL = "gpt-5.2"
DEFAULT_COMPLETION_WINDOW = "24h"
DEFAULT_ENDPOINT = "/v1/chat/completions"
DEFAULT_MAX_FIELD_LENGTH = 200

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


def _resolve_output_dir(output_dir: Path, *, create: bool) -> Path:
    if create:
        output_dir.mkdir(parents=True, exist_ok=True)
    elif not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    return output_dir


def _load_prompt_text(prompt_file: Optional[Path]) -> str:
    if prompt_file is None:
        return ACTIVE_LEARNING_SYSTEM_PROMPT
    return prompt_file.read_text(encoding="utf-8").strip()


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


def _prepare_single_dataset(
    *,
    run_dir: Path,
    spec: TestSetSpec,
    model: str,
    system_prompt: str,
    max_field_length: int,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    dataset_dir = _dataset_dir(run_dir, spec)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    records = _read_jsonl_gz(Path(spec.data_path))
    metadata_rows: List[Dict[str, Any]] = []
    requests: List[Dict[str, Any]] = []

    for idx, record in enumerate(records):
        custom_id = f"{spec.output_slug}__req_{idx}"
        messages = build_messages(
            record,
            prompt_fields=spec.prompt_fields,
            system_prompt=system_prompt,
            max_field_length=max_field_length,
        )
        requests.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": DEFAULT_ENDPOINT,
                "body": {
                    "model": model,
                    #"temperature": 0,
                    "messages": messages,
                },
            }
        )

        gold_bool = _normalize_gold_label(record.get("label"))
        metadata_rows.append(
            {
                "custom_id": custom_id,
                "pair_index": idx,
                "pair_id": _normalize_text(record.get("pair_id")),
                "id_left": _normalize_text(record.get("id_left")),
                "id_right": _normalize_text(record.get("id_right")),
                "gold_label": "TRUE" if gold_bool else "FALSE",
                "gold_match": gold_bool,
            }
        )

    pd.DataFrame(metadata_rows).to_csv(dataset_dir / "metadata.csv", index=False)

    dataset_manifest = {
        "benchmark": spec.benchmark,
        "dataset_name": spec.dataset_name,
        "data_path": spec.data_path,
        "output_slug": spec.output_slug,
        "pair_count": len(metadata_rows),
        "model": model,
        "system_prompt": system_prompt,
        "max_field_length": int(max_field_length),
        "prompt_fields": list(spec.prompt_fields),
    }
    _write_json(dataset_dir / "dataset_manifest.json", dataset_manifest)
    return dataset_manifest, requests


def prepare_run(
    *,
    output_dir: Path,
    model: str,
    benchmarks: Optional[Iterable[str]] = None,
) -> Path:
    resolved_run_dir = _resolve_output_dir(output_dir, create=True)
    system_prompt = _load_prompt_text(None)
    specs = discover_test_sets(
        config_path=DEFAULT_CONFIG_PATH,
        data_root=DEFAULT_DATA_ROOT,
        benchmarks=benchmarks,
    )

    dataset_manifests: List[Dict[str, Any]] = []
    combined_requests: List[Dict[str, Any]] = []
    for spec in specs:
        dataset_manifest, requests = _prepare_single_dataset(
            run_dir=resolved_run_dir,
            spec=spec,
            model=model,
            system_prompt=system_prompt,
            max_field_length=DEFAULT_MAX_FIELD_LENGTH,
        )
        dataset_manifests.append(dataset_manifest)
        combined_requests.extend(requests)

    batch_input_path = resolved_run_dir / "batch_input.jsonl"
    with batch_input_path.open("w", encoding="utf-8") as handle:
        for request in combined_requests:
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")

    run_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(DEFAULT_CONFIG_PATH),
        "data_root": str(DEFAULT_DATA_ROOT),
        "model": model,
        "benchmarks": list(benchmarks or []),
        "system_prompt_source": "active_learning_default",
        "system_prompt": system_prompt,
        "max_field_length": int(DEFAULT_MAX_FIELD_LENGTH),
        "batch_input_path": str(batch_input_path),
        "request_count": len(combined_requests),
        "datasets": dataset_manifests,
    }
    _write_json(resolved_run_dir / "run_manifest.json", run_manifest)
    return resolved_run_dir


def submit_run(run_dir: Path) -> None:
    from openai import OpenAI

    client = OpenAI()
    run_manifest = _load_json(run_dir / "run_manifest.json")
    batch_info_path = run_dir / "batch_info.json"
    if batch_info_path.exists():
        batch_info = _load_json(batch_info_path)
        status = str(batch_info.get("status", "")).lower()
        if status not in {"failed", "expired", "cancelled"}:
            return

    batch_input_path = run_dir / "batch_input.jsonl"
    with batch_input_path.open("rb") as handle:
        file_obj = client.files.create(file=handle, purpose="batch")

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint=DEFAULT_ENDPOINT,
        completion_window=DEFAULT_COMPLETION_WINDOW,
        metadata={
            "run_dir": run_dir.name,
            "request_count": str(run_manifest.get("request_count", "")),
            "dataset_count": str(len(run_manifest.get("datasets") or [])),
        },
    )
    batch_info = {
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
        "status": batch.status,
        "file_id": file_obj.id,
        "batch_id": batch.id,
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
    }
    _write_json(batch_info_path, batch_info)


def _download_file_if_missing(client: Any, file_id: Optional[str], target_path: Path) -> None:
    if not file_id or target_path.exists():
        return
    content = client.files.content(file_id)
    with target_path.open("wb") as handle:
        handle.write(content.read())


def _legacy_batch_info_paths(run_dir: Path) -> List[Path]:
    return sorted(run_dir.glob("*/batch_info.json"))


def refresh_status(run_dir: Path) -> List[Dict[str, Any]]:
    from openai import OpenAI

    client = OpenAI()
    run_manifest = _load_json(run_dir / "run_manifest.json")
    batch_info_path = run_dir / "batch_info.json"
    if not batch_info_path.exists():
        legacy_paths = _legacy_batch_info_paths(run_dir)
        if legacy_paths:
            statuses: List[Dict[str, Any]] = []
            dataset_map = {
                str(dataset["output_slug"]): dataset for dataset in (run_manifest.get("datasets") or [])
            }
            for info_path in legacy_paths:
                slug = info_path.parent.name
                dataset = dataset_map.get(slug, {})
                batch_info = _load_json(info_path)
                batch = client.batches.retrieve(str(batch_info["batch_id"]))
                request_counts = getattr(batch, "request_counts", None)
                updated = {
                    **batch_info,
                    "checked_at": datetime.now().isoformat(timespec="seconds"),
                    "status": batch.status,
                    "output_file_id": getattr(batch, "output_file_id", None),
                    "error_file_id": getattr(batch, "error_file_id", None),
                    "request_counts": {
                        "completed": int(getattr(request_counts, "completed", 0) or 0),
                        "failed": int(getattr(request_counts, "failed", 0) or 0),
                        "total": int(getattr(request_counts, "total", 0) or 0),
                    },
                }
                _write_json(info_path, updated)
                _download_file_if_missing(client, updated.get("output_file_id"), info_path.parent / "batch_output.jsonl")
                _download_file_if_missing(client, updated.get("error_file_id"), info_path.parent / "batch_error.jsonl")
                statuses.append(
                    {
                        "output_slug": slug,
                        "benchmark": dataset.get("benchmark", ""),
                        "dataset_name": dataset.get("dataset_name", ""),
                        "status": updated["status"],
                        "completed": updated["request_counts"]["completed"],
                        "failed": updated["request_counts"]["failed"],
                        "total": updated["request_counts"]["total"],
                    }
                )
            return statuses
        return [
            {
                "run_dir": run_dir.name,
                "status": "not_submitted",
                "datasets": len(run_manifest.get("datasets") or []),
                "requests": int(run_manifest.get("request_count", 0) or 0),
            }
        ]

    batch_info = _load_json(batch_info_path)
    batch = client.batches.retrieve(str(batch_info["batch_id"]))
    request_counts = getattr(batch, "request_counts", None)
    updated = {
        **batch_info,
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "status": batch.status,
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
        "request_counts": {
            "completed": int(getattr(request_counts, "completed", 0) or 0),
            "failed": int(getattr(request_counts, "failed", 0) or 0),
            "total": int(getattr(request_counts, "total", 0) or 0),
        },
    }
    _write_json(batch_info_path, updated)

    _download_file_if_missing(client, updated.get("output_file_id"), run_dir / "batch_output.jsonl")
    _download_file_if_missing(client, updated.get("error_file_id"), run_dir / "batch_error.jsonl")

    return [
        {
            "run_dir": run_dir.name,
            "status": updated["status"],
            "datasets": len(run_manifest.get("datasets") or []),
            "requests": int(run_manifest.get("request_count", 0) or 0),
            "completed": updated["request_counts"]["completed"],
            "failed": updated["request_counts"]["failed"],
            "total": updated["request_counts"]["total"],
        }
    ]


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


def _parse_batch_output(output_path: Path) -> Dict[str, Dict[str, Any]]:
    parsed: Dict[str, Dict[str, Any]] = {}
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            custom_id = str(row.get("custom_id", ""))
            if not custom_id:
                continue
            if row.get("error"):
                parsed[custom_id] = {
                    "predicted_match": None,
                    "response_text": "",
                    "error": json.dumps(row["error"]),
                }
                continue
            try:
                content = row["response"]["body"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                parsed[custom_id] = {
                    "predicted_match": None,
                    "response_text": "",
                    "error": "missing_response_content",
                }
                continue
            parsed[custom_id] = {
                "predicted_match": parse_match_from_content(content),
                "response_text": content,
                "error": "",
            }
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


def evaluate_run(run_dir: Path) -> pd.DataFrame:
    run_manifest = _load_json(run_dir / "run_manifest.json")
    datasets = run_manifest.get("datasets") or []
    summary_rows: List[Dict[str, Any]] = []
    root_predictions: Dict[str, Dict[str, Any]] = {}
    root_output_path = run_dir / "batch_output.jsonl"
    if root_output_path.exists():
        root_predictions = _parse_batch_output(root_output_path)

    for dataset in datasets:
        slug = str(dataset["output_slug"])
        dataset_dir = _dataset_dir(run_dir, slug)
        metadata_path = dataset_dir / "metadata.csv"
        metadata = pd.read_csv(metadata_path)
        predictions = root_predictions
        legacy_output_path = dataset_dir / "batch_output.jsonl"
        if not predictions and legacy_output_path.exists():
            predictions = _parse_batch_output(legacy_output_path)

        predicted_matches: List[Optional[bool]] = []
        response_texts: List[str] = []
        errors: List[str] = []
        for custom_id in metadata["custom_id"].astype(str):
            payload = predictions.get(custom_id, {})
            predicted_matches.append(payload.get("predicted_match"))
            response_texts.append(str(payload.get("response_text", "")))
            errors.append(str(payload.get("error", "")))

        eval_df = metadata.copy()
        eval_df["predicted_match"] = predicted_matches
        eval_df["predicted_label"] = eval_df["predicted_match"].map(
            lambda value: "TRUE" if value is True else ("FALSE" if value is False else "")
        )
        eval_df["response_text"] = response_texts
        eval_df["error"] = errors
        eval_df.to_csv(dataset_dir / "predictions.csv", index=False)

        scored_df = eval_df[eval_df["predicted_match"].notna()].copy()
        gold = scored_df["gold_match"].astype(bool).tolist()
        pred = scored_df["predicted_match"].astype(bool).tolist()
        metrics = compute_binary_metrics(gold, pred)
        total_pairs = int(len(eval_df))
        parsed_pairs = int(len(scored_df))
        parse_failures = int(total_pairs - parsed_pairs)
        coverage = (parsed_pairs / total_pairs) if total_pairs else 0.0

        summary_rows.append(
            {
                "benchmark": dataset["benchmark"],
                "dataset_name": dataset["dataset_name"],
                "output_slug": slug,
                "model": dataset["model"],
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
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(run_dir / "summary.csv", index=False)
    _write_json(
        run_dir / "summary.json",
        {
            "run_dir": str(run_dir),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "results": summary_rows,
        },
    )
    return summary_df


def print_status_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No datasets found.")
        return
    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate all configured benchmark test sets with the active-learning prompt."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create batch JSONL files for all configured test sets.")
    prepare.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    prepare.add_argument("--model", default=DEFAULT_MODEL)
    prepare.add_argument("--benchmarks", default="", help="Comma-separated benchmark keys, e.g. abt-buy")

    submit = subparsers.add_parser("submit", help="Upload and submit all prepared batches in a run.")
    submit.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    status = subparsers.add_parser("status", help="Refresh batch status and download completed outputs.")
    status.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    evaluate = subparsers.add_parser("evaluate", help="Score completed batch outputs against gold labels.")
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

    resolved_run_dir = _resolve_output_dir(Path(args.output_dir), create=False)

    if args.command == "submit":
        submit_run(resolved_run_dir)
        print(resolved_run_dir)
        return

    if args.command == "status":
        rows = refresh_status(resolved_run_dir)
        print_status_table(rows)
        return

    if args.command == "evaluate":
        summary_df = evaluate_run(resolved_run_dir)
        print_summary_table(summary_df)
        return

    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
