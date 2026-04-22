#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import run_benchmark_batch_eval as batch_eval  # noqa: E402


DEFAULT_INPUT_ROOT = ROOT / "generated_labels" / "three_phase_labeling_ditto_only_v2"
DEFAULT_OUTPUT_ROOT = ROOT / "generated_labels" / "three_phase_labeling_ditto_only_v2_relabel_batch_gpt-5-mini_agent_precision"
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_PROMPT_FILE = (
    ROOT / "scripts" / "experiments" / "evidence_first_abstain" / "prompts" / "agent_precision_system_prompt.txt"
)
DEFAULT_MAX_FIELD_LENGTH = 100000
DEFAULT_COMPLETION_WINDOW = "24h"
DEFAULT_ENDPOINT = "/v1/chat/completions"
WDC_SPECIAL_RELATIVE = Path("wdc") / "benchmark_wdc_20260323_202820"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_jsonl_gz(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl_gz(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def _dataset_dirs(input_root: Path) -> List[Path]:
    return sorted(path for path in input_root.iterdir() if path.is_dir() and (path / "profile_manifest.json").exists())


def _dataset_key(path: Path) -> str:
    return path.name


def _source_profile_dir(dataset_dir: Path, profile_name: str) -> Path:
    return dataset_dir / profile_name


def _master_train_path(dataset_dir: Path) -> Path:
    matches = sorted((dataset_dir / "all_plus20random").glob("*train.json.gz"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one train json.gz in {dataset_dir / 'all_plus20random'}")
    return matches[0]


def _active_key(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(row.get("id1", "")),
        str(row.get("id2", "")),
        str(row.get("rid1", "")),
        str(row.get("rid2", "")),
    )


def _active_key_from_series(row: pd.Series) -> Tuple[str, str, str, str]:
    return (
        str(row.get("id1", "")),
        str(row.get("id2", "")),
        str(row.get("rid1", "")),
        str(row.get("rid2", "")),
    )


def _extract_usage(row: Dict[str, Any]) -> Dict[str, int]:
    try:
        usage = row["response"]["body"]["usage"]
    except (KeyError, TypeError):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping at {path}")
    return payload


def _wdc_source_record(
    active_row: Dict[str, Any],
    left_rows: List[Dict[str, Any]],
    right_rows: List[Dict[str, Any]],
    *,
    prompt_fields: Iterable[str],
) -> Dict[str, Any]:
    rid1 = str(active_row.get("rid1", ""))
    rid2 = str(active_row.get("rid2", ""))
    side1, idx1_text = rid1.split(":", 1)
    side2, idx2_text = rid2.split(":", 1)
    if side1 != "L" or side2 != "R":
        raise ValueError(f"Unexpected WDC rid format: {rid1}, {rid2}")
    left_row = left_rows[int(idx1_text)]
    right_row = right_rows[int(idx2_text)]
    record: Dict[str, Any] = {
        "pair_id": f"{active_row.get('id1','')}__{active_row.get('id2','')}__{rid1}__{rid2}",
        "label": 1 if bool(active_row.get("label", False)) else 0,
    }
    for field in prompt_fields:
        record[f"{field}_left"] = left_row.get(field)
        record[f"{field}_right"] = right_row.get(field)
    return record


def _parse_batch_output(output_path: Path) -> Dict[str, Dict[str, Any]]:
    parsed: Dict[str, Dict[str, Any]] = {}
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            custom_id = str(row.get("custom_id", ""))
            if not custom_id:
                continue
            usage = _extract_usage(row)
            if row.get("error"):
                parsed[custom_id] = {
                    "predicted_match": None,
                    "response_text": "",
                    "error": json.dumps(row["error"]),
                    **usage,
                }
                continue
            try:
                content = row["response"]["body"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                parsed[custom_id] = {
                    "predicted_match": None,
                    "response_text": "",
                    "error": "missing_response_content",
                    **usage,
                }
                continue
            parsed[custom_id] = {
                "predicted_match": batch_eval.parse_match_from_content(content),
                "response_text": content,
                "error": "",
                **usage,
            }
    return parsed


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.write_bytes(src.read_bytes())


def prepare_run(
    *,
    input_root: Path,
    output_root: Path,
    model: str,
    prompt_file: Path,
    max_field_length: int,
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    system_prompt = prompt_file.read_text(encoding="utf-8").strip()
    datasets_summary: List[Dict[str, Any]] = []
    total_requests = 0

    for dataset_dir in _dataset_dirs(input_root):
        source_manifest = _load_json(dataset_dir / "profile_manifest.json")
        prompt_fields = list(source_manifest.get("ditto_fields") or ["title", "description", "price"])
        master_active_path = dataset_dir / "all_plus20random" / "active_labels_latest.csv"
        master_final_path = dataset_dir / "all_plus20random" / "labels_final.csv"
        master_train_path = _master_train_path(dataset_dir)

        active_df = pd.read_csv(master_active_path)
        final_df = pd.read_csv(master_final_path)
        train_rows = _read_jsonl_gz(master_train_path)
        if len(active_df) != len(final_df) or len(active_df) != len(train_rows):
            raise ValueError(f"Row count mismatch in {dataset_dir}")

        dataset_key = _dataset_key(dataset_dir)
        dataset_out_dir = output_root / dataset_key
        dataset_out_dir.mkdir(parents=True, exist_ok=True)

        metadata_rows: List[Dict[str, Any]] = []
        requests: List[Dict[str, Any]] = []
        for idx, (active_row, record) in enumerate(zip(active_df.to_dict(orient="records"), train_rows)):
            custom_id = f"{dataset_key}__req_{idx}"
            messages = batch_eval.build_messages(
                record,
                prompt_fields=prompt_fields,
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
                        "messages": messages,
                    },
                }
            )
            metadata_rows.append(
                {
                    "custom_id": custom_id,
                    "pair_index": idx,
                    "pair_id": str(record.get("pair_id", "")),
                    "id1": str(active_row.get("id1", "")),
                    "id2": str(active_row.get("id2", "")),
                    "rid1": str(active_row.get("rid1", "")),
                    "rid2": str(active_row.get("rid2", "")),
                    "original_label": bool(active_row.get("label", False)),
                }
            )

        batch_input_path = dataset_out_dir / "batch_input.jsonl"
        with batch_input_path.open("w", encoding="utf-8") as handle:
            for request in requests:
                handle.write(json.dumps(request, ensure_ascii=False) + "\n")
        pd.DataFrame(metadata_rows).to_csv(dataset_out_dir / "metadata.csv", index=False)

        dataset_manifest = {
            "dataset_key": dataset_key,
            "benchmark": source_manifest.get("benchmark", dataset_key),
            "source_dataset_dir": str(dataset_dir),
            "source_profile_manifest": str(dataset_dir / "profile_manifest.json"),
            "source_master_active_csv": str(master_active_path),
            "source_master_final_csv": str(master_final_path),
            "source_master_train_json_gz": str(master_train_path),
            "pair_count": len(train_rows),
            "model": model,
            "prompt_file": str(prompt_file),
            "system_prompt": system_prompt,
            "prompt_fields": prompt_fields,
            "max_field_length": int(max_field_length),
            "batch_input_path": str(batch_input_path),
        }
        _write_json(dataset_out_dir / "dataset_manifest.json", dataset_manifest)
        _copy_if_exists(dataset_dir / "profile_manifest.json", dataset_out_dir / "source_profile_manifest.json")

        datasets_summary.append(
            {
                "dataset_key": dataset_key,
                "benchmark": source_manifest.get("benchmark", dataset_key),
                "dataset_type": "profile_manifest",
                "request_count": len(requests),
                "source_dataset_dir": str(dataset_dir),
            }
        )
        total_requests += len(requests)

    wdc_special_path = input_root / WDC_SPECIAL_RELATIVE
    if wdc_special_path.exists():
        cfg = _load_yaml(ROOT / "configs" / "labeling" / "benchmarks_active.yaml")
        wdc_cfg = dict((cfg.get("benchmarks") or {}).get("wdc") or {})
        left_csv = ROOT / str(wdc_cfg["left_csv"])
        right_csv = ROOT / str(wdc_cfg["right_csv"])
        prompt_fields = list((wdc_cfg.get("fields") or {}).keys())
        left_rows = pd.read_csv(left_csv).to_dict(orient="records")
        right_rows = pd.read_csv(right_csv).to_dict(orient="records")
        active_df = pd.read_csv(wdc_special_path)
        dataset_key = "wdc"
        dataset_out_dir = output_root / dataset_key
        dataset_out_dir.mkdir(parents=True, exist_ok=True)
        metadata_rows: List[Dict[str, Any]] = []
        requests: List[Dict[str, Any]] = []
        for idx, active_row in enumerate(active_df.to_dict(orient="records")):
            record = _wdc_source_record(active_row, left_rows, right_rows, prompt_fields=prompt_fields)
            custom_id = f"{dataset_key}__req_{idx}"
            messages = batch_eval.build_messages(
                record,
                prompt_fields=prompt_fields,
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
                        "messages": messages,
                    },
                }
            )
            metadata_rows.append(
                {
                    "custom_id": custom_id,
                    "pair_index": idx,
                    "pair_id": str(record.get("pair_id", "")),
                    "id1": str(active_row.get("id1", "")),
                    "id2": str(active_row.get("id2", "")),
                    "rid1": str(active_row.get("rid1", "")),
                    "rid2": str(active_row.get("rid2", "")),
                    "original_label": bool(active_row.get("label", False)),
                }
            )
        batch_input_path = dataset_out_dir / "batch_input.jsonl"
        with batch_input_path.open("w", encoding="utf-8") as handle:
            for request in requests:
                handle.write(json.dumps(request, ensure_ascii=False) + "\n")
        pd.DataFrame(metadata_rows).to_csv(dataset_out_dir / "metadata.csv", index=False)
        dataset_manifest = {
            "dataset_key": dataset_key,
            "benchmark": "wdc",
            "dataset_type": "wdc_flat",
            "source_dataset_file": str(wdc_special_path),
            "pair_count": len(requests),
            "model": model,
            "prompt_file": str(prompt_file),
            "system_prompt": system_prompt,
            "prompt_fields": prompt_fields,
            "max_field_length": int(max_field_length),
            "batch_input_path": str(batch_input_path),
        }
        _write_json(dataset_out_dir / "dataset_manifest.json", dataset_manifest)
        datasets_summary.append(
            {
                "dataset_key": dataset_key,
                "benchmark": "wdc",
                "dataset_type": "wdc_flat",
                "request_count": len(requests),
                "source_dataset_dir": str(wdc_special_path),
            }
        )
        total_requests += len(requests)

    run_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "model": model,
        "prompt_file": str(prompt_file),
        "max_field_length": int(max_field_length),
        "request_count": int(total_requests),
        "datasets": datasets_summary,
        "notes": "Only each dataset's all_plus20random master split is batched. Smaller profiles are rebuilt from the relabeled master by (id1,id2,rid1,rid2).",
    }
    _write_json(output_root / "run_manifest.json", run_manifest)
    return run_manifest


def submit_run(output_root: Path) -> None:
    from openai import OpenAI

    client = OpenAI()
    run_manifest = _load_json(output_root / "run_manifest.json")
    for dataset in run_manifest.get("datasets", []):
        dataset_dir = output_root / str(dataset["dataset_key"])
        batch_info_path = dataset_dir / "batch_info.json"
        if batch_info_path.exists():
            batch_info = _load_json(batch_info_path)
            status = str(batch_info.get("status", "")).lower()
            if status not in {"failed", "expired", "cancelled"}:
                continue
        batch_input_path = dataset_dir / "batch_input.jsonl"
        with batch_input_path.open("rb") as handle:
            file_obj = client.files.create(file=handle, purpose="batch")
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint=DEFAULT_ENDPOINT,
            completion_window=DEFAULT_COMPLETION_WINDOW,
            metadata={
                "dataset_key": str(dataset["dataset_key"]),
                "request_count": str(dataset["request_count"]),
                "relabel_run_root": output_root.name,
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


def refresh_status(output_root: Path) -> List[Dict[str, Any]]:
    from openai import OpenAI

    client = OpenAI()
    run_manifest = _load_json(output_root / "run_manifest.json")
    rows: List[Dict[str, Any]] = []
    for dataset in run_manifest.get("datasets", []):
        dataset_dir = output_root / str(dataset["dataset_key"])
        batch_info_path = dataset_dir / "batch_info.json"
        if not batch_info_path.exists():
            rows.append(
                {
                    "dataset_key": dataset["dataset_key"],
                    "status": "not_submitted",
                    "requests": int(dataset["request_count"]),
                }
            )
            continue
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
        batch_eval._download_file_if_missing(client, updated.get("output_file_id"), dataset_dir / "batch_output.jsonl")
        batch_eval._download_file_if_missing(client, updated.get("error_file_id"), dataset_dir / "batch_error.jsonl")
        rows.append(
            {
                "dataset_key": dataset["dataset_key"],
                "status": updated["status"],
                "requests": int(dataset["request_count"]),
                "completed": updated["request_counts"]["completed"],
                "failed": updated["request_counts"]["failed"],
                "total": updated["request_counts"]["total"],
            }
        )
    return rows


def _materialize_dataset(dataset_dir: Path) -> Dict[str, Any]:
    dataset_manifest = _load_json(dataset_dir / "dataset_manifest.json")
    if dataset_manifest.get("dataset_type") == "wdc_flat":
        source_file = Path(dataset_manifest["source_dataset_file"])
        active_df = pd.read_csv(source_file)
        metadata_df = pd.read_csv(dataset_dir / "metadata.csv")
        batch_output_path = dataset_dir / "batch_output.jsonl"
        if not batch_output_path.exists():
            raise FileNotFoundError(f"Missing batch output: {batch_output_path}")
        predictions = _parse_batch_output(batch_output_path)
        new_labels: List[bool] = []
        parse_failures = 0
        prompt_tokens = completion_tokens = total_tokens = 0
        for _, meta_row in metadata_df.iterrows():
            payload = predictions.get(str(meta_row["custom_id"]), {})
            predicted = payload.get("predicted_match")
            if predicted is None:
                predicted = bool(meta_row["original_label"])
                parse_failures += 1
            new_labels.append(bool(predicted))
            prompt_tokens += int(payload.get("prompt_tokens", 0) or 0)
            completion_tokens += int(payload.get("completion_tokens", 0) or 0)
            total_tokens += int(payload.get("total_tokens", 0) or 0)
        out_df = active_df.copy()
        out_df["label"] = new_labels
        out_df.to_csv(dataset_dir / "active_labels_latest.csv", index=False)
        out_df.to_csv(dataset_dir / "labels_final.csv", index=False)
        summary = {
            "dataset_key": dataset_manifest["dataset_key"],
            "benchmark": "wdc",
            "pair_count": int(len(out_df)),
            "parse_failures": int(parse_failures),
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(total_tokens),
            "output_active_csv": str(dataset_dir / "active_labels_latest.csv"),
            "output_final_csv": str(dataset_dir / "labels_final.csv"),
        }
        _write_json(dataset_dir / "materialize_summary.json", summary)
        return summary

    source_dataset_dir = Path(dataset_manifest["source_dataset_dir"])
    source_profile_manifest = _load_json(source_dataset_dir / "profile_manifest.json")
    master_active_df = pd.read_csv(Path(dataset_manifest["source_master_active_csv"]))
    master_final_df = pd.read_csv(Path(dataset_manifest["source_master_final_csv"]))
    master_train_rows = _read_jsonl_gz(Path(dataset_manifest["source_master_train_json_gz"]))
    metadata_df = pd.read_csv(dataset_dir / "metadata.csv")
    batch_output_path = dataset_dir / "batch_output.jsonl"
    if not batch_output_path.exists():
        raise FileNotFoundError(f"Missing batch output: {batch_output_path}")

    predictions = _parse_batch_output(batch_output_path)
    predicted_labels: List[bool] = []
    parse_failures = 0
    prompt_tokens = completion_tokens = total_tokens = 0
    for _, meta_row in metadata_df.iterrows():
        payload = predictions.get(str(meta_row["custom_id"]), {})
        predicted = payload.get("predicted_match")
        if predicted is None:
            predicted = bool(meta_row["original_label"])
            parse_failures += 1
        predicted_labels.append(bool(predicted))
        prompt_tokens += int(payload.get("prompt_tokens", 0) or 0)
        completion_tokens += int(payload.get("completion_tokens", 0) or 0)
        total_tokens += int(payload.get("total_tokens", 0) or 0)

    relabeled_master_active = master_active_df.copy()
    relabeled_master_final = master_final_df.copy()
    relabeled_master_active["label"] = predicted_labels
    relabeled_master_final["label"] = predicted_labels
    relabeled_master_train: List[Dict[str, Any]] = []
    for row, label in zip(master_train_rows, predicted_labels):
        updated = dict(row)
        updated["label"] = 1 if label else 0
        relabeled_master_train.append(updated)

    out_master_dir = dataset_dir / "all_plus20random"
    out_master_dir.mkdir(parents=True, exist_ok=True)
    train_name = Path(dataset_manifest["source_master_train_json_gz"]).name
    (out_master_dir / "active_labels_latest.csv").write_text(relabeled_master_active.to_csv(index=False), encoding="utf-8")
    (out_master_dir / "labels_final.csv").write_text(relabeled_master_final.to_csv(index=False), encoding="utf-8")
    _write_jsonl_gz(out_master_dir / train_name, relabeled_master_train)

    master_lookup: Dict[Tuple[str, str, str, str], Tuple[int, Dict[str, Any]]] = {}
    for idx, (active_row, train_row) in enumerate(zip(relabeled_master_active.to_dict(orient="records"), relabeled_master_train)):
        master_lookup[_active_key(active_row)] = (idx, train_row)

    updated_manifest = deepcopy(source_profile_manifest)
    updated_manifest["runner_script"] = "scripts/labeling/relabel_three_phase_generated_labels_batch.py"
    updated_manifest["run_dir"] = str(dataset_dir)
    updated_manifest["run_summary_json"] = str(dataset_dir / "materialize_summary.json")
    updated_manifest["source_train_file"] = str(dataset_dir / "labels_final.csv")
    updated_manifest["source_all_examples_file"] = str(dataset_dir / "active_labels_latest.csv")
    updated_manifest["labeling_cost"] = {
        "model": dataset_manifest["model"],
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
        "available": True,
    }

    root_active = relabeled_master_active.copy()
    root_final = relabeled_master_final.copy()
    root_active.to_csv(dataset_dir / "active_labels_latest.csv", index=False)
    root_final.to_csv(dataset_dir / "labels_final.csv", index=False)

    profile_summaries: Dict[str, Any] = {}
    for profile_name in updated_manifest.get("profiles", {}):
        source_profile_dir = source_dataset_dir / profile_name
        source_active = pd.read_csv(source_profile_dir / "active_labels_latest.csv")
        source_final = pd.read_csv(source_profile_dir / "labels_final.csv")
        source_train_path = next(source_profile_dir.glob("*train.json.gz"))

        new_labels: List[bool] = []
        rebuilt_train: List[Dict[str, Any]] = []
        for _, row in source_active.iterrows():
            key = _active_key_from_series(row)
            if key not in master_lookup:
                raise KeyError(f"Profile row missing from relabeled master: {profile_name} {key}")
            _, train_row = master_lookup[key]
            label_bool = bool(train_row.get("label", 0))
            new_labels.append(label_bool)
            rebuilt_train.append(dict(train_row))

        rebuilt_active = source_active.copy()
        rebuilt_final = source_final.copy()
        rebuilt_active["label"] = new_labels
        rebuilt_final["label"] = new_labels

        out_profile_dir = dataset_dir / profile_name
        out_profile_dir.mkdir(parents=True, exist_ok=True)
        rebuilt_active.to_csv(out_profile_dir / "active_labels_latest.csv", index=False)
        rebuilt_final.to_csv(out_profile_dir / "labels_final.csv", index=False)
        _write_jsonl_gz(out_profile_dir / source_train_path.name, rebuilt_train)

        pos = int(sum(1 for value in new_labels if value))
        neg = int(len(new_labels) - pos)
        updated_manifest["profiles"][profile_name]["actual_total"] = int(len(new_labels))
        updated_manifest["profiles"][profile_name]["actual_pos"] = pos
        updated_manifest["profiles"][profile_name]["actual_neg"] = neg
        updated_manifest["profiles"][profile_name]["labels_csv"] = str(out_profile_dir / "active_labels_latest.csv")
        updated_manifest["profiles"][profile_name]["ditto_train_json_gz"] = str(out_profile_dir / source_train_path.name)
        profile_summaries[profile_name] = {
            "row_count": int(len(new_labels)),
            "positive_labels": pos,
            "negative_labels": neg,
            "active_labels_latest": str(out_profile_dir / "active_labels_latest.csv"),
            "labels_final": str(out_profile_dir / "labels_final.csv"),
            "train_json_gz": str(out_profile_dir / source_train_path.name),
        }

    _write_json(dataset_dir / "profile_manifest.json", updated_manifest)
    materialize_summary = {
        "dataset_key": dataset_manifest["dataset_key"],
        "benchmark": dataset_manifest["benchmark"],
        "pair_count": int(len(relabeled_master_train)),
        "parse_failures": int(parse_failures),
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
        "profiles": profile_summaries,
    }
    _write_json(dataset_dir / "materialize_summary.json", materialize_summary)
    return materialize_summary


def materialize_run(output_root: Path) -> Dict[str, Any]:
    run_manifest = _load_json(output_root / "run_manifest.json")
    dataset_summaries: List[Dict[str, Any]] = []
    for dataset in run_manifest.get("datasets", []):
        dataset_summaries.append(_materialize_dataset(output_root / str(dataset["dataset_key"])))
    summary = {
        "output_root": str(output_root),
        "dataset_count": len(dataset_summaries),
        "datasets": dataset_summaries,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(output_root / "materialize_run_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch relabel three-phase generated label profiles from their all_plus20random master split.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    prepare.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    prepare.add_argument("--model", default=DEFAULT_MODEL)
    prepare.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_FILE))
    prepare.add_argument("--max-field-length", type=int, default=DEFAULT_MAX_FIELD_LENGTH)

    submit = subparsers.add_parser("submit")
    submit.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))

    status = subparsers.add_parser("status")
    status.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))

    args = parser.parse_args()
    load_dotenv(ROOT / ".env")

    if args.command == "prepare":
        summary = prepare_run(
            input_root=Path(args.input_root),
            output_root=Path(args.output_root),
            model=args.model,
            prompt_file=Path(args.prompt_file),
            max_field_length=int(args.max_field_length),
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "submit":
        submit_run(Path(args.output_root))
        return

    if args.command == "status":
        print(json.dumps(refresh_status(Path(args.output_root)), indent=2))
        return

    if args.command == "materialize":
        print(json.dumps(materialize_run(Path(args.output_root)), indent=2))
        return


if __name__ == "__main__":
    main()
