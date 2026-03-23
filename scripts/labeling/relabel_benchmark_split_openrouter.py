#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from third_party.ditto_modern.data import load_wdc_json_gz, serialize_entity

CONFIG_PATH = ROOT / "configs" / "labeling" / "benchmarks_active.yaml"
DEFAULT_BENCHMARKS = "wdc"
DEFAULT_SPLIT = "test"
DEFAULT_MODEL = "openai/gpt-oss-20b:nitro"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_FIELD_LENGTH = 200
CHECKPOINT_EVERY = 250
SYSTEM_PROMPT = (
    "You are an expert entity matcher. Decide if two records refer to the same real-world entity. Return only valid JSON with exactly one field: {\"match\": true|false}."
)

load_dotenv(ROOT / ".env")

_thread_local = threading.local()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping at {path}")
    return payload


def _load_benchmark_specs() -> Dict[str, Dict[str, Any]]:
    config = _load_yaml(CONFIG_PATH)
    benchmarks = config.get("benchmarks") or {}
    if not isinstance(benchmarks, dict) or not benchmarks:
        raise ValueError(f"No benchmarks found in {CONFIG_PATH}")
    out: Dict[str, Dict[str, Any]] = {}
    for name, spec in benchmarks.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Benchmark '{name}' config must be a mapping")
        out[str(name)] = spec
    return out


def _parse_benchmark_arg(raw: str, available: Dict[str, Dict[str, Any]]) -> List[str]:
    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not parts:
        raise ValueError("No benchmarks requested")
    if len(parts) == 1 and parts[0].lower() == "all":
        return sorted(available.keys())
    unknown = [part for part in parts if part not in available]
    if unknown:
        raise ValueError(f"Unknown benchmarks: {', '.join(unknown)}")
    return parts


def _fields_from_spec(benchmark: str, spec: Dict[str, Any]) -> List[str]:
    fields = spec.get("fields") or {}
    if not isinstance(fields, dict) or not fields:
        raise ValueError(f"Benchmark '{benchmark}' has no fields in {CONFIG_PATH}")
    return [str(field).strip() for field in fields.keys() if str(field).strip()]


def _path_for_split(benchmark: str, spec: Dict[str, Any], split: str) -> Path:
    key = f"{split}_path"
    raw = spec.get(key)
    if not raw:
        raise ValueError(f"Benchmark '{benchmark}' has no '{key}' in {CONFIG_PATH}")
    path = ROOT / str(raw)
    if not path.exists():
        raise FileNotFoundError(f"Configured path does not exist for {benchmark}/{split}: {path}")
    return path


def _load_pair_df(path: Path) -> pd.DataFrame:
    if path.name.endswith(".json.gz"):
        return load_wdc_json_gz(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "label" not in df.columns:
            raise ValueError(f"Missing label column in {path}")
        if "pair_id" not in df.columns:
            left_ids = df["id_left"].astype(str) if "id_left" in df.columns else pd.Series([f"left-{i}" for i in range(len(df))])
            right_ids = df["id_right"].astype(str) if "id_right" in df.columns else pd.Series([f"right-{i}" for i in range(len(df))])
            df["pair_id"] = left_ids + "#" + right_ids
        if "is_hard_negative" not in df.columns:
            df["is_hard_negative"] = 0
        return df
    raise ValueError(f"Unsupported dataset format: {path}")


def _model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def _benchmark_output_dir(benchmark: str, split: str, model: str) -> Path:
    return ROOT / "output" / "benchmark_split_openrouter" / benchmark / split / _model_slug(model)


def _run_output_dir(split: str, model: str) -> Path:
    return ROOT / "output" / "benchmark_split_openrouter" / "_runs" / split / _model_slug(model)


def _get_client() -> OpenAI:
    client = getattr(_thread_local, "client", None)
    if client is None:
        api_key = os.getenv("OPEN_ROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPEN_ROUTER_API_KEY in environment or .env")
        client = OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            default_headers={"X-Title": "automatic-data-labeling-benchmark-split"},
        )
        _thread_local.client = client
    return client


def _json_from_text(text: str) -> Dict[str, Any]:
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
        raise ValueError("No JSON object found in model response")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Model response JSON is not an object")
    return payload


def _coerce_match(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(bool(int(value)))
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return 1
    if text in {"false", "0", "no", "n"}:
        return 0
    raise ValueError(f"Unsupported match value: {value!r}")


def _extract_usage(response: Any) -> Dict[str, Optional[int]]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _build_messages(record: Dict[str, Any], fields: List[str]) -> List[Dict[str, str]]:
    left = serialize_entity(record, "left", fields, MAX_FIELD_LENGTH)
    right = serialize_entity(record, "right", fields, MAX_FIELD_LENGTH)
    user_prompt = (
        "Do the two serialized records refer to the same real-world entity? "
        f"Entity 1: {left}\n"
        f"Entity 2: {right}"
    )
    
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _call_model(record: Dict[str, Any], fields: List[str], model: str, max_retries: int = 3) -> Dict[str, Any]:
    client = _get_client()
    messages = _build_messages(record, fields)
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=messages,
            )
            content = response.choices[0].message.content or ""
            payload = _json_from_text(content)
            return {
                "pred_label": _coerce_match(payload.get("match")),
                "raw_response": content,
                **_extract_usage(response),
            }
        except Exception as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Model call failed after retries: {last_error}")


def _compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    ok = df[df["status"] == "ok"].copy()
    prompt_tokens = int(pd.to_numeric(df.get("prompt_tokens"), errors="coerce").fillna(0).sum())
    completion_tokens = int(pd.to_numeric(df.get("completion_tokens"), errors="coerce").fillna(0).sum())
    total_tokens = int(pd.to_numeric(df.get("total_tokens"), errors="coerce").fillna(0).sum())
    if ok.empty:
        return {
            "pair_count": int(len(df)),
            "pairs_scored": 0,
            "parse_failures": int((df["status"] != "ok").sum()),
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
    y_true = ok["gold_label"].astype(int)
    y_pred = ok["pred_label"].astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    accuracy = float((tp + tn) / len(ok)) if len(ok) else None
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0
    return {
        "pair_count": int(len(df)),
        "pairs_scored": int(len(ok)),
        "parse_failures": int((df["status"] != "ok").sum()),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _checkpoint_metrics(results_jsonl: Path) -> Optional[Dict[str, Any]]:
    if not results_jsonl.exists():
        return None
    rows = [json.loads(line) for line in results_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return None
    return _compute_metrics(pd.DataFrame(rows))


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _run_single_benchmark(
    benchmark: str,
    spec: Dict[str, Any],
    *,
    split: str,
    model: str,
    max_workers: int,
    limit: Optional[int],
) -> Dict[str, Any]:
    fields = _fields_from_spec(benchmark, spec)
    data_path = _path_for_split(benchmark, spec, split)
    output_dir = _benchmark_output_dir(benchmark, split, model)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_jsonl = output_dir / "results.jsonl"
    predictions_csv = output_dir / "predictions.csv.gz"
    predictions_xlsx = output_dir / "predictions.xlsx"
    summary_json = output_dir / "summary.json"
    manifest_json = output_dir / "manifest.json"

    df = _load_pair_df(data_path)
    if limit is not None:
        df = df.head(limit).copy()
    records = df.to_dict(orient="records")

    manifest_json.write_text(
        json.dumps(
            {
                "benchmark": benchmark,
                "split": split,
                "model": model,
                "config_path": str(CONFIG_PATH),
                "data_path": str(data_path),
                "fields": fields,
                "pair_count": len(records),
                "max_workers": max_workers,
                "max_field_length": MAX_FIELD_LENGTH,
                "prompt_style": "Ditto serialization via third_party.ditto_modern.data.serialize_entity",
                "system_prompt": SYSTEM_PROMPT,
                "token_accounting": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    existing: Dict[str, Dict[str, Any]] = {}
    if results_jsonl.exists():
        for line in results_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            existing[str(payload["pair_id"])] = payload

    pending = [record for record in records if str(record.get("pair_id", "")) not in existing]
    next_checkpoint = ((len(records) - len(pending)) // CHECKPOINT_EVERY + 1) * CHECKPOINT_EVERY

    with tqdm(total=len(records), initial=len(records) - len(pending), desc=f"Labeling {benchmark} {split}", unit="pair") as progress:
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            futures = {
                executor.submit(_call_model, record, fields, model): record
                for record in pending
            }
            for future in as_completed(futures):
                record = futures[future]
                pair_id = str(record.get("pair_id", ""))
                try:
                    payload = future.result()
                    result_row = {
                        "pair_id": pair_id,
                        "gold_label": int(record.get("label", 0)),
                        "pred_label": int(payload["pred_label"]),
                        "status": "ok",
                        "prompt_tokens": payload.get("prompt_tokens"),
                        "completion_tokens": payload.get("completion_tokens"),
                        "total_tokens": payload.get("total_tokens"),
                        "raw_response": payload.get("raw_response", ""),
                    }
                except Exception as exc:
                    result_row = {
                        "pair_id": pair_id,
                        "gold_label": int(record.get("label", 0)),
                        "pred_label": None,
                        "status": "error",
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "total_tokens": None,
                        "raw_response": str(exc),
                    }
                with results_jsonl.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result_row, ensure_ascii=False) + "\n")
                progress.update(1)

                completed = progress.n
                if completed >= next_checkpoint or completed == len(records):
                    metrics = _checkpoint_metrics(results_jsonl)
                    if metrics is not None:
                        tqdm.write(
                            f"[{benchmark}/{split}] scored={metrics['pairs_scored']} "
                            f"failures={metrics['parse_failures']} "
                            f"f1={_fmt_metric(metrics['f1'])} "
                            f"precision={_fmt_metric(metrics['precision'])} "
                            f"recall={_fmt_metric(metrics['recall'])} "
                            f"tokens={metrics['total_tokens']}"
                        )
                        progress.set_postfix(
                            f1=_fmt_metric(metrics["f1"]),
                            scored=metrics["pairs_scored"],
                        )
                    next_checkpoint += CHECKPOINT_EVERY

    rows = [json.loads(line) for line in results_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    results_df = pd.DataFrame(rows).drop_duplicates(subset=["pair_id"], keep="last")

    base_df = df.copy()
    base_df["pair_id"] = base_df["pair_id"].astype(str)
    keep_cols = ["pair_id", "label"] + [f"{field}_{side}" for field in fields for side in ("left", "right")]
    keep_cols = [col for col in keep_cols if col in base_df.columns]
    merged = base_df[keep_cols].rename(columns={"label": "gold_label"}).merge(results_df, on=["pair_id", "gold_label"], how="left")
    merged["is_correct"] = merged["status"].eq("ok") & (merged["gold_label"] == merged["pred_label"])

    merged.to_csv(predictions_csv, index=False, compression="gzip")
    with pd.ExcelWriter(predictions_xlsx) as writer:
        merged.to_excel(writer, sheet_name="predictions", index=False)

    summary = _compute_metrics(merged)
    summary.update(
        {
            "benchmark": benchmark,
            "split": split,
            "model": model,
            "config_path": str(CONFIG_PATH),
            "data_path": str(data_path),
            "fields": fields,
            "checkpoint_every": CHECKPOINT_EVERY,
            "output_dir": str(output_dir),
        }
    )
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Relabel benchmark valid/test splits with an OpenRouter model.")
    parser.add_argument("--benchmarks", default=DEFAULT_BENCHMARKS)
    parser.add_argument("--split", choices=["valid", "test"], default=DEFAULT_SPLIT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    specs = _load_benchmark_specs()
    selected = _parse_benchmark_arg(args.benchmarks, specs)
    summaries: List[Dict[str, Any]] = []
    for benchmark in selected:
        summaries.append(
            _run_single_benchmark(
                benchmark,
                specs[benchmark],
                split=args.split,
                model=args.model,
                max_workers=args.max_workers,
                limit=args.limit,
            )
        )

    run_dir = _run_output_dir(args.split, args.model)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = run_dir / "run_summary.csv"
    summary_json = run_dir / "run_summary.json"
    pd.DataFrame(summaries).to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Wrote run summary to {summary_csv}")


if __name__ == "__main__":
    main()
