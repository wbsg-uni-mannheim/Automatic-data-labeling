#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = ROOT / "reports" / "benchmark_model_test_set_comparison.csv"


def _strip_json_gz_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".json.gz"):
        return name[: -len(".json.gz")]
    return path.stem


def _dataset_name_from_path(data_path: str) -> str:
    return _strip_json_gz_suffix(Path(str(data_path)))


def _benchmark_from_path(data_path: str) -> str:
    return Path(str(data_path)).parent.name


def _classify_model_type(model: str) -> str:
    value = str(model).lower()
    open_weight_markers = (
        "gpt-oss",
        "qwen/",
        "llama",
        "mistral",
        "deepseek",
        "gemma",
    )
    closed_source_markers = (
        "gpt-5",
        "minimax/",
        "claude",
        "gemini",
        "o1",
        "o3",
        "o4",
    )
    if any(marker in value for marker in open_weight_markers):
        return "open_weight"
    if any(marker in value for marker in closed_source_markers):
        return "closed_source"
    return "unknown"


def _model_column_prefix(model: str) -> str:
    out = []
    prev_sep = False
    for ch in str(model).lower():
        if ch.isalnum():
            out.append(ch)
            prev_sep = False
            continue
        if not prev_sep:
            out.append("_")
        prev_sep = True
    return "".join(out).strip("_")


def _safe_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return 0
        return int(value)
    text = str(value).strip()
    if not text:
        return 0
    return int(float(text))


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _metrics_from_openrouter_results(pair_count: int, rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    latest_by_pair: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        pair_id = str(row.get("pair_id", "")).strip()
        if not pair_id:
            continue
        latest_by_pair[pair_id] = row

    deduped = list(latest_by_pair.values())
    ok_rows = [row for row in deduped if str(row.get("status", "")).strip().lower() == "ok"]

    tp = tn = fp = fn = 0
    prompt_tokens = completion_tokens = total_tokens = 0
    for row in deduped:
        prompt_tokens += _safe_int(row.get("prompt_tokens"))
        completion_tokens += _safe_int(row.get("completion_tokens"))
        total_tokens += _safe_int(row.get("total_tokens"))

    for row in ok_rows:
        gold = _safe_int(row.get("gold_label"))
        pred = _safe_int(row.get("pred_label"))
        if gold == 1 and pred == 1:
            tp += 1
        elif gold == 0 and pred == 0:
            tn += 1
        elif gold == 0 and pred == 1:
            fp += 1
        else:
            fn += 1

    pairs_scored = len(ok_rows)
    parse_failures = max(int(pair_count) - pairs_scored, 0)
    accuracy = ((tp + tn) / pairs_scored) if pairs_scored else None
    precision = (tp / (tp + fp)) if (tp + fp) else (0.0 if pairs_scored else None)
    recall = (tp / (tp + fn)) if (tp + fn) else (0.0 if pairs_scored else None)
    f1 = (
        (2 * precision * recall / (precision + recall))
        if pairs_scored and precision is not None and recall is not None and (precision + recall)
        else (0.0 if pairs_scored else None)
    )

    return {
        "pair_count": int(pair_count),
        "pairs_scored": int(pairs_scored),
        "parse_failures": int(parse_failures),
        "coverage": (pairs_scored / pair_count) if pair_count else None,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
    }


def _build_dataset_row(
    *,
    model: str,
    benchmark: str,
    dataset_name: str,
    pair_count: int,
    pairs_scored: int,
    parse_failures: int,
    accuracy: Any,
    precision: Any,
    recall: Any,
    f1: Any,
    tp: int,
    tn: int,
    fp: int,
    fn: int,
    run_family: str,
    execution_provider: str,
    source_path: Path,
    prompt_tokens: Any = None,
    completion_tokens: Any = None,
    total_tokens: Any = None,
) -> Dict[str, Any]:
    coverage = (pairs_scored / pair_count) if pair_count else None
    return {
        "row_type": "dataset",
        "model": model,
        "model_type": _classify_model_type(model),
        "run_family": run_family,
        "execution_provider": execution_provider,
        "benchmark": benchmark,
        "dataset_name": dataset_name,
        "dataset_key": f"{benchmark}/{dataset_name}",
        "pair_count": int(pair_count),
        "pairs_scored": int(pairs_scored),
        "coverage": coverage,
        "parse_failures": int(parse_failures),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "status": "complete" if pair_count and pairs_scored == pair_count and parse_failures == 0 else "partial",
        "dataset_count": 1,
        "complete_dataset_count": 1 if pair_count and pairs_scored == pair_count and parse_failures == 0 else 0,
        "macro_accuracy": None,
        "macro_precision": None,
        "macro_recall": None,
        "macro_f1": None,
        "source_path": str(source_path.resolve()),
    }


def _collect_openrouter_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    root = ROOT / "output" / "benchmark_split_openrouter"
    for manifest_path in sorted(root.glob("*/test/*/manifest.json")):
        manifest = _read_json(manifest_path)
        source_path = manifest_path.parent / "summary.json"
        if source_path.exists():
            summary = _read_json(source_path)
            metrics = {
                "pair_count": _safe_int(summary.get("pair_count")),
                "pairs_scored": _safe_int(summary.get("pairs_scored")),
                "parse_failures": _safe_int(summary.get("parse_failures")),
                "coverage": (
                    _safe_int(summary.get("pairs_scored")) / _safe_int(summary.get("pair_count"))
                    if _safe_int(summary.get("pair_count"))
                    else None
                ),
                "accuracy": summary.get("accuracy"),
                "precision": summary.get("precision"),
                "recall": summary.get("recall"),
                "f1": summary.get("f1"),
                "tp": _safe_int(summary.get("tp")),
                "tn": _safe_int(summary.get("tn")),
                "fp": _safe_int(summary.get("fp")),
                "fn": _safe_int(summary.get("fn")),
                "prompt_tokens": _safe_int(summary.get("prompt_tokens")),
                "completion_tokens": _safe_int(summary.get("completion_tokens")),
                "total_tokens": _safe_int(summary.get("total_tokens")),
            }
        else:
            result_path = manifest_path.parent / "results.jsonl"
            metrics = _metrics_from_openrouter_results(
                pair_count=_safe_int(manifest.get("pair_count")),
                rows=_read_jsonl(result_path) if result_path.exists() else [],
            )
            source_path = result_path

        data_path = str(manifest.get("data_path", ""))
        rows.append(
            _build_dataset_row(
                model=str(manifest.get("model", "")),
                benchmark=str(manifest.get("benchmark") or _benchmark_from_path(data_path)),
                dataset_name=_dataset_name_from_path(data_path),
                pair_count=metrics["pair_count"],
                pairs_scored=metrics["pairs_scored"],
                parse_failures=metrics["parse_failures"],
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1=metrics["f1"],
                tp=metrics["tp"],
                tn=metrics["tn"],
                fp=metrics["fp"],
                fn=metrics["fn"],
                prompt_tokens=metrics["prompt_tokens"],
                completion_tokens=metrics["completion_tokens"],
                total_tokens=metrics["total_tokens"],
                run_family="openrouter_direct",
                execution_provider="openrouter",
                source_path=source_path,
            )
        )
    return rows


def _collect_minimax_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    root = ROOT / "output" / "wdc_test_minimax_openrouter"
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = _read_json(manifest_path)
        source_path = manifest_path.parent / "summary.json"
        if source_path.exists():
            summary = _read_json(source_path)
            metrics = {
                "pair_count": _safe_int(summary.get("pair_count")),
                "pairs_scored": _safe_int(summary.get("pairs_scored")),
                "parse_failures": _safe_int(summary.get("parse_failures")),
                "accuracy": summary.get("accuracy"),
                "precision": summary.get("precision"),
                "recall": summary.get("recall"),
                "f1": summary.get("f1"),
                "tp": _safe_int(summary.get("tp")),
                "tn": _safe_int(summary.get("tn")),
                "fp": _safe_int(summary.get("fp")),
                "fn": _safe_int(summary.get("fn")),
                "prompt_tokens": _safe_int(summary.get("prompt_tokens")),
                "completion_tokens": _safe_int(summary.get("completion_tokens")),
                "total_tokens": _safe_int(summary.get("total_tokens")),
            }
        else:
            result_path = manifest_path.parent / "results.jsonl"
            metrics = _metrics_from_openrouter_results(
                pair_count=_safe_int(manifest.get("pair_count")),
                rows=_read_jsonl(result_path) if result_path.exists() else [],
            )
            source_path = result_path

        data_path = str(manifest.get("data_path", ""))
        rows.append(
            _build_dataset_row(
                model=str(manifest.get("model", "")),
                benchmark=_benchmark_from_path(data_path),
                dataset_name=_dataset_name_from_path(data_path),
                pair_count=metrics["pair_count"],
                pairs_scored=metrics["pairs_scored"],
                parse_failures=metrics["parse_failures"],
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1=metrics["f1"],
                tp=metrics["tp"],
                tn=metrics["tn"],
                fp=metrics["fp"],
                fn=metrics["fn"],
                prompt_tokens=metrics["prompt_tokens"],
                completion_tokens=metrics["completion_tokens"],
                total_tokens=metrics["total_tokens"],
                run_family="openrouter_direct",
                execution_provider="openrouter",
                source_path=source_path,
            )
        )
    return rows


def _collect_batch_eval_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for summary_path in sorted((ROOT / "output").glob("batch_eval_*/summary.csv")):
        summary_df = pd.read_csv(summary_path)
        for _, record in summary_df.iterrows():
            rows.append(
                _build_dataset_row(
                    model=str(record.get("model", "")),
                    benchmark=str(record.get("benchmark", "")),
                    dataset_name=str(record.get("dataset_name", "")),
                    pair_count=_safe_int(record.get("pair_count")),
                    pairs_scored=_safe_int(record.get("pairs_scored")),
                    parse_failures=_safe_int(record.get("parse_failures")),
                    accuracy=record.get("accuracy"),
                    precision=record.get("precision"),
                    recall=record.get("recall"),
                    f1=record.get("f1"),
                    tp=_safe_int(record.get("tp")),
                    tn=_safe_int(record.get("tn")),
                    fp=_safe_int(record.get("fp")),
                    fn=_safe_int(record.get("fn")),
                    run_family=summary_path.parent.name,
                    execution_provider="openai_batch",
                    source_path=summary_path,
                )
            )
    return rows


def _pick_best_row(group: pd.DataFrame) -> pd.Series | None:
    scored = group[pd.to_numeric(group["f1"], errors="coerce").notna()].copy()
    if scored.empty:
        return None
    scored["_f1_sort"] = pd.to_numeric(scored["f1"], errors="coerce")
    scored = scored.sort_values(by=["_f1_sort", "coverage", "model"], ascending=[False, False, True])
    return scored.iloc[0]


def _build_dataset_wide_frame(dataset_rows: pd.DataFrame) -> pd.DataFrame:
    model_order = sorted(
        dataset_rows[["model", "model_type"]].drop_duplicates().itertuples(index=False, name=None),
        key=lambda item: (str(item[1]), str(item[0])),
    )

    wide_rows: List[Dict[str, Any]] = []
    grouped = dataset_rows.groupby(["benchmark", "dataset_name", "dataset_key"], dropna=False)
    for (benchmark, dataset_name, dataset_key), group in grouped:
        group = group.sort_values(by=["model"])
        pair_count = int(pd.to_numeric(group["pair_count"], errors="coerce").max())
        row: Dict[str, Any] = {
            "benchmark": benchmark,
            "dataset_name": dataset_name,
            "dataset_key": dataset_key,
            "pair_count": pair_count,
            "models_present": int(len(group)),
            "complete_models_present": int(group["status"].eq("complete").sum()),
        }

        best_any = _pick_best_row(group)
        if best_any is not None:
            row["best_model_by_f1_any"] = best_any["model"]
            row["best_f1_any"] = best_any["f1"]
            row["best_accuracy_any"] = best_any["accuracy"]
            row["best_coverage_any"] = best_any["coverage"]
        else:
            row["best_model_by_f1_any"] = ""
            row["best_f1_any"] = None
            row["best_accuracy_any"] = None
            row["best_coverage_any"] = None

        complete_group = group[group["status"].eq("complete")].copy()
        best_complete = _pick_best_row(complete_group) if not complete_group.empty else None
        if best_complete is not None:
            row["best_model_by_f1_complete"] = best_complete["model"]
            row["best_f1_complete"] = best_complete["f1"]
            row["best_accuracy_complete"] = best_complete["accuracy"]
        else:
            row["best_model_by_f1_complete"] = ""
            row["best_f1_complete"] = None
            row["best_accuracy_complete"] = None

        by_model = {str(rec["model"]): rec for _, rec in group.iterrows()}
        for model, model_type in model_order:
            prefix = _model_column_prefix(model)
            rec = by_model.get(str(model))
            row[f"{prefix}__model"] = model
            row[f"{prefix}__model_type"] = model_type
            if rec is None:
                row[f"{prefix}__f1"] = None
                row[f"{prefix}__accuracy"] = None
                row[f"{prefix}__precision"] = None
                row[f"{prefix}__recall"] = None
                row[f"{prefix}__coverage"] = None
                row[f"{prefix}__status"] = ""
                row[f"{prefix}__pairs_scored"] = None
                row[f"{prefix}__parse_failures"] = None
                continue

            row[f"{prefix}__f1"] = rec["f1"]
            row[f"{prefix}__accuracy"] = rec["accuracy"]
            row[f"{prefix}__precision"] = rec["precision"]
            row[f"{prefix}__recall"] = rec["recall"]
            row[f"{prefix}__coverage"] = rec["coverage"]
            row[f"{prefix}__status"] = rec["status"]
            row[f"{prefix}__pairs_scored"] = rec["pairs_scored"]
            row[f"{prefix}__parse_failures"] = rec["parse_failures"]

        wide_rows.append(row)

    wide_df = pd.DataFrame(wide_rows)
    return wide_df.sort_values(by=["benchmark", "dataset_name"]).reset_index(drop=True)


def _build_dataset_model_long_frame(dataset_rows: pd.DataFrame) -> pd.DataFrame:
    output_rows: List[Dict[str, Any]] = []
    grouped = dataset_rows.groupby(["benchmark", "dataset_name", "dataset_key"], dropna=False)
    for (_, dataset_name, dataset_key), group in grouped:
        best_any = _pick_best_row(group)
        complete_group = group[group["status"].eq("complete")].copy()
        best_complete = _pick_best_row(complete_group) if not complete_group.empty else None

        for _, row in group.iterrows():
            output_rows.append(
                {
                    "benchmark": row["benchmark"],
                    "dataset_name": dataset_name,
                    "dataset_key": dataset_key,
                    "model": row["model"],
                    "model_type": row["model_type"],
                    "run_family": row["run_family"],
                    "execution_provider": row["execution_provider"],
                    "pair_count": row["pair_count"],
                    "pairs_scored": row["pairs_scored"],
                    "coverage": row["coverage"],
                    "parse_failures": row["parse_failures"],
                    "accuracy": row["accuracy"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": row["f1"],
                    "tp": row["tp"],
                    "tn": row["tn"],
                    "fp": row["fp"],
                    "fn": row["fn"],
                    "status": row["status"],
                    "best_model_by_f1_any": best_any["model"] if best_any is not None else "",
                    "best_f1_any": best_any["f1"] if best_any is not None else None,
                    "best_model_by_f1_complete": best_complete["model"] if best_complete is not None else "",
                    "best_f1_complete": best_complete["f1"] if best_complete is not None else None,
                    "source_path": row["source_path"],
                }
            )

    output_df = pd.DataFrame(output_rows)
    output_df["_benchmark"] = output_df["benchmark"].astype(str)
    output_df["_dataset"] = output_df["dataset_name"].astype(str)
    output_df["_f1"] = pd.to_numeric(output_df["f1"], errors="coerce").fillna(-1.0)
    output_df = output_df.sort_values(
        by=["_benchmark", "_dataset", "_f1", "model"],
        ascending=[True, True, False, True],
    ).drop(columns=["_benchmark", "_dataset", "_f1"])
    return output_df.reset_index(drop=True)


def _select_best_row_per_benchmark_model(long_df: pd.DataFrame) -> pd.DataFrame:
    selected_rows: List[pd.Series] = []
    grouped = long_df.groupby(["benchmark", "model"], dropna=False)
    for _, group in grouped:
        ranked = group.copy()
        ranked["_f1"] = pd.to_numeric(ranked["f1"], errors="coerce").fillna(-1.0)
        ranked["_coverage"] = pd.to_numeric(ranked["coverage"], errors="coerce").fillna(-1.0)
        ranked["_pairs_scored"] = pd.to_numeric(ranked["pairs_scored"], errors="coerce").fillna(-1)
        ranked = ranked.sort_values(
            by=["_f1", "_coverage", "_pairs_scored", "dataset_name"],
            ascending=[False, False, False, True],
        )
        selected_rows.append(ranked.iloc[0].drop(labels=["_f1", "_coverage", "_pairs_scored"]))

    out = pd.DataFrame(selected_rows).reset_index(drop=True)
    out["_benchmark"] = out["benchmark"].astype(str)
    out["_f1"] = pd.to_numeric(out["f1"], errors="coerce").fillna(-1.0)
    out = out.sort_values(
        by=["_benchmark", "_f1", "model"],
        ascending=[True, False, True],
    ).drop(columns=["_benchmark", "_f1"])
    return out.reset_index(drop=True)


def build_comparison_frame() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    rows.extend(_collect_openrouter_rows())
    rows.extend(_collect_minimax_rows())
    rows.extend(_collect_batch_eval_rows())

    dataset_df = pd.DataFrame(rows)
    if dataset_df.empty:
        raise FileNotFoundError("No benchmark test-set model results were found under output/")

    long_df = _build_dataset_model_long_frame(dataset_df)
    return _select_best_row_per_benchmark_model(long_df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CSV comparing benchmark test-set performance across model runs.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comparison_df = build_comparison_frame()
    comparison_df.to_csv(output_path, index=False)
    print(output_path)


if __name__ == "__main__":
    main()
