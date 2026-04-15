#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _extract_split(summary: Dict[str, Any]) -> str:
    config = str(summary.get("config", "")).strip()
    if config:
        stem = Path(config).stem
        if "__" in stem:
            return stem.rsplit("__", 1)[-1]
    run_root = str(summary.get("run_root", "")).strip()
    if run_root:
        stem = Path(run_root).name
        parts = stem.split("_")
        known = {
            "small",
            "small_plus20random",
            "medium",
            "medium_plus20random",
            "large",
            "all",
            "all_plus20random",
        }
        for part in parts:
            if part in known:
                return part
    return ""


def _summary_files(input_root: Path) -> List[Path]:
    return sorted(path for path in input_root.rglob("summary.json") if path.is_file())


def _row_from_result(summary_path: Path, summary: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    metrics = dict(result.get("metrics") or {})
    test = dict(metrics.get("test") or {})
    return {
        "benchmark": str(result.get("benchmark") or ""),
        "split": _extract_split(summary),
        "status": str(result.get("status") or ""),
        "train_rows": int(result.get("train_rows") or 0),
        "valid_rows": int(result.get("valid_rows") or 0),
        "test_rows": int(result.get("test_rows") or 0),
        "accuracy": test.get("accuracy"),
        "precision": test.get("precision"),
        "recall": test.get("recall"),
        "f1": test.get("f1"),
        "best_epoch": metrics.get("best_epoch"),
        "best_val_f1": metrics.get("best_val_f1"),
        "best_threshold": metrics.get("best_threshold"),
        "summary_path": str(summary_path),
        "run_root": str(summary.get("run_root") or ""),
        "run_dir": str(result.get("run_dir") or ""),
        "error": str(result.get("error") or ""),
    }


def aggregate(input_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for summary_path in _summary_files(input_root):
        summary = _load_json(summary_path)
        results = summary.get("results")
        if not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, dict):
                continue
            rows.append(_row_from_result(summary_path, summary, result))
    rows.sort(key=lambda row: (str(row["benchmark"]), str(row["split"]), str(row["summary_path"])))
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "benchmark",
        "split",
        "status",
        "train_rows",
        "valid_rows",
        "test_rows",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "best_epoch",
        "best_val_f1",
        "best_threshold",
        "summary_path",
        "run_root",
        "run_dir",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate generated-label training summary.json files into one CSV.")
    parser.add_argument("input_root", help="Root directory containing generated training run folders")
    parser.add_argument("--output-csv", default="", help="Destination CSV path; defaults to <input_root>/aggregated_training_metrics.csv")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    output_csv = Path(args.output_csv) if str(args.output_csv).strip() else (input_root / "aggregated_training_metrics.csv")

    rows = aggregate(input_root)
    write_csv(output_csv, rows)
    print(json.dumps({"input_root": str(input_root), "rows": len(rows), "output_csv": str(output_csv)}, indent=2))


if __name__ == "__main__":
    main()
