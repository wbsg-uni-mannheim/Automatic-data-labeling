#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = ROOT / "output"
DEFAULT_DATA_DIR = ROOT / "data"

CSV_ALLOWLIST = {
    "summary.csv",
    "run_summary.csv",
    "metadata.csv",
    "metrics.tsv",
    "overall_results.tsv",
}
CSV_KEYWORDS = ("score", "summary", "metric", "metadata")
METADATA_ALLOWLIST = {
    "latest_run",
    "run_manifest.json",
    "dataset_manifest.json",
    "batch_info.json",
    "summary.json",
    "run_summary.json",
    "metrics.json",
    "config.json",
    "manifest.json",
    "run_state.json",
    "benchmark_report.json",
    "tribunal_agent_stats.json",
    "train_config.yaml",
    "train_config.yml",
    "abt_buy_train_config.yaml",
    "profile_manifest.json",
    "source_profile_manifest.json",
    "materialize_summary.json",
}
EXCLUDED_DIR_NAMES = {
    "__pycache__",
    ".ipynb_checkpoints",
    "checkpoints",
    "checkpoint",
    "splits",
}
EXCLUDED_FILE_NAMES = {
    ".ds_store",
}
ERROR_ANALYSIS_RESULT_FILES = {
    "predictions.csv",
    "predictions.csv.gz",
    "results.jsonl",
    "results.jsonl.gz",
}
ERROR_ANALYSIS_BENCHMARK_GLOB = "*gs.json.gz"
ERROR_ANALYSIS_SUPPORT_FILES = [
    ROOT / "scripts" / "run_error_analysis.py",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy only compact result CSVs and run metadata from an output folder to another directory. "
            "Large artifacts such as checkpoints or jsonl batch dumps are skipped."
        )
    )
    parser.add_argument(
        "destination",
        help="Target directory for the exported files.",
    )
    parser.add_argument(
        "--source-dir",
        default=str(DEFAULT_SOURCE_DIR),
        help="Source directory to scan. Default: %(default)s",
    )
    parser.add_argument(
        "--include-predictions",
        action="store_true",
        help="Also copy predictions.csv / predictions.csv.gz files.",
    )
    parser.add_argument(
        "--include-error-analysis",
        action="store_true",
        help=(
            "Export everything needed by scripts/run_error_analysis.py: predictions/results files, "
            "benchmark *gs.json.gz data files, and the error-analysis script itself."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show which files would be copied.",
    )
    return parser.parse_args()


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def _is_csv_like(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".csv") or name.endswith(".csv.gz") or name.endswith(".tsv")


def _is_excluded(rel_path: Path) -> bool:
    for part in rel_path.parts:
        lowered = part.lower()
        if lowered in EXCLUDED_DIR_NAMES:
            return True
    return rel_path.name.lower() in EXCLUDED_FILE_NAMES


def _should_copy(rel_path: Path, include_predictions: bool, include_error_analysis: bool) -> Tuple[bool, str]:
    if _is_excluded(rel_path):
        return False, ""

    name = rel_path.name.lower()

    if name in {"predictions.csv", "predictions.csv.gz"}:
        if include_predictions or include_error_analysis:
            return True, "predictions"
        return False, ""

    if name in {"results.jsonl", "results.jsonl.gz"}:
        return (include_error_analysis, "results_jsonl" if include_error_analysis else "")

    if _is_csv_like(rel_path):
        if name in CSV_ALLOWLIST or any(keyword in name for keyword in CSV_KEYWORDS):
            return True, "results_csv"
        return False, ""

    if name in METADATA_ALLOWLIST:
        return True, "metadata"

    return False, ""


def _collect_output_files(
    source_dir: Path,
    include_predictions: bool,
    include_error_analysis: bool,
) -> List[Tuple[Path, Path, str]]:
    selected: List[Tuple[Path, Path, str]] = []
    for path in sorted(p for p in source_dir.rglob("*") if p.is_file()):
        rel_path = path.relative_to(source_dir)
        should_copy, category = _should_copy(
            rel_path,
            include_predictions=include_predictions,
            include_error_analysis=include_error_analysis,
        )
        if should_copy:
            selected.append((path, rel_path, category))
    return selected


def _collect_error_analysis_support_files() -> List[Tuple[Path, Path, str]]:
    selected: List[Tuple[Path, Path, str]] = []

    if DEFAULT_DATA_DIR.exists():
        for path in sorted(DEFAULT_DATA_DIR.rglob(ERROR_ANALYSIS_BENCHMARK_GLOB)):
            rel_path = path.relative_to(ROOT)
            selected.append((path, rel_path, "benchmark_data"))

    for path in ERROR_ANALYSIS_SUPPORT_FILES:
        if path.exists():
            rel_path = path.relative_to(ROOT)
            selected.append((path, rel_path, "script"))

    return selected


def _format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{size} B"


def _copy_files(
    destination_dir: Path,
    selected_files: Iterable[Tuple[Path, Path, str]],
    dry_run: bool,
) -> Tuple[int, int]:
    file_count = 0
    total_bytes = 0

    for source_path, rel_path, category in selected_files:
        destination_path = destination_dir / rel_path
        size = source_path.stat().st_size
        total_bytes += size
        file_count += 1
        print(f"[{category}] {rel_path} ({_format_bytes(size)})")
        if dry_run:
            continue
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)

    return file_count, total_bytes


def main() -> None:
    args = _parse_args()
    source_dir = Path(args.source_dir).expanduser().resolve()
    destination_dir = Path(args.destination).expanduser().resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_dir}")
    if source_dir == destination_dir:
        raise ValueError("Source and destination must be different directories.")
    if _is_relative_to(destination_dir, source_dir):
        raise ValueError(
            f"Destination must not be inside the source directory: {destination_dir}"
        )

    selected_files = _collect_output_files(
        source_dir=source_dir,
        include_predictions=args.include_predictions,
        include_error_analysis=args.include_error_analysis,
    )
    if args.include_error_analysis:
        selected_files.extend(_collect_error_analysis_support_files())
        dedup: dict[Path, Tuple[Path, Path, str]] = {}
        for source_path, rel_path, category in selected_files:
            dedup[rel_path] = (source_path, rel_path, category)
        selected_files = [dedup[key] for key in sorted(dedup.keys())]
    if not selected_files:
        print("No matching result files found.")
        return

    if not args.dry_run:
        destination_dir.mkdir(parents=True, exist_ok=True)

    file_count, total_bytes = _copy_files(
        destination_dir=destination_dir,
        selected_files=selected_files,
        dry_run=args.dry_run,
    )

    action = "Would copy" if args.dry_run else "Copied"
    print()
    print(f"{action} {file_count} files.")
    print(f"Total size: {_format_bytes(total_bytes)}")
    print(f"Source: {source_dir}")
    print(f"Destination: {destination_dir}")


if __name__ == "__main__":
    main()
