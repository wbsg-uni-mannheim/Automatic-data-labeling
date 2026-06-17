#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "paper_artifacts" / "training_data" / "MANIFEST.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that the materialized artifact training files listed in "
            "paper_artifacts/training_data/MANIFEST.csv are present, readable, and count-consistent."
        )
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="CSV manifest to verify. Default: %(default)s",
    )
    parser.add_argument(
        "--no-gzip-check",
        action="store_true",
        help="Only check file presence and skip gzip stream validation.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON summary instead of the human-readable report.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl_gzip_count(path: Path) -> tuple[int, str | None]:
    count = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
    except Exception as exc:  # noqa: BLE001 - the report should capture any gzip failure.
        return count, f"{type(exc).__name__}: {exc}"
    return count, None


def verify(rows: list[dict[str, str]], *, check_gzip: bool) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    missing = 0
    corrupt = 0
    present = 0

    for row in rows:
        rel_path = row.get("path", "").strip()
        path = ROOT / rel_path
        status = "present"
        detail = ""
        size_bytes = 0
        observed_rows: int | None = None

        if not rel_path:
            status = "invalid_manifest_row"
            detail = "Missing path"
            corrupt += 1
        elif not path.exists():
            status = "missing"
            detail = "file not found"
            missing += 1
        else:
            present += 1
            size_bytes = path.stat().st_size
            if check_gzip and path.suffix == ".gz":
                observed_rows, error = read_jsonl_gzip_count(path)
                if error:
                    status = "corrupt"
                    detail = error
                    corrupt += 1
                else:
                    expected_rows_raw = row.get("n_pairs", "").strip()
                    if expected_rows_raw:
                        expected_rows = int(expected_rows_raw)
                        if observed_rows != expected_rows:
                            status = "row_count_mismatch"
                            detail = f"expected {expected_rows}, observed {observed_rows}"
                            corrupt += 1

        results.append(
            {
                "benchmark": row.get("benchmark", ""),
                "labeler": row.get("labeler", ""),
                "method": row.get("method", ""),
                "expected_rows": row.get("n_pairs", ""),
                "observed_rows": observed_rows,
                "path": rel_path,
                "status": status,
                "detail": detail,
                "size_bytes": size_bytes,
            }
        )

    return {
        "checked": len(rows),
        "present": present,
        "missing": missing,
        "corrupt": corrupt,
        "ok": missing == 0 and corrupt == 0,
        "results": results,
    }


def print_text_report(summary: dict[str, Any]) -> None:
    print(
        "checked={checked} present={present} missing={missing} corrupt={corrupt}".format(
            **summary
        )
    )
    problems = [row for row in summary["results"] if row["status"] != "present"]
    if not problems:
        print("All expected artifact files are present and readable.")
        return

    print()
    print("Problems:")
    for row in problems:
        print(
            "{status}\t{benchmark}\t{labeler}\t{method}\t{path}\t{detail}".format(
                **row
            )
        )


def main() -> int:
    args = parse_args()
    rows = load_rows(Path(args.manifest).expanduser().resolve())
    summary = verify(rows, check_gzip=not args.no_gzip_check)
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print_text_report(summary)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
