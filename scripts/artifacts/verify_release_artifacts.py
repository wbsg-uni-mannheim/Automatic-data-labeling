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
DEFAULT_MANIFEST = ROOT / "paper_artifacts" / "EXPECTED_TRAINING_SETS.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that the paper artifact training files listed in "
            "paper_artifacts/EXPECTED_TRAINING_SETS.csv are present and readable."
        )
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="CSV manifest to verify. Default: %(default)s",
    )
    parser.add_argument(
        "--tier",
        choices=("core", "optional", "all"),
        default="core",
        help="Which manifest tier to verify. Default: %(default)s",
    )
    parser.add_argument(
        "--layout",
        choices=("release", "source"),
        default="release",
        help="Check release_path or source_path from the manifest. Default: %(default)s",
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


def load_rows(path: Path, tier: str) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if tier == "all":
        return rows
    return [row for row in rows if row.get("tier") == tier]


def validate_gzip(path: Path) -> str | None:
    try:
        with gzip.open(path, "rb") as handle:
            while handle.read(1024 * 1024):
                pass
    except Exception as exc:  # noqa: BLE001 - the report should capture any gzip failure.
        return f"{type(exc).__name__}: {exc}"
    return None


def verify(rows: list[dict[str, str]], *, layout: str, check_gzip: bool) -> dict[str, Any]:
    path_column = f"{layout}_path"
    results: list[dict[str, Any]] = []
    missing = 0
    corrupt = 0
    present = 0

    for row in rows:
        rel_path = row.get(path_column, "").strip()
        path = ROOT / rel_path
        status = "present"
        detail = ""
        size_bytes = 0

        if not rel_path:
            status = "invalid_manifest_row"
            detail = f"Missing {path_column}"
            corrupt += 1
        elif not path.exists():
            status = "missing"
            detail = "file not found"
            missing += 1
        else:
            present += 1
            size_bytes = path.stat().st_size
            if check_gzip and path.suffix == ".gz":
                error = validate_gzip(path)
                if error:
                    status = "corrupt"
                    detail = error
                    corrupt += 1

        results.append(
            {
                "tier": row.get("tier", ""),
                "scenario": row.get("scenario", ""),
                "method": row.get("method", ""),
                "teacher": row.get("teacher", ""),
                "benchmark": row.get("benchmark", ""),
                "profile": row.get("profile", ""),
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
            "{status}\t{scenario}\t{method}\t{teacher}\t{benchmark}\t{profile}\t{path}\t{detail}".format(
                **row
            )
        )


def main() -> int:
    args = parse_args()
    rows = load_rows(Path(args.manifest).expanduser().resolve(), args.tier)
    summary = verify(rows, layout=args.layout, check_gzip=not args.no_gzip_check)
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print_text_report(summary)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
