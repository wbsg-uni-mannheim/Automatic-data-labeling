"""
Scan Entity Matching Sets for Data Quality Issues

This script scans entity matching files and reports:
1. Duplicate IDs in datasets
2. IDs in matching files that don't exist in datasets
3. Malformed lines (missing labels, etc.)

Usage:
    python scan_matching_sets.py --data-dir usecases/input/companies/data \
        --matching-dir usecases/input/companies/entitymatching \
        --output-dir scripts/output/companies_scan
"""

import argparse
import html
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from PyDI.io.loaders import load_xml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan entity matching sets for data quality issues.",
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Path to directory containing input XML data files",
    )
    parser.add_argument(
        "--matching-dir", type=str, required=True,
        help="Path to directory containing entity matching CSV files",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Path to output directory for issue reports",
    )
    return parser.parse_args()


def load_datasets(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all XML datasets from the data directory."""
    datasets = {}
    for xml_file in data_dir.glob("*.xml"):
        name = xml_file.stem
        print(f"Loading {name}...")
        df = load_xml(xml_file, nested_handling="aggregate")
        # Normalize IDs
        df["id"] = df["id"].astype(str).apply(html.unescape)
        datasets[name] = df
        print(f"  {len(df)} records, columns: {list(df.columns)}")
    return datasets


def parse_matching_filename(filename: str) -> Optional[Tuple[str, str, str]]:
    """Parse entity matching filename to extract dataset names and split type."""
    match = re.match(r"(.+)_2_(.+)_(train|test|val|all)\.csv", filename)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None


def load_matching_file(filepath: Path) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load an entity matching CSV file.
    Returns (dataframe, list of malformed lines).
    """
    rows = []
    malformed = []

    with open(filepath, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # Try to parse: id1,id2,label
            parts = line.rsplit(",", 1)
            if len(parts) != 2:
                malformed.append(f"Line {line_num}: No comma found - {line[:80]}")
                continue

            label = parts[1].strip().lower()
            if label not in ("true", "false"):
                malformed.append(f"Line {line_num}: Invalid label '{label}' - {line[:80]}")
                continue

            # Split the IDs part
            ids_part = parts[0]
            id_parts = ids_part.split(",", 1)
            if len(id_parts) != 2:
                malformed.append(f"Line {line_num}: Cannot split IDs - {line[:80]}")
                continue

            id1, id2 = id_parts
            rows.append({
                "id1": html.unescape(id1),
                "id2": html.unescape(id2),
                "label": label.upper(),
                "raw_line": line,
            })

    return pd.DataFrame(rows), malformed


def scan_matching_file(
    matching_path: Path,
    datasets: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> Dict:
    """Scan a single matching file for issues."""
    parsed = parse_matching_filename(matching_path.name)
    if not parsed:
        return {"file": matching_path.name, "error": "Could not parse filename"}

    left_name, right_name, split_type = parsed

    result = {
        "file": matching_path.name,
        "left_dataset": left_name,
        "right_dataset": right_name,
        "split": split_type,
    }

    # Check datasets exist
    if left_name not in datasets:
        result["error"] = f"Left dataset '{left_name}' not found"
        return result
    if right_name not in datasets:
        result["error"] = f"Right dataset '{right_name}' not found"
        return result

    df_left = datasets[left_name]
    df_right = datasets[right_name]

    # Load matching file
    matching_df, malformed = load_matching_file(matching_path)
    result["total_lines"] = len(matching_df) + len(malformed)
    result["valid_pairs"] = len(matching_df)
    result["malformed_lines"] = len(malformed)

    if malformed:
        malformed_path = output_dir / f"{matching_path.stem}_malformed.txt"
        with open(malformed_path, "w") as f:
            f.write("\n".join(malformed))
        result["malformed_file"] = str(malformed_path)

    if len(matching_df) == 0:
        result["error"] = "No valid pairs found"
        return result

    # Check for IDs not found
    left_ids = set(df_left["id"].astype(str))
    right_ids = set(df_right["id"].astype(str))

    left_not_found = matching_df[~matching_df["id1"].astype(str).isin(left_ids)]
    right_not_found = matching_df[~matching_df["id2"].astype(str).isin(right_ids)]

    # Combine - pairs where either ID is not found
    not_found_mask = (
        ~matching_df["id1"].astype(str).isin(left_ids) |
        ~matching_df["id2"].astype(str).isin(right_ids)
    )
    not_found = matching_df[not_found_mask]

    result["ids_not_found"] = len(not_found)
    result["left_ids_not_found"] = len(left_not_found)
    result["right_ids_not_found"] = len(right_not_found)

    if len(not_found) > 0:
        not_found_path = output_dir / f"{matching_path.stem}_ids_not_found.csv"
        not_found.to_csv(not_found_path, index=False)
        result["not_found_file"] = str(not_found_path)

    # Count valid pairs (both IDs exist)
    valid_pairs = matching_df[~not_found_mask]
    result["usable_pairs"] = len(valid_pairs)
    result["usable_true"] = (valid_pairs["label"] == "TRUE").sum()
    result["usable_false"] = (valid_pairs["label"] == "FALSE").sum()

    return result


def scan_dataset_duplicates(
    datasets: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> Dict[str, Dict]:
    """Scan all datasets for duplicate IDs."""
    results = {}

    for name, df in datasets.items():
        dups = df[df.duplicated(subset=["id"], keep=False)]
        result = {
            "dataset": name,
            "total_records": len(df),
            "unique_ids": df["id"].nunique(),
            "duplicate_ids": len(dups),
        }

        if len(dups) > 0:
            dup_path = output_dir / f"{name}_duplicate_ids.csv"
            dups.to_csv(dup_path, index=False)
            result["duplicates_file"] = str(dup_path)

            # Show which IDs are duplicated
            dup_id_counts = df["id"].value_counts()
            dup_id_counts = dup_id_counts[dup_id_counts > 1]
            result["duplicate_id_list"] = dup_id_counts.to_dict()

        results[name] = result

    return results


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    matching_dir = Path(args.matching_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Entity Matching Data Quality Scan")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Matching directory: {matching_dir}")
    print(f"Output directory: {output_dir}")

    # Load datasets
    print("\n--- Loading Datasets ---")
    datasets = load_datasets(data_dir)
    print(f"Loaded {len(datasets)} datasets: {list(datasets.keys())}")

    # Scan for duplicate IDs in datasets
    print("\n--- Scanning for Duplicate IDs ---")
    dup_results = scan_dataset_duplicates(datasets, output_dir)
    for name, result in dup_results.items():
        if result["duplicate_ids"] > 0:
            print(f"  {name}: {result['duplicate_ids']} duplicate records")
            if "duplicate_id_list" in result:
                for dup_id, count in list(result["duplicate_id_list"].items())[:5]:
                    print(f"    - {dup_id[:60]}... ({count} occurrences)")
        else:
            print(f"  {name}: No duplicates")

    # Scan matching files
    print("\n--- Scanning Matching Files ---")
    matching_files = sorted(matching_dir.glob("*.csv"))
    print(f"Found {len(matching_files)} matching files")

    scan_results = []
    for matching_path in matching_files:
        print(f"\n{matching_path.name}:")
        result = scan_matching_file(matching_path, datasets, output_dir)
        scan_results.append(result)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue

        print(f"  Total lines: {result['total_lines']}")
        if result.get("malformed_lines", 0) > 0:
            print(f"  Malformed: {result['malformed_lines']} (see {result.get('malformed_file', 'N/A')})")
        if result.get("ids_not_found", 0) > 0:
            print(f"  IDs not found: {result['ids_not_found']} (left: {result['left_ids_not_found']}, right: {result['right_ids_not_found']})")
            print(f"    Saved to: {result.get('not_found_file', 'N/A')}")
        print(f"  Usable pairs: {result['usable_pairs']} (TRUE: {result.get('usable_true', 0)}, FALSE: {result.get('usable_false', 0)})")

    # Save summary
    summary_df = pd.DataFrame(scan_results)
    summary_path = output_dir / "scan_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n--- Summary saved to {summary_path} ---")

    print("\n=== Scan Complete ===")


if __name__ == "__main__":
    main()
