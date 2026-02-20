#!/usr/bin/env python3
"""
Compare fusion validation ground truth across different modes.

This script loads ground truth CSVs from each validation mode (llm, llm_omit, web, web_omit)
and the test set XML, then creates a comparison table to easily spot differences.

Usage:
    python scripts/compare_fusion_validation.py <output_dir>

Example:
    python scripts/compare_fusion_validation.py scripts/output/music_optimized_active_learning
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

# ANSI color codes for terminal output
class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def load_test_set_xml(xml_path: Path) -> pd.DataFrame:
    """Load test set from XML format."""
    if not xml_path.exists():
        return pd.DataFrame()

    tree = ET.parse(xml_path)
    root = tree.getroot()

    records = []
    for entity in root:
        record = {"id": entity.find("id").text if entity.find("id") is not None else None}

        # Get simple attributes
        for child in entity:
            if child.tag == "id":
                continue
            if child.tag == "tracks":
                # Handle nested tracks
                track_names = []
                track_durations = []
                track_positions = []
                for track in child.findall("track"):
                    name_el = track.find("name")
                    dur_el = track.find("duration")
                    pos_el = track.find("position")
                    if name_el is not None:
                        track_names.append(name_el.text)
                    if dur_el is not None:
                        track_durations.append(dur_el.text)
                    if pos_el is not None:
                        track_positions.append(pos_el.text)
                record["tracks_track_name"] = str(track_names)
                record["tracks_track_duration"] = str(track_durations)
                record["tracks_track_position"] = str(track_positions)
            else:
                record[child.tag] = child.text

        records.append(record)

    return pd.DataFrame(records)


def load_ground_truth(csv_path: Path) -> pd.DataFrame:
    """Load ground truth CSV."""
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def extract_entity_ids(source_ids_str: str) -> set:
    """Extract individual IDs from source_ids string."""
    if pd.isna(source_ids_str):
        return set()
    return set(id.strip() for id in source_ids_str.split(","))


def normalize_source_ids(source_ids_str: str) -> str:
    """Normalize source_ids to a canonical form for matching."""
    ids = extract_entity_ids(source_ids_str)
    return "|".join(sorted(ids))


def match_entities(gt_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Match entities between ground truth and test set by ID overlap."""
    matches = {}

    if gt_df.empty or test_df.empty:
        return matches

    for _, gt_row in gt_df.iterrows():
        gt_ids = extract_entity_ids(gt_row.get("source_ids", ""))
        gt_entity_id = gt_row["entity_id"]

        for _, test_row in test_df.iterrows():
            test_id = test_row.get("id", "")
            if test_id in gt_ids:
                matches[gt_entity_id] = test_id
                break

    return matches


def truncate(s: str, max_len: int = 30) -> str:
    """Truncate string with ellipsis."""
    s = str(s) if pd.notna(s) else "-"
    if len(s) > max_len:
        return s[:max_len-3] + "..."
    return s


def values_match(v1, v2) -> bool:
    """Check if two values match (case-insensitive, ignoring UNKNOWN)."""
    s1 = str(v1).strip().lower() if pd.notna(v1) else ""
    s2 = str(v2).strip().lower() if pd.notna(v2) else ""
    if not s1 or not s2 or s1 == "unknown" or s2 == "unknown":
        return True  # Can't compare
    return s1 == s2


def build_entity_index(modes: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Build an index of entities by their normalized source_ids.

    Returns a dict mapping normalized_source_ids -> {
        'source_ids': original source_ids string,
        'modes': {mode_name: row_data}
    }
    """
    index = {}

    for mode_name, df in modes.items():
        if df.empty:
            continue

        for _, row in df.iterrows():
            source_ids = row.get("source_ids", "")
            norm_key = normalize_source_ids(source_ids)

            if norm_key not in index:
                index[norm_key] = {
                    "source_ids": source_ids,
                    "modes": {},
                }
            index[norm_key]["modes"][mode_name] = row.to_dict()

    return index


def print_comparison_table(
    modes: dict[str, pd.DataFrame],
    test_df: pd.DataFrame,
    attributes: list[str],
):
    """Print a nicely formatted comparison table per attribute."""

    # Build index by source_ids (not entity_id)
    entity_index = build_entity_index(modes)

    # Build test set index
    test_index = {}
    if not test_df.empty:
        for _, row in test_df.iterrows():
            test_id = row.get("id", "")
            if test_id:
                test_index[test_id] = row.to_dict()

    mode_names = [m for m, df in modes.items() if not df.empty]

    for attr in attributes:
        print(f"\n{Colors.BOLD}{'='*120}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}ATTRIBUTE: {attr}{Colors.RESET}")
        print(f"{'='*120}")

        # Header
        header = f"{'Source IDs':<40} | {'Test':<25}"
        for mode in mode_names:
            header += f" | {mode:<20}"
        header += " | Status"
        print(header)
        print("-" * 150)

        for norm_key in sorted(entity_index.keys()):
            entity_data = entity_index[norm_key]
            source_ids_set = extract_entity_ids(entity_data["source_ids"])

            # Find matching test entity
            test_val = "-"
            for test_id, test_row in test_index.items():
                if test_id in source_ids_set:
                    # Try both naming conventions
                    for attr_name in [attr, attr.replace("_", "-"), attr.replace("-", "_")]:
                        if attr_name in test_row:
                            test_val = test_row[attr_name]
                            break
                    break

            # Get mode values
            mode_vals = {}
            for mode in mode_names:
                if mode in entity_data["modes"]:
                    mode_vals[mode] = entity_data["modes"][mode].get(attr)
                else:
                    mode_vals[mode] = None

            # Determine status
            non_empty_vals = [v for v in mode_vals.values() if pd.notna(v) and str(v).lower() != "unknown"]
            unique_vals = set(str(v).strip().lower() for v in non_empty_vals)

            if len(unique_vals) == 0:
                status = f"{Colors.YELLOW}NO DATA{Colors.RESET}"
            elif len(unique_vals) == 1:
                # Check against test
                if test_val != "-" and pd.notna(test_val):
                    if values_match(list(non_empty_vals)[0], test_val):
                        status = f"{Colors.GREEN}MATCH{Colors.RESET}"
                    else:
                        status = f"{Colors.RED}DIFF vs TEST{Colors.RESET}"
                else:
                    status = f"{Colors.GREEN}AGREE{Colors.RESET}"
            else:
                status = f"{Colors.RED}MODES DIFFER{Colors.RESET}"

            # Format row - show abbreviated source IDs
            source_ids_short = truncate(entity_data["source_ids"], 40)
            row = f"{source_ids_short:<40} | {truncate(test_val, 25):<25}"
            for mode in mode_names:
                val = mode_vals.get(mode)
                val_str = truncate(val, 20)
                # Color code if different from test
                if test_val != "-" and pd.notna(test_val) and pd.notna(val):
                    if not values_match(val, test_val):
                        val_str = f"{Colors.RED}{val_str}{Colors.RESET}"
                row += f" | {val_str:<20}"
            row += f" | {status}"

            print(row)


def create_wide_csv(
    modes: dict[str, pd.DataFrame],
    test_df: pd.DataFrame,
    attributes: list[str],
    output_path: Path,
):
    """Create a wide-format CSV with all comparisons."""

    # Build index by source_ids
    entity_index = build_entity_index(modes)

    # Build test set index
    test_index = {}
    if not test_df.empty:
        for _, row in test_df.iterrows():
            test_id = row.get("id", "")
            if test_id:
                test_index[test_id] = row.to_dict()

    mode_names = [m for m, df in modes.items() if not df.empty]

    rows = []
    for norm_key in sorted(entity_index.keys()):
        entity_data = entity_index[norm_key]
        source_ids_set = extract_entity_ids(entity_data["source_ids"])

        row = {"source_ids": entity_data["source_ids"]}

        # Find matching test entity
        test_id = ""
        test_row_data = {}
        for tid, trow in test_index.items():
            if tid in source_ids_set:
                test_id = tid
                test_row_data = trow
                break
        row["test_id"] = test_id

        # For each attribute
        for attr in attributes:
            # Test value
            test_val = None
            if test_row_data:
                for attr_name in [attr, attr.replace("_", "-"), attr.replace("-", "_")]:
                    if attr_name in test_row_data:
                        test_val = test_row_data[attr_name]
                        break
            row[f"{attr}_test"] = test_val

            # Mode values
            for mode in mode_names:
                if mode in entity_data["modes"]:
                    row[f"{attr}_{mode}"] = entity_data["modes"][mode].get(attr)
                else:
                    row[f"{attr}_{mode}"] = None

        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_path, index=False)
    return result_df


def main():
    parser = argparse.ArgumentParser(description="Compare fusion validation results across modes")
    parser.add_argument("output_dir", help="Pipeline output directory")
    parser.add_argument("--test-set", help="Path to test set XML (default: auto-detect)")
    parser.add_argument("--out", help="Output CSV path")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    args = parser.parse_args()

    # Disable colors if requested
    if args.no_color:
        Colors.RED = Colors.GREEN = Colors.YELLOW = Colors.BLUE = Colors.CYAN = Colors.RESET = Colors.BOLD = ""

    output_dir = Path(args.output_dir)
    fusion_dir = output_dir / "fusion"

    if not fusion_dir.exists():
        print(f"Error: Fusion directory not found: {fusion_dir}")
        sys.exit(1)

    # Define modes
    mode_dirs = {
        "llm": fusion_dir / "validation_llm",
        "llm_omit": fusion_dir / "validation_llm_omit",
        "web": fusion_dir / "validation_web",
        "web_omit": fusion_dir / "validation_web_omit",
    }

    # Load ground truth from each mode
    print(f"\n{Colors.BOLD}Loading validation results...{Colors.RESET}")
    modes = {}
    for mode_name, mode_dir in mode_dirs.items():
        gt_path = mode_dir / "fusion_ground_truth.csv"
        modes[mode_name] = load_ground_truth(gt_path)
        if not modes[mode_name].empty:
            print(f"  {Colors.GREEN}✓{Colors.RESET} {mode_name}: {len(modes[mode_name])} entities")
        else:
            print(f"  {Colors.YELLOW}✗{Colors.RESET} {mode_name}: not found")

    # Load test set
    if args.test_set:
        test_set_path = Path(args.test_set)
    else:
        # Auto-detect
        if "music" in str(output_dir).lower():
            test_set_path = Path("usecases/input/music/fusion/test_set.xml")
        elif "movies" in str(output_dir).lower():
            test_set_path = Path("usecases/input/movies/fusion/test_set.xml")
        elif "companies" in str(output_dir).lower():
            test_set_path = Path("usecases/input/companies/fusion/test_set.xml")
        else:
            test_set_path = None

    test_df = pd.DataFrame()
    if test_set_path and test_set_path.exists():
        test_df = load_test_set_xml(test_set_path)
        print(f"  {Colors.GREEN}✓{Colors.RESET} test_set: {len(test_df)} entities from {test_set_path}")
    else:
        print(f"  {Colors.YELLOW}✗{Colors.RESET} test_set: not found")

    # Get attributes
    attributes = set()
    for df in modes.values():
        if not df.empty:
            attrs = [c for c in df.columns if c not in ("entity_id", "source_ids")]
            attributes.update(attrs)
    attributes = sorted(attributes)

    print(f"\n{Colors.BOLD}Attributes:{Colors.RESET} {', '.join(attributes)}")

    # Print comparison table
    print_comparison_table(modes, test_df, attributes)

    # Save CSV
    out_path = Path(args.out) if args.out else fusion_dir / "validation_comparison.csv"
    result_df = create_wide_csv(modes, test_df, attributes, out_path)
    print(f"\n{Colors.BOLD}Saved comparison to:{Colors.RESET} {out_path}")

    # Summary statistics
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}SUMMARY - Mode Agreement{Colors.RESET}")
    print(f"{'='*60}")

    mode_names = [m for m, df in modes.items() if not df.empty]
    for attr in attributes:
        cols = [f"{attr}_{m}" for m in mode_names]
        cols = [c for c in cols if c in result_df.columns]

        if len(cols) < 2:
            continue

        # Count agreements/disagreements
        agree = 0
        disagree = 0
        for _, row in result_df.iterrows():
            vals = [str(row[c]).strip().lower() for c in cols if pd.notna(row[c]) and str(row[c]).lower() != "unknown"]
            if len(vals) >= 2:
                if len(set(vals)) == 1:
                    agree += 1
                else:
                    disagree += 1

        total = agree + disagree
        if total > 0:
            pct = 100 * agree / total
            color = Colors.GREEN if pct >= 80 else Colors.YELLOW if pct >= 50 else Colors.RED
            print(f"  {attr}: {color}{agree}/{total} agree ({pct:.0f}%){Colors.RESET}")

    # Test set accuracy (for entities that appear in test set)
    test_col_suffix = "_test"
    print(f"\n{Colors.BOLD}SUMMARY - Test Set Accuracy{Colors.RESET}")
    print(f"{'='*60}")

    for mode in mode_names:
        correct = 0
        total = 0
        for attr in attributes:
            test_col = f"{attr}_test"
            mode_col = f"{attr}_{mode}"
            if test_col not in result_df.columns or mode_col not in result_df.columns:
                continue

            for _, row in result_df.iterrows():
                test_val = row[test_col]
                mode_val = row[mode_col]
                if pd.notna(test_val) and pd.notna(mode_val) and str(mode_val).lower() != "unknown":
                    total += 1
                    if values_match(test_val, mode_val):
                        correct += 1

        if total > 0:
            pct = 100 * correct / total
            color = Colors.GREEN if pct >= 70 else Colors.YELLOW if pct >= 50 else Colors.RED
            print(f"  {mode}: {color}{correct}/{total} correct ({pct:.0f}%){Colors.RESET}")


if __name__ == "__main__":
    main()
