#!/usr/bin/env python3
"""
Update source entity matching files based on manual decisions from disagreements review.

This script reads a disagreements CSV file (which contains a 'decision' column with the
final human decision) and updates the corresponding source matching file accordingly.
"""

import argparse
import pandas as pd
from pathlib import Path


def update_matching_file(disagreements_path: str, source_path: str, output_path: str = None):
    """
    Update source matching file based on decisions in disagreements file.

    Args:
        disagreements_path: Path to disagreements CSV with 'decision' column
        source_path: Path to source matching file to update
        output_path: Path for output file (defaults to overwriting source)
    """
    # Load disagreements with decisions
    disagreements = pd.read_csv(disagreements_path)

    # Validate required columns
    required_cols = ["id1", "id2", "decision"]
    missing = [col for col in required_cols if col not in disagreements.columns]
    if missing:
        raise ValueError(f"Missing required columns in disagreements file: {missing}")

    # Load source matching file
    source = pd.read_csv(source_path, header=None, names=["id1", "id2", "label"])

    print(f"Loaded {len(disagreements)} disagreements from {disagreements_path}")
    print(f"Loaded {len(source)} pairs from {source_path}")

    # Create lookup dict from disagreements: (id1, id2) -> decision
    decisions = {}
    for _, row in disagreements.iterrows():
        key = (row["id1"], row["id2"])
        decision = row["decision"]
        # Normalize decision to match source file format
        if isinstance(decision, bool):
            decisions[key] = decision
        elif isinstance(decision, str):
            decisions[key] = decision.upper() == "TRUE"
        else:
            decisions[key] = bool(decision)

    # Update source file
    updates = 0
    for idx, row in source.iterrows():
        key = (row["id1"], row["id2"])
        if key in decisions:
            old_label = row["label"]
            new_label = decisions[key]
            if old_label != new_label:
                source.at[idx, "label"] = new_label
                updates += 1
                print(f"  Updated: {row['id1'][:50]}... | {old_label} -> {new_label}")

    print(f"\nTotal updates: {updates}")

    # Save updated file
    if output_path is None:
        output_path = source_path

    source.to_csv(output_path, header=False, index=False)
    print(f"Saved updated file to: {output_path}")

    return updates


def main():
    parser = argparse.ArgumentParser(
        description="Update source matching file based on manual decisions from disagreements review"
    )
    parser.add_argument(
        "disagreements_path",
        help="Path to disagreements CSV file with 'decision' column"
    )
    parser.add_argument(
        "source_path",
        help="Path to source matching file to update"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path (defaults to overwriting source file)",
        default=None
    )

    args = parser.parse_args()

    update_matching_file(args.disagreements_path, args.source_path, args.output)


if __name__ == "__main__":
    main()
