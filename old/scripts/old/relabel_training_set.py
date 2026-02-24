"""Relabel labeled set samples using an LLM to check labeling accuracy.

Usage:
    python relabel_training_set.py path/to/labeled_set.csv --sample-size 25
    python relabel_training_set.py path/to/labeled_set.csv --full  # relabel entire set
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI

# Add parent directory to path for PyDI imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from PyDI.entitymatching import LLMBasedMatcher

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relabel labeled set samples with an LLM to check accuracy."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to labeled set CSV file (must have id1, id2, label columns)",
    )
    parser.add_argument(
        "--left-dataset",
        type=str,
        required=True,
        help="Path to left dataset CSV file",
    )
    parser.add_argument(
        "--right-dataset",
        type=str,
        required=True,
        help="Path to right dataset CSV file",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=25,
        help="Number of pairs to sample and relabel (default: 25)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Relabel the entire training set (overrides --sample-size)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for relabeled data (default: auto-generated)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.2",
        help="OpenAI model to use (default: gpt-5.2)",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace the original training set with the relabeled version",
    )
    return parser.parse_args()


def load_labeled_set(file_path: Path) -> pd.DataFrame:
    """Load a labeled set CSV file."""
    df = pd.read_csv(file_path, dtype={"label": str})
    df["label"] = df["label"].str.upper().str.strip()
    return df


def relabel_pairs(
    pairs: pd.DataFrame,
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    chat_model,
    id_column: str = "id",
) -> pd.DataFrame:
    """Relabel pairs using LLM and return results with original labels."""
    matcher = LLMBasedMatcher()

    # Run matching
    results = matcher.match(
        df_left=df_left,
        df_right=df_right,
        candidates=pairs[["id1", "id2"]],
        id_column=id_column,
        chat_model=chat_model,
        generate_explanations=True,
        parse_strictness="skip",
    )

    if results is None or results.empty:
        print("Warning: LLM returned no results")
        return pd.DataFrame()

    # Merge with original labels
    merged = pairs[["id1", "id2", "label"]].merge(
        results[["id1", "id2", "match", "explanation"]],
        on=["id1", "id2"],
        how="left",
    )

    # Convert match bool to label string
    merged["new_label"] = merged["match"].apply(lambda x: "TRUE" if x else "FALSE")

    # Flag disagreements
    merged["disagreement"] = merged["label"] != merged["new_label"]

    return merged


def print_analysis(results: pd.DataFrame, df_left: pd.DataFrame, df_right: pd.DataFrame, id_column: str = "id"):
    """Print analysis of relabeling results."""
    total = len(results)
    disagreements = results["disagreement"].sum()
    agreement_rate = (total - disagreements) / total * 100 if total > 0 else 0

    print("\n" + "=" * 70)
    print("RELABELING ANALYSIS")
    print("=" * 70)
    print(f"\nTotal pairs relabeled: {total}")
    print(f"Agreements: {total - disagreements} ({agreement_rate:.1f}%)")
    print(f"Disagreements: {disagreements} ({100 - agreement_rate:.1f}%)")

    # Breakdown by original label
    print("\n--- By Original Label ---")
    for orig_label in ["TRUE", "FALSE"]:
        subset = results[results["label"] == orig_label]
        if len(subset) > 0:
            n_disagree = subset["disagreement"].sum()
            print(f"  Original {orig_label}: {len(subset)} pairs, {n_disagree} changed ({n_disagree/len(subset)*100:.1f}%)")

    # Show disagreements with details
    if disagreements > 0:
        print("\n--- Disagreements (LLM thinks original label is wrong) ---")
        disagreement_rows = results[results["disagreement"]]

        for i, (_, row) in enumerate(disagreement_rows.iterrows(), 1):
            print(f"\n[{i}] {row['id1']} <-> {row['id2']}")
            print(f"    Original: {row['label']} -> LLM says: {row['new_label']}")

            # Show record details
            left_record = df_left[df_left[id_column] == row["id1"]]
            right_record = df_right[df_right[id_column] == row["id2"]]

            if not left_record.empty:
                left_record = left_record.iloc[0]
                cols = [c for c in df_left.columns if c != id_column][:4]
                attrs = [f"{c}={left_record[c]}" for c in cols if pd.notna(left_record.get(c))]
                print(f"    Left:  {', '.join(attrs)}")

            if not right_record.empty:
                right_record = right_record.iloc[0]
                cols = [c for c in df_right.columns if c != id_column][:4]
                attrs = [f"{c}={right_record[c]}" for c in cols if pd.notna(right_record.get(c))]
                print(f"    Right: {', '.join(attrs)}")

            if pd.notna(row.get("explanation")):
                print(f"    LLM explanation: {row['explanation']}")

    print("\n" + "=" * 70)
    print(f"ESTIMATED ERROR RATE: {100 - agreement_rate:.1f}%")
    print("=" * 70)

    if disagreements > 0:
        print("\nNext steps:")
        print("  - Review the disagreements above")
        print("  - If LLM is mostly correct, run with --full to relabel entire set")
        print("  - Relabeled data will be saved for you to review/use")


def main():
    args = parse_args()

    input_file = Path(args.input_file)
    left_dataset = Path(args.left_dataset)
    right_dataset = Path(args.right_dataset)

    # Validate paths
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    if not left_dataset.exists():
        print(f"Error: Left dataset not found: {left_dataset}")
        sys.exit(1)
    if not right_dataset.exists():
        print(f"Error: Right dataset not found: {right_dataset}")
        sys.exit(1)

    # Load data
    print(f"Loading labeled set: {input_file}")
    labeled_df = load_labeled_set(input_file)

    required_cols = {"id1", "id2", "label"}
    if not required_cols.issubset(labeled_df.columns):
        print(f"Error: Input file must contain columns: {required_cols}")
        print(f"Found: {set(labeled_df.columns)}")
        sys.exit(1)

    print(f"Loading left dataset: {left_dataset}")
    df_left = pd.read_csv(left_dataset)

    print(f"Loading right dataset: {right_dataset}")
    df_right = pd.read_csv(right_dataset)

    # Initialize LLM
    print(f"\nUsing model: {args.model}")
    llm = ChatOpenAI(model=args.model, temperature=0)

    print(f"\n{'=' * 70}")
    print(f"Processing: {input_file.name}")
    print(f"{'=' * 70}")

    print(f"Labeled set size: {len(labeled_df)}")
    n_pos = (labeled_df["label"] == "TRUE").sum()
    n_neg = (labeled_df["label"] == "FALSE").sum()
    print(f"Original distribution: {n_pos} TRUE, {n_neg} FALSE")

    # Sample or use full set
    if args.full:
        sample = labeled_df
        print(f"Relabeling entire set ({len(sample)} pairs)...")
    else:
        # Stratified sample
        sample_size = min(args.sample_size, len(labeled_df))
        pos_sample = labeled_df[labeled_df["label"] == "TRUE"].sample(
            n=min(sample_size // 2, n_pos), random_state=42
        )
        neg_sample = labeled_df[labeled_df["label"] == "FALSE"].sample(
            n=min(sample_size - len(pos_sample), n_neg), random_state=42
        )
        sample = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=42)
        print(f"Sampling {len(sample)} pairs (stratified: {len(pos_sample)} TRUE, {len(neg_sample)} FALSE)...")

    # Relabel with token tracking
    print("Relabeling with LLM...")
    with get_openai_callback() as cb:
        results = relabel_pairs(sample, df_left, df_right, llm)

    if results.empty:
        print("No results returned")
        sys.exit(1)

    # Print analysis
    print_analysis(results, df_left, df_right)

    # Print token usage
    print("\n--- Token Usage ---")
    print(f"  Prompt tokens:     {cb.prompt_tokens:,}")
    print(f"  Completion tokens: {cb.completion_tokens:,}")
    print(f"  Total tokens:      {cb.total_tokens:,}")
    print(f"  Total cost:        ${cb.total_cost:.4f}")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        suffix = "full" if args.full else f"sample{len(sample)}"
        output_path = input_file.parent / f"relabeled_{input_file.stem}_{suffix}.csv"

    results.to_csv(output_path, index=False)
    print(f"\nSaved relabeling results to: {output_path}")

    # Replace original if requested
    if args.replace:
        if not args.full:
            print("\nWarning: --replace requires --full to replace the entire set")
            print("Skipping replacement (only a sample was relabeled)")
        else:
            import shutil

            backup_path = input_file.with_suffix(".backup.csv")
            shutil.copy(input_file, backup_path)
            print(f"\nBacked up original to: {backup_path}")

            # Write new labeled set with new_label as the label
            new_labeled = results[["id1", "id2", "new_label"]].copy()
            new_labeled = new_labeled.rename(columns={"new_label": "label"})
            new_labeled.to_csv(input_file, index=False)
            print(f"Replaced labeled set: {input_file}")

            # Summary of changes
            n_changed = results["disagreement"].sum()
            print(f"  {n_changed} labels changed out of {len(results)} pairs")


if __name__ == "__main__":
    main()
