"""Re-evaluate error analysis files using a different LLM model.

Usage:
    python scripts/relabel_error_analysis.py ksenia/error_analysis_products_80cc20_100un_non-matches_gpt-5-mini_DE.csv --model gpt-5.2
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate entity pairs from error analysis file with a different LLM."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to error analysis CSV file (must have Entity1, Entity2, Label columns)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.2",
        help="OpenAI model to use (default: gpt-5.2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results (default: auto-generated)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for LLM calls (default: 10)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only process first N rows (for testing)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of times to evaluate each pair (default: 1). Use 3+ for consistency analysis.",
    )
    return parser.parse_args()


# PyDI default system prompt with explanations enabled
# Note: curly braces must be escaped as {{ }} for LangChain templates
SYSTEM_PROMPT = """You are an expert product matcher. Your task is to decide if two product records refer to the EXACT same product (same GTIN/SKU).

Analyze the provided records carefully and return your decision as strict JSON in this format:
{{"match": true|false, "explanation": "<brief explanation>"}}

CRITICAL: Product variants are NOT matches. Different sizes, colors, configurations, or package quantities are DIFFERENT products with different GTINs.

Guidelines:
- match: true ONLY if records refer to the exact same product that would have the same GTIN/barcode
- match: false if they are variants of the same product line (different size, color, capacity, etc.)
- Consider: exact model numbers, dimensions, capacity, color, and configuration
- explanation should be concise (1-2 sentences)
- Respond with ONLY the JSON object and nothing else."""


def extract_json_from_response(response_text: str) -> dict | None:
    """Extract JSON object from response text, handling extra text."""
    start = response_text.find('{')
    if start == -1:
        return None

    brace_count = 0
    for i, char in enumerate(response_text[start:], start):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                try:
                    return json.loads(response_text[start:i+1])
                except json.JSONDecodeError:
                    return None
    return None


def evaluate_pair(llm, entity1: str, entity2: str) -> tuple[bool, str, str]:
    """Evaluate a single pair and return (is_match, explanation, raw_response)."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Left record: {left_record}\nRight record: {right_record}\n\nReturn JSON matching the schema described above."),
    ])

    chain = prompt | llm
    response = chain.invoke({"left_record": entity1, "right_record": entity2})
    raw = response.content.strip()

    # Parse JSON response
    parsed = extract_json_from_response(raw)
    if parsed:
        is_match = bool(parsed.get("match", False))
        explanation = parsed.get("explanation", "")
    else:
        # Fallback: try to detect yes/no in raw response
        raw_lower = raw.lower()
        is_match = "true" in raw_lower or '"match": true' in raw_lower
        explanation = f"(parse failed) {raw}"

    return is_match, explanation, raw


def main():
    args = parse_args()

    input_file = Path(args.input_file)
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    # Load data
    print(f"Loading: {input_file}")
    df = pd.read_csv(input_file, encoding='utf-8-sig')

    # Check required columns
    required = {"Entity1", "Entity2", "Label"}
    if not required.issubset(df.columns):
        print(f"Error: Missing required columns. Found: {list(df.columns)}")
        print(f"Required: {required}")
        sys.exit(1)

    if args.sample:
        df = df.head(args.sample)
        print(f"Processing first {args.sample} rows (sample mode)")

    print(f"Total pairs: {len(df)}")
    print(f"Ground truth distribution: {df['Label'].value_counts().to_dict()}")

    # Initialize LLM
    print(f"\nUsing model: {args.model}")
    llm = ChatOpenAI(model=args.model, temperature=0)

    num_runs = args.runs
    if num_runs > 1:
        print(f"Running {num_runs} evaluations per pair for consistency analysis")

    # Process pairs
    results = []
    print("\nEvaluating pairs...")

    with get_openai_callback() as cb:
        for idx, row in df.iterrows():
            ground_truth = int(row["Label"])
            pair_id = row.get("Pair_ID", "")

            # Run evaluation multiple times
            run_results = []
            for run_num in range(num_runs):
                try:
                    is_match, explanation, raw_response = evaluate_pair(llm, row["Entity1"], row["Entity2"])
                    run_results.append({
                        "run": run_num + 1,
                        "prediction": 1 if is_match else 0,
                        "explanation": explanation,
                        "raw_response": raw_response,
                        "error": None,
                    })
                except Exception as e:
                    run_results.append({
                        "run": run_num + 1,
                        "prediction": -1,
                        "explanation": "",
                        "raw_response": f"ERROR: {e}",
                        "error": str(e),
                    })

            # Aggregate results across runs
            valid_predictions = [r["prediction"] for r in run_results if r["prediction"] != -1]

            if valid_predictions:
                # Majority vote for final prediction
                match_votes = sum(1 for p in valid_predictions if p == 1)
                no_match_votes = len(valid_predictions) - match_votes
                final_prediction = 1 if match_votes > no_match_votes else 0

                # Check consistency
                all_same = len(set(valid_predictions)) == 1
                consistency = "consistent" if all_same else "inconsistent"

                # Use explanation from first successful run
                first_valid = next((r for r in run_results if r["prediction"] != -1), None)
                explanation = first_valid["explanation"] if first_valid else ""
                raw_response = first_valid["raw_response"] if first_valid else ""
            else:
                final_prediction = -1
                consistency = "error"
                explanation = ""
                raw_response = run_results[0]["raw_response"] if run_results else "ERROR"

            result_entry = {
                "idx": idx,
                "ground_truth": ground_truth,
                "new_prediction": final_prediction,
                "explanation": explanation,
                "raw_response": raw_response,
                "correct": final_prediction == ground_truth,
                "Pair_ID": pair_id,
            }

            # Add per-run details if multiple runs
            if num_runs > 1:
                result_entry["consistency"] = consistency
                result_entry["match_votes"] = sum(1 for p in valid_predictions if p == 1) if valid_predictions else 0
                result_entry["no_match_votes"] = sum(1 for p in valid_predictions if p == 0) if valid_predictions else 0
                result_entry["run_predictions"] = str(valid_predictions)
                # Store all explanations
                all_explanations = [r["explanation"] for r in run_results if r["prediction"] != -1]
                result_entry["all_explanations"] = " | ".join(all_explanations)

            results.append(result_entry)

            if (idx + 1) % 10 == 0:
                correct_so_far = sum(1 for r in results if r["correct"])
                print(f"  Processed {idx + 1}/{len(df)} - Accuracy so far: {correct_so_far}/{len(results)} ({correct_so_far/len(results)*100:.1f}%)")

    # Calculate metrics
    results_df = pd.DataFrame(results)
    valid_results = results_df[results_df["new_prediction"] != -1]

    total = len(valid_results)
    correct = int(valid_results["correct"].sum())
    accuracy = correct / total if total > 0 else 0

    # Confusion matrix - cast to int for cleaner output
    tp = int(((valid_results["ground_truth"] == 1) & (valid_results["new_prediction"] == 1)).sum())
    tn = int(((valid_results["ground_truth"] == 0) & (valid_results["new_prediction"] == 0)).sum())
    fp = int(((valid_results["ground_truth"] == 0) & (valid_results["new_prediction"] == 1)).sum())
    fn = int(((valid_results["ground_truth"] == 1) & (valid_results["new_prediction"] == 0)).sum())

    # Metrics - handle edge cases properly
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n" + "=" * 70)
    print(f"RESULTS ({args.model})")
    print("=" * 70)
    print(f"\nTotal evaluated: {total}")
    print(f"Accuracy: {accuracy:.1%} ({correct}/{total})")
    print(f"\nConfusion Matrix:")
    print(f"  TP (correct matches):     {tp}")
    print(f"  TN (correct non-matches): {tn}")
    print(f"  FP (false matches):       {fp}")
    print(f"  FN (missed matches):      {fn}")
    print(f"\nMetrics (positive class = match):")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")

    # Consistency analysis (if multiple runs)
    if num_runs > 1 and "consistency" in results_df.columns:
        consistent_count = int((results_df["consistency"] == "consistent").sum())
        inconsistent_count = int((results_df["consistency"] == "inconsistent").sum())
        consistency_rate = consistent_count / total if total > 0 else 0

        print(f"\n--- Consistency Analysis ({num_runs} runs per pair) ---")
        print(f"  Consistent pairs:   {consistent_count} ({consistency_rate:.1%})")
        print(f"  Inconsistent pairs: {inconsistent_count} ({1-consistency_rate:.1%})")

        # Show inconsistent pairs
        inconsistent = results_df[results_df["consistency"] == "inconsistent"]
        if len(inconsistent) > 0:
            print(f"\n--- Inconsistent Pairs (LLM gave different answers across runs) ---")
            for _, row in inconsistent.iterrows():
                idx = row["idx"]
                orig_row = df.iloc[idx]
                gt_label = "MATCH" if row["ground_truth"] == 1 else "NO MATCH"
                votes = f"Match: {row['match_votes']}, No Match: {row['no_match_votes']}"
                print(f"\n[{row['Pair_ID']}] Ground truth: {gt_label}")
                print(f"  Votes: {votes}")
                print(f"  Run predictions: {row['run_predictions']}")
                # Show truncated entities
                e1 = str(orig_row["Entity1"])[:120] + "..." if len(str(orig_row["Entity1"])) > 120 else str(orig_row["Entity1"])
                e2 = str(orig_row["Entity2"])[:120] + "..." if len(str(orig_row["Entity2"])) > 120 else str(orig_row["Entity2"])
                print(f"  E1: {e1}")
                print(f"  E2: {e2}")
                print(f"  Explanations: {row['all_explanations'][:300]}...")

    # Token usage
    print(f"\n--- Token Usage ---")
    print(f"  Prompt tokens:     {cb.prompt_tokens:,}")
    print(f"  Completion tokens: {cb.completion_tokens:,}")
    print(f"  Total tokens:      {cb.total_tokens:,}")
    print(f"  Total cost:        ${cb.total_cost:.4f}")

    # Show disagreements with ground truth
    disagreements = valid_results[~valid_results["correct"]]
    if len(disagreements) > 0:
        print(f"\n--- Sample Disagreements with Ground Truth ({len(disagreements)} total) ---")
        for _, row in disagreements.head(10).iterrows():
            idx = row["idx"]
            orig_row = df.iloc[idx]
            gt_label = "MATCH" if row["ground_truth"] == 1 else "NO MATCH"
            pred_label = "MATCH" if row["new_prediction"] == 1 else "NO MATCH"
            print(f"\n[{row['Pair_ID']}]")
            print(f"  Ground truth: {gt_label}, LLM predicted: {pred_label}")
            if num_runs > 1 and "consistency" in row:
                print(f"  Consistency: {row['consistency']} (votes: Match={row.get('match_votes', '?')}, No Match={row.get('no_match_votes', '?')})")
            # Show truncated entities
            e1 = str(orig_row["Entity1"])[:150] + "..." if len(str(orig_row["Entity1"])) > 150 else str(orig_row["Entity1"])
            e2 = str(orig_row["Entity2"])[:150] + "..." if len(str(orig_row["Entity2"])) > 150 else str(orig_row["Entity2"])
            print(f"  E1: {e1}")
            print(f"  E2: {e2}")
            print(f"  Explanation: {row['explanation']}")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_file.parent / f"relabeled_{input_file.stem}_{args.model.replace('.', '-')}.csv"

    # Merge results back with original data
    output_df = df.copy()
    output_df["new_prediction"] = results_df["new_prediction"]
    output_df["new_explanation"] = results_df["explanation"]
    output_df["new_raw_response"] = results_df["raw_response"]
    output_df["new_correct"] = results_df["correct"]

    output_df.to_csv(output_path, index=False)
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
