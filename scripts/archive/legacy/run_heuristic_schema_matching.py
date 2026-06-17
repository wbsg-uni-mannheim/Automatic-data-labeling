"""
Heuristic Schema Matching Script

Runs label-based and instance-based schema matching (no LLM) on the
games, music, and companies use cases, and evaluates accuracy against
ground truth (LLM-based mappings from pipeline runs).

Usage:
    # Run all use cases with all matchers
    python run_heuristic_schema_matching.py

    # Run specific use case(s)
    python run_heuristic_schema_matching.py --use-cases music games

    # Run specific matcher(s)
    python run_heuristic_schema_matching.py --matchers label instance

    # Custom threshold
    python run_heuristic_schema_matching.py --threshold 0.3
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from PyDI.pipeline.run import load_data_file
from PyDI.schemamatching import (
    LabelBasedSchemaMatcher,
    InstanceBasedSchemaMatcher,
    SchemaMapping,
    get_schema_columns,
)

# Base directory for use cases and ground truth
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
BASE_DIR = PROJECT_DIR / "usecases" / "input"
GT_DIR = SCRIPT_DIR / "output"

# Use case definitions with ground truth paths
USE_CASES = {
    "music": {
        "schema": BASE_DIR / "music" / "schemamatching" / "target_schema.json",
        "data": BASE_DIR / "music" / "data",
        "ground_truth": GT_DIR / "music_0302" / "schema_matching" / "mappings",
        "fusion_validation": GT_DIR / "music_0302" / "fusion" / "validation_web" / "fusion_validation_set.csv",
    },
    "games": {
        "schema": BASE_DIR / "games" / "schemamatching" / "target_schema.json",
        "data": BASE_DIR / "games" / "data",
        "ground_truth": GT_DIR / "games_0302" / "schema_matching" / "mappings",
        "fusion_validation": GT_DIR / "games_0302" / "fusion" / "validation_web" / "fusion_validation_set.csv",
    },
    "companies": {
        "schema": BASE_DIR / "companies" / "schemamatching" / "target_schema_flat.json",
        "data": BASE_DIR / "companies" / "data",
        "ground_truth": GT_DIR / "companies_0302" / "schema_matching" / "mappings",
        "fusion_validation": GT_DIR / "companies_0302" / "fusion" / "validation_web" / "fusion_validation_set.csv",
    },
}

# Matcher configurations to test
MATCHER_CONFIGS = {
    # Label-based matchers
    "label_jaccard": {
        "type": "label",
        "params": {"similarity_function": "jaccard", "tokenize": True},
    },
    "label_levenshtein": {
        "type": "label",
        "params": {"similarity_function": "levenshtein", "tokenize": False},
    },
    "label_jaro_winkler": {
        "type": "label",
        "params": {"similarity_function": "jaro_winkler", "tokenize": False},
    },
    "label_cosine": {
        "type": "label",
        "params": {"similarity_function": "cosine", "tokenize": True},
    },
    # Instance-based matchers
    "instance_tfidf_cosine": {
        "type": "instance",
        "params": {"vector_creation_method": "tfidf", "similarity_function": "cosine"},
    },
    "instance_tf_cosine": {
        "type": "instance",
        "params": {"vector_creation_method": "term_frequencies", "similarity_function": "cosine"},
    },
    "instance_binary_jaccard": {
        "type": "instance",
        "params": {"vector_creation_method": "binary_occurrence", "similarity_function": "jaccard"},
    },
}


def setup_logging(output_dir: Path):
    """Configure logging to file and console."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "heuristic_schema_matching.log"

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def build_target_df(target_schema: dict) -> pd.DataFrame:
    """Create an empty target DataFrame from a JSON Schema, matching pipeline behavior."""
    target_columns = list(target_schema.get("properties", {}).keys())
    df_target = pd.DataFrame(columns=target_columns)
    df_target.attrs["dataset_name"] = target_schema.get("title", "target")
    return df_target


def load_fusion_validation_as_target(val_path: Path, schema_name: str = "target") -> Optional[pd.DataFrame]:
    """Load a fusion validation CSV and pivot it into a target DataFrame.

    The fusion validation set has rows (entity_id, attribute, correct_value).
    We pivot so each entity becomes a row and each attribute becomes a column,
    giving instance-based matchers real data to compare against.
    """
    if not val_path.exists():
        return None

    val_df = pd.read_csv(val_path)
    if val_df.empty or "attribute" not in val_df.columns:
        return None

    pivoted = val_df.pivot_table(
        index="entity_id",
        columns="attribute",
        values="correct_value",
        aggfunc="first",
    ).reset_index(drop=True)

    pivoted.attrs["dataset_name"] = schema_name
    return pivoted


def create_matcher(config: dict):
    """Create a matcher instance from a config dict."""
    matcher_type = config["type"]
    params = config["params"]

    if matcher_type == "label":
        return LabelBasedSchemaMatcher(**params)
    elif matcher_type == "instance":
        return InstanceBasedSchemaMatcher(**params)
    else:
        raise ValueError(f"Unknown matcher type: {matcher_type}")


def run_matcher_on_dataset(
    matcher,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    threshold: float,
) -> SchemaMapping:
    """Run a single matcher on a single dataset pair.

    Note: Instance-based matchers compare value distributions across columns.
    When the target is an empty schema DataFrame (no rows), they will find no
    matches.  They are included here for completeness but are most useful when
    both source and target contain actual data (e.g., cross-source matching).
    """
    try:
        return matcher.match(source_df, target_df, threshold=threshold)
    except Exception:
        return pd.DataFrame()


def format_mapping_table(mapping: pd.DataFrame, gt_pairs: Set[Tuple[str, str]]) -> str:
    """Format a mapping DataFrame as a readable string, marking correct/wrong."""
    if mapping.empty:
        return "    (no mappings found)"
    lines = []
    for _, row in mapping.iterrows():
        src = row.get("source_column", "?")
        tgt = row.get("target_column", "?")
        score = row.get("score", 0)
        mark = "OK" if (src, tgt) in gt_pairs else "XX"
        lines.append(f"    [{mark}] {src:30s} -> {tgt:20s}  ({score:.3f})")
    return "\n".join(lines)


def load_ground_truth(gt_dir: Path) -> Dict[str, Set[Tuple[str, str]]]:
    """Load ground truth mappings from a directory of *_mapping.csv files.

    Returns {dataset_name: {(source_col, target_col), ...}}.
    """
    gt = {}
    if not gt_dir.exists():
        return gt
    for csv_path in gt_dir.glob("*_mapping.csv"):
        dataset_name = csv_path.stem.replace("_mapping", "")
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        gt[dataset_name] = set(zip(df["source_column"], df["target_column"]))
    return gt


def evaluate(
    predicted: pd.DataFrame,
    gt_pairs: Set[Tuple[str, str]],
) -> dict:
    """Compute precision, recall, F1 of predicted mapping vs ground truth pairs."""
    if predicted.empty:
        return {
            "tp": 0, "fp": 0, "fn": len(gt_pairs),
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
        }

    pred_pairs = set(zip(predicted["source_column"], predicted["target_column"]))
    tp = len(pred_pairs & gt_pairs)
    fp = len(pred_pairs - gt_pairs)
    fn = len(gt_pairs - pred_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run heuristic (non-LLM) schema matching on use cases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--use-cases", type=str, nargs="*", default=None,
        choices=list(USE_CASES.keys()),
        help="Use cases to test (default: all)",
    )
    parser.add_argument(
        "--matchers", type=str, nargs="*", default=None,
        choices=["label", "instance", "all"],
        help="Matcher families to run: label, instance, or all (default: all)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.3,
        help="Minimum similarity threshold (default: 0.3)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="scripts/output/heuristic_schema_matching",
        help="Output directory (default: scripts/output/heuristic_schema_matching)",
    )
    return parser.parse_args()


def select_configs(matchers_arg: Optional[List[str]]) -> Dict[str, dict]:
    """Filter MATCHER_CONFIGS based on user selection."""
    if matchers_arg is None or "all" in matchers_arg:
        return MATCHER_CONFIGS

    selected = {}
    for name, config in MATCHER_CONFIGS.items():
        if config["type"] in matchers_arg:
            selected[name] = config
    return selected


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir)

    use_cases = args.use_cases or list(USE_CASES.keys())
    configs = select_configs(args.matchers)
    threshold = args.threshold

    logger.info("=" * 70)
    logger.info("HEURISTIC SCHEMA MATCHING")
    logger.info("=" * 70)
    logger.info(f"Use cases:  {use_cases}")
    logger.info(f"Matchers:   {list(configs.keys())}")
    logger.info(f"Threshold:  {threshold}")
    logger.info(f"Output dir: {output_dir}")

    # Collect all results for summary
    all_results = []

    for use_case in use_cases:
        if use_case not in USE_CASES:
            logger.warning(f"Unknown use case: {use_case}, skipping")
            continue

        uc_config = USE_CASES[use_case]
        schema_path = uc_config["schema"]
        data_dir = uc_config["data"]
        gt_dir = uc_config["ground_truth"]

        if not data_dir.exists():
            logger.warning(f"Data directory not found: {data_dir}, skipping")
            continue
        if not schema_path.exists():
            logger.warning(f"Schema not found: {schema_path}, skipping")
            continue

        # Load ground truth
        ground_truth = load_ground_truth(gt_dir)
        if ground_truth:
            logger.info(f"Ground truth loaded for {use_case}: {list(ground_truth.keys())}")
        else:
            logger.warning(f"No ground truth found at {gt_dir}")

        # Load schema
        with open(schema_path) as f:
            target_schema = json.load(f)
        target_df = build_target_df(target_schema)

        # Load fusion validation set as populated target for instance-based matchers
        instance_target_df = None
        fusion_val_path = uc_config.get("fusion_validation")
        if fusion_val_path:
            instance_target_df = load_fusion_validation_as_target(
                fusion_val_path,
                schema_name=target_schema.get("title", "target"),
            )
            if instance_target_df is not None:
                logger.info(f"  Loaded fusion validation as instance target: "
                            f"{len(instance_target_df)} rows, "
                            f"{list(instance_target_df.columns)}")
            else:
                logger.warning(f"  Fusion validation not found at {fusion_val_path}; "
                               f"instance-based matchers will use empty target")

        # Find and load data files
        data_files = sorted(data_dir.glob("*.csv")) + sorted(data_dir.glob("*.xml"))
        if not data_files:
            logger.warning(f"No data files in {data_dir}")
            continue

        logger.info("")
        logger.info("=" * 70)
        logger.info(f"USE CASE: {use_case.upper()}")
        logger.info(f"  Schema: {schema_path.name}")
        logger.info(f"  Target columns: {list(target_df.columns)}")
        logger.info(f"  Data files: {[f.name for f in data_files]}")
        logger.info("=" * 70)

        uc_output_dir = output_dir / use_case
        uc_output_dir.mkdir(parents=True, exist_ok=True)
        (uc_output_dir / "mappings").mkdir(parents=True, exist_ok=True)

        for data_path in data_files:
            dataset_name = data_path.stem

            logger.info(f"\n--- {dataset_name} ---")
            try:
                source_df = load_data_file(data_path)
            except Exception as e:
                logger.error(f"  Failed to load {data_path.name}: {e}")
                continue

            source_cols = get_schema_columns(source_df)
            gt_pairs = ground_truth.get(dataset_name, set())
            logger.info(f"  Source columns: {source_cols}")
            if gt_pairs:
                logger.info(f"  Ground truth:   {len(gt_pairs)} mappings")

            for config_name, config in configs.items():
                logger.info(f"\n  Matcher: {config_name}")
                try:
                    matcher = create_matcher(config)
                    # Use populated target for instance-based matchers
                    effective_target = target_df
                    if config["type"] == "instance" and instance_target_df is not None:
                        effective_target = instance_target_df
                    mapping = run_matcher_on_dataset(
                        matcher, source_df, effective_target, threshold
                    )
                except Exception as e:
                    logger.error(f"    Error: {e}")
                    mapping = pd.DataFrame()

                # Evaluate against ground truth
                metrics = evaluate(mapping, gt_pairs) if gt_pairs else {}

                logger.info(f"    Results ({len(mapping)} mappings):")
                logger.info(format_mapping_table(mapping, gt_pairs))

                if metrics:
                    logger.info(
                        f"    -> P={metrics['precision']:.2f}  "
                        f"R={metrics['recall']:.2f}  "
                        f"F1={metrics['f1']:.2f}  "
                        f"(TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']})"
                    )

                # Save mapping
                mapping_filename = f"{dataset_name}_{config_name}_mapping.csv"
                mapping.to_csv(uc_output_dir / "mappings" / mapping_filename, index=False)

                # Record result
                all_results.append({
                    "use_case": use_case,
                    "dataset": dataset_name,
                    "matcher": config_name,
                    "matcher_type": config["type"],
                    "threshold": threshold,
                    "num_predicted": len(mapping),
                    "num_ground_truth": len(gt_pairs),
                    "tp": metrics.get("tp", 0),
                    "fp": metrics.get("fp", 0),
                    "fn": metrics.get("fn", 0),
                    "precision": metrics.get("precision", 0.0),
                    "recall": metrics.get("recall", 0.0),
                    "f1": metrics.get("f1", 0.0),
                })

    # =========================================================================
    # FINAL ACCURACY REPORT
    # =========================================================================
    if not all_results:
        logger.info("No results to report.")
        return

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "detailed_results.csv", index=False)

    logger.info("")
    logger.info("")
    logger.info("=" * 90)
    logger.info("ACCURACY REPORT  (ground truth = LLM-based mappings)")
    logger.info("=" * 90)

    # --- Per-dataset breakdown ---
    logger.info("")
    logger.info("-" * 90)
    logger.info(f"{'Matcher':<25s} {'Use Case':<12s} {'Dataset':<15s} "
                f"{'Prec':>5s} {'Rec':>5s} {'F1':>5s}  "
                f"{'TP':>3s} {'FP':>3s} {'FN':>3s}")
    logger.info("-" * 90)

    for _, row in results_df.iterrows():
        logger.info(
            f"{row['matcher']:<25s} {row['use_case']:<12s} {row['dataset']:<15s} "
            f"{row['precision']:5.2f} {row['recall']:5.2f} {row['f1']:5.2f}  "
            f"{row['tp']:3d} {row['fp']:3d} {row['fn']:3d}"
        )

    # --- Aggregated per-matcher (macro-average across all datasets) ---
    logger.info("")
    logger.info("-" * 90)
    logger.info("MACRO-AVERAGE PER MATCHER (averaged across all datasets)")
    logger.info("-" * 90)
    logger.info(f"{'Matcher':<25s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}  "
                f"{'TP':>4s} {'FP':>4s} {'FN':>4s}  "
                f"{'#Datasets':>9s}")
    logger.info("-" * 90)

    for config_name in configs:
        subset = results_df[results_df["matcher"] == config_name]
        n = len(subset)
        avg_p = subset["precision"].mean()
        avg_r = subset["recall"].mean()
        avg_f1 = subset["f1"].mean()
        total_tp = subset["tp"].sum()
        total_fp = subset["fp"].sum()
        total_fn = subset["fn"].sum()
        logger.info(
            f"{config_name:<25s} {avg_p:6.2f} {avg_r:6.2f} {avg_f1:6.2f}  "
            f"{total_tp:4d} {total_fp:4d} {total_fn:4d}  "
            f"{n:9d}"
        )

    # --- Micro-average (pooled TP/FP/FN) per matcher ---
    logger.info("")
    logger.info("-" * 90)
    logger.info("MICRO-AVERAGE PER MATCHER (pooled TP/FP/FN)")
    logger.info("-" * 90)
    logger.info(f"{'Matcher':<25s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}")
    logger.info("-" * 90)

    for config_name in configs:
        subset = results_df[results_df["matcher"] == config_name]
        tp = subset["tp"].sum()
        fp = subset["fp"].sum()
        fn = subset["fn"].sum()
        micro_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        micro_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
        logger.info(f"{config_name:<25s} {micro_p:6.2f} {micro_r:6.2f} {micro_f1:6.2f}")

    # --- Per use-case summary (best matcher) ---
    logger.info("")
    logger.info("-" * 90)
    logger.info("BEST MATCHER PER USE CASE (by micro-F1)")
    logger.info("-" * 90)

    for use_case in use_cases:
        uc_rows = results_df[results_df["use_case"] == use_case]
        if uc_rows.empty:
            continue
        best_f1 = -1
        best_name = ""
        for config_name in configs:
            subset = uc_rows[uc_rows["matcher"] == config_name]
            tp = subset["tp"].sum()
            fp = subset["fp"].sum()
            fn = subset["fn"].sum()
            micro_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            micro_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
            if micro_f1 > best_f1:
                best_f1 = micro_f1
                best_name = config_name
                best_p = micro_p
                best_r = micro_r
        logger.info(
            f"  {use_case:<12s}  best={best_name:<25s}  "
            f"P={best_p:.2f}  R={best_r:.2f}  F1={best_f1:.2f}"
        )

    logger.info("")
    logger.info(f"Detailed results saved to {output_dir / 'detailed_results.csv'}")
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
