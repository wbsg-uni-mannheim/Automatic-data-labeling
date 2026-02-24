"""
Schema Matching Test Script

This script runs the schema matching step of the pipeline in isolation,
testing across all use cases and header variations.

Usage:
    # Run all combinations (default)
    python test_schema_matching.py

    # Run specific use case(s)
    python test_schema_matching.py --use-cases music games

    # Run specific header type(s)
    python test_schema_matching.py --header-types original challenging

    # Run single specific test
    python test_schema_matching.py --data-dir <path> --schema <path> --output-dir <path>
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from PyDI.pipeline.run import discover_files, load_data_file
from PyDI.pipeline.schema_matching import auto_match_schema
from PyDI.pipeline.normalization import auto_normalize
from PyDI.pipeline.run import _ensure_stable_id

load_dotenv()

# LLM Model
LLM_MODEL = "gpt-5.2"

# Base directory for use cases
BASE_DIR = Path(__file__).parent.parent / "usecases" / "input"

# Available use cases and their schema paths
USE_CASES = {
    "music": {
        "schema": BASE_DIR / "music" / "schemamatching" / "target_schema.json",
        "variants": {
            "original": BASE_DIR / "music" / "data",
            "challenging": BASE_DIR / "music_challenging" / "data",
            "no_headers": BASE_DIR / "music_no_headers" / "data",
        },
    },
    "games": {
        "schema": BASE_DIR / "games" / "schemamatching" / "target_schema.json",
        "variants": {
            "original": BASE_DIR / "games" / "data",
            "challenging": BASE_DIR / "games_challenging" / "data",
            "no_headers": BASE_DIR / "games_no_headers" / "data",
        },
    },
    "companies": {
        "schema": BASE_DIR / "companies" / "schemamatching" / "target_schema_flat.json",
        "variants": {
            "original": BASE_DIR / "companies" / "data",
            "challenging": BASE_DIR / "companies_challenging" / "data",
            "no_headers": BASE_DIR / "companies_no_headers" / "data",
        },
    },
}

HEADER_TYPES = ["original", "challenging", "no_headers"]


def setup_logging(output_dir: Path, log_name: str = "schema_matching_test"):
    """Configure logging to file and console."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"{log_name}.log"

    # Clear existing handlers
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test schema matching step in isolation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Run all combinations mode (default)
    parser.add_argument(
        "--use-cases", type=str, nargs="*", default=None,
        choices=list(USE_CASES.keys()),
        help="Use cases to test (default: all). Options: music, games, companies",
    )
    parser.add_argument(
        "--header-types", type=str, nargs="*", default=None,
        choices=HEADER_TYPES,
        help="Header types to test (default: all). Options: original, challenging, no_headers",
    )

    # Single run mode (for backwards compatibility)
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to directory containing input CSV/XML files (single run mode)",
    )
    parser.add_argument(
        "--schema", type=str, default=None,
        help="Path to target schema JSON file (single run mode)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Path to output directory for results",
    )

    # Common options
    parser.add_argument(
        "--num-rows", type=int, default=30,
        help="Number of sample rows to show the LLM (default: 30)",
    )
    parser.add_argument(
        "--force-rematch", action="store_true",
        help="Force re-matching even if cached results exist",
    )
    parser.add_argument(
        "--skip-normalization", action="store_true",
        help="Skip normalization step, only run schema matching",
    )

    return parser.parse_args()


def run_schema_matching_test(
    data_dir: Path,
    schema_path: Path,
    output_dir: Path,
    chat_model,
    num_rows: int = 30,
    force_rematch: bool = False,
    skip_normalization: bool = False,
    logger=None,
) -> dict:
    """
    Run schema matching (and optionally normalization) on all data files.

    Returns a dictionary with results for each dataset.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Find data files
    data_files = list(data_dir.glob("*.xml")) + list(data_dir.glob("*.csv"))

    if not data_files:
        logger.warning(f"No data files found in {data_dir}")
        return {"error": f"No data files found in {data_dir}"}

    if not schema_path.exists():
        logger.warning(f"Schema file not found: {schema_path}")
        return {"error": f"Schema file not found: {schema_path}"}

    # Set up output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    mappings_dir = output_dir / "mappings"
    mappings_dir.mkdir(parents=True, exist_ok=True)

    # Load target schema
    with open(schema_path) as f:
        target_schema = json.load(f)

    logger.info("-" * 50)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Data files: {[p.name for p in data_files]}")
    logger.info("-" * 50)

    results = {}
    summary = []

    for data_path in data_files:
        name = data_path.stem
        mapping_path = mappings_dir / f"{name}_mapping.csv"

        logger.info(f"\n  Processing: {name}")

        # Check cache
        if not force_rematch and mapping_path.exists():
            logger.info(f"    Loading cached mapping...")
            mapping = pd.read_csv(mapping_path)
            if mapping.empty:
                logger.warning(f"    Cached mapping is empty, will re-match")
            else:
                logger.info(f"    Loaded {len(mapping)} mappings from cache")
                results[name] = {
                    "mapping": mapping,
                    "status": "cached",
                }
                summary.append({
                    "dataset": name,
                    "status": "cached",
                    "num_mappings": len(mapping),
                    "source_columns": None,
                    "mapped_columns": list(mapping["target_column"].unique()) if "target_column" in mapping.columns else [],
                })
                continue

        # Load data
        logger.info(f"    Loading data...")
        try:
            df = load_data_file(data_path)
            logger.info(f"    Loaded {len(df)} rows, {len(df.columns)} columns")
            logger.info(f"    Columns: {list(df.columns)}")
        except Exception as e:
            logger.error(f"    Failed to load data: {e}")
            results[name] = {"status": "load_error", "error": str(e)}
            summary.append({
                "dataset": name,
                "status": "load_error",
                "error": str(e),
            })
            continue

        # Schema matching
        logger.info("    Running schema matching...")
        try:
            mapping = auto_match_schema(df, target_schema, chat_model, num_rows=num_rows)

            if mapping.empty:
                logger.warning(f"    No mappings found")
                results[name] = {"mapping": mapping, "status": "no_mappings"}
                summary.append({
                    "dataset": name,
                    "status": "no_mappings",
                    "num_mappings": 0,
                    "source_columns": list(df.columns),
                    "mapped_columns": [],
                })
            else:
                logger.info(f"    Found {len(mapping)} mappings:")
                for _, row in mapping.iterrows():
                    src = row.get("source_column", "?")
                    tgt = row.get("target_column", "?")
                    score = row.get("score", "?")
                    logger.info(f"      {src} -> {tgt} (score: {score})")

                # Save mapping
                mapping.to_csv(mapping_path, index=False)

                results[name] = {"mapping": mapping, "status": "matched"}
                summary.append({
                    "dataset": name,
                    "status": "matched",
                    "num_mappings": len(mapping),
                    "source_columns": list(df.columns),
                    "mapped_columns": list(mapping["target_column"].unique()) if "target_column" in mapping.columns else [],
                })

        except Exception as e:
            logger.error(f"    Schema matching failed: {e}")
            results[name] = {"status": "match_error", "error": str(e)}
            summary.append({
                "dataset": name,
                "status": "match_error",
                "error": str(e),
            })
            continue

        # Normalization (optional)
        if not skip_normalization and results[name]["status"] == "matched":
            logger.info("    Running normalization...")
            try:
                normalized, transform_result = auto_normalize(
                    df, mapping, target_schema,
                    on_failure="null",
                    chat_model=chat_model,
                    schema_base_path=str(schema_path.parent),
                    taxonomy_cache_dir=str(output_dir),
                )
                normalized = _ensure_stable_id(normalized, dataset_name=name, id_column="id")

                # Save normalized data
                normalized_path = output_dir / f"{name}.csv"
                normalized.to_csv(normalized_path, index=False)

                logger.info(f"    Normalized: {len(normalized)} rows, {transform_result.total_transformed} transformed, {transform_result.total_failed} failed")

                results[name]["normalized"] = normalized
                results[name]["transform_result"] = transform_result

            except Exception as e:
                logger.error(f"    Normalization failed: {e}")
                results[name]["normalization_error"] = str(e)

    # Save summary
    summary_df = pd.DataFrame(summary)
    if not summary_df.empty:
        summary_df.to_csv(output_dir / "schema_matching_summary.csv", index=False)

    # Calculate success rate
    total = len(summary)
    matched = sum(1 for s in summary if s.get("status") == "matched")
    cached = sum(1 for s in summary if s.get("status") == "cached")
    failed = total - matched - cached

    return {
        "results": results,
        "summary": summary,
        "stats": {
            "total": total,
            "matched": matched,
            "cached": cached,
            "failed": failed,
        },
    }


def run_all_combinations(
    use_cases: List[str],
    header_types: List[str],
    output_base_dir: Path,
    chat_model,
    num_rows: int = 30,
    force_rematch: bool = False,
    skip_normalization: bool = False,
    logger=None,
) -> dict:
    """
    Run schema matching tests for all combinations of use cases and header types.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    all_results = {}
    overall_summary = []

    total_combinations = len(use_cases) * len(header_types)
    current = 0

    for use_case in use_cases:
        if use_case not in USE_CASES:
            logger.warning(f"Unknown use case: {use_case}, skipping")
            continue

        config = USE_CASES[use_case]
        schema_path = config["schema"]

        for header_type in header_types:
            current += 1
            if header_type not in config["variants"]:
                logger.warning(f"Header type {header_type} not available for {use_case}, skipping")
                continue

            data_dir = config["variants"][header_type]

            if not data_dir.exists():
                logger.warning(f"Data directory not found: {data_dir}, skipping")
                continue

            combination_key = f"{use_case}_{header_type}"
            output_dir = output_base_dir / combination_key

            logger.info("")
            logger.info("=" * 70)
            logger.info(f"[{current}/{total_combinations}] {use_case.upper()} - {header_type.upper()}")
            logger.info("=" * 70)

            result = run_schema_matching_test(
                data_dir=data_dir,
                schema_path=schema_path,
                output_dir=output_dir,
                chat_model=chat_model,
                num_rows=num_rows,
                force_rematch=force_rematch,
                skip_normalization=skip_normalization,
                logger=logger,
            )

            all_results[combination_key] = result

            # Add to overall summary
            if "stats" in result:
                stats = result["stats"]
                overall_summary.append({
                    "use_case": use_case,
                    "header_type": header_type,
                    "total_datasets": stats["total"],
                    "matched": stats["matched"],
                    "cached": stats["cached"],
                    "failed": stats["failed"],
                    "success_rate": f"{(stats['matched'] + stats['cached']) / max(stats['total'], 1) * 100:.1f}%",
                })

                logger.info(f"\n  Results: {stats['matched']} matched, {stats['cached']} cached, {stats['failed']} failed")

    return {
        "results": all_results,
        "summary": overall_summary,
    }


def print_final_summary(summary: List[dict], logger):
    """Print a formatted summary table."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)

    if not summary:
        logger.info("No results to summarize")
        return

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary)

    # Print table
    logger.info("")
    logger.info(summary_df.to_string(index=False))

    # Print totals
    total_datasets = sum(s.get("total_datasets", 0) for s in summary)
    total_matched = sum(s.get("matched", 0) for s in summary)
    total_cached = sum(s.get("cached", 0) for s in summary)
    total_failed = sum(s.get("failed", 0) for s in summary)

    logger.info("")
    logger.info(f"TOTALS: {total_matched} matched, {total_cached} cached, {total_failed} failed out of {total_datasets} datasets")
    logger.info(f"Overall success rate: {(total_matched + total_cached) / max(total_datasets, 1) * 100:.1f}%")

    return summary_df


def main():
    args = parse_args()

    # Determine run mode
    single_run_mode = args.data_dir is not None and args.schema is not None

    if single_run_mode:
        # Single run mode (backwards compatible)
        data_dir = Path(args.data_dir)
        schema_path = Path(args.schema)
        output_dir = Path(args.output_dir) if args.output_dir else Path("scripts/output/schema_test")

        logger = setup_logging(output_dir)
        llm = ChatOpenAI(model=LLM_MODEL)

        logger.info("=" * 70)
        logger.info("SCHEMA MATCHING TEST - Single Run Mode")
        logger.info("=" * 70)

        result = run_schema_matching_test(
            data_dir=data_dir,
            schema_path=schema_path,
            output_dir=output_dir,
            chat_model=llm,
            num_rows=args.num_rows,
            force_rematch=args.force_rematch,
            skip_normalization=args.skip_normalization,
            logger=logger,
        )

        if "stats" in result:
            stats = result["stats"]
            logger.info("")
            logger.info(f"Results: {stats['matched']} matched, {stats['cached']} cached, {stats['failed']} failed")

    else:
        # All combinations mode
        use_cases = args.use_cases if args.use_cases else list(USE_CASES.keys())
        header_types = args.header_types if args.header_types else HEADER_TYPES
        output_dir = Path(args.output_dir) if args.output_dir else Path("scripts/output/schema_test")

        logger = setup_logging(output_dir)
        llm = ChatOpenAI(model=LLM_MODEL)

        logger.info("=" * 70)
        logger.info("SCHEMA MATCHING TEST - All Combinations Mode")
        logger.info("=" * 70)
        logger.info(f"Use cases: {use_cases}")
        logger.info(f"Header types: {header_types}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Skip normalization: {args.skip_normalization}")
        logger.info(f"Force rematch: {args.force_rematch}")

        result = run_all_combinations(
            use_cases=use_cases,
            header_types=header_types,
            output_base_dir=output_dir,
            chat_model=llm,
            num_rows=args.num_rows,
            force_rematch=args.force_rematch,
            skip_normalization=args.skip_normalization,
            logger=logger,
        )

        # Print final summary
        summary_df = print_final_summary(result["summary"], logger)

        # Save overall summary
        if summary_df is not None:
            summary_df.to_csv(output_dir / "overall_summary.csv", index=False)
            logger.info(f"\nSummary saved to {output_dir / 'overall_summary.csv'}")

    logger.info("")
    logger.info("=== Schema Matching Test Complete ===")


if __name__ == "__main__":
    main()
