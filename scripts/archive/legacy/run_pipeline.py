"""
PyDI Data Integration Pipeline

Usage:
    python run_pipeline.py --data-dir <path> --schema <path> --output-dir <path>

Example:
    python run_pipeline.py \
        --data-dir usecases/input/music/data \
        --schema usecases/input/music/schemamatching/target_schema.json \
        --output-dir scripts/output/music
"""

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from PyDI.pipeline import (
    run_pipeline,
    select_dataset_pairs,
    load_or_generate_similarity_validation_set,
    load_or_generate_similarity_training_set,
    run_active_learning,
    optimize_matching,
    FusionConfig,
    run_data_fusion,
    generate_faiss_candidates,
    generate_training_variants,
    compare_training_variants,
)
from PyDI.pipeline.fusion_validation_generation import (
    generate_fusion_validation_set,
    load_fusion_validation_from_cache,
    convert_to_tabular_format,
    identify_identifying_columns,
    build_entity_groups,
    select_well_known_entities,
)
from PyDI.pipeline.fusion_optimization import (
    run_fusion_case,
    run_all_fusion_cases,
    evaluate_fusion_accuracy,
    compute_source_accuracy_from_validation_set,
    convert_provided_validation_to_csv_format,
)
from PyDI.pipeline.end_to_end_metrics import (
    EndToEndMetrics,
    calculate_structural_metrics,
    generate_end_to_end_report,
    save_end_to_end_report,
)
from PyDI.pipeline.labeling_cost_estimation import save_cost_report
from PyDI.pipeline.normalization_stats import save_normalization_stats
from PyDI.pipeline.fusion_comparison import generate_fusion_comparison_report
from PyDI.pipeline.cluster_cleaning import (
    clean_oversized_clusters,
    save_cluster_cleaning_report,
    save_cluster_cleaning_latex,
)
from PyDI.io.loaders import load_xml
from PyDI.pipeline.reporting import (
    PipelineMetrics,
    create_step_tracker,
    print_source_overview,
    save_source_overview,
    generate_source_overview_table,
    generate_schema_matching_report_for_pipeline,
    save_training_comparison_summary,
    save_cluster_size_report,
)
from PyDI.pipeline.utils import report_correspondences
from PyDI.normalization import load_normalization_spec
from PyDI.normalization.transform import transform_dataframe

from pipeline_utils import (
    PipelineDirectories,
    check_validation_cache,
    check_training_cache,
    check_matching_cache,
    check_fusion_cache,
    load_json_cache,
    save_json_cache,
    normalize_blocking_cache,
    setup_logging,
    pair_key,
    find_training_file,
)

import time

start_time = time.time()

load_dotenv()


# =============================================================================
# CONFIGURATION - Edit these values to customize the pipeline
# =============================================================================

# LLM Model
LLM_MODEL = "gpt-5.2"

# --- Validation Set Generation ---
# Target number of positive (matching) pairs
VALIDATION_TARGET_POSITIVES = 100
# Target number of negative (non-matching) pairs
VALIDATION_TARGET_NEGATIVES = 200
VALIDATION_K = 20                   # Number of FAISS neighbors to consider per query
# (positives_per_query, negatives_per_query)
VALIDATION_NEIGHBORS_PER_QUERY = (1, 3)
VALIDATION_BATCH_SIZE = 3           # Label batch size
VALIDATION_MAX_LABELS = 1_500        # Max LLM labeling calls

# --- Training Set Generation (small = step 2, large = active learning / comparison) ---
# Positives for small training set (step 2 / faiss_small)
TRAINING_SMALL_TARGET_POSITIVES = 34
TRAINING_SMALL_TARGET_NEGATIVES = 66    # Negatives for small training set
# Positives for large training set (active / faiss_large)
TRAINING_LARGE_TARGET_POSITIVES = 200
TRAINING_LARGE_TARGET_NEGATIVES = 400   # Negatives for large training set
TRAINING_K = 20                         # Number of FAISS neighbors to consider
# (positives_per_query, negatives_per_query)
TRAINING_NEIGHBORS_PER_QUERY = (1, 3)
TRAINING_BATCH_SIZE = 3                 # Label batch size
TRAINING_MAX_LABELS = 3_000               # Max LLM labeling calls

# --- FAISS Candidate Generation ---
FAISS_K = 20                        # Number of neighbors per query record

# --- Matcher Optimization ---
MATCHER_THRESHOLDS = [0.3, 0.4, 0.5, 0.6,
                      0.7, 0.8, 0.9]  # Thresholds to evaluate
INCLUDE_RULE_BASED = True           # Include rule-based matchers
INCLUDE_ML_BASED = True             # Include ML-based matchers
INCLUDE_LLM_BASED = False           # Include LLM-based matchers (expensive)

# --- Active Learning ---
ACTIVE_LEARNING_MAX_LABELS = 5000       # Max total labels in active learning
ACTIVE_LEARNING_LABELS_PER_ITER = 100   # Labels per iteration
ACTIVE_LEARNING_MAX_CANDIDATES = 5000   # Max candidates to consider
ACTIVE_LEARNING_MAX_ITERATIONS = 30     # Max active learning iterations
ACTIVE_LEARNING_BATCH_SIZE = 25         # LLM labeling batch size

# --- Training Set Comparison ---
# Random pairs to add as fraction of base set (20%)
COMPARISON_RANDOM_SAMPLE_RATIO = 0.2

# --- Data Fusion ---
FUSION_INCLUDE_SINGLETONS = True    # Include unmatched records in fused output
FUSION_USE_LLM = True               # Use LLM for fusion strategy planning

# --- Fusion Validation ---
FUSION_VALIDATION_N_ENTITIES = 30   # Number of entities to validate
# Entities to sample for well-known selection
FUSION_VALIDATION_SAMPLE_SIZE = 100

# --- Diagnostics ---
DIAGNOSTICS_TOP_EXAMPLES = 10       # Number of examples to show in reports


# =============================================================================
# CLI Arguments
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PyDI data integration pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Path to directory containing input CSV/XML files",
    )
    parser.add_argument(
        "--schema", type=str, required=True,
        help="Path to target schema JSON file",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Path to output directory for results",
    )
    parser.add_argument(
        "--test-dir", type=str, default=None,
        help="Path to directory containing held-out test sets",
    )
    parser.add_argument(
        "--fusion-val-generation-mode", type=str, default="llm",
        choices=["llm", "llm_omit", "web", "web_omit", "all"],
        help="Fusion validation mode: llm (LLM only), llm_omit (LLM + omit target), "
             "web (web search), web_omit (web search + omit target), all (generate all 4 variants)",
    )
    parser.add_argument(
        "--fusion-use-test-set", action="store_true", default=False,
        help="Filter fusion validation to entities in the test set",
    )
    parser.add_argument(
        "--fusion-case", type=str, default=None,
        choices=["heuristic", "llm_no_val", "llm_val",
                 "llm_val_stats", "heuristic_stats", "iterative", "all"],
        help="Fusion rule selection strategy: "
             "heuristic (data types only), "
             "llm_no_val (LLM without validation), "
             "llm_val (LLM + validation feedback), "
             "llm_val_stats (LLM + validation + accuracy stats), "
             "heuristic_stats (heuristic + accuracy stats), "
             "iterative (LLM iteratively improving), "
             "all (run all cases for comparison)",
    )
    parser.add_argument(
        "--fusion-iterations", type=int, default=3,
        help="Number of iterations for iterative fusion case (default: 3)",
    )
    parser.add_argument(
        "--fusion-primary-val-mode", type=str, default="llm_omit",
        choices=["llm", "llm_omit", "web", "web_omit"],
        help="Primary validation mode for computing source accuracy stats (default: llm_omit)",
    )
    # Training set comparison
    parser.add_argument(
        "--compare-training-sets", action="store_true", default=False,
        help="Generate and compare multiple training set variants (faiss_small, faiss_large, active, +random)",
    )
    parser.add_argument(
        "--comparison-pairs", type=str, nargs="*", default=None,
        help="Specific dataset pairs to compare (format: 'left_right' e.g., 'dbpedia_forbes forbes_fullcontact'). "
             "If not specified, compares all pairs from step 2.",
    )
    # Provided validation comparison
    parser.add_argument(
        "--compare-provided-validation", action="store_true", default=False,
        help="Compare fusion strategies from auto-generated validation sets vs provided validation_set.xml. "
             "Requires --test-dir to point to directory containing validation_set.xml.",
    )
    return parser.parse_args()


# =============================================================================
# Pipeline Steps
# =============================================================================

def load_raw_sources(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load raw source files before any transformation.

    Args:
        data_dir: Directory containing data files (*.xml, *.csv)

    Returns:
        Dictionary mapping source names to raw DataFrames
    """
    data_dir = Path(data_dir)
    data_files = list(data_dir.glob("*.xml")) + list(data_dir.glob("*.csv"))

    sources = {}
    for data_path in data_files:
        name = data_path.stem
        if data_path.suffix == ".csv":
            sources[name] = pd.read_csv(data_path)
        elif data_path.suffix == ".xml":
            sources[name] = load_xml(data_path, nested_handling="aggregate")

    return sources


def print_and_save_source_overview(
    args,
    dirs: PipelineDirectories,
    target_columns: int = 9,
) -> dict[str, pd.DataFrame]:
    """Load raw sources, print overview table, and save to file.

    Args:
        args: Command line arguments (needs args.data_dir, args.schema)
        dirs: Pipeline directories
        target_columns: Number of columns in target schema

    Returns:
        Dictionary of raw source DataFrames
    """
    import json

    # Determine target schema size
    schema_path = Path(args.schema)
    if schema_path.exists():
        with open(schema_path) as f:
            schema = json.load(f)
        target_columns = len(schema.get("properties", {}))

    # Load raw sources
    sources = load_raw_sources(args.data_dir)

    if sources:
        # Print to console
        print_source_overview(sources, target_columns)

        # Save to reporting folder
        reporting_dir = dirs.output / "reporting"
        reporting_dir.mkdir(parents=True, exist_ok=True)
        save_source_overview(
            sources,
            reporting_dir,
            target_columns,
            filename="source_overview.csv",
        )
        print(f"  Saved source overview to: {reporting_dir / 'source_overview.csv'}")

    return sources


def step1_schema_matching(args, dirs: PipelineDirectories, llm, track_step):
    """Step 1: Schema Matching (1a) + Normalization (1b)"""
    print("\n=== Step 1: Schema Matching + Normalization ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Schema: {args.schema}")

    results = run_pipeline(
        data_dir=args.data_dir,
        schema_path=args.schema,
        chat_model=llm,
        output_dir=dirs.schema_matching,
        force_rematch=False,
        track_step=track_step,
    )

    print("\n--- Normalized Datasets ---")
    for name, df in results.items():
        print(f"{name}: {len(df)} rows, {len(df.columns)} columns")

    return results


def _parse_comparison_pairs(args, results):
    """Parse --comparison-pairs into list of (left, right) tuples."""
    if not args.comparison_pairs:
        return None

    parsed = []
    available_names = set(results.keys())

    for pair_str in args.comparison_pairs:
        parts = pair_str.split("_")
        if len(parts) == 2:
            left, right = parts
            # Normalize order: larger dataset first (consistent with select_dataset_pairs)
            if left in available_names and right in available_names:
                if len(results[left]) >= len(results[right]):
                    parsed.append((left, right))
                else:
                    parsed.append((right, left))
            else:
                missing = [n for n in [left, right]
                           if n not in available_names]
                print(f"  Warning: dataset(s) not found: {missing}")
        else:
            print(
                f"  Warning: invalid pair format '{pair_str}', expected 'left_right'")

    return parsed if parsed else None


def step2_validation_training(args, dirs: PipelineDirectories, results, llm, track_step):
    """Step 2: Validation & Training Set Generation"""
    print("\n=== Step 2: Validation & Training Set Generation ===")

    # Use custom pairs if specified, otherwise auto-select
    custom_pairs = _parse_comparison_pairs(args, results)
    if custom_pairs:
        pairs = custom_pairs
        print(f"Using custom pairs: {pairs}")
    else:
        pairs = select_dataset_pairs(results)
    validation_sets = {}
    training_sets = {}

    # Check for pre-computed embeddings in input directory
    input_embeddings_dir = Path(
        args.data_dir).parent / "entitymatching" / "embeddings"
    if input_embeddings_dir.exists():
        print(f"Found pre-computed embeddings: {input_embeddings_dir}")
    else:
        input_embeddings_dir = None

    val_cached = check_validation_cache(pairs, dirs.validation)
    train_cached = check_training_cache(pairs, dirs.training)

    with track_step("Step 2: Validation & Training", cached=val_cached and train_cached):
        for left_name, right_name in pairs:
            print(f"\n--- {left_name} <-> {right_name} ---")

            # Validation set
            print("  [Validation] Loading or generating...")
            val_set = load_or_generate_similarity_validation_set(
                df_left=results[left_name],
                df_right=results[right_name],
                left_name=left_name,
                right_name=right_name,
                chat_model=llm,
                output_dir=dirs.validation,
                id_column="id",
                target_size=VALIDATION_TARGET_POSITIVES + VALIDATION_TARGET_NEGATIVES,
                target_positives=VALIDATION_TARGET_POSITIVES,
                k=VALIDATION_K,
                neighbors_per_query=VALIDATION_NEIGHBORS_PER_QUERY,
                batch_size=VALIDATION_BATCH_SIZE,
                query_order="similarity",
                force_regenerate=False,
                generate_guidelines=False,
                retrieval_method="faiss",
                max_labels=VALIDATION_MAX_LABELS,
                input_embeddings_dir=input_embeddings_dir,
            )
            validation_sets[(left_name, right_name)] = val_set
            n_pos = (val_set["label"] == "TRUE").sum()
            n_neg = (val_set["label"] == "FALSE").sum()
            print(
                f"  Validation: {len(val_set)} pairs ({n_pos} pos, {n_neg} neg)")

            # Training set
            print("  [Training] Loading or generating...")
            train_set = load_or_generate_similarity_training_set(
                df_left=results[left_name],
                df_right=results[right_name],
                left_name=left_name,
                right_name=right_name,
                chat_model=llm,
                output_dir=dirs.training,
                id_column="id",
                target_size=TRAINING_SMALL_TARGET_POSITIVES + TRAINING_SMALL_TARGET_NEGATIVES,
                target_positives=TRAINING_SMALL_TARGET_POSITIVES,
                k=TRAINING_K,
                neighbors_per_query=TRAINING_NEIGHBORS_PER_QUERY,
                batch_size=TRAINING_BATCH_SIZE,
                query_order="similarity",
                exclude_pairs=val_set,
                force_regenerate=False,
                generate_guidelines=False,
                retrieval_method="faiss",
                max_labels=TRAINING_MAX_LABELS,
                input_embeddings_dir=input_embeddings_dir,
            )
            training_sets[(left_name, right_name)] = train_set
            n_pos = (train_set["label"] == "TRUE").sum()
            n_neg = (train_set["label"] == "FALSE").sum()
            print(
                f"  Training: {len(train_set)} pairs ({n_pos} pos, {n_neg} neg)")

    return pairs, validation_sets, training_sets


def step3_faiss_candidates(args, dirs: PipelineDirectories, pairs, results, llm, track_step):
    """Step 3: FAISS Candidate Generation"""
    print("\n=== Step 3: FAISS Candidate Generation ===")

    # Check for pre-computed embeddings in input directory
    input_embeddings_dir = Path(
        args.data_dir).parent / "entitymatching" / "embeddings"
    if not input_embeddings_dir.exists():
        input_embeddings_dir = None

    faiss_candidates = {}

    with track_step("Step 3: FAISS Candidate Generation"):
        for left_name, right_name in pairs:
            candidates = generate_faiss_candidates(
                df_left=results[left_name],
                df_right=results[right_name],
                left_name=left_name,
                right_name=right_name,
                id_column="id",
                k=FAISS_K,
                output_dir=dirs.validation,
                chat_model=llm,
                input_embeddings_dir=input_embeddings_dir,
            )
            faiss_candidates[(left_name, right_name)] = candidates
            print(f"{left_name} <-> {right_name}: {len(candidates)} candidates")

    return faiss_candidates


def _load_training_comparison_cache(
    comparison_dir: Path,
    training_dir: Path,
    left_name: str,
    right_name: str,
) -> tuple:
    """Load cached training comparison results if available.

    Returns
    -------
    tuple
        (best_training_set, best_variant, cached) or (None, None, False) if not cached
    """
    import json

    # Try both orderings of the pair names
    pair_orderings = [
        (left_name, right_name),
        (right_name, left_name),
    ]

    details_path = None
    actual_left, actual_right = left_name, right_name

    for l, r in pair_orderings:
        candidate_path = comparison_dir / f"{l}_{r}" / "comparison_details.json"
        if candidate_path.exists():
            details_path = candidate_path
            actual_left, actual_right = l, r
            break

    if details_path is None:
        return None, None, False

    try:
        with open(details_path) as f:
            details = json.load(f)

        best_variant = details.get("best_variant")
        if not best_variant:
            return None, None, False

        # Load the best variant's training set CSV - try multiple path patterns
        possible_paths = [
            training_dir / f"similarity_training_{best_variant}_{actual_left}_{actual_right}.csv",
            training_dir / f"similarity_training_{best_variant}_{actual_right}_{actual_left}.csv",
            training_dir / f"training_{actual_left}_{actual_right}_{best_variant}.csv",
            training_dir / f"training_{actual_left}_{actual_right}_augmented.csv",
            training_dir / f"training_{actual_left}_{actual_right}_latest.csv",
        ]

        for variant_path in possible_paths:
            if variant_path.exists():
                best_training_set = pd.read_csv(variant_path, dtype={"label": str})
                return best_training_set, best_variant, True

        return None, None, False

    except Exception as e:
        print(f"  Warning: Failed to load cache: {e}")
        return None, None, False


def step4_training_comparison(
    args,
    dirs: PipelineDirectories,
    pairs,
    results,
    validation_sets,
    training_sets,
    faiss_candidates,
    llm,
    track_step,
):
    """Step 4: Training Set Comparison (optional)

    Generates multiple training set variants (faiss_small, faiss_large, active, *_plus_random)
    and compares their performance on the provided test set (or validation set if no test set).

    Returns the best training sets for each pair.
    """
    if not args.compare_training_sets:
        return training_sets

    print("\n=== Step 4: Training Set Comparison ===")

    comparison_dir = dirs.training / "training_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Look for test sets in the entitymatching directory
    test_sets_dir = Path(args.data_dir).parent / "entitymatching"

    best_training_sets = {}

    # Check if all pairs are cached
    all_cached = True
    for (left_name, right_name) in validation_sets.keys():
        _, _, cached = _load_training_comparison_cache(
            comparison_dir, dirs.training, left_name, right_name
        )
        if not cached:
            all_cached = False
            break

    with track_step("Step 4: Training Set Comparison", cached=all_cached):
        for (left_name, right_name), val_set in validation_sets.items():
            print(f"\n--- {left_name} <-> {right_name} ---")

            # Check cache first
            cached_training_set, cached_best_variant, is_cached = _load_training_comparison_cache(
                comparison_dir, dirs.training, left_name, right_name
            )

            if is_cached:
                best_training_sets[(left_name, right_name)] = cached_training_set
                n_pos = (cached_training_set["label"].astype(str).str.upper() == "TRUE").sum()
                n_neg = (cached_training_set["label"].astype(str).str.upper() == "FALSE").sum()
                print(f"  Loaded from cache: {cached_best_variant}")
                print(f"  Training set: {len(cached_training_set)} pairs ({n_pos} pos, {n_neg} neg)")
                continue

            candidates = faiss_candidates.get((left_name, right_name))
            if candidates is None or candidates.empty:
                print("  Skipping: no FAISS candidates")
                best_training_sets[(left_name, right_name)] = training_sets.get(
                    (left_name, right_name), pd.DataFrame()
                )
                continue

            # Try to find a test set for this pair
            test_set = None
            test_set_patterns = [
                f"{left_name}_2_{right_name}_test.csv",
                f"{right_name}_2_{left_name}_test.csv",
                f"{left_name}_{right_name}_test.csv",
                f"{right_name}_{left_name}_test.csv",
            ]
            for pattern in test_set_patterns:
                test_path = test_sets_dir / pattern
                if test_path.exists():
                    # Detect if file has header (same approach as test_evaluation.py)
                    with open(test_path, 'r') as f:
                        first_line = f.readline().strip()

                    if 'id1' in first_line.lower() or 'label' in first_line.lower():
                        test_set = pd.read_csv(test_path)
                        test_set.columns = [c.lower()
                                            for c in test_set.columns]
                    else:
                        test_set = pd.read_csv(test_path, header=None, names=[
                                               "id1", "id2", "label"])

                    # Normalize
                    test_set["id1"] = test_set["id1"].astype(str)
                    test_set["id2"] = test_set["id2"].astype(str)
                    test_set["label"] = test_set["label"].astype(
                        str).str.upper().str.strip()

                    # Normalize column names if needed
                    if "ltable_id" in test_set.columns:
                        test_set = test_set.rename(
                            columns={"ltable_id": "id1", "rtable_id": "id2"})

                    print(
                        f"  Found test set: {test_path.name} ({len(test_set)} pairs)")
                    break

            if test_set is None:
                print("  No test set found, will evaluate on validation set")

            # Get existing training set from step 2 (to reuse for faiss_small)
            existing_train_set = training_sets.get((left_name, right_name))

            # Generate all variants
            variants = generate_training_variants(
                df_left=results[left_name],
                df_right=results[right_name],
                left_name=left_name,
                right_name=right_name,
                validation_set=val_set,
                faiss_candidates=candidates,
                chat_model=llm,
                output_dir=dirs.training,
                id_column="id",
                small_target_positives=TRAINING_SMALL_TARGET_POSITIVES,
                small_target_negatives=TRAINING_SMALL_TARGET_NEGATIVES,
                large_target_positives=TRAINING_LARGE_TARGET_POSITIVES,
                large_target_negatives=TRAINING_LARGE_TARGET_NEGATIVES,
                random_sample_ratio=COMPARISON_RANDOM_SAMPLE_RATIO,
                existing_training_set=existing_train_set,
            )

            # Compare variants (using test set if available)
            pair_comparison_dir = comparison_dir / f"{left_name}_{right_name}"
            pair_comparison_dir.mkdir(parents=True, exist_ok=True)

            comparison_df, best_variant = compare_training_variants(
                df_left=results[left_name],
                df_right=results[right_name],
                variants=variants,
                validation_set=val_set,
                output_dir=pair_comparison_dir,
                id_column="id",
                test_set=test_set,
                left_name=left_name,
                right_name=right_name,
                entitymatching_dir=test_sets_dir,
            )

            # Use the best variant's training set
            best_training_sets[(left_name, right_name)
                               ] = variants[best_variant].training_set

            print(f"  Best variant: {best_variant}")
            print(f"  Comparison saved to: {pair_comparison_dir}")

    # Save overall summary
    print("\n--- Training Comparison Summary ---")
    for (left_name, right_name), train_set in best_training_sets.items():
        n_pos = (train_set["label"].astype(str).str.upper() == "TRUE").sum()
        n_neg = (train_set["label"].astype(str).str.upper() == "FALSE").sum()
        print(
            f"  {left_name} <-> {right_name}: {len(train_set)} pairs ({n_pos} pos, {n_neg} neg)")

    # Generate Auto vs Provided comparison summary report
    summary_path = save_training_comparison_summary(dirs.output)
    if summary_path:
        print(f"\n  Saved Auto vs Provided summary to: {summary_path}")

    return best_training_sets


def step5_matcher_optimization(
    dirs: PipelineDirectories,
    pairs,
    results,
    validation_sets,
    training_sets,
    faiss_candidates,
    track_step,
):
    """Step 5: Matcher Optimization"""
    print("\n=== Step 5: Matcher Optimization ===")

    # Load matcher cache
    matcher_cache_path = dirs.matching / "matcher_configs.json"
    matcher_cache = load_json_cache(matcher_cache_path)
    blocking_cache = normalize_blocking_cache(
        load_json_cache(dirs.blocking / "blocking_configs.json")
    )

    fusion_correspondences = []
    best_matchers_summary = []

    cached = check_matching_cache(pairs, dirs.matching)
    with track_step("Step 5: Matcher Optimization", cached=cached):
        for (left_name, right_name), val_set in validation_sets.items():
            cache_key = pair_key(left_name, right_name)
            corr_path = dirs.matching / \
                f"correspondences_{left_name}_{right_name}.csv"

            # Check cache
            cached_matcher = matcher_cache.get(cache_key)
            if cached_matcher and corr_path.exists():
                print(f"\n{left_name} <-> {right_name}: Using cached matcher")
                print(
                    f"  {cached_matcher.get('matcher')} (F1={cached_matcher.get('f1', 0):.3f})")

                matched = pd.read_csv(corr_path)
                if not matched.empty:
                    fusion_correspondences.append(matched)

                best_matchers_summary.append({
                    "left": left_name, "right": right_name,
                    "best_matcher": cached_matcher.get("matcher"),
                    "f1": cached_matcher.get("f1"),
                    "precision": cached_matcher.get("precision"),
                    "recall": cached_matcher.get("recall"),
                    "threshold": cached_matcher.get("threshold"),
                })
                continue

            print(f"\nOptimizing matchers for {left_name} <-> {right_name}...")

            train_set = training_sets[(left_name, right_name)]
            df_left = results[left_name]
            df_right = results[right_name]

            # Get matching columns from blocking cache if available
            matching_columns = None
            cached_blocking = blocking_cache.get(cache_key)
            if isinstance(cached_blocking, dict):
                cols = cached_blocking.get("blocking_columns") or []
                if isinstance(cols, str):
                    cols = [cols]
                cols = [
                    c for c in cols if c in df_left.columns and c in df_right.columns and c != "id"]
                if cols:
                    matching_columns = cols

            match_opt = optimize_matching(
                df_left=df_left,
                df_right=df_right,
                validation_set=val_set,
                id_column="id",
                matching_columns=matching_columns,
                include_rule_based=INCLUDE_RULE_BASED,
                include_ml_based=INCLUDE_ML_BASED,
                include_llm_based=INCLUDE_LLM_BASED,
                training_set=train_set,
                thresholds=MATCHER_THRESHOLDS,
                out_dir=dirs.matching,
                out_path=dirs.matching /
                f"matcher_optimization_{left_name}_{right_name}.csv",
            )

            best = match_opt["best"]
            best_artifacts = match_opt["best_artifacts"]

            if best:
                print(f"  Best: {best['matcher']} (F1={best['f1']:.3f})")
                best_matchers_summary.append({
                    "left": left_name, "right": right_name,
                    "best_matcher": best.get("matcher"),
                    "f1": best.get("f1"),
                    "precision": best.get("precision"),
                    "recall": best.get("recall"),
                    "threshold": best.get("threshold"),
                })

            # Run best matcher on FAISS candidates
            candidates_df = faiss_candidates.get((left_name, right_name))
            if candidates_df is None or candidates_df.empty:
                continue

            matched = _run_best_matcher(
                best, best_artifacts,
                df_left, df_right, candidates_df,
                dirs.matching, left_name, right_name,
            )

            if matched is not None and not matched.empty:
                fusion_correspondences.append(matched)
                # Update cache
                matcher_cache[cache_key] = {
                    "matcher": best.get("matcher"),
                    "threshold": best.get("threshold"),
                    "f1": best.get("f1"),
                    "precision": best.get("precision"),
                    "recall": best.get("recall"),
                }
                save_json_cache(matcher_cache_path, matcher_cache)

    # Save summary
    if best_matchers_summary:
        summary_df = pd.DataFrame(best_matchers_summary)
        summary_df.to_csv(dirs.matching / "matching_summary.csv", index=False)
        print("\n--- Best Matchers ---")
        for row in best_matchers_summary:
            print(
                f"  {row['left']} <-> {row['right']}: {row['best_matcher']} (F1={row['f1']:.3f})")

    return fusion_correspondences, best_matchers_summary


def _run_best_matcher(
    best, best_artifacts,
    df_left, df_right, candidates_df,
    matching_dir, left_name, right_name,
):
    """Run the best matcher on candidates and save correspondences."""
    if not best or not best_artifacts:
        return None

    mname = str(best.get("matcher") or "")
    thr = float(best.get("threshold") or 0.5)
    matcher = best_artifacts.get("matcher")

    if matcher is None:
        return None

    cand_batches = [candidates_df[["id1", "id2"]]]
    display_cols = [
        c for c in df_left.columns if c in df_right.columns and c != "id"][:4]

    try:
        if mname == "RuleBasedMatcher":
            comparators = best_artifacts.get("comparators")
            if not comparators:
                return None
            corr = matcher.match(
                df_left=df_left, df_right=df_right,
                candidates=cand_batches, id_column="id",
                comparators=comparators, threshold=0.0,
            )
        elif mname == "MLBasedMatcher":
            classifier = best_artifacts.get("classifier")
            if not classifier:
                return None
            corr = matcher.match(
                df_left=df_left, df_right=df_right,
                candidates=cand_batches, id_column="id",
                trained_classifier=classifier, threshold=0.0,
                use_probabilities=bool(best.get("use_probabilities", True)),
            )
        elif mname == "LLMBasedMatcher":
            corr = matcher.match(
                df_left=df_left, df_right=df_right,
                candidates=cand_batches, id_column="id",
                threshold=0.0,
            )
        else:
            return None

        report_correspondences(
            title=mname, corr=corr, threshold=thr,
            df_left=df_left, df_right=df_right,
            id_column="id", display_cols=display_cols,
            top_examples=DIAGNOSTICS_TOP_EXAMPLES,
        )

        matched = corr[pd.to_numeric(corr["score"], errors="coerce") >= thr]
        matched = matched[["id1", "id2", "score"]].dropna(
        ).drop_duplicates().reset_index(drop=True)

        if not matched.empty:
            out_path = matching_dir / \
                f"correspondences_{left_name}_{right_name}.csv"
            matched.to_csv(out_path, index=False)
            return matched

    except Exception as e:
        print(f"  Matcher failed: {e}")

    return None


def step6_active_learning(
    dirs: PipelineDirectories,
    results,
    validation_sets,
    training_sets,
    faiss_candidates,
    llm,
    track_step,
):
    """Step 6: Active Learning"""
    print("\n=== Step 6: Active Learning ===")

    augmentation_summary = []

    with track_step("Step 6: Active Learning"):
        for (left_name, right_name), train_set in training_sets.items():
            print(f"\n{left_name} <-> {right_name}:")

            val_set = validation_sets.get(
                (left_name, right_name), pd.DataFrame())
            candidates = faiss_candidates.get((left_name, right_name))

            if candidates is None or candidates.empty:
                print("  Skipping: no candidates")
                continue

            # Load training file
            training_path = find_training_file(
                dirs.training, left_name, right_name)
            if training_path and training_path.exists():
                original_training = pd.read_csv(training_path)
                print(
                    f"  Loaded: {training_path.name} ({len(original_training)} rows)")
            else:
                original_training = train_set

            original_pos = (original_training["label"].astype(
                str).str.upper() == "TRUE").sum()
            original_neg = (original_training["label"].astype(
                str).str.upper() == "FALSE").sum()
            print(
                f"  Current: {original_pos} pos, {original_neg} neg | Target: {TRAINING_LARGE_TARGET_POSITIVES} pos, {TRAINING_LARGE_TARGET_NEGATIVES} neg")

            # Check if already have enough
            if original_pos >= TRAINING_LARGE_TARGET_POSITIVES and (TRAINING_LARGE_TARGET_NEGATIVES == 0 or original_neg >= TRAINING_LARGE_TARGET_NEGATIVES):
                print("  Skipping: targets already met")
                continue

            augmented, was_augmented, summary = run_active_learning(
                df_left=results[left_name],
                df_right=results[right_name],
                left_name=left_name,
                right_name=right_name,
                training_set=original_training,
                validation_set=val_set,
                candidates=candidates,
                chat_model=llm,
                output_dir=dirs.training,
                id_column="id",
                target_positives=TRAINING_LARGE_TARGET_POSITIVES,
                target_negatives=TRAINING_LARGE_TARGET_NEGATIVES if TRAINING_LARGE_TARGET_NEGATIVES > 0 else None,
                max_total_labels=ACTIVE_LEARNING_MAX_LABELS,
                labels_per_iteration=ACTIVE_LEARNING_LABELS_PER_ITER,
                max_candidates=ACTIVE_LEARNING_MAX_CANDIDATES,
                max_iterations=ACTIVE_LEARNING_MAX_ITERATIONS,
                label_batch_size=ACTIVE_LEARNING_BATCH_SIZE,
            )

            if was_augmented:
                # Update in-memory training set
                if not val_set.empty:
                    val_pairs = set(zip(val_set["id1"].astype(
                        str), val_set["id2"].astype(str)))
                    filtered = augmented[
                        ~augmented.apply(lambda r: (
                            str(r["id1"]), str(r["id2"])) in val_pairs, axis=1)
                    ].reset_index(drop=True)
                    training_sets[(left_name, right_name)] = filtered

                augmentation_summary.append(summary)

                # Save shuffled augmented set
                augmented_shuffled = augmented.sample(
                    frac=1, random_state=42).reset_index(drop=True)
                augmented_shuffled.to_csv(
                    dirs.training / f"training_{left_name}_{right_name}_augmented.csv", index=False)
                augmented_shuffled.to_csv(
                    dirs.training / f"training_{left_name}_{right_name}_latest.csv", index=False)

    if augmentation_summary:
        aug_df = pd.DataFrame(augmentation_summary)
        aug_df.to_csv(dirs.training /
                      "active_learning_summary.csv", index=False)
        print("\n--- Active Learning Summary ---")
        print(aug_df.to_string(index=False))


def step7_data_fusion(
    dirs: PipelineDirectories,
    results,
    fusion_correspondences,
    llm,
    track_step,
):
    """Step 7: Data Fusion"""
    print("\n=== Step 7: Data Fusion ===")

    cached = check_fusion_cache(dirs.fusion)
    with track_step("Step 7: Data Fusion", cached=cached):
        if cached:
            print("Fusion output cached; skipping.")
            return

        if not fusion_correspondences:
            print("No correspondences; skipping fusion.")
            return

        all_corr = pd.concat(fusion_correspondences, ignore_index=True)
        all_corr = all_corr.drop_duplicates(
            subset=["id1", "id2"]).reset_index(drop=True)
        all_corr.to_csv(dirs.matching / "correspondences_all.csv", index=False)
        print(f"Fusing with {len(all_corr)} correspondences")

        fused, strategy, _ = run_data_fusion(
            results,
            correspondences=all_corr,
            config=FusionConfig(
                id_column="id",
                include_singletons=FUSION_INCLUDE_SINGLETONS,
                debug=True,
                use_llm=FUSION_USE_LLM,
            ),
            chat_model=llm,
            output_dir=dirs.fusion,
        )
        print(f"Fused: {len(fused)} rows (strategy={strategy.name})")

        # Save clean version
        metadata_cols = ["_fusion_source_datasets",
                         "_fusion_confidence", "_fusion_metadata"]
        clean_cols = [c for c in fused.columns if c not in metadata_cols]
        fused[clean_cols].to_csv(dirs.fusion / "fused_clean.csv", index=False)


def step7_5_fusion_validation(
    args,
    dirs: PipelineDirectories,
    results,
    fusion_correspondences,
    llm,
    track_step,
):
    """Step 7.5: Fusion Validation Set Generation"""
    print("\n=== Step 7.5: Fusion Validation Set Generation ===")

    if not fusion_correspondences:
        print("No correspondences; skipping.")
        return

    # Load target entity IDs from test set if enabled
    target_entity_ids = None
    if args.fusion_use_test_set:
        fusion_test_set_path = Path(
            args.data_dir).parent / "fusion" / "test_set.xml"
        if fusion_test_set_path.exists():
            from PyDI.io import load_xml
            print(f"Loading test set from {fusion_test_set_path}...")
            test_set_df = load_xml(fusion_test_set_path,
                                   nested_handling="aggregate")
            if "id" in test_set_df.columns:
                target_entity_ids = set(test_set_df["id"].astype(str).tolist())
                print(f"  >> {len(target_entity_ids)} entity IDs loaded")

    # Parse fusion mode into variants to generate
    mode_configs = {
        "llm": [(False, False, "validation_llm")],
        "llm_omit": [(False, True, "validation_llm_omit")],
        "web": [(True, False, "validation_web")],
        "web_omit": [(True, True, "validation_web_omit")],
        "all": [
            (False, False, "validation_llm"),
            (False, True, "validation_llm_omit"),
            (True, False, "validation_web"),
            (True, True, "validation_web_omit"),
        ],
    }
    variants = mode_configs.get(args.fusion_val_generation_mode, [
                                (False, False, "validation")])

    all_corr = pd.concat(fusion_correspondences, ignore_index=True)
    all_corr = all_corr.drop_duplicates(
        subset=["id1", "id2"]).reset_index(drop=True)

    # Pre-compute identifying columns once if any variant needs omit mode
    identifying_columns = None
    needs_omit = any(omit for _, omit, _ in variants)
    if needs_omit:
        all_attributes = set()
        for df in results.values():
            all_attributes.update(
                c for c in df.columns if c not in ("id", "_id"))
        print(
            f"Pre-computing identifying columns from {len(all_attributes)} attributes...")
        identifying_columns = identify_identifying_columns(
            list(all_attributes), llm)
        print(
            f"  >> Identified {len(identifying_columns)} identifying columns: {identifying_columns}")

    # Pre-select entity groups once if running multiple variants
    # This ensures the same entities are used across all validation variants
    selected_groups = None
    if len(variants) > 1:
        print("\nPre-selecting entity groups for all validation variants...")
        entity_groups = build_entity_groups(all_corr, results)
        print(
            f"  >> Found {len(entity_groups)} entity groups spanning multiple datasets")
        if entity_groups:
            if target_entity_ids is not None:
                # Filter to test set entities
                selected_groups = []
                for group in entity_groups:
                    for record in group.records:
                        record_id = str(record.get(
                            "_id", record.get("id", "")))
                        if record_id in target_entity_ids:
                            selected_groups.append(group)
                            break
                print(
                    f"  >> Filtered to {len(selected_groups)} groups matching test set IDs (will reuse for all variants)")
            else:
                # Select well-known entities
                selected_groups = select_well_known_entities(
                    entity_groups,
                    llm,
                    n_select=FUSION_VALIDATION_N_ENTITIES,
                    sample_size=FUSION_VALIDATION_SAMPLE_SIZE,
                )
                print(
                    f"  >> Selected {len(selected_groups)} well-known entities (will reuse for all variants)")

    for use_web_search, omit_target, subdir_name in variants:
        variant_dir = dirs.fusion_validation.parent / subdir_name
        variant_label = f"web_search={use_web_search}, omit_target={omit_target}"
        print(f"\n--- Generating variant: {subdir_name} ({variant_label}) ---")

        cached = load_fusion_validation_from_cache(variant_dir) is not None
        with track_step(f"Step 7.5: Fusion Validation ({subdir_name})", cached=cached):
            cached_val = load_fusion_validation_from_cache(variant_dir)
            if cached_val is not None:
                print(f"Loaded cached: {len(cached_val)} records")
                continue

            validation_set = generate_fusion_validation_set(
                correspondences=all_corr,
                datasets=results,
                chat_model=llm,
                n_entities=FUSION_VALIDATION_N_ENTITIES,
                sample_size=FUSION_VALIDATION_SAMPLE_SIZE,
                output_dir=variant_dir,
                use_web_search=use_web_search,
                omit_target_attribute=omit_target,
                target_entity_ids=target_entity_ids,
                identifying_columns=identifying_columns if omit_target else None,
                selected_groups=selected_groups,
            )
            print(f"Generated: {len(validation_set)} records")

            if not validation_set.empty:
                ground_truth_df, _ = convert_to_tabular_format(
                    validation_set, output_dir=variant_dir
                )
                print(f"Tabular format: {len(ground_truth_df)} entities")


def step7_6_end_to_end_metrics(
    dirs: PipelineDirectories,
    results,
    track_step,
):
    """Step 7.6: Calculate End-to-End Integration Metrics"""
    print("\n=== Step 7.6: End-to-End Integration Metrics ===")

    fused_path = dirs.fusion / "fused_clean.csv"
    if not fused_path.exists():
        print("No fused output found; skipping metrics.")
        return None

    # Look for fusion debug file (may be in optimization or main fusion dir)
    debug_file = None
    optimization_dir = dirs.fusion / "optimization"
    if optimization_dir.exists():
        # Find the most recent fusion_debug.jsonl in optimization subdirs
        for case_dir in optimization_dir.iterdir():
            if case_dir.is_dir():
                candidate = case_dir / "fusion_debug.jsonl"
                if candidate.exists():
                    debug_file = candidate
                    break
    if debug_file is None:
        # Check main fusion directory
        main_debug = dirs.fusion / "fusion_debug.jsonl"
        if main_debug.exists():
            debug_file = main_debug

    with track_step("Step 7.6: End-to-End Metrics"):
        fused_df = pd.read_csv(fused_path)

        # Also load the full fused file with metadata columns for metrics
        full_fused_path = dirs.fusion / "fused.csv"
        if not full_fused_path.exists():
            # Try optimization directory
            if optimization_dir.exists():
                for case_dir in optimization_dir.iterdir():
                    if case_dir.is_dir():
                        candidate = case_dir / "fused.csv"
                        if candidate.exists():
                            full_fused_path = candidate
                            break

        if full_fused_path.exists():
            fused_df_with_metadata = pd.read_csv(full_fused_path)
        else:
            fused_df_with_metadata = fused_df

        metrics = calculate_structural_metrics(
            datasets=results,
            fused_df=fused_df_with_metadata,
            debug_file=debug_file,
        )

        # Save metrics
        metrics_path = dirs.output / "end_to_end_metrics.json"
        metrics.save(metrics_path)
        print(f"Saved metrics to {metrics_path}")

        # Print and save end-to-end report
        report_text = generate_end_to_end_report(metrics)
        print(report_text)

        # Save report files
        txt_path, csv_path = save_end_to_end_report(metrics, dirs.output)
        print(f"\nSaved report to: {txt_path}")
        print(f"Saved report CSV to: {csv_path}")

        # Print per-source statistics
        print("\n--- Per-Source Statistics ---")
        print(metrics.per_source_table().to_string(index=False))

    return metrics


def step7_7_fusion_optimization(
    args,
    dirs: PipelineDirectories,
    results,
    fusion_correspondences,
    llm,
    track_step,
):
    """Step 7.7: Fusion Rule Optimization - runs multiple strategies and picks the best."""
    print("\n=== Step 7: Data Fusion (Optimization) ===")

    if not fusion_correspondences:
        print("No correspondences; skipping fusion optimization.")
        return None

    all_corr = pd.concat(fusion_correspondences, ignore_index=True)
    all_corr = all_corr.drop_duplicates(
        subset=["id1", "id2"]).reset_index(drop=True)

    # Load validation sets from all available modes
    validation_sets = {}
    mode_dirs = {
        "llm": dirs.fusion / "validation_llm",
        "llm_omit": dirs.fusion / "validation_llm_omit",
        "web": dirs.fusion / "validation_web",
        "web_omit": dirs.fusion / "validation_web_omit",
    }

    for mode, mode_dir in mode_dirs.items():
        val_path = mode_dir / "fusion_validation_set.csv"
        if val_path.exists():
            validation_sets[mode] = pd.read_csv(val_path)
            print(
                f"  Loaded validation set: {mode} ({len(validation_sets[mode])} records)")
        else:
            print(f"  Validation set not found: {mode}")

    if not validation_sets:
        print("No validation sets available; skipping fusion optimization.")
        print(
            "Run with --fusion-val-generation-mode all first to generate validation sets.")
        return None

    # Load test set if provided (for final evaluation only, not optimization)
    test_set = None
    if args.test_dir:
        test_dir = Path(args.test_dir)
        # Look for test_set.xml file (standard format)
        test_xml = test_dir / "test_set.xml"
        if test_xml.exists():
            test_set = load_xml(test_xml, nested_handling="aggregate")
            print(
                f"  Loaded test set: {len(test_set)} records from {test_xml}")
        else:
            # Fallback to CSV/JSON files
            test_files = list(test_dir.glob("*.csv")) + \
                list(test_dir.glob("*.json"))
            if test_files:
                test_dfs = []
                for tf in test_files:
                    if tf.suffix == ".csv":
                        test_dfs.append(pd.read_csv(tf))
                    elif tf.suffix == ".json":
                        test_dfs.append(pd.read_json(tf))
                if test_dfs:
                    test_set = pd.concat(test_dfs, ignore_index=True)
                    print(
                        f"  Loaded test set: {len(test_set)} records from {len(test_files)} files")

    # Normalize test set using same normalization as Step 1 (including taxonomy mapping)
    if test_set is not None:
        import json
        schema_path = Path(args.schema)
        with open(schema_path) as f:
            target_schema = json.load(f)

        spec = load_normalization_spec(target_schema)
        result = transform_dataframe(
            test_set,
            spec,
            chat_model=llm,  # For taxonomy mapping
            taxonomy_cache_dir=str(dirs.schema_matching),  # Reuse Step 1's cache
            schema_base_path=str(schema_path.parent),  # For relative taxonomy paths
        )
        test_set = result.dataframe
        print(f"  Normalized test set: {len(test_set)} records")

    config = FusionConfig(
        id_column="id",
        include_singletons=FUSION_INCLUDE_SINGLETONS,
        debug=False,  # Disable verbose fusion logging; evaluation logs are separate
        use_llm=True,
    )

    optimization_dir = dirs.fusion / "optimization"
    optimization_dir.mkdir(parents=True, exist_ok=True)

    # Determine which cases to run (default to all if not specified)
    fusion_case = args.fusion_case or "all"
    if fusion_case == "all":
        cases_to_run = ["heuristic", "llm_no_val", "llm_val",
                        "llm_val_stats", "heuristic_stats", "iterative"]
    else:
        cases_to_run = [fusion_case]

    with track_step(f"Step 7: Fusion Optimization ({fusion_case})"):
        case_results, best_case_key, best_case_dir = run_all_fusion_cases(
            datasets=results,
            validation_sets=validation_sets,
            correspondences=all_corr,
            chat_model=llm,
            config=config,
            output_dir=optimization_dir,
            iterations=args.fusion_iterations,
            primary_validation_mode=args.fusion_primary_val_mode,
            cases=cases_to_run,
            test_set=test_set,
        )

    # Print summary
    print("\n--- Fusion Optimization Summary ---")
    for case_name, result in case_results.items():
        print(f"\n{case_name}:")
        for mode, acc in result.accuracy_by_mode.items():
            print(f"  {mode}: {acc:.1%}")

    # Create fused_clean.csv from the best case
    if best_case_dir and best_case_dir.exists():
        best_fused_path = best_case_dir / "fused.csv"
        if best_fused_path.exists():
            print(f"\nCreating fused_clean.csv from best case: {best_case_key}")
            best_fused = pd.read_csv(best_fused_path)

            # Remove metadata columns for clean version
            metadata_cols = ["_fusion_source_datasets", "_fusion_confidence", "_fusion_metadata"]
            clean_cols = [c for c in best_fused.columns if c not in metadata_cols]
            best_fused[clean_cols].to_csv(dirs.fusion / "fused_clean.csv", index=False)

            # Also copy the fusion rules used
            best_rules_path = best_case_dir / "fusion_rules.json"
            if best_rules_path.exists():
                import shutil
                shutil.copy(best_rules_path, dirs.fusion / "fusion_rules.json")

            print(f"  Saved: {dirs.fusion / 'fused_clean.csv'}")
        else:
            print(f"Warning: Best case fused output not found at {best_fused_path}")

    return case_results


def step7_8_provided_validation_comparison(
    args,
    dirs: PipelineDirectories,
    results,
    fusion_correspondences,
    llm,
    track_step,
):
    """Step 7.8: Compare auto-generated vs provided validation sets for fusion optimization.

    This step:
    1. Loads the provided validation_set.xml from the test directory
    2. Converts it to the CSV format used by fusion optimization
    3. Runs fusion optimization using the provided validation set
    4. Compares results between auto-generated and provided validation approaches
    5. Saves comparison metrics and both fusion outputs
    """
    if not args.compare_provided_validation:
        return None

    print("\n=== Step 7.8: Provided Validation Comparison ===")

    if not args.test_dir:
        print("Warning: --test-dir required for --compare-provided-validation")
        return None

    # Check for provided validation set
    test_dir = Path(args.test_dir)
    provided_val_path = test_dir / "validation_set.xml"
    if not provided_val_path.exists():
        print(f"Warning: No validation_set.xml found at {provided_val_path}")
        return None

    if not fusion_correspondences:
        print("No correspondences; skipping provided validation comparison.")
        return None

    all_corr = pd.concat(fusion_correspondences, ignore_index=True)
    all_corr = all_corr.drop_duplicates(subset=["id1", "id2"]).reset_index(drop=True)

    # Load and normalize the provided validation set
    print(f"Loading provided validation set from {provided_val_path}...")
    provided_val_raw = load_xml(provided_val_path, nested_handling="aggregate")
    print(f"  Loaded {len(provided_val_raw)} entities from provided validation set")

    # Normalize using same normalization as Step 1
    import json as json_module
    schema_path = Path(args.schema)
    with open(schema_path) as f:
        target_schema = json_module.load(f)

    spec = load_normalization_spec(target_schema)
    result = transform_dataframe(
        provided_val_raw,
        spec,
        chat_model=llm,
        taxonomy_cache_dir=str(dirs.schema_matching),
        schema_base_path=str(schema_path.parent),
    )
    provided_val_normalized = result.dataframe
    print(f"  Normalized: {len(provided_val_normalized)} entities")

    # Convert to CSV format for fusion optimization
    print("Converting provided validation set to optimization format...")
    provided_val_csv = convert_provided_validation_to_csv_format(
        validation_xml_df=provided_val_normalized,
        datasets=results,
        correspondences=all_corr,
        id_column="id",
    )
    print(f"  Converted: {len(provided_val_csv)} validation records")

    if provided_val_csv.empty:
        print("Warning: No validation records could be extracted from provided validation set")
        return None

    # Save the converted validation set for inspection
    provided_val_dir = dirs.fusion / "validation_provided"
    provided_val_dir.mkdir(parents=True, exist_ok=True)
    provided_val_csv.to_csv(provided_val_dir / "fusion_validation_set.csv", index=False)

    # Also create ground truth in tabular format (for evaluation)
    from PyDI.pipeline.fusion_validation_generation import convert_to_tabular_format
    gt_df, _ = convert_to_tabular_format(provided_val_csv, output_dir=provided_val_dir)
    print(f"  Ground truth: {len(gt_df)} entities")

    # Create validation_sets dict for provided validation
    # We'll use "provided" as the mode name
    provided_validation_sets = {"provided": provided_val_csv}

    # Also load the auto-generated validation sets for comparison
    auto_validation_sets = {}
    mode_dirs = {
        "llm": dirs.fusion / "validation_llm",
        "llm_omit": dirs.fusion / "validation_llm_omit",
        "web": dirs.fusion / "validation_web",
        "web_omit": dirs.fusion / "validation_web_omit",
    }
    for mode, mode_dir in mode_dirs.items():
        val_path = mode_dir / "fusion_validation_set.csv"
        if val_path.exists():
            auto_validation_sets[mode] = pd.read_csv(val_path)

    # Load test set for final evaluation
    test_set = None
    test_xml = test_dir / "test_set.xml"
    if test_xml.exists():
        test_set = load_xml(test_xml, nested_handling="aggregate")
        # Normalize test set
        test_result = transform_dataframe(
            test_set,
            spec,
            chat_model=llm,
            taxonomy_cache_dir=str(dirs.schema_matching),
            schema_base_path=str(schema_path.parent),
        )
        test_set = test_result.dataframe
        print(f"  Loaded test set: {len(test_set)} records")

    config = FusionConfig(
        id_column="id",
        include_singletons=FUSION_INCLUDE_SINGLETONS,
        debug=False,
        use_llm=True,
    )

    # Determine which cases to run
    fusion_case = args.fusion_case or "all"
    if fusion_case == "all":
        cases_to_run = ["heuristic", "llm_no_val", "llm_val",
                        "llm_val_stats", "heuristic_stats", "iterative"]
    else:
        cases_to_run = [fusion_case]

    # Run fusion optimization using provided validation set
    print("\n--- Running fusion optimization with PROVIDED validation set ---")
    provided_output_dir = dirs.fusion / "optimization_provided_val"
    provided_output_dir.mkdir(parents=True, exist_ok=True)

    with track_step("Step 7.8: Fusion Optimization (Provided Val)"):
        provided_results, provided_best_key, provided_best_dir = run_all_fusion_cases(
            datasets=results,
            validation_sets=provided_validation_sets,
            correspondences=all_corr,
            chat_model=llm,
            config=config,
            output_dir=provided_output_dir,
            iterations=args.fusion_iterations,
            primary_validation_mode="provided",
            cases=cases_to_run,
            test_set=test_set,
        )

    # Load auto-generated results from step 7.7
    auto_output_dir = dirs.fusion / "optimization"
    auto_best_case_path = auto_output_dir / "best_case.json"
    auto_best_info = None
    if auto_best_case_path.exists():
        with open(auto_best_case_path) as f:
            auto_best_info = json_module.load(f)

    # Create comparison summary
    print("\n--- Creating Comparison Summary ---")
    comparison_rows = []

    # Collect results from auto-generated validation
    auto_summary_path = auto_output_dir / "comparison_summary_validation.csv"
    if auto_summary_path.exists():
        auto_summary = pd.read_csv(auto_summary_path)
        for _, row in auto_summary.iterrows():
            comparison_rows.append({
                "validation_source": f"auto_{row['optimization_mode']}",
                "case": row["case"],
                "eval_mode": row["eval_mode"],
                "accuracy": row["accuracy"],
                "correct": row.get("correct", ""),
                "total": row.get("total", ""),
            })

    # Collect results from provided validation
    for result_key, result in provided_results.items():
        for mode, acc in result.accuracy_by_mode.items():
            comparison_rows.append({
                "validation_source": "provided",
                "case": result.case_name,
                "eval_mode": mode,
                "accuracy": acc,
                "correct": "",
                "total": "",
            })

    # Save comparison CSV
    if comparison_rows:
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_path = dirs.fusion / "validation_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"Saved comparison to {comparison_path}")

    # Create best case comparison
    best_comparison = {
        "auto_generated": {
            "best_case_key": auto_best_info.get("best_case_key") if auto_best_info else None,
            "validation_accuracy": auto_best_info.get("best_validation_accuracy") if auto_best_info else None,
            "test_accuracy": auto_best_info.get("best_test_accuracy") if auto_best_info else None,
            "rules": auto_best_info.get("rules") if auto_best_info else None,
        },
        "provided": {
            "best_case_key": provided_best_key,
            "validation_accuracy": provided_results[provided_best_key].accuracy_by_mode.get("provided") if provided_best_key else None,
            "test_accuracy": provided_results[provided_best_key].accuracy_by_mode.get("test") if provided_best_key else None,
            "rules": provided_results[provided_best_key].rules if provided_best_key else None,
        },
    }

    best_comparison_path = dirs.fusion / "best_case_comparison.json"
    with open(best_comparison_path, "w") as f:
        json_module.dump(best_comparison, f, indent=2, default=str)
    print(f"Saved best case comparison to {best_comparison_path}")

    # Generate fusion comparison report with per-attribute accuracy
    auto_best_dir = Path(auto_best_info.get("best_case_dir")) if auto_best_info and auto_best_info.get("best_case_dir") else None
    fusion_report_dir = dirs.output / "reporting" / "fusion"
    comparison_report_path = generate_fusion_comparison_report(
        output_dir=fusion_report_dir,
        auto_dir=auto_best_dir,
        provided_dir=provided_best_dir,
        auto_best_info=auto_best_info,
        provided_best_info={
            "best_case_key": provided_best_key,
            "best_validation_accuracy": provided_results[provided_best_key].accuracy_by_mode.get("provided") if provided_best_key else None,
            "best_test_accuracy": provided_results[provided_best_key].accuracy_by_mode.get("test") if provided_best_key else None,
            "rules": provided_results[provided_best_key].rules if provided_best_key else None,
        } if provided_best_key else None,
    )
    if comparison_report_path:
        print(f"Saved fusion comparison report to {comparison_report_path}")

    # Save best provided validation fused output
    if provided_best_dir and provided_best_dir.exists():
        best_fused_path = provided_best_dir / "fused.csv"
        if best_fused_path.exists():
            import shutil
            # Copy to main fusion directory with distinct name
            dest_path = dirs.fusion / "fused_provided_val.csv"
            best_fused = pd.read_csv(best_fused_path)
            metadata_cols = ["_fusion_source_datasets", "_fusion_confidence", "_fusion_metadata"]
            clean_cols = [c for c in best_fused.columns if c not in metadata_cols]
            best_fused[clean_cols].to_csv(dest_path, index=False)
            print(f"Saved provided validation fused output to {dest_path}")

            # Also copy fusion rules
            rules_path = provided_best_dir / "fusion_rules.json"
            if rules_path.exists():
                shutil.copy(rules_path, dirs.fusion / "fusion_rules_provided_val.json")

    # Rename auto-generated output for clarity (if not already named)
    auto_fused_path = dirs.fusion / "fused_clean.csv"
    if auto_fused_path.exists():
        # Determine auto validation mode used
        auto_val_mode = args.fusion_primary_val_mode or "llm_omit"
        dest_auto_path = dirs.fusion / f"fused_auto_{auto_val_mode}.csv"
        if not dest_auto_path.exists():
            import shutil
            shutil.copy(auto_fused_path, dest_auto_path)
            print(f"Copied auto-generated fused output to {dest_auto_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION COMPARISON SUMMARY")
    print("=" * 70)

    if auto_best_info:
        print(f"\nAuto-generated validation (mode: {auto_best_info.get('best_case_key', 'N/A')}):")
        print(f"  Validation accuracy: {auto_best_info.get('best_validation_accuracy', 'N/A'):.1%}" if auto_best_info.get('best_validation_accuracy') else "  Validation accuracy: N/A")
        print(f"  Test accuracy: {auto_best_info.get('best_test_accuracy', 'N/A'):.1%}" if auto_best_info.get('best_test_accuracy') else "  Test accuracy: N/A")

    if provided_best_key:
        prov_val_acc = provided_results[provided_best_key].accuracy_by_mode.get("provided")
        prov_test_acc = provided_results[provided_best_key].accuracy_by_mode.get("test")
        print(f"\nProvided validation (case: {provided_best_key}):")
        print(f"  Validation accuracy: {prov_val_acc:.1%}" if prov_val_acc else "  Validation accuracy: N/A")
        print(f"  Test accuracy: {prov_test_acc:.1%}" if prov_test_acc else "  Test accuracy: N/A")

    return provided_results


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # Setup
    dirs = PipelineDirectories(Path(args.output_dir))
    dirs.create_all()
    setup_logging(dirs.output)

    # Load metrics
    metrics = PipelineMetrics.load(dirs.output / "pipeline_metrics.json")
    _, track_step = create_step_tracker(metrics)

    # LLM
    llm = ChatOpenAI(model=LLM_MODEL)

    # Print source overview (raw data statistics before any transformation)
    print_and_save_source_overview(args, dirs)

    # Pipeline steps
    results = step1_schema_matching(args, dirs, llm, track_step)

    # Generate schema matching report
    generate_schema_matching_report_for_pipeline(
        schema_path=args.schema,
        mappings_dir=dirs.schema_matching / "mappings",
        output_dir=dirs.output,
        normalized_results=results,
        raw_data_dir=args.data_dir,
    )

    # Generate normalization statistics (LLM from cache, builtin from pipeline_stats)
    norm_stats_path = save_normalization_stats(output_dir=dirs.output)
    if norm_stats_path:
        print(f"\n  Saved normalization stats to: {norm_stats_path}")

    pairs, validation_sets, training_sets = step2_validation_training(
        args, dirs, results, llm, track_step
    )

    faiss_candidates = step3_faiss_candidates(
        args, dirs, pairs, results, llm, track_step)

    # Generate estimated labeling cost report
    cost_report_path = save_cost_report(
        results=results,
        candidates_dict=faiss_candidates,
        output_dir=dirs.output,
        model=LLM_MODEL,
    )
    if cost_report_path:
        print(f"\n  Saved labeling cost estimate to: {cost_report_path}")

    # Step 4: Training set comparison (optional)
    # If --compare-training-sets is passed, generates and compares multiple variants
    # and returns the best training sets for each pair
    training_sets = step4_training_comparison(
        args, dirs, pairs, results, validation_sets, training_sets,
        faiss_candidates, llm, track_step
    )

    fusion_correspondences, _ = step5_matcher_optimization(
        dirs, pairs, results, validation_sets, training_sets,
        faiss_candidates, track_step
    )

    # Save cluster size distribution report
    cluster_report_path = save_cluster_size_report(
        fusion_correspondences, num_datasets=len(results), output_dir=dirs.output,
    )
    if cluster_report_path:
        print(f"  Saved cluster size distribution to: {cluster_report_path}")

    # Clean oversized clusters using post-clustering algorithms
    fusion_correspondences, cleaning_report = clean_oversized_clusters(
        fusion_correspondences, num_datasets=len(results),
    )
    if cleaning_report["best_strategy"] is not None:
        report_dir = save_cluster_cleaning_report(cleaning_report, dirs.output)
        tex_path = save_cluster_cleaning_latex(
            cleaning_report, num_datasets=len(results), output_dir=dirs.output,
        )
        print(f"  Cluster cleaning reports saved to: {report_dir}")
        print(f"  LaTeX table saved to: {tex_path}")

    # Step 6: Active learning (skipped if --compare-training-sets was used,
    # since active variant was already generated and compared)
    if not args.compare_training_sets:
        step6_active_learning(
            dirs, results, validation_sets, training_sets,
            faiss_candidates, llm, track_step
        )

    # Step 7: Data Fusion with optimization
    # Runs multiple fusion strategies, evaluates with type-aware matching, and picks the best
    step7_5_fusion_validation(
        args, dirs, results, fusion_correspondences, llm, track_step)

    step7_7_fusion_optimization(
        args, dirs, results, fusion_correspondences, llm, track_step)

    # Step 7.8: Compare auto-generated vs provided validation (optional)
    # If --compare-provided-validation is passed, runs fusion optimization with
    # the provided validation_set.xml and compares to auto-generated results
    step7_8_provided_validation_comparison(
        args, dirs, results, fusion_correspondences, llm, track_step)

    step7_6_end_to_end_metrics(dirs, results, track_step)

    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(metrics.summary_table().to_string(index=False))
    metrics.summary_table().to_csv(dirs.output / "pipeline_metrics.csv", index=False)
    print(f"\nTotal Runtime: {metrics.total_runtime():.1f}s")
    print(f"Total Tokens: {metrics.total_tokens():,}")
    print(f"Total Cost: ${metrics.total_cost():.4f}")
    print("\n=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
    exectution_time = time.time() - start_time
    print(f"\nTotal execution time: {exectution_time:.1f} seconds")