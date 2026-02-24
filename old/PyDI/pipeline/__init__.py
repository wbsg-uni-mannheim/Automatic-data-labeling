"""
Pipeline module for end-to-end data integration.

Usage:
    from PyDI.pipeline import run_pipeline

    # Process all data files in a directory
    results = run_pipeline(
        data_dir="usecases/input/movies/data",
        schema_path="usecases/input/movies/target_schema.json",
        chat_model=llm,
    )

    # Or use individual functions
    from PyDI.pipeline import auto_match_schema, auto_normalize

    mapping = auto_match_schema(source_df, target_schema, chat_model=llm)
    normalized_df = auto_normalize(source_df, mapping, target_schema)

Module structure:
- entity_resolution.py - Core entity resolution functions (blocking, labeling)
- blocking_optimization.py - Blocker optimization and evaluation
- matching_optimization.py - Matcher optimization
- labeled_set_generation.py - Training and validation set generation
- optimization.py - Re-exports from above for backward compatibility
- fusion.py - Data fusion
- schema_matching.py - Schema matching
- normalization.py - Data normalization
- run.py - Pipeline runner
"""

from .schema_matching import auto_match_schema
from .normalization import auto_normalize
from .run import run_pipeline, discover_files
from .entity_resolution import (
    select_dataset_pairs,
    select_blocking_columns,
    parse_blocking_strategy,
    generate_candidates_multi_blocker,
    label_candidates_with_llm,
    generate_validation_set,
)
from .labeled_set_generation import (
    load_or_generate_validation_set,
    load_or_generate_training_set,
    load_labeled_set_from_cache,
    save_labeled_set_to_cache,
    generate_labeled_set,
    collect_candidates_from_blockers,
    drop_overlapping_pairs,
    run_active_learning,
    augment_training_set_with_disagreements,
    find_matcher_disagreements,
    generate_completely_random_pairs,
)
from .training_comparison import (
    TrainingSetVariant,
    ComparisonResult,
    generate_training_variants,
    compare_training_variants,
)
from .optimization import (
    get_default_blocker_specs,
    optimize_matching,
    optimize_blocking,
    evaluate_blocker_types,
)
from .fusion import (
    FusionConfig,
    auto_build_fusion_strategy,
    select_fusion_rules_with_llm,
    run_data_fusion,
)
from .similarity_set_generation import (
    SimilarityBasedSetGenerator,
    generate_similarity_based_labeled_set,
    generate_similarity_based_validation_set,
    generate_similarity_based_training_set,
    load_or_generate_similarity_labeled_set,
    load_or_generate_similarity_validation_set,
    load_or_generate_similarity_training_set,
    generate_faiss_candidates,
)
from .end_to_end_metrics import (
    EndToEndMetrics,
    SourceStats,
    calculate_structural_metrics,
    calculate_density,
    generate_end_to_end_report,
    save_end_to_end_report,
)
from .fusion_optimization import (
    SourceAccuracyStats,
    FusionCaseResult,
    compute_source_accuracy_from_validation_set,
    evaluate_fusion_accuracy,
    evaluate_fusion_against_test_set,
    run_fusion_case,
    run_all_fusion_cases,
)
from .labeling_cost_estimation import (
    estimate_labeling_cost_for_pair,
    estimate_labeling_costs,
    generate_cost_report,
    save_cost_report,
)
from .normalization_stats import (
    generate_normalization_stats,
    save_normalization_stats,
)
from .fusion_comparison import (
    generate_fusion_comparison_report,
)
from .cluster_cleaning import (
    clean_oversized_clusters,
    save_cluster_cleaning_report,
    save_cluster_cleaning_latex,
)

__all__ = [
    # Pipeline runner
    "run_pipeline",
    "discover_files",
    # Schema matching and normalization
    "auto_match_schema",
    "auto_normalize",
    # Entity resolution - core functions
    "select_dataset_pairs",
    "select_blocking_columns",
    "parse_blocking_strategy",
    "generate_candidates_multi_blocker",
    "label_candidates_with_llm",
    "generate_validation_set",
    # Labeled set generation
    "load_or_generate_validation_set",
    "load_or_generate_training_set",
    "load_labeled_set_from_cache",
    "save_labeled_set_to_cache",
    "generate_labeled_set",
    "collect_candidates_from_blockers",
    "drop_overlapping_pairs",
    # Optimization
    "get_default_blocker_specs",
    "optimize_matching",
    "optimize_blocking",
    "evaluate_blocker_types",
    # Active learning
    "run_active_learning",
    "augment_training_set_with_disagreements",
    "find_matcher_disagreements",
    # Data fusion
    "FusionConfig",
    "auto_build_fusion_strategy",
    "select_fusion_rules_with_llm",
    "run_data_fusion",
    # Similarity-based set generation
    "SimilarityBasedSetGenerator",
    "generate_similarity_based_labeled_set",
    "generate_similarity_based_validation_set",
    "generate_similarity_based_training_set",
    "load_or_generate_similarity_labeled_set",
    "load_or_generate_similarity_validation_set",
    "load_or_generate_similarity_training_set",
    # FAISS candidate generation
    "generate_faiss_candidates",
    # End-to-end metrics
    "EndToEndMetrics",
    "SourceStats",
    "calculate_structural_metrics",
    "calculate_density",
    "generate_end_to_end_report",
    "save_end_to_end_report",
    # Fusion optimization
    "SourceAccuracyStats",
    "FusionCaseResult",
    "compute_source_accuracy_from_validation_set",
    "evaluate_fusion_accuracy",
    "evaluate_fusion_against_test_set",
    "run_fusion_case",
    "run_all_fusion_cases",
    # Training comparison
    "TrainingSetVariant",
    "ComparisonResult",
    "generate_training_variants",
    "compare_training_variants",
    "generate_completely_random_pairs",
    # Labeling cost estimation
    "estimate_labeling_cost_for_pair",
    "estimate_labeling_costs",
    "generate_cost_report",
    "save_cost_report",
    # Normalization stats
    "generate_normalization_stats",
    "save_normalization_stats",
    # Fusion comparison
    "generate_fusion_comparison_report",
    # Cluster cleaning
    "clean_oversized_clusters",
    "save_cluster_cleaning_report",
    "save_cluster_cleaning_latex",
]
