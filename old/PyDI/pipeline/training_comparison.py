"""
Training set comparison for entity matching.

This module generates multiple training set variants and compares their
performance on a validation set to identify the best configuration.

Variants:
- faiss_small: Small FAISS-based training set
- faiss_large: Large FAISS-based training set (same size as active learning target)
- active: Active learning augmented training set
- *_plus_random: Any above with 20% additional random pairs
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..entitymatching import LLMBasedMatcher
from .labeled_set_generation import generate_completely_random_pairs
from .similarity_set_generation import load_or_generate_similarity_training_set
from .labeled_set_generation import run_active_learning
from .matching_optimization import optimize_matching


logger = logging.getLogger(__name__)


@dataclass
class TrainingSetVariant:
    """Container for a training set variant with metadata."""

    name: str
    training_set: pd.DataFrame
    positives: int
    negatives: int
    token_usage: Dict[str, int] = field(default_factory=dict)
    generation_time_sec: float = 0.0


@dataclass
class ComparisonResult:
    """Result of comparing a training variant with a specific model."""

    variant: str
    model: str
    train_total: int
    train_positives: int
    train_negatives: int
    f1: float
    precision: float
    recall: float
    tokens_used: int = 0


def _label_random_pairs(
    random_pairs: pd.DataFrame,
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    chat_model,
    *,
    id_column: str = "id",
    batch_size: int = 25,
    phase: str = "random",
) -> pd.DataFrame:
    """Label random pairs using LLM-based matcher.

    Parameters
    ----------
    random_pairs : pd.DataFrame
        Unlabeled pairs with columns [id1, id2]
    df_left, df_right : pd.DataFrame
        Source datasets
    chat_model
        LLM chat model for labeling
    id_column : str
        ID column name
    batch_size : int
        Number of pairs to label per LLM call
    phase : str
        Phase identifier for token logging

    Returns
    -------
    pd.DataFrame
        Labeled pairs with columns [id1, id2, label]
    """
    if random_pairs.empty:
        return pd.DataFrame(columns=["id1", "id2", "label"])

    # Create matcher for labeling
    matcher = LLMBasedMatcher()

    # Label in batches
    labeled_rows: List[Dict[str, Any]] = []
    for i in range(0, len(random_pairs), batch_size):
        batch = random_pairs.iloc[i : i + batch_size]

        try:
            results = matcher.match(
                df_left=df_left,
                df_right=df_right,
                candidates=batch[["id1", "id2"]],
                id_column=id_column,
                chat_model=chat_model,
                generate_explanations=False,
                parse_strictness="skip",
            )

            if results is not None and not results.empty:
                for _, row in results.iterrows():
                    match_val = row.get("match", False)
                    label = "TRUE" if bool(match_val) else "FALSE"
                    labeled_rows.append({"id1": row["id1"], "id2": row["id2"], "label": label})
        except Exception as e:
            logger.warning(f"Failed to label batch {i}-{i+batch_size}: {e}")

    return pd.DataFrame(labeled_rows)


def generate_training_variants(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    left_name: str,
    right_name: str,
    validation_set: pd.DataFrame,
    faiss_candidates: pd.DataFrame,
    chat_model,
    output_dir: Path,
    *,
    id_column: str = "id",
    small_target_positives: int = 34,
    small_target_negatives: int = 66,
    large_target_positives: int = 200,
    large_target_negatives: int = 400,
    random_sample_ratio: float = 0.2,
    force_regenerate: bool = False,
    existing_training_set: Optional[pd.DataFrame] = None,
) -> Dict[str, TrainingSetVariant]:
    """Generate all training set variants for comparison.

    Generates:
    - faiss_small: Small FAISS-based set
    - faiss_large: Large FAISS-based set (matching active learning targets)
    - active: Active learning augmented set
    - *_plus_random: Each of the above with 20% additional random pairs

    Parameters
    ----------
    df_left, df_right : pd.DataFrame
        Source datasets
    left_name, right_name : str
        Dataset names for caching
    validation_set : pd.DataFrame
        Validation set to exclude from training
    faiss_candidates : pd.DataFrame
        Pre-generated FAISS candidates
    chat_model
        LLM model for labeling
    output_dir : Path
        Directory for output files
    id_column : str
        ID column name
    small_target_positives : int
        Target positives for faiss_small
    small_target_negatives : int
        Target negatives for faiss_small
    large_target_positives : int
        Target positives for faiss_large and active
    large_target_negatives : int
        Target negatives for faiss_large and active
    random_sample_ratio : float
        Ratio of random pairs to add for *_plus_random variants
    force_regenerate : bool
        If True, regenerate even if cached
    existing_training_set : pd.DataFrame, optional
        If provided, reuse this as faiss_small instead of generating fresh.
        This avoids redundant LLM calls when step 2 already generated a
        training set with the same target sizes.

    Returns
    -------
    Dict[str, TrainingSetVariant]
        Dictionary mapping variant name to TrainingSetVariant
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants: Dict[str, TrainingSetVariant] = {}

    # Helper to check cache
    def _load_cached(variant_name: str) -> Optional[pd.DataFrame]:
        if force_regenerate:
            return None
        cache_path = output_dir / f"similarity_training_{variant_name}_{left_name}_{right_name}.csv"
        if cache_path.exists():
            logger.info(f"Loading cached variant: {variant_name}")
            return pd.read_csv(cache_path, dtype={"label": str})
        return None

    def _save_variant(variant_name: str, df: pd.DataFrame) -> None:
        cache_path = output_dir / f"similarity_training_{variant_name}_{left_name}_{right_name}.csv"
        df.to_csv(cache_path, index=False)
        logger.info(f"Saved variant {variant_name} to {cache_path}")

    def _count_labels(df: pd.DataFrame) -> Tuple[int, int]:
        if df.empty:
            return 0, 0
        labels = df["label"].astype(str).str.upper()
        pos = (labels == "TRUE").sum()
        neg = (labels == "FALSE").sum()
        return pos, neg

    # 1. Generate faiss_small (or reuse existing training set from step 2)
    print("\n--- Generating faiss_small variant ---")
    start_time = time.time()

    cached = _load_cached("faiss_small")
    if cached is not None:
        faiss_small_df = cached
        print("  Loaded from cache")
    elif existing_training_set is not None and not existing_training_set.empty:
        # Reuse the training set from step 2 (avoids redundant LLM calls)
        faiss_small_df = existing_training_set.copy()
        _save_variant("faiss_small", faiss_small_df)
        print("  Reused existing training set from step 2")
    else:
        faiss_small_df = load_or_generate_similarity_training_set(
            df_left=df_left,
            df_right=df_right,
            left_name=left_name,
            right_name=right_name,
            chat_model=chat_model,
            output_dir=output_dir,
            id_column=id_column,
            target_size=small_target_positives + small_target_negatives,
            target_positives=small_target_positives,
            exclude_pairs=validation_set,
            retrieval_method="faiss",
            query_order="similarity",
        )
        _save_variant("faiss_small", faiss_small_df)

    pos, neg = _count_labels(faiss_small_df)
    variants["faiss_small"] = TrainingSetVariant(
        name="faiss_small",
        training_set=faiss_small_df,
        positives=pos,
        negatives=neg,
        generation_time_sec=time.time() - start_time,
    )
    print(f"  faiss_small: {len(faiss_small_df)} pairs ({pos} pos, {neg} neg)")

    # 2. Generate faiss_large (continue from faiss_small)
    print("\n--- Generating faiss_large variant ---")
    start_time = time.time()

    cached = _load_cached("faiss_large")
    if cached is not None:
        # Check if cached has enough pairs (with 5% tolerance)
        cached_pos, cached_neg = _count_labels(cached)
        pos_tolerance = max(1, int(large_target_positives * 0.05))
        neg_tolerance = max(1, int(large_target_negatives * 0.05))

        pos_ok = cached_pos >= large_target_positives - pos_tolerance
        neg_ok = cached_neg >= large_target_negatives - neg_tolerance

        if pos_ok and neg_ok:
            faiss_large_df = cached
            if cached_pos >= large_target_positives and cached_neg >= large_target_negatives:
                print("  Loaded from cache")
            else:
                print(f"  Loaded from cache (close enough: {cached_pos} pos, {cached_neg} neg)")
        else:
            print(f"  Cache has {cached_pos} pos, {cached_neg} neg (need {large_target_positives} pos, {large_target_negatives} neg) - continuing")
            # Use cached as starting point
            starting_set = cached
            starting_pos, starting_neg = cached_pos, cached_neg
            cached = None  # Signal to continue generating
    else:
        # Start from faiss_small
        starting_set = faiss_small_df
        starting_pos, starting_neg = _count_labels(faiss_small_df)
        print(f"  Starting from faiss_small: {starting_pos} pos, {starting_neg} neg")

    if cached is None:
        # Calculate how many more pairs we need
        additional_positives = max(0, large_target_positives - starting_pos)
        additional_negatives = max(0, large_target_negatives - starting_neg)

        if additional_positives == 0 and additional_negatives == 0:
            faiss_large_df = starting_set
        else:
            # Exclude validation set AND already-labeled pairs from starting_set
            exclude_combined = pd.concat([validation_set, starting_set[["id1", "id2", "label"]]], ignore_index=True)
            print(len(exclude_combined), "pairs to exclude when generating additional pairs")

            # Generate additional pairs
            try:
                additional_pairs = load_or_generate_similarity_training_set(
                    df_left=df_left,
                    df_right=df_right,
                    left_name=left_name,
                    right_name=right_name,
                    chat_model=chat_model,
                    output_dir=output_dir,
                    id_column=id_column,
                    target_size=additional_positives + additional_negatives,
                    target_positives=additional_positives,
                    exclude_pairs=exclude_combined,
                    retrieval_method="faiss",
                    query_order="similarity",
                    force_regenerate=True,
                )
                # Combine starting set with additional pairs
                faiss_large_df = pd.concat([starting_set, additional_pairs], ignore_index=True)
            except ValueError as e:
                # If no more pairs can be labeled, use what we have
                logger.warning(f"Could not generate additional pairs: {e}. Using existing {starting_pos} pos, {starting_neg} neg.")
                faiss_large_df = starting_set

        _save_variant("faiss_large", faiss_large_df)

    pos, neg = _count_labels(faiss_large_df)
    variants["faiss_large"] = TrainingSetVariant(
        name="faiss_large",
        training_set=faiss_large_df,
        positives=pos,
        negatives=neg,
        generation_time_sec=time.time() - start_time,
    )
    print(f"  faiss_large: {len(faiss_large_df)} pairs ({pos} pos, {neg} neg)")

    # 3. Generate active (active learning on faiss_small)
    print("\n--- Generating active variant ---")
    start_time = time.time()

    cached = _load_cached("active")
    if cached is not None:
        active_df = cached
    else:
        active_df, was_augmented, _ = run_active_learning(
            df_left=df_left,
            df_right=df_right,
            left_name=left_name,
            right_name=right_name,
            training_set=faiss_small_df.copy(),
            validation_set=validation_set,
            candidates=faiss_candidates,
            chat_model=chat_model,
            output_dir=output_dir,
            id_column=id_column,
            target_positives=large_target_positives,
            target_negatives=large_target_negatives,
            max_total_labels=5000,
            labels_per_iteration=100,
            max_candidates=5000,
            max_iterations=30,
        )
        _save_variant("active", active_df)

    pos, neg = _count_labels(active_df)
    variants["active"] = TrainingSetVariant(
        name="active",
        training_set=active_df,
        positives=pos,
        negatives=neg,
        generation_time_sec=time.time() - start_time,
    )
    print(f"  active: {len(active_df)} pairs ({pos} pos, {neg} neg)")

    # 4. Generate *_plus_random variants
    base_variants = ["faiss_small", "faiss_large", "active"]
    for base_name in base_variants:
        variant_name = f"{base_name}_plus_random"
        print(f"\n--- Generating {variant_name} variant ---")
        start_time = time.time()

        cached = _load_cached(variant_name)
        if cached is not None:
            plus_random_df = cached
        else:
            base_df = variants[base_name].training_set
            n_random = int(len(base_df) * random_sample_ratio)

            # Collect all existing pairs to exclude
            all_existing = pd.concat(
                [validation_set, base_df], ignore_index=True
            )

            # Generate random pairs
            random_pairs = generate_completely_random_pairs(
                df_left=df_left,
                df_right=df_right,
                n_pairs=n_random,
                exclude_pairs=all_existing,
                id_column=id_column,
            )

            if not random_pairs.empty:
                # Label random pairs
                labeled_random = _label_random_pairs(
                    random_pairs,
                    df_left,
                    df_right,
                    chat_model,
                    id_column=id_column,
                    phase="random",
                )

                # Combine with base
                plus_random_df = pd.concat([base_df, labeled_random], ignore_index=True)
                plus_random_df = plus_random_df.drop_duplicates(
                    subset=["id1", "id2"]
                ).reset_index(drop=True)
            else:
                plus_random_df = base_df.copy()

            _save_variant(variant_name, plus_random_df)

        pos, neg = _count_labels(plus_random_df)
        variants[variant_name] = TrainingSetVariant(
            name=variant_name,
            training_set=plus_random_df,
            positives=pos,
            negatives=neg,
            generation_time_sec=time.time() - start_time,
        )
        print(f"  {variant_name}: {len(plus_random_df)} pairs ({pos} pos, {neg} neg)")

    return variants


def _load_provided_training_set(
    entitymatching_dir: Path,
    left_name: str,
    right_name: str,
) -> Optional[pd.DataFrame]:
    """Load provided/official training set from entitymatching directory.

    Looks for files matching patterns like:
    - {left_name}_2_{right_name}_train.csv
    - {right_name}_2_{left_name}_train.csv (swapped order)

    Parameters
    ----------
    entitymatching_dir : Path
        Directory containing entitymatching files
    left_name, right_name : str
        Dataset names

    Returns
    -------
    pd.DataFrame or None
        Loaded training set or None if not found
    """
    if not entitymatching_dir or not entitymatching_dir.exists():
        return None

    # Try both orderings
    patterns = [
        entitymatching_dir / f"{left_name}_2_{right_name}_train.csv",
        entitymatching_dir / f"{right_name}_2_{left_name}_train.csv",
    ]

    for train_path in patterns:
        if train_path.exists():
            logger.info(f"Loading provided training set from {train_path}")

            # Detect if file has header
            with open(train_path, "r") as f:
                first_line = f.readline().strip()

            if "id1" in first_line.lower() or "label" in first_line.lower():
                df = pd.read_csv(train_path)
                df.columns = [c.lower() for c in df.columns]
            else:
                df = pd.read_csv(train_path, header=None, names=["id1", "id2", "label"])

            # Normalize
            df["id1"] = df["id1"].astype(str)
            df["id2"] = df["id2"].astype(str)
            df["label"] = df["label"].astype(str).str.upper().str.strip()

            return df

    return None


def compare_training_variants(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    variants: Dict[str, TrainingSetVariant],
    validation_set: pd.DataFrame,
    output_dir: Path,
    *,
    id_column: str = "id",
    test_set: Optional[pd.DataFrame] = None,
    left_name: Optional[str] = None,
    right_name: Optional[str] = None,
    entitymatching_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, str]:
    """Compare training variants by training matchers and evaluating on test set.

    Trains multiple classifier types on each variant and evaluates on the
    test set (or validation set if no test set provided) to find the best
    performing combination.

    Parameters
    ----------
    df_left, df_right : pd.DataFrame
        Source datasets
    variants : Dict[str, TrainingSetVariant]
        Training set variants to compare
    validation_set : pd.DataFrame
        Validation set (used if no test_set provided)
    output_dir : Path
        Directory for output files
    id_column : str
        ID column name
    test_set : pd.DataFrame, optional
        Ground truth test set for evaluation. If provided, this is used instead
        of the validation set. Should have columns: id1, id2, label (TRUE/FALSE).
    left_name, right_name : str, optional
        Dataset names (required if entitymatching_dir is provided)
    entitymatching_dir : Path, optional
        Directory containing official/provided training sets (*_train.csv files).
        If provided, loads these as the "provided" variant for comparison.

    Returns
    -------
    Tuple[pd.DataFrame, str]
        (comparison_results_df, best_variant_name)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    left_ids = set(df_left[id_column].astype(str))
    right_ids = set(df_right[id_column].astype(str))

    def _fix_id_ordering(df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Ensure id1 matches df_left and id2 matches df_right."""
        df = df.copy()
        df["id1"] = df["id1"].astype(str)
        df["id2"] = df["id2"].astype(str)
        sample_id1s = df["id1"].head(10).tolist()
        sample_id2s = df["id2"].head(10).tolist()
        id1_in_left = sum(1 for x in sample_id1s if x in left_ids)
        id1_in_right = sum(1 for x in sample_id1s if x in right_ids)
        id2_in_left = sum(1 for x in sample_id2s if x in left_ids)
        id2_in_right = sum(1 for x in sample_id2s if x in right_ids)
        if id1_in_right > id1_in_left and id2_in_left > id2_in_right:
            logger.info(f"Swapping id1/id2 columns in {name} to match df_left/df_right ordering")
            df["id1"], df["id2"] = df["id2"].copy(), df["id1"].copy()
        return df

    # Always select on validation set
    eval_set = _fix_id_ordering(validation_set, "validation_set")
    print(f"Selecting best variant on validation set ({len(eval_set)} pairs)")

    # Prepare test set for held-out evaluation if provided
    has_test_set = test_set is not None
    if has_test_set:
        test_eval_set = _fix_id_ordering(test_set, "test_set")
        print(f"Will evaluate best models on held-out test set ({len(test_eval_set)} pairs)")

    # Load provided training set if entitymatching_dir is specified
    if entitymatching_dir and left_name and right_name:
        provided_df = _load_provided_training_set(entitymatching_dir, left_name, right_name)
        if provided_df is not None:
            provided_df = _fix_id_ordering(provided_df, "provided_training_set")

            labels = provided_df["label"].astype(str).str.upper()
            full_pos = (labels == "TRUE").sum()
            full_neg = (labels == "FALSE").sum()
            print(f"  Loaded provided training set: {len(provided_df)} pairs ({full_pos} pos, {full_neg} neg)")

            # Create subsampled variants of the provided training set to match each generated variant's size
            # This allows fair comparison across different training set generation methods
            def _subsample_provided(
                df: pd.DataFrame, target_pos: int, target_neg: int, variant_suffix: str
            ) -> None:
                """Subsample provided training set to match target sizes. Caches to disk."""
                variant_name = f"provided_{variant_suffix}"

                # Check cache first
                cache_path = output_dir / f"provided_{variant_suffix}_{left_name}_{right_name}.csv"
                if cache_path.exists():
                    cached_df = pd.read_csv(cache_path, dtype={"label": str})
                    cached_labels = cached_df["label"].astype(str).str.upper()
                    n_pos = (cached_labels == "TRUE").sum()
                    n_neg = (cached_labels == "FALSE").sum()
                    variants[variant_name] = TrainingSetVariant(
                        name=variant_name,
                        training_set=cached_df,
                        positives=n_pos,
                        negatives=n_neg,
                    )
                    print(f"  Loaded {variant_name} from cache: {len(cached_df)} pairs ({n_pos} pos, {n_neg} neg)")
                    return

                df_labels = df["label"].astype(str).str.upper()
                positives = df[df_labels == "TRUE"]
                negatives = df[df_labels == "FALSE"]

                # Sample up to target sizes (or use all if fewer available)
                n_pos = min(target_pos, len(positives))
                n_neg = min(target_neg, len(negatives))

                sampled_pos = positives.sample(n=n_pos, random_state=42)
                sampled_neg = negatives.sample(n=n_neg, random_state=42)

                subsampled = pd.concat([sampled_pos, sampled_neg], ignore_index=True)
                # Shuffle the subsampled set
                subsampled = subsampled.sample(frac=1.0, random_state=42).reset_index(drop=True)

                # Save to cache
                subsampled.to_csv(cache_path, index=False)

                variants[variant_name] = TrainingSetVariant(
                    name=variant_name,
                    training_set=subsampled,
                    positives=n_pos,
                    negatives=n_neg,
                )
                print(f"  Created {variant_name}: {len(subsampled)} pairs ({n_pos} pos, {n_neg} neg)")

            # Create a subsampled "provided" variant for each generated variant
            for variant_name, variant in list(variants.items()):
                _subsample_provided(
                    provided_df,
                    variant.positives,
                    variant.negatives,
                    variant_name,
                )

    results: List[ComparisonResult] = []
    # Store artifacts keyed by (variant_name, model_type) for test-set evaluation
    variant_artifacts: Dict[Tuple[str, str], Dict[str, Any]] = {}
    best_f1 = -1.0
    best_variant = ""
    best_model = ""

    model_name_map = {
        "logreg": "logreg",
        "rf": "rf",
        "gbdt": "gb",
        "xgboost": "xgb",
        "hist_gbdt": "hist_gb",
    }

    for variant_name, variant in variants.items():
        print(f"\n--- Evaluating {variant_name} (on validation set) ---")
        train_df = variant.training_set.copy()

        if train_df.empty:
            logger.warning(f"Skipping {variant_name}: empty training set")
            continue

        # Shuffle the training set before training to avoid any ordering bias
        train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        try:
            # Use optimize_matching which has proper class balancing
            opt_result = optimize_matching(
                df_left=df_left,
                df_right=df_right,
                validation_set=eval_set,
                id_column=id_column,
                include_rule_based=False,
                include_ml_based=True,
                training_set=train_df,
                include_llm_based=False,
            )

            # Extract results for each model type from optimize_matching output
            all_results_df = opt_result.get("all_results", pd.DataFrame())
            all_artifacts_list = opt_result.get("all_artifacts", [])
            if all_results_df.empty:
                logger.warning(f"No results from optimize_matching for {variant_name}")
                continue

            # Get best result per classifier at optimal threshold
            for classifier_name in all_results_df["classifier"].dropna().unique():
                clf_results = all_results_df[all_results_df["classifier"] == classifier_name]
                if clf_results.empty:
                    continue

                # Get the best threshold for this classifier
                best_idx = clf_results["f1"].idxmax()
                best_row = clf_results.loc[best_idx]

                model_type = model_name_map.get(classifier_name, classifier_name)

                result = ComparisonResult(
                    variant=variant_name,
                    model=model_type,
                    train_total=len(train_df),
                    train_positives=variant.positives,
                    train_negatives=variant.negatives,
                    f1=float(best_row["f1"]),
                    precision=float(best_row["precision"]),
                    recall=float(best_row["recall"]),
                    tokens_used=variant.token_usage.get("total_tokens", 0),
                )
                results.append(result)

                # Store artifacts for potential test-set evaluation
                # best_idx is the position in the sorted results_df from optimize_matching
                if best_idx < len(all_artifacts_list):
                    variant_artifacts[(variant_name, model_type)] = all_artifacts_list[best_idx]

                print(f"  {model_type}: val_F1={result.f1:.3f}, val_P={result.precision:.3f}, val_R={result.recall:.3f}")

                # Track best (selected on validation)
                if result.f1 > best_f1:
                    best_f1 = result.f1
                    best_variant = variant_name
                    best_model = model_type

        except Exception as e:
            import traceback
            print(f"  ERROR: Failed to evaluate {variant_name}: {e}")
            print(traceback.format_exc())
            logger.warning(f"Failed to evaluate {variant_name}: {e}")

    # --- Held-out test set evaluation ---
    # Only evaluate the best model per variant (selected on validation) on the
    # held-out test set. This avoids inflating test scores through selection.
    test_results_map: Dict[Tuple[str, str], Dict[str, float]] = {}
    if has_test_set and variant_artifacts:
        # Find best model per variant based on validation F1
        best_model_per_variant: Dict[str, Tuple[str, float]] = {}
        for r in results:
            prev = best_model_per_variant.get(r.variant)
            if prev is None or r.f1 > prev[1]:
                best_model_per_variant[r.variant] = (r.model, r.f1)

        # Only keep artifacts for the best model of each variant
        test_candidates = {
            (vname, mtype): variant_artifacts[(vname, mtype)]
            for vname, (mtype, _) in best_model_per_variant.items()
            if (vname, mtype) in variant_artifacts
        }

        print(f"\n--- Evaluating best-per-variant on held-out test set ({len(test_eval_set)} pairs) ---")
        for (vname, mtype), artifacts in test_candidates.items():
            clf = artifacts.get("classifier")
            extractor = artifacts.get("feature_extractor")
            if clf is None or extractor is None:
                continue

            try:
                # Extract features for test pairs
                test_pairs = test_eval_set[["id1", "id2"]]
                features_df = extractor.create_features(
                    df_left, df_right, test_pairs, id_column, labels=None
                )

                id_cols = ["id1", "id2", "label"]
                feature_cols = [c for c in features_df.columns if c not in id_cols]

                # Align with test labels (some pairs may be missing if IDs not found)
                features_df["id1"] = features_df["id1"].astype(str)
                features_df["id2"] = features_df["id2"].astype(str)
                test_with_labels = test_eval_set.copy()
                test_with_labels["label_binary"] = (
                    test_with_labels["label"].astype(str).str.upper() == "TRUE"
                ).astype(int)

                merged = features_df.merge(
                    test_with_labels[["id1", "id2", "label_binary"]],
                    on=["id1", "id2"],
                    how="inner",
                )

                if merged.empty:
                    continue

                X_test = merged[feature_cols].values
                y_true = merged["label_binary"].values

                # Use predict_proba with 0.5 threshold if available, else predict
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(X_test)
                    # Get probability of positive class
                    pos_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else -1
                    if pos_idx >= 0:
                        scores = probs[:, pos_idx]
                    else:
                        scores = probs[:, -1]
                    y_pred = (scores >= 0.5).astype(int)
                else:
                    y_pred = clf.predict(X_test)

                from sklearn.metrics import f1_score, precision_score, recall_score
                test_f1 = float(f1_score(y_true, y_pred, zero_division=0))
                test_p = float(precision_score(y_true, y_pred, zero_division=0))
                test_r = float(recall_score(y_true, y_pred, zero_division=0))

                test_results_map[(vname, mtype)] = {
                    "test_f1": test_f1,
                    "test_precision": test_p,
                    "test_recall": test_r,
                }
                print(f"  {vname}/{mtype}: test_F1={test_f1:.3f}, test_P={test_p:.3f}, test_R={test_r:.3f}")

            except Exception as e:
                logger.warning(f"Test evaluation failed for {vname}/{mtype}: {e}")

    # Create results DataFrame
    rows = []
    for r in results:
        row = {
            "variant": r.variant,
            "model": r.model,
            "train_total": r.train_total,
            "train_positives": r.train_positives,
            "train_negatives": r.train_negatives,
            "val_f1": r.f1,
            "val_precision": r.precision,
            "val_recall": r.recall,
            "tokens_used": r.tokens_used,
        }
        # Add test metrics if available
        test_metrics = test_results_map.get((r.variant, r.model))
        if test_metrics:
            row["test_f1"] = test_metrics["test_f1"]
            row["test_precision"] = test_metrics["test_precision"]
            row["test_recall"] = test_metrics["test_recall"]
        rows.append(row)

    results_df = pd.DataFrame(rows)

    # For backward compat, also include "f1" column pointing to val_f1
    if "val_f1" in results_df.columns:
        results_df["f1"] = results_df["val_f1"]
        results_df["precision"] = results_df["val_precision"]
        results_df["recall"] = results_df["val_recall"]

    # Save results
    results_df.to_csv(output_dir / "comparison_summary.csv", index=False)

    # Save detailed JSON (convert numpy types to Python types for JSON serialization)
    best_test_metrics = test_results_map.get((best_variant, best_model), {})
    details = {
        "best_variant": best_variant,
        "best_model": best_model,
        "best_val_f1": float(best_f1),
        "best_test_f1": best_test_metrics.get("test_f1"),
        "variants": {
            name: {
                "positives": int(v.positives),
                "negatives": int(v.negatives),
                "total": int(len(v.training_set)),
                "generation_time_sec": float(v.generation_time_sec),
                "token_usage": {k: int(val) for k, val in v.token_usage.items()} if v.token_usage else {},
            }
            for name, v in variants.items()
        },
        "results": [
            {
                "variant": r.variant,
                "model": r.model,
                "val_f1": float(r.f1),
                "val_precision": float(r.precision),
                "val_recall": float(r.recall),
                **(test_results_map.get((r.variant, r.model), {})),
            }
            for r in results
        ],
    }
    with open(output_dir / "comparison_details.json", "w") as f:
        json.dump(details, f, indent=2)

    if best_test_metrics:
        print(f"\n=== Best Variant: {best_variant} with {best_model} "
              f"(val_F1={best_f1:.3f}, test_F1={best_test_metrics['test_f1']:.3f}) ===")
    else:
        print(f"\n=== Best Variant: {best_variant} with {best_model} (val_F1={best_f1:.3f}) ===")

    return results_df, best_variant


__all__ = [
    "TrainingSetVariant",
    "ComparisonResult",
    "generate_training_variants",
    "compare_training_variants",
]
