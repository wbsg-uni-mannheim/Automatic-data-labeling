"""
Blocking optimization using validation sets.

This module provides heuristics to optimize blocking parameters
by evaluating different blocking strategies against labeled validation data.

Optimization Strategy
---------------------
The optimizer prioritizes HIGH REDUCTION RATIO (fewer candidates) while
maintaining a minimum PAIR COMPLETENESS (recall) threshold of 97%.

This means:
- We want to generate as few candidate pairs as possible (high reduction)
- But we must find at least 97% of true matches (high recall)
- Among configurations meeting the recall constraint, we pick the one
  with the highest reduction ratio

Why this strategy?
- Blocking is a filtering step before expensive matching
- Missing true matches at blocking stage cannot be recovered
- But having too many candidates wastes compute on the matcher
- 97% recall ensures we catch almost all matches while still filtering aggressively
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..entitymatching.blocking import blocker_from_spec
from ..entitymatching.evaluation import EntityMatchingEvaluator
from .entity_resolution import select_blocking_columns

logger = logging.getLogger(__name__)


def get_default_blocker_specs(
    blocking_columns: List[str],
    include_embedding: bool = False,
    embedding_thresholds: Optional[List[float]] = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_top_k: int = 30,
    embedding_backend: str = "sklearn",
    embedding_device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Generate default blocker specifications for evaluation.

    Parameters
    ----------
    blocking_columns : list of str
        Columns to use for blocking. For token/SNB blockers, these are
        concatenated into a single blocking key.
    include_embedding : bool
        Whether to include EmbeddingBlocker in the specs.
    embedding_thresholds : list of float, optional
        Similarity thresholds to test for EmbeddingBlocker. If provided,
        generates one EmbeddingBlocker spec per threshold.
        Default: [0.5] (single threshold).
    embedding_model : str
        Model for EmbeddingBlocker.
    embedding_top_k : int
        top_k parameter for EmbeddingBlocker.
    embedding_backend : str
        Backend for EmbeddingBlocker (sklearn/faiss/hnsw).
    embedding_device : str, optional
        Device for embedding model.

    Returns
    -------
    list of dict
        Blocker specifications that can be passed to evaluate_blocker_types
        or used with blocker_from_spec.
    """
    specs = [
        {
            "blocker_type": "TokenBlocker",
            "blocker_name": "TokenBlocker_default",
            "blocking_columns": blocking_columns,
            "min_token_len": 3,
            "min_overlap": 2,
        },
        {
            "blocker_type": "TokenBlocker",
            "blocker_name": "TokenBlocker_5gram",
            "blocking_columns": blocking_columns,
            "ngram_size": 5,
            "ngram_type": "character",
        },
        {
            "blocker_type": "SortedNeighbourhoodBlocker",
            "blocker_name": "SortedNeighbourhood_w5",
            "blocking_columns": blocking_columns,
            "window": 5,
        },
        {
            "blocker_type": "SortedNeighbourhoodBlocker",
            "blocker_name": "SortedNeighbourhood_w7",
            "blocking_columns": blocking_columns,
            "window": 7,
        },
        {
            "blocker_type": "SortedNeighbourhoodBlocker",
            "blocker_name": "SortedNeighbourhood_w10",
            "blocking_columns": blocking_columns,
            "window": 10,
        },
    ]

    if include_embedding:
        thresholds = embedding_thresholds or [0.5]
        for threshold in thresholds:
            name = f"EmbeddingBlocker_t{threshold}"
            specs.append({
                "blocker_type": "EmbeddingBlocker",
                "blocker_name": name,
                "text_cols": blocking_columns,
                "model": embedding_model,
                "top_k": embedding_top_k,
                "threshold": threshold,
                "index_backend": embedding_backend,
                "device": embedding_device,
            })

    return specs


def _prepare_blocking_column(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    blocking_columns: List[str],
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Prepare dataframes for blocking by creating a combined column if needed.

    If blocking_columns has multiple columns, creates a "_blocking_key_" column
    by concatenating them. Returns the modified dataframes and the actual
    column name to use for blocking.

    Parameters
    ----------
    df_left, df_right : pd.DataFrame
        Source dataframes.
    blocking_columns : list of str
        Columns to use for blocking.

    Returns
    -------
    tuple of (df_left, df_right, actual_blocking_col)
    """
    if len(blocking_columns) > 1:
        combined_col = "_blocking_key_"
        df_left = df_left.copy()
        df_right = df_right.copy()
        df_left[combined_col] = df_left[blocking_columns].astype(
            str).agg(" ".join, axis=1)
        df_right[combined_col] = df_right[blocking_columns].astype(
            str).agg(" ".join, axis=1)
        return df_left, df_right, combined_col
    else:
        return df_left, df_right, blocking_columns[0]


def _spec_to_blocker_spec(
    spec: Dict[str, Any],
    actual_blocking_col: str,
) -> Dict[str, Any]:
    """
    Convert an evaluation spec to a blocker_from_spec compatible spec.

    The evaluation spec uses 'blocking_columns' (list), but blocker_from_spec
    expects 'blocking_column' (str) for Token/SNB blockers.

    Parameters
    ----------
    spec : dict
        Evaluation spec with blocking_columns list.
    actual_blocking_col : str
        The actual column name to use (either original or combined).

    Returns
    -------
    dict
        Spec compatible with blocker_from_spec.
    """
    blocker_spec = spec.copy()
    blocker_type = spec["blocker_type"]

    # Remove our custom fields
    blocker_spec.pop("blocker_name", None)
    blocker_spec.pop("blocking_columns", None)

    # Set the appropriate column field based on blocker type
    if blocker_type == "EmbeddingBlocker":
        # EmbeddingBlocker uses text_cols natively (already in spec)
        pass
    else:
        # Token/SNB blockers use blocking_column (single column)
        blocker_spec["blocking_column"] = actual_blocking_col

    return blocker_spec


def evaluate_blocker_types(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    validation_set: pd.DataFrame,
    blocker_specs: List[Dict[str, Any]],
    id_column: str = "id",
    min_pair_completeness: float = 0.97,
    out_dir: Optional[Path] = None,
) -> List[Dict]:
    """
    Evaluate blocker specifications against a validation set.

    Parameters
    ----------
    df_left, df_right : pd.DataFrame
        Datasets to block.
    validation_set : pd.DataFrame
        Labeled validation set with columns [id1, id2, label].
    blocker_specs : list of dict
        Blocker specifications to evaluate. Each spec should have:
        - blocker_type: str (TokenBlocker, SortedNeighbourhoodBlocker, EmbeddingBlocker)
        - blocker_name: str (display name for results)
        - blocking_columns: list of str (columns to use)
        - Additional blocker-specific parameters (ngram_size, window, etc.)

        Use get_default_blocker_specs() to generate standard specs.
    id_column : str
        ID column name.
    min_pair_completeness : float
        Minimum required pair completeness (recall). Default 0.97 (97%).
    out_dir : Path, optional
        Directory to write results.

    Returns
    -------
    list of dict
        Evaluation results for each blocker, including:
        - blocker_name: str
        - blocker_spec: dict (the input spec)
        - pair_completeness, pair_quality, reduction_ratio: float
        - num_candidates: int
        - meets_constraint: bool
    """
    logger.info("=" * 60)
    logger.info("BLOCKER TYPE EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Evaluating {len(blocker_specs)} blocker configurations")
    logger.info(f"Min pair completeness: {min_pair_completeness:.0%}")
    logger.info("")

    results = []
    total_blockers = len(blocker_specs)

    for idx, spec in enumerate(blocker_specs):
        blocker_name = spec.get("blocker_name", f"Blocker_{idx}")
        blocking_columns = spec.get(
            "blocking_columns") or spec.get("text_cols", [])
        blocker_spec: dict | None = None

        try:
            logger.info(f"--- {blocker_name} ---")
            print(
                f"    Blocker {idx+1}/{total_blockers}: {blocker_name}...", end=" ", flush=True)

            # Prepare dataframes (create combined column if needed)
            df_left_prep, df_right_prep, actual_col = _prepare_blocking_column(
                df_left, df_right, blocking_columns
            )

            # Convert to blocker_from_spec compatible format
            blocker_spec = _spec_to_blocker_spec(spec, actual_col)

            # Create and run blocker
            start = time.time()
            blocker = blocker_from_spec(
                blocker_spec, df_left_prep, df_right_prep, id_column)
            candidates = blocker.materialize()
            elapsed = time.time() - start

            # Evaluate
            eval_result = EntityMatchingEvaluator.evaluate_blocking(
                candidate_pairs=candidates[["id1", "id2"]] if len(
                    candidates) > 0 else pd.DataFrame(columns=["id1", "id2"]),
                test_pairs=validation_set,
                blocker=blocker,
            )
            eval_result["num_candidates"] = len(candidates)

            pc = eval_result.get("pair_completeness", 0)
            pq = eval_result.get("pair_quality", 0)
            rr = eval_result.get("reduction_ratio", 0)

            meets_constraint = pc >= min_pair_completeness
            eval_result["blocker_name"] = blocker_name
            # Store both:
            # - blocker_spec: compatible with blocker_from_spec (used downstream)
            # - evaluation_spec: original evaluation spec (includes blocking_columns, blocker_name, etc.)
            eval_result["blocker_spec"] = blocker_spec
            eval_result["evaluation_spec"] = spec
            eval_result["meets_constraint"] = meets_constraint
            results.append(eval_result)

            status = "✓" if meets_constraint else "✗"
            print(
                f"{len(candidates)} candidates, PC={pc:.3f}, RR={rr:.4f} {status} ({elapsed:.1f}s)")

            logger.info(f"  {status}")
            logger.info(f"  Candidates: {len(candidates)}")
            logger.info(
                f"  Pair Completeness: {pc:.3f} (min: {min_pair_completeness:.3f})")
            logger.info(f"  Pair Quality: {pq:.3f}")
            logger.info(f"  Reduction Ratio: {rr:.6f}")
            logger.info("")

        except Exception as e:
            print(f"FAILED: {e}")
            logger.warning(f"  FAILED: {e}")
            results.append({
                "blocker_name": blocker_name,
                "blocker_spec": blocker_spec,
                "evaluation_spec": spec,
                "error": str(e),
            })

    # Find best result
    valid_results = [r for r in results if "error" not in r and r.get(
        "meets_constraint", False)]
    if valid_results:
        best = max(valid_results, key=lambda x: x.get("reduction_ratio", 0))
        logger.info(f"Best blocker: {best['blocker_name']}")
        logger.info(f"  Reduction Ratio: {best['reduction_ratio']:.6f}")
        logger.info(f"  Pair Completeness: {best['pair_completeness']:.3f}")
    else:
        all_valid = [r for r in results if "error" not in r]
        if all_valid:
            best = max(all_valid, key=lambda x: x.get("pair_completeness", 0))
            logger.warning(
                f"No blocker meets {min_pair_completeness:.0%} recall constraint")
            logger.info(
                f"Fallback best: {best['blocker_name']} (recall={best['pair_completeness']:.3f})")

    # Save results
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results_df = pd.DataFrame(results)
        results_path = out_dir / "blocker_type_evaluation.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")

    return results


def optimize_blocking(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    validation_set: pd.DataFrame,
    chat_model,
    id_column: str = "id",
    min_pair_completeness: float = 0.97,
    out_dir: Optional[Path] = None,
    include_embedding: bool = False,
    embedding_thresholds: Optional[List[float]] = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_top_k: int = 30,
    embedding_backend: str = "sklearn",
    embedding_device: Optional[str] = None,
) -> Dict:
    """
    Find optimal blocking parameters using validation set.

    This function:
    1. Uses LLM to select blocking strategies (column combinations)
    2. For each strategy, generates blocker specs using get_default_blocker_specs
    3. Evaluates all specs using evaluate_blocker_types
    4. Returns the best configuration

    Optimization Strategy:
    - Prioritize HIGH REDUCTION RATIO (fewer candidates)
    - While maintaining PAIR COMPLETENESS >= min_pair_completeness (default 97%)

    Parameters
    ----------
    df_left, df_right : pd.DataFrame
        Datasets to block.
    validation_set : pd.DataFrame
        Labeled validation set.
    chat_model : BaseChatModel
        LangChain chat model for selecting blocking columns.
    id_column : str
        ID column name.
    min_pair_completeness : float
        Minimum required pair completeness (recall). Default 0.97 (97%).
    out_dir : Path, optional
        Directory to write optimization results.
    include_embedding : bool
        If True, include EmbeddingBlocker in evaluation.
    embedding_thresholds : list of float, optional
        Similarity thresholds to test for EmbeddingBlocker.
        Default: [0.5] if include_embedding is True.
    embedding_* : various
        Other parameters for EmbeddingBlocker.

    Returns
    -------
    dict
        Contains:
        - best: dict with best blocker result
        - all_results: list of all evaluation results
        - blocking_strategies: list of strategies tested
        - best_spec: blocker_from_spec-compatible spec for the best blocker
        - best_eval_spec: original evaluation spec (human-readable; includes blocking_columns)
    """
    from .entity_resolution import parse_blocking_strategy

    logger.info("=" * 60)
    logger.info("BLOCKING OPTIMIZATION")
    logger.info("=" * 60)
    logger.info(
        f"Strategy: Maximize reduction ratio while pair_completeness >= {min_pair_completeness:.0%}")
    logger.info("")

    # Get blocking strategies from LLM (e.g., ["title", "title+year"])
    blocking_strategies = select_blocking_columns(
        df_left, df_right, chat_model, id_column)
    if not blocking_strategies:
        raise ValueError("No suitable blocking strategies found")

    logger.info(f"Blocking strategies to test: {blocking_strategies}")
    logger.info("")

    # Generate specs for all strategies
    all_specs = []
    for strategy in blocking_strategies:
        blocking_cols = parse_blocking_strategy(strategy)
        specs = get_default_blocker_specs(
            blocking_columns=blocking_cols,
            include_embedding=include_embedding,
            embedding_thresholds=embedding_thresholds,
            embedding_model=embedding_model,
            embedding_top_k=embedding_top_k,
            embedding_backend=embedding_backend,
            embedding_device=embedding_device,
        )
        # Add strategy info to each spec
        for spec in specs:
            spec["blocking_strategy"] = strategy
            spec["blocker_name"] = f"{spec['blocker_name']}_{strategy}"
        all_specs.extend(specs)

    print(
        f"  Testing {len(all_specs)} blocker configurations across {len(blocking_strategies)} strategies")

    # Evaluate all specs
    results = evaluate_blocker_types(
        df_left=df_left,
        df_right=df_right,
        validation_set=validation_set,
        blocker_specs=all_specs,
        id_column=id_column,
        min_pair_completeness=min_pair_completeness,
        out_dir=out_dir,
    )

    # Find best
    valid_results = [r for r in results if "error" not in r and r.get(
        "meets_constraint", False)]
    if valid_results:
        best = max(valid_results, key=lambda x: x.get("reduction_ratio", 0))
        logger.info("=" * 60)
        logger.info("OPTIMIZATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Best blocker: {best['blocker_name']}")
        logger.info(
            f"  Pair Completeness: {best['pair_completeness']:.3f} (>= {min_pair_completeness:.3f} ✓)")
        logger.info(f"  Pair Quality: {best['pair_quality']:.3f}")
        logger.info(f"  Reduction Ratio: {best['reduction_ratio']:.6f}")
        logger.info(f"  Candidates: {best['num_candidates']}")
    else:
        logger.warning(
            f"No configuration meets min_pair_completeness={min_pair_completeness:.0%}")
        logger.warning(
            "Falling back to configuration with highest pair_completeness")
        all_valid = [r for r in results if "error" not in r]
        best = max(all_valid, key=lambda x: x.get("pair_completeness", 0))
        logger.info("=" * 60)
        logger.info("OPTIMIZATION RESULTS (FALLBACK)")
        logger.info("=" * 60)
        logger.info(f"Best blocker: {best['blocker_name']}")
        logger.info(
            f"  Pair Completeness: {best['pair_completeness']:.3f} (below target {min_pair_completeness:.3f})")

    # Save summary
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results_df = pd.DataFrame(results)
        results_path = out_dir / "blocking_optimization_results.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")

    return {
        "best": best,
        "best_spec": best.get("blocker_spec"),  # blocker_from_spec compatible
        "best_eval_spec": best.get("evaluation_spec"),  # original evaluation spec (human-readable)
        "all_results": results,
        "blocking_strategies": blocking_strategies,
        "min_pair_completeness": min_pair_completeness,
    }


__all__ = [
    "get_default_blocker_specs",
    "evaluate_blocker_types",
    "optimize_blocking",
]
