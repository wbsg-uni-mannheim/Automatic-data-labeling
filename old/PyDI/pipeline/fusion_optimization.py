"""
Fusion optimization experimental framework.

This module provides 6 experimental cases for comparing fusion rule selection strategies:
1. heuristic: Rule-based fusion based on data types only (no LLM)
2. llm_no_val: LLM-based with heuristic hints, no validation set
3. llm_val: LLM rules + validation set, no source accuracy statistics
4. llm_val_stats: LLM rules + validation set + source accuracy statistics
5. heuristic_stats: Heuristic rules + source accuracy statistics (no LLM)
6. iterative: Run case 3 → case 4 with LLM choosing final config

Each case can be evaluated against all 4 validation modes (llm, llm_omit, web, web_omit).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .fusion import (
    FusionConfig,
    _heuristic_recommendation,
    _resolver_registry,
    _sample_values,
    _compute_trust_scores,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Evaluation Function Configuration
# =============================================================================


def _infer_attribute_type(
    datasets: Dict[str, pd.DataFrame],
    attribute: str,
) -> str:
    """
    Infer the data type of an attribute from sample values.

    Returns one of: 'list', 'date', 'numeric', 'string'
    """
    sample_values = []
    for df in datasets.values():
        if attribute in df.columns:
            non_null = df[attribute].dropna().head(10).tolist()
            sample_values.extend(non_null)

    if not sample_values:
        return "string"

    # Check if values are lists/sets
    list_count = sum(1 for v in sample_values if isinstance(v, (list, tuple, set)))
    if list_count > len(sample_values) / 2:
        return "list"

    # Check for dates
    date_count = 0
    for v in sample_values:
        if isinstance(v, (pd.Timestamp, np.datetime64)):
            date_count += 1
        elif isinstance(v, str):
            # Try to detect date-like strings
            v_lower = str(v).lower().strip()
            if any(x in v_lower for x in ["-", "/", "date", "year"]):
                try:
                    parsed = pd.to_datetime(v, errors="coerce")
                    if not pd.isna(parsed):
                        date_count += 1
                except Exception:
                    pass
    if date_count > len(sample_values) / 2:
        return "date"

    # Check for numeric
    numeric_count = 0
    for v in sample_values:
        if isinstance(v, (int, float, np.number)):
            numeric_count += 1
        elif isinstance(v, str):
            try:
                float(v.replace(",", "").replace("$", "").replace("%", ""))
                numeric_count += 1
            except (ValueError, AttributeError):
                pass
    if numeric_count > len(sample_values) / 2:
        return "numeric"

    return "string"


def _numeric_tolerance_match_relative(
    fused_value,
    expected_value,
    tolerance: float = 0.1,
) -> bool:
    """
    Numeric tolerance match with relative (percentage) tolerance.

    Parameters
    ----------
    tolerance : float
        Relative tolerance as a fraction (e.g., 0.1 for 10%).
    """
    from PyDI.fusion.evaluation import _is_missing_value

    if _is_missing_value(fused_value) and _is_missing_value(expected_value):
        return True
    if _is_missing_value(fused_value) or _is_missing_value(expected_value):
        return False

    try:
        fused_num = float(fused_value)
        expected_num = float(expected_value)
        diff = abs(fused_num - expected_num)

        if expected_num == 0:
            # If expected is 0, use small absolute tolerance
            return diff <= tolerance
        return diff <= abs(expected_num * tolerance)
    except (ValueError, TypeError):
        return str(fused_value).strip() == str(expected_value).strip()


def add_evaluation_functions_to_strategy(
    strategy: Any,
    datasets: Dict[str, pd.DataFrame],
    config: Any,
    numeric_tolerance: float = 0.2,
) -> None:
    """
    Add type-aware evaluation functions to a DataFusionStrategy.

    Assigns evaluation functions based on inferred data types:
    - Lists: set_equality_match
    - Dates: year_only_match
    - Numerics: numeric tolerance match with relative tolerance
    - Strings: tokenized_match

    Parameters
    ----------
    strategy : DataFusionStrategy
        The strategy to add evaluation functions to.
    datasets : Dict[str, pd.DataFrame]
        The datasets to infer types from.
    config : FusionConfig
        Fusion configuration (for id_column).
    numeric_tolerance : float
        Relative tolerance for numeric comparisons (default: 0.2 = 20%).
    """
    from functools import partial
    from PyDI.fusion.evaluation import (
        tokenized_match,
        year_only_match,
        set_equality_match,
    )

    # Get all attributes from datasets
    all_attrs = set()
    for df in datasets.values():
        all_attrs.update(df.columns)

    # Exclude id and metadata columns
    excluded = {config.id_column, "_id", "id"}
    attrs = [a for a in all_attrs if a not in excluded and not a.startswith("_fusion_")]

    for attr in attrs:
        attr_type = _infer_attribute_type(datasets, attr)

        if attr_type == "list":
            strategy.add_evaluation_function(attr, set_equality_match)
            logger.debug(f"Evaluation function for '{attr}': set_equality_match (list)")
        elif attr_type == "date":
            strategy.add_evaluation_function(attr, year_only_match)
            logger.debug(f"Evaluation function for '{attr}': year_only_match (date)")
        elif attr_type == "numeric":
            strategy.add_evaluation_function(
                attr,
                partial(_numeric_tolerance_match_relative, tolerance=numeric_tolerance),
            )
            logger.debug(f"Evaluation function for '{attr}': numeric_tolerance_match ({numeric_tolerance:.0%})")
        else:
            strategy.add_evaluation_function(attr, tokenized_match)
            logger.debug(f"Evaluation function for '{attr}': tokenized_match (string)")


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SourceAccuracyStats:
    """Accuracy statistics for a single source on a single attribute."""
    source_name: str
    attribute: str
    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_name": self.source_name,
            "attribute": self.attribute,
            "correct": self.correct,
            "total": self.total,
            "accuracy": round(self.accuracy, 4),
        }


@dataclass
class FusionCaseResult:
    """Result of running a fusion case."""
    case_name: str
    rules: Dict[str, Dict[str, Any]]  # attribute -> {resolver, kwargs}
    accuracy_by_mode: Dict[str, float] = field(default_factory=dict)  # validation_mode -> accuracy
    source_stats: Optional[Dict[str, Dict[str, SourceAccuracyStats]]] = None
    iterations_used: int = 1
    optimization_mode: Optional[str] = None  # which validation mode was used for optimization

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_name": self.case_name,
            "rules": self.rules,
            "accuracy_by_mode": self.accuracy_by_mode,
            "source_stats": {
                attr: {src: stats.to_dict() for src, stats in sources.items()}
                for attr, sources in (self.source_stats or {}).items()
            } if self.source_stats else None,
            "iterations_used": self.iterations_used,
            "optimization_mode": self.optimization_mode,
        }


# =============================================================================
# Source Accuracy Computation
# =============================================================================


def compute_source_accuracy(
    ground_truth_df: pd.DataFrame,
) -> Dict[str, Dict[str, SourceAccuracyStats]]:
    """
    Compute per-source accuracy for each attribute from validation ground truth.

    Parameters
    ----------
    ground_truth_df : pd.DataFrame
        Ground truth from fusion validation with columns:
        [entity_id, source_ids, <attr1>, <attr2>, ...]
        This is the tabular format from convert_to_tabular_format().

    Returns
    -------
    Dict[str, Dict[str, SourceAccuracyStats]]
        {attribute: {source_name: SourceAccuracyStats}}
    """
    if ground_truth_df.empty:
        return {}

    # We need the fusion_validation_set.csv which has source_values per attribute
    # But we're given ground_truth_df which is the converted tabular format
    # We need to re-load the original validation set to get source_values
    logger.warning("compute_source_accuracy requires original validation set with source_values")
    return {}


def _get_match_function_for_value(correct_value: Any, source_value: Any) -> callable:
    """
    Determine the appropriate match function based on value types.

    Returns a function that takes (source_value, correct_value) and returns bool.
    """
    from functools import partial
    from PyDI.fusion.evaluation import (
        tokenized_match,
        year_only_match,
        set_equality_match,
    )

    # Check for list/set types
    if isinstance(correct_value, (list, tuple, set)) or isinstance(source_value, (list, tuple, set)):
        return set_equality_match

    # Check for date-like strings
    correct_str = str(correct_value).strip().lower()
    if any(x in correct_str for x in ["-", "/", "date"]):
        try:
            parsed = pd.to_datetime(correct_value, errors="coerce")
            if not pd.isna(parsed):
                return year_only_match
        except Exception:
            pass

    # Check for numeric values
    try:
        float(str(correct_value).replace(",", "").replace("$", "").replace("%", ""))
        return partial(_numeric_tolerance_match_relative, tolerance=0.2)
    except (ValueError, TypeError, AttributeError):
        pass

    # Default to tokenized match for strings
    return tokenized_match


def compute_source_accuracy_from_validation_set(
    validation_df: pd.DataFrame,
) -> Dict[str, Dict[str, SourceAccuracyStats]]:
    """
    Compute per-source accuracy for each attribute from the original validation set.

    Uses type-aware matching:
    - Lists: set equality
    - Dates: year-only comparison
    - Numbers: 20% relative tolerance
    - Strings: tokenized match

    Parameters
    ----------
    validation_df : pd.DataFrame
        Original validation set with columns:
        [entity_id, dataset_ids, attribute, correct_value, source_values, reasoning]

    Returns
    -------
    Dict[str, Dict[str, SourceAccuracyStats]]
        {attribute: {source_name: SourceAccuracyStats}}
    """
    if validation_df.empty:
        return {}

    stats: Dict[str, Dict[str, SourceAccuracyStats]] = {}

    for _, row in validation_df.iterrows():
        attr = row["attribute"]
        correct_value = row["correct_value"]
        source_values_str = row["source_values"]

        # Skip unknown values
        if str(correct_value).strip().lower() == "unknown":
            continue

        # Parse source values
        try:
            source_values = json.loads(source_values_str) if isinstance(source_values_str, str) else source_values_str
        except (json.JSONDecodeError, TypeError):
            continue

        if not isinstance(source_values, dict):
            continue

        # Initialize attribute entry
        if attr not in stats:
            stats[attr] = {}

        # Check each source using type-aware matching
        for source_name, source_value in source_values.items():
            if source_name not in stats[attr]:
                stats[attr][source_name] = SourceAccuracyStats(
                    source_name=source_name,
                    attribute=attr,
                )

            stats[attr][source_name].total += 1

            # Get appropriate match function and check if values match
            match_func = _get_match_function_for_value(correct_value, source_value)
            try:
                if match_func(source_value, correct_value):
                    stats[attr][source_name].correct += 1
            except Exception:
                # If matching fails, fall back to string comparison
                if str(source_value).strip().lower() == str(correct_value).strip().lower():
                    stats[attr][source_name].correct += 1

    return stats


def format_source_accuracy_for_llm(
    stats: Dict[str, Dict[str, SourceAccuracyStats]],
) -> str:
    """
    Format source accuracy statistics for inclusion in LLM prompt.

    Parameters
    ----------
    stats : Dict[str, Dict[str, SourceAccuracyStats]]
        Output from compute_source_accuracy_from_validation_set()

    Returns
    -------
    str
        Formatted string for LLM prompt
    """
    if not stats:
        return "No source accuracy statistics available."

    lines = ["SOURCE ACCURACY STATISTICS (from validation set):"]
    lines.append("")

    for attr in sorted(stats.keys()):
        sources = stats[attr]
        lines.append(f"Attribute \"{attr}\":")
        for source_name in sorted(sources.keys()):
            s = sources[source_name]
            pct = s.accuracy * 100
            lines.append(f"  - {source_name}: {pct:.0f}% correct ({s.correct}/{s.total} values)")
        lines.append("")

    lines.append("RECOMMENDATION: For attributes where one source is significantly more accurate (>80%),")
    lines.append("consider using 'favour_sources' to prefer that source's values.")

    return "\n".join(lines)


def format_validation_feedback_for_llm(
    validation_df: pd.DataFrame,
    max_examples: int = 5,
) -> str:
    """
    Format validation set as feedback for LLM (without accuracy percentages).

    Shows which values were correct/incorrect per attribute.

    Parameters
    ----------
    validation_df : pd.DataFrame
        Original validation set with columns:
        [entity_id, dataset_ids, attribute, correct_value, source_values, reasoning]
    max_examples : int
        Maximum number of examples to show per attribute

    Returns
    -------
    str
        Formatted string for LLM prompt
    """
    if validation_df.empty:
        return "No validation feedback available."

    lines = ["VALIDATION FEEDBACK:"]
    lines.append("(Showing which source values were correct for sample entities)")
    lines.append("")

    # Group by attribute
    by_attr = validation_df.groupby("attribute")

    for attr, group in by_attr:
        lines.append(f"Attribute \"{attr}\":")
        examples = list(group.iterrows())[:max_examples]

        for _, row in examples:
            correct_value = row["correct_value"]
            source_values_str = row["source_values"]

            try:
                source_values = json.loads(source_values_str) if isinstance(source_values_str, str) else source_values_str
            except (json.JSONDecodeError, TypeError):
                continue

            if not isinstance(source_values, dict):
                continue

            # Format source values
            sources_str = ", ".join(f"{src}={val}" for src, val in source_values.items())
            lines.append(f"  - Sources: {sources_str}")
            lines.append(f"    → Correct: {correct_value}")

        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Provided Validation Set Conversion
# =============================================================================


def convert_provided_validation_to_csv_format(
    validation_xml_df: pd.DataFrame,
    datasets: Dict[str, pd.DataFrame],
    correspondences: pd.DataFrame,
    id_column: str = "id",
) -> pd.DataFrame:
    """
    Convert a provided validation set (XML format) to the auto-generated validation set format.

    The XML format has one row per entity with attribute columns.
    The CSV format has one row per (entity, attribute) with source_values dict.

    Parameters
    ----------
    validation_xml_df : pd.DataFrame
        Validation set loaded from XML with 'id' column and attribute columns.
    datasets : Dict[str, pd.DataFrame]
        Original datasets (used to look up source values).
    correspondences : pd.DataFrame
        Entity correspondences with 'id1', 'id2' columns.
    id_column : str
        Name of the ID column.

    Returns
    -------
    pd.DataFrame
        Validation set in CSV format with columns:
        [entity_id, dataset_ids, attribute, correct_value, source_values, reasoning]
    """
    if validation_xml_df.empty:
        return pd.DataFrame(columns=[
            "entity_id", "dataset_ids", "attribute", "correct_value", "source_values", "reasoning"
        ])

    # Build mapping from record ID to (dataset_name, row)
    id_to_dataset: Dict[str, Tuple[str, pd.Series]] = {}
    for name, df in datasets.items():
        for _, row in df.iterrows():
            record_id = str(row.get(id_column, row.get("_id", "")))
            if record_id:
                id_to_dataset[record_id] = (name, row)

    # Build entity groups from correspondences using Union-Find
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: str, y: str) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Add all IDs from correspondences
    for _, row in correspondences.iterrows():
        id1, id2 = str(row["id1"]), str(row["id2"])
        union(id1, id2)

    # Group IDs by their root
    groups: Dict[str, set] = {}
    for id_ in parent.keys():
        root = find(id_)
        if root not in groups:
            groups[root] = set()
        groups[root].add(id_)

    # Build reverse mapping: any ID -> all IDs in its group
    id_to_group: Dict[str, set] = {}
    for root, group_ids in groups.items():
        for id_ in group_ids:
            id_to_group[id_] = group_ids

    # Get attribute columns (exclude id and metadata columns)
    attr_columns = [
        c for c in validation_xml_df.columns
        if c not in (id_column, "_id") and not c.startswith("_fusion_")
    ]

    rows = []
    entity_counter = 0

    for _, val_row in validation_xml_df.iterrows():
        entity_id_raw = str(val_row.get(id_column, ""))
        if not entity_id_raw:
            continue

        # Find all corresponding records for this entity
        group_ids = id_to_group.get(entity_id_raw, {entity_id_raw})

        # Build dataset_ids dict and collect source data
        dataset_ids: Dict[str, str] = {}
        source_records: Dict[str, pd.Series] = {}

        for record_id in group_ids:
            if record_id in id_to_dataset:
                ds_name, ds_row = id_to_dataset[record_id]
                dataset_ids[ds_name] = record_id
                source_records[ds_name] = ds_row

        # Also check the validation entity ID itself (might not be in correspondences)
        if entity_id_raw in id_to_dataset:
            ds_name, ds_row = id_to_dataset[entity_id_raw]
            if ds_name not in dataset_ids:
                dataset_ids[ds_name] = entity_id_raw
                source_records[ds_name] = ds_row

        if not source_records:
            # No matching source records found, skip this entity
            logger.warning(f"No source records found for validation entity {entity_id_raw}")
            continue

        entity_id = f"entity_{entity_counter}"
        entity_counter += 1

        for attr in attr_columns:
            correct_value = val_row.get(attr)

            # Skip missing values (handle arrays properly)
            try:
                is_missing = pd.isna(correct_value) if not isinstance(correct_value, (list, np.ndarray)) else False
            except ValueError:
                is_missing = False
            if is_missing:
                continue
            correct_str = str(correct_value).strip()
            if not correct_str:
                continue

            # Build source_values dict from original datasets
            source_values: Dict[str, Any] = {}
            for ds_name, ds_row in source_records.items():
                if attr in ds_row.index:
                    src_val = ds_row[attr]
                    # Handle arrays properly for isna check
                    try:
                        src_is_missing = pd.isna(src_val) if not isinstance(src_val, (list, np.ndarray)) else False
                    except ValueError:
                        src_is_missing = False
                    if not src_is_missing:
                        source_values[ds_name] = src_val

            # Skip if no sources have this attribute
            if not source_values:
                continue

            rows.append({
                "entity_id": entity_id,
                "dataset_ids": json.dumps(dataset_ids),
                "attribute": attr,
                "correct_value": correct_value,
                "source_values": json.dumps(source_values, default=str),
                "reasoning": "from provided validation set",
            })

    return pd.DataFrame(rows)


# =============================================================================
# Fusion Accuracy Evaluation
# =============================================================================


def evaluate_fusion_accuracy(
    fused_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    id_column: str = "id",
) -> Dict[str, Any]:
    """
    Compare fused output against ground truth.

    Parameters
    ----------
    fused_df : pd.DataFrame
        Fused dataset output
    ground_truth_df : pd.DataFrame
        Ground truth with columns [entity_id, source_ids, <attributes>...]
        The source_ids column contains comma-separated record IDs

    Returns
    -------
    Dict[str, Any]
        {
            'overall_accuracy': float,
            'total_comparisons': int,
            'correct_comparisons': int,
            'per_attribute_accuracy': {attr: {'accuracy': float, 'correct': int, 'total': int}},
            'per_entity_results': [{entity_id, accuracy, correct, total}],
        }
    """
    if fused_df.empty or ground_truth_df.empty:
        return {
            "overall_accuracy": 0.0,
            "total_comparisons": 0,
            "correct_comparisons": 0,
            "per_attribute_accuracy": {},
            "per_entity_results": [],
        }

    # Get attributes to compare (exclude id-like columns)
    gt_attrs = [c for c in ground_truth_df.columns if c not in ("entity_id", "source_ids")]
    fused_attrs = [c for c in fused_df.columns if c not in (id_column, "_id")]
    common_attrs = set(gt_attrs) & set(fused_attrs)

    if not common_attrs:
        logger.warning("No common attributes between fused output and ground truth")
        return {
            "overall_accuracy": 0.0,
            "total_comparisons": 0,
            "correct_comparisons": 0,
            "per_attribute_accuracy": {},
            "per_entity_results": [],
        }

    # Build index from fused_df: record_id -> row
    fused_index = {}
    for _, row in fused_df.iterrows():
        record_id = str(row.get(id_column, row.get("_id", "")))
        if record_id:
            fused_index[record_id] = row

    total_correct = 0
    total_comparisons = 0
    per_attr_stats: Dict[str, Dict[str, int]] = {attr: {"correct": 0, "total": 0} for attr in common_attrs}
    per_entity_results = []

    for _, gt_row in ground_truth_df.iterrows():
        entity_id = gt_row["entity_id"]
        source_ids_str = gt_row["source_ids"]

        # Parse source IDs
        source_ids = [s.strip() for s in str(source_ids_str).split(",")]

        # Find matching fused record
        fused_row = None
        for sid in source_ids:
            if sid in fused_index:
                fused_row = fused_index[sid]
                break

        if fused_row is None:
            continue

        entity_correct = 0
        entity_total = 0

        for attr in common_attrs:
            gt_value = gt_row.get(attr)
            fused_value = fused_row.get(attr)

            # Skip if either is missing or UNKNOWN
            # Handle arrays/lists by checking if any element is na
            try:
                gt_is_na = pd.isna(gt_value) if not isinstance(gt_value, (list, tuple, np.ndarray)) else False
                fused_is_na = pd.isna(fused_value) if not isinstance(fused_value, (list, tuple, np.ndarray)) else False
            except ValueError:
                # Handle arrays with ambiguous truth values
                gt_is_na = False
                fused_is_na = False

            if gt_is_na or fused_is_na:
                continue
            gt_str = str(gt_value).strip().lower()
            fused_str = str(fused_value).strip().lower()
            if gt_str == "unknown" or not gt_str:
                continue

            per_attr_stats[attr]["total"] += 1
            total_comparisons += 1
            entity_total += 1

            if gt_str == fused_str:
                per_attr_stats[attr]["correct"] += 1
                total_correct += 1
                entity_correct += 1

        if entity_total > 0:
            per_entity_results.append({
                "entity_id": entity_id,
                "accuracy": entity_correct / entity_total,
                "correct": entity_correct,
                "total": entity_total,
            })

    # Compute per-attribute accuracy
    per_attr_accuracy = {}
    for attr, counts in per_attr_stats.items():
        if counts["total"] > 0:
            per_attr_accuracy[attr] = {
                "accuracy": counts["correct"] / counts["total"],
                "correct": counts["correct"],
                "total": counts["total"],
            }

    overall_accuracy = total_correct / total_comparisons if total_comparisons > 0 else 0.0

    return {
        "overall_accuracy": overall_accuracy,
        "total_comparisons": total_comparisons,
        "correct_comparisons": total_correct,
        "per_attribute_accuracy": per_attr_accuracy,
        "per_entity_results": per_entity_results,
    }


def evaluate_fusion_against_test_set(
    fused_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_column: str = "id",
) -> Dict[str, Any]:
    """
    Compare fused output against a test set (ground truth).

    Unlike evaluate_fusion_accuracy which expects validation set format,
    this function expects a simple test set where each row has an 'id'
    column that matches source record IDs.

    Parameters
    ----------
    fused_df : pd.DataFrame
        Fused dataset output
    test_df : pd.DataFrame
        Test set with 'id' column and attribute columns

    Returns
    -------
    Dict[str, Any]
        Same format as evaluate_fusion_accuracy
    """
    if fused_df.empty or test_df.empty:
        return {
            "overall_accuracy": 0.0,
            "total_comparisons": 0,
            "correct_comparisons": 0,
            "per_attribute_accuracy": {},
            "per_entity_results": [],
        }

    # Get attributes to compare (exclude id-like columns)
    test_attrs = [c for c in test_df.columns if c not in ("id", "_id")]
    fused_attrs = [c for c in fused_df.columns if c not in (id_column, "_id")]

    # Handle attribute name variations (underscores vs hyphens)
    common_attrs = set()
    attr_mapping = {}  # test_attr -> fused_attr
    for test_attr in test_attrs:
        if test_attr in fused_attrs:
            common_attrs.add(test_attr)
            attr_mapping[test_attr] = test_attr
        else:
            # Try with underscores/hyphens swapped
            alt_attr = test_attr.replace("-", "_")
            if alt_attr in fused_attrs:
                common_attrs.add(test_attr)
                attr_mapping[test_attr] = alt_attr
            else:
                alt_attr = test_attr.replace("_", "-")
                if alt_attr in fused_attrs:
                    common_attrs.add(test_attr)
                    attr_mapping[test_attr] = alt_attr

    if not common_attrs:
        logger.warning("No common attributes between fused output and test set")
        return {
            "overall_accuracy": 0.0,
            "total_comparisons": 0,
            "correct_comparisons": 0,
            "per_attribute_accuracy": {},
            "per_entity_results": [],
        }

    # Build index from fused_df: any source_id -> row
    # This allows matching test records by any of the source IDs that were fused together
    import ast

    fused_index = {}
    for _, row in fused_df.iterrows():
        # Index by primary ID
        record_id = str(row.get(id_column, row.get("_id", "")))
        if record_id:
            fused_index[record_id] = row

        # Also index by all IDs in _fusion_sources (handles ID alignment)
        fusion_sources = row.get("_fusion_sources")
        if fusion_sources:
            if isinstance(fusion_sources, str):
                try:
                    sources = ast.literal_eval(fusion_sources)
                except (ValueError, SyntaxError):
                    sources = [fusion_sources]
            else:
                sources = list(fusion_sources) if fusion_sources else []

            for src_id in sources:
                if isinstance(src_id, str) and src_id:
                    fused_index[src_id] = row

    total_correct = 0
    total_comparisons = 0
    per_attr_stats: Dict[str, Dict[str, int]] = {attr: {"correct": 0, "total": 0} for attr in common_attrs}
    per_entity_results = []

    for _, test_row in test_df.iterrows():
        test_id = str(test_row.get("id", ""))

        # Find matching fused record (now works with any source ID)
        fused_row = fused_index.get(test_id)
        if fused_row is None:
            continue

        entity_correct = 0
        entity_total = 0

        for test_attr in common_attrs:
            fused_attr = attr_mapping[test_attr]
            test_value = test_row.get(test_attr)
            fused_value = fused_row.get(fused_attr)

            # Skip if either is missing
            try:
                test_is_na = pd.isna(test_value) if not isinstance(test_value, (list, tuple, np.ndarray)) else False
                fused_is_na = pd.isna(fused_value) if not isinstance(fused_value, (list, tuple, np.ndarray)) else False
            except ValueError:
                test_is_na = False
                fused_is_na = False

            if test_is_na or fused_is_na:
                continue

            test_str = str(test_value).strip().lower()
            fused_str = str(fused_value).strip().lower()

            if not test_str:
                continue

            per_attr_stats[test_attr]["total"] += 1
            total_comparisons += 1
            entity_total += 1

            if test_str == fused_str:
                per_attr_stats[test_attr]["correct"] += 1
                total_correct += 1
                entity_correct += 1

        if entity_total > 0:
            per_entity_results.append({
                "entity_id": test_id,
                "accuracy": entity_correct / entity_total,
                "correct": entity_correct,
                "total": entity_total,
            })

    # Compute per-attribute accuracy
    per_attr_accuracy = {}
    for attr, counts in per_attr_stats.items():
        if counts["total"] > 0:
            per_attr_accuracy[attr] = {
                "accuracy": counts["correct"] / counts["total"],
                "correct": counts["correct"],
                "total": counts["total"],
            }

    overall_accuracy = total_correct / total_comparisons if total_comparisons > 0 else 0.0

    return {
        "overall_accuracy": overall_accuracy,
        "total_comparisons": total_comparisons,
        "correct_comparisons": total_correct,
        "per_attribute_accuracy": per_attr_accuracy,
        "per_entity_results": per_entity_results,
    }


# =============================================================================
# Fusion Case Implementations
# =============================================================================


def _case_heuristic(
    datasets: Dict[str, pd.DataFrame],
    config: FusionConfig,
) -> Dict[str, Dict[str, Any]]:
    """
    Case 1: Pure heuristic-based fusion rules (no LLM, no validation).

    Uses data type inference to select appropriate resolvers.
    """
    ds_list = list(datasets.values())
    all_cols: set[str] = set()
    for df in ds_list:
        all_cols.update(df.columns)

    excluded = {config.id_column, "_id"}
    cols = [c for c in sorted(all_cols) if c not in excluded and not str(c).startswith("_fusion_")]

    rules = {}
    for col in cols:
        resolver, kwargs, _ = _heuristic_recommendation(ds_list, col, config=config)
        rules[col] = {"resolver": resolver, "kwargs": kwargs}

    return rules


def _case_llm_no_val(
    datasets: Dict[str, pd.DataFrame],
    chat_model: Any,
    config: FusionConfig,
    output_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Case 2: LLM-based with heuristic hints, no validation set.

    This is the existing select_fusion_rules_with_llm() behavior.
    """
    from .fusion import select_fusion_rules_with_llm

    return select_fusion_rules_with_llm(
        datasets,
        chat_model=chat_model,
        config=config,
        output_dir=output_dir,
    )


def _case_llm_val(
    datasets: Dict[str, pd.DataFrame],
    validation_df: pd.DataFrame,
    chat_model: Any,
    config: FusionConfig,
    output_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Case 3: LLM rules + validation feedback (no accuracy statistics).

    Shows the LLM which values were correct/incorrect without percentages.
    """
    return select_fusion_rules_with_validation(
        datasets,
        validation_df=validation_df,
        chat_model=chat_model,
        config=config,
        output_dir=output_dir,
        include_stats=False,
    )


def _case_llm_val_stats(
    datasets: Dict[str, pd.DataFrame],
    validation_df: pd.DataFrame,
    chat_model: Any,
    config: FusionConfig,
    output_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Case 4: LLM rules + validation + source accuracy statistics.

    Shows the LLM accuracy percentages per source per attribute.
    """
    return select_fusion_rules_with_validation(
        datasets,
        validation_df=validation_df,
        chat_model=chat_model,
        config=config,
        output_dir=output_dir,
        include_stats=True,
    )


def _case_heuristic_stats(
    datasets: Dict[str, pd.DataFrame],
    validation_df: pd.DataFrame,
    config: FusionConfig,
    accuracy_threshold: float = 0.8,
) -> Dict[str, Dict[str, Any]]:
    """
    Case 5: Heuristic rules + source accuracy statistics (no LLM).

    Override heuristic with favour_sources when one source has ≥threshold accuracy.
    """
    # Start with heuristic rules
    rules = _case_heuristic(datasets, config)

    # Compute source accuracy
    stats = compute_source_accuracy_from_validation_set(validation_df)

    # Override rules for attributes where one source is significantly better
    for attr, sources in stats.items():
        if attr not in rules:
            continue

        # Find best source
        best_source = None
        best_accuracy = 0.0
        for source_name, source_stats in sources.items():
            if source_stats.total >= 2 and source_stats.accuracy > best_accuracy:
                best_accuracy = source_stats.accuracy
                best_source = source_name

        # Override if best source exceeds threshold
        if best_source and best_accuracy >= accuracy_threshold:
            rules[attr] = {
                "resolver": "favour_sources",
                "kwargs": {"sources": [best_source]},
            }
            logger.info(
                f"Case heuristic_stats: Override {attr} to favour_sources({best_source}) "
                f"with accuracy {best_accuracy:.0%}"
            )

    return rules


def _case_iterative(
    datasets: Dict[str, pd.DataFrame],
    validation_df: pd.DataFrame,
    chat_model: Any,
    config: FusionConfig,
    output_dir: Optional[Path] = None,
    iterations: int = 3,
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    """
    Case 6: Iterative refinement.

    Run case 3 (llm_val), then case 4 (llm_val_stats), let LLM choose.
    Repeat for N iterations.

    Returns (rules, iterations_used).
    """
    # Start with case 3 (LLM + validation feedback, no stats)
    current_rules = _case_llm_val(
        datasets,
        validation_df,
        chat_model,
        config,
        output_dir=output_dir / "iteration_0" if output_dir else None,
    )

    for i in range(1, iterations):
        iter_dir = output_dir / f"iteration_{i}" if output_dir else None

        # Run case 4 (LLM + validation + stats) with previous rules as context
        new_rules = select_fusion_rules_with_validation(
            datasets,
            validation_df=validation_df,
            chat_model=chat_model,
            config=config,
            output_dir=iter_dir,
            include_stats=True,
            previous_rules=current_rules,
        )

        # Check if rules changed
        if new_rules == current_rules:
            logger.info(f"Iterative case converged after {i} iterations")
            return new_rules, i

        current_rules = new_rules

    return current_rules, iterations


# =============================================================================
# LLM Rule Selection with Validation
# =============================================================================


def select_fusion_rules_with_validation(
    datasets: Dict[str, pd.DataFrame],
    validation_df: pd.DataFrame,
    chat_model: Any,
    config: FusionConfig,
    output_dir: Optional[Path] = None,
    include_stats: bool = False,
    previous_rules: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Ask LLM to choose fusion rules with validation feedback.

    Parameters
    ----------
    datasets : Dict[str, pd.DataFrame]
        Datasets to fuse
    validation_df : pd.DataFrame
        Validation set with ground truth values
    chat_model : Any
        LLM chat model
    config : FusionConfig
        Fusion configuration
    output_dir : Path, optional
        Directory to save prompt and result
    include_stats : bool
        If True, include source accuracy statistics in prompt
    previous_rules : Dict, optional
        Previous rules for iterative mode

    Returns
    -------
    Dict[str, Dict[str, Any]]
        {attribute: {"resolver": "<name>", "kwargs": {...}}}
    """
    from langchain_core.messages import HumanMessage

    out_dir = Path(output_dir) if output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    ds_list = list(datasets.values())
    all_cols: set[str] = set()
    for df in ds_list:
        all_cols.update(df.columns)

    excluded = {config.id_column, "_id"}
    cols = [c for c in sorted(all_cols) if c not in excluded and not str(c).startswith("_fusion_")]

    # Build heuristic starting plan
    heuristic: Dict[str, Dict[str, Any]] = {}
    per_attr: List[Dict[str, Any]] = []
    for col in cols:
        resolver, kwargs, diag = _heuristic_recommendation(ds_list, col, config=config)
        heuristic[col] = {"resolver": resolver, "kwargs": kwargs}
        per_attr.append(diag)

    resolver_names = sorted(_resolver_registry().keys())

    # Build dataset overview
    trust_scores = _compute_trust_scores(datasets, id_column=config.id_column)
    dataset_overview = []
    for name, df in datasets.items():
        dataset_overview.append({
            "dataset_name": name,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "trust_score": trust_scores.get(name),
        })

    # Build validation feedback section
    validation_feedback = format_validation_feedback_for_llm(validation_df)

    # Build statistics section if requested
    stats_section = ""
    if include_stats:
        stats = compute_source_accuracy_from_validation_set(validation_df)
        stats_section = "\n\n" + format_source_accuracy_for_llm(stats)

    # Build previous rules section for iterative mode
    previous_section = ""
    if previous_rules:
        previous_section = f"""
PREVIOUS ITERATION RULES (you may keep, modify, or discard):
{json.dumps(previous_rules, indent=2, sort_keys=True)}
"""

    prompt = f"""We are doing DATA FUSION: merge multiple records that refer to the same real-world entity.

Task:
1) Infer what the MAIN ENTITY is from dataset names and attribute names.
2) For EACH attribute, choose EXACTLY ONE conflict resolution function (resolver) from the allowed list.
3) Use the validation feedback to understand which sources tend to be correct for each attribute.

Allowed resolvers (choose only from this list):
{resolver_names}

Resolver guidance:
- Lists/Sets: union, intersection, intersection_k_sources (if you choose union for string-lists, you may set kwargs.separator to ',', ';', or '|')
- Strings: most_complete, longest_string, shortest_string
- Numerics: median, average, maximum, minimum, sum_values
- Dates: most_recent, earliest
- Source-aware: prefer_higher_trust (use kwargs.trust_key='{config.trust_key}'), voting, weighted_voting, favour_sources (use kwargs.sources=['source1']), random_value

Starting point (heuristic plan you can override):
{json.dumps(heuristic, indent=2, sort_keys=True)}

Attribute diagnostics:
{pd.DataFrame(per_attr).to_string(index=False)}

Dataset overview:
{json.dumps(dataset_overview, indent=2)}

{validation_feedback}
{stats_section}
{previous_section}

Return ONLY strict JSON in this schema:
{{
  "main_entity": "<string>",
  "attribute_rules": {{
    "<attribute>": {{"resolver": "<resolver_name>", "kwargs": {{...}}}},
    ...
  }}
}}
Notes:
- Include every attribute listed in the heuristic plan.
- Use kwargs only when needed. Do not invent new resolvers.
- For attributes where validation shows one source is consistently correct, consider using favour_sources.
"""

    if out_dir:
        prompt_path = out_dir / "fusion_rules_prompt.txt"
        prompt_path.write_text(prompt)

    response = chat_model.invoke([HumanMessage(content=prompt)])
    content = getattr(response, "content", str(response)).strip()

    try:
        parsed = json.loads(content)
    except Exception as e:
        raise ValueError(f"LLM fusion rule selection did not return valid JSON: {e}")

    main_entity = parsed.get("main_entity")
    rules = parsed.get("attribute_rules")
    if not isinstance(rules, dict) or not rules:
        raise ValueError("LLM fusion rule selection returned no attribute_rules")

    # Validate + normalize
    registry = _resolver_registry()
    out_rules: Dict[str, Dict[str, Any]] = {}
    for attr, spec in rules.items():
        if not isinstance(attr, str) or attr not in heuristic:
            continue
        if not isinstance(spec, dict):
            continue
        resolver_name = str(spec.get("resolver") or "").strip()
        kwargs = spec.get("kwargs") or {}
        if resolver_name not in registry:
            resolver_name = heuristic[attr]["resolver"]
            kwargs = heuristic[attr].get("kwargs") or {}
        if not isinstance(kwargs, dict):
            kwargs = {}
        if resolver_name == "prefer_higher_trust":
            kwargs = dict(kwargs)
            kwargs.setdefault("trust_key", config.trust_key)
        out_rules[attr] = {"resolver": resolver_name, "kwargs": kwargs}

    # Ensure we return something for every attribute
    for attr in heuristic.keys():
        if attr not in out_rules:
            out_rules[attr] = heuristic[attr]

    if out_dir:
        cache_path = out_dir / "fusion_rules.json"
        cache_path.write_text(
            json.dumps(
                {"main_entity": main_entity, "attribute_rules": out_rules},
                indent=2,
                sort_keys=True,
                default=str,
            )
        )

    return out_rules


# =============================================================================
# Main Runner
# =============================================================================


def run_fusion_case(
    case: str,
    datasets: Dict[str, pd.DataFrame],
    validation_sets: Dict[str, pd.DataFrame],
    chat_model: Any,
    config: FusionConfig,
    output_dir: Path,
    iterations: int = 3,
    primary_validation_mode: str = "llm_omit",
) -> FusionCaseResult:
    """
    Run a specific fusion case and evaluate against all validation modes.

    Parameters
    ----------
    case : str
        One of: 'heuristic', 'llm_no_val', 'llm_val', 'llm_val_stats', 'heuristic_stats', 'iterative'
    datasets : Dict[str, pd.DataFrame]
        Datasets to fuse
    validation_sets : Dict[str, pd.DataFrame]
        {mode: validation_df} for modes: llm, llm_omit, web, web_omit
    chat_model : Any
        LLM chat model
    config : FusionConfig
        Fusion configuration
    output_dir : Path
        Directory to save outputs
    iterations : int
        Number of iterations for iterative case
    primary_validation_mode : str
        Which validation mode to use for computing stats (used by cases 3-6)

    Returns
    -------
    FusionCaseResult
        Result including rules and accuracy by validation mode
    """
    case_dir = output_dir / f"case_{case}"
    case_dir.mkdir(parents=True, exist_ok=True)

    # Get primary validation set for cases that need it
    primary_val_df = validation_sets.get(primary_validation_mode, pd.DataFrame())

    iterations_used = 1
    source_stats = None

    # Run the appropriate case
    if case == "heuristic":
        rules = _case_heuristic(datasets, config)

    elif case == "llm_no_val":
        rules = _case_llm_no_val(datasets, chat_model, config, case_dir)

    elif case == "llm_val":
        if primary_val_df.empty:
            logger.warning(f"No validation set for mode {primary_validation_mode}, falling back to heuristic")
            rules = _case_heuristic(datasets, config)
        else:
            rules = _case_llm_val(datasets, primary_val_df, chat_model, config, case_dir)

    elif case == "llm_val_stats":
        if primary_val_df.empty:
            logger.warning(f"No validation set for mode {primary_validation_mode}, falling back to heuristic")
            rules = _case_heuristic(datasets, config)
        else:
            rules = _case_llm_val_stats(datasets, primary_val_df, chat_model, config, case_dir)
            source_stats = compute_source_accuracy_from_validation_set(primary_val_df)

    elif case == "heuristic_stats":
        if primary_val_df.empty:
            logger.warning(f"No validation set for mode {primary_validation_mode}, falling back to heuristic")
            rules = _case_heuristic(datasets, config)
        else:
            rules = _case_heuristic_stats(datasets, primary_val_df, config)
            source_stats = compute_source_accuracy_from_validation_set(primary_val_df)

    elif case == "iterative":
        if primary_val_df.empty:
            logger.warning(f"No validation set for mode {primary_validation_mode}, falling back to heuristic")
            rules = _case_heuristic(datasets, config)
        else:
            rules, iterations_used = _case_iterative(
                datasets, primary_val_df, chat_model, config, case_dir, iterations
            )
            source_stats = compute_source_accuracy_from_validation_set(primary_val_df)

    else:
        raise ValueError(f"Unknown fusion case: {case}")

    # Save rules
    rules_path = case_dir / "fusion_rules.json"
    rules_path.write_text(json.dumps({"attribute_rules": rules}, indent=2, sort_keys=True))

    result = FusionCaseResult(
        case_name=case,
        rules=rules,
        source_stats=source_stats,
        iterations_used=iterations_used,
        optimization_mode=primary_validation_mode if case not in ("heuristic", "llm_no_val") else None,
    )

    # Save result summary
    result_path = case_dir / "case_result.json"
    result_path.write_text(json.dumps(result.to_dict(), indent=2, default=str))

    return result


def run_all_fusion_cases(
    datasets: Dict[str, pd.DataFrame],
    validation_sets: Dict[str, pd.DataFrame],
    correspondences: pd.DataFrame,
    chat_model: Any,
    config: FusionConfig,
    output_dir: Path,
    iterations: int = 3,
    primary_validation_mode: str = "llm_omit",
    cases: Optional[List[str]] = None,
    test_set: Optional[pd.DataFrame] = None,
) -> Dict[str, FusionCaseResult]:
    """
    Run all or selected fusion cases and create comparison summary.

    For cases that don't need validation (heuristic, llm_no_val), runs once.
    For cases that need validation (llm_val, llm_val_stats, heuristic_stats, iterative),
    runs once per available validation mode to create 6 cases × 4 modes = 24 configurations.

    Parameters
    ----------
    datasets : Dict[str, pd.DataFrame]
        Datasets to fuse
    validation_sets : Dict[str, pd.DataFrame]
        {mode: validation_df} for modes (used for optimization)
    correspondences : pd.DataFrame
        Entity correspondences for fusion
    chat_model : Any
        LLM chat model
    config : FusionConfig
        Fusion configuration
    output_dir : Path
        Directory to save outputs
    iterations : int
        Number of iterations for iterative case
    primary_validation_mode : str
        Default validation mode (used when running a single mode)
    cases : List[str], optional
        List of cases to run. If None, runs all cases.
    test_set : pd.DataFrame, optional
        Ground truth test set for final evaluation only (not used for optimization).
        Should have 'id' column matching source record IDs.

    Returns
    -------
    Dict[str, FusionCaseResult]
        {case_name_opt_mode: result} - key includes optimization mode for validation-dependent cases
    """
    from .fusion import run_data_fusion
    from PyDI.fusion import DataFusionStrategy

    all_cases = ["heuristic", "llm_no_val", "llm_val", "llm_val_stats", "heuristic_stats", "iterative"]
    cases_to_run = cases if cases else all_cases

    # Cases that don't need validation - run once
    no_val_cases = {"heuristic", "llm_no_val"}

    # Available validation modes (only non-empty ones)
    available_val_modes = [mode for mode, df in validation_sets.items() if not df.empty]

    results: Dict[str, FusionCaseResult] = {}

    for case in cases_to_run:
        # Determine which optimization modes to run for this case
        if case in no_val_cases:
            # Cases without validation: run once with no optimization mode
            opt_modes_to_run = [None]
        else:
            # Cases with validation: run once per available validation mode
            opt_modes_to_run = available_val_modes if available_val_modes else [None]

        for opt_mode in opt_modes_to_run:
            # Create unique key for this configuration
            if opt_mode is None:
                result_key = case
                case_dir_name = f"case_{case}"
            else:
                result_key = f"{case}__opt_{opt_mode}"
                case_dir_name = f"case_{case}__opt_{opt_mode}"

            logger.info(f"Running fusion case: {case} (optimization_mode={opt_mode})")
            print(f"\n{'='*60}")
            print(f"Running fusion case: {case}" + (f" (optimized on {opt_mode})" if opt_mode else ""))
            print(f"{'='*60}")

            # Run the case to get rules
            result = run_fusion_case(
                case=case,
                datasets=datasets,
                validation_sets=validation_sets,
                chat_model=chat_model,
                config=config,
                output_dir=output_dir / case_dir_name.replace(f"case_{case}", ""),  # adjust path
                iterations=iterations,
                primary_validation_mode=opt_mode or primary_validation_mode,
            )

            # Override case_dir since run_fusion_case creates its own subdirectory
            case_dir = output_dir / case_dir_name
            case_dir.mkdir(parents=True, exist_ok=True)

            # Build strategy from rules
            from PyDI.fusion import DataFusionStrategy
            strategy = DataFusionStrategy(f"{case}_strategy")
            registry = _resolver_registry()
            for attr, spec in result.rules.items():
                resolver = registry.get(spec.get("resolver"))
                if resolver:
                    kwargs = spec.get("kwargs") or {}
                    strategy.add_attribute_fuser(attr, resolver, **kwargs)

            # Add type-aware evaluation functions to strategy
            add_evaluation_functions_to_strategy(
                strategy=strategy,
                datasets=datasets,
                config=config,
                numeric_tolerance=0.2,  # 20% tolerance for numerics
            )

            # Run fusion with this strategy
            fused_df, _, _ = run_data_fusion(
                datasets,
                correspondences=correspondences,
                config=config,
                strategy=strategy,
                output_dir=case_dir,
            )

            # Evaluate against each validation mode using DataFusionEvaluator
            eval_dir = case_dir / "evaluation"
            eval_dir.mkdir(parents=True, exist_ok=True)

            # Create evaluator WITHOUT debug for validation sets (to avoid large logs)
            from PyDI.fusion import DataFusionEvaluator
            evaluator = DataFusionEvaluator(
                strategy,
                debug=False,  # No debug for validation set comparisons
            )

            for mode, val_df in validation_sets.items():
                if val_df.empty:
                    continue

                # Convert validation set to tabular ground truth format
                from .fusion_validation_generation import convert_to_tabular_format
                gt_df, _ = convert_to_tabular_format(val_df, output_dir=None)

                if gt_df.empty:
                    continue

                # Evaluate using DataFusionEvaluator with type-aware matching
                eval_metrics = evaluator.evaluate(
                    fused_df=fused_df,
                    fused_id_column=config.id_column,
                    expected_df=gt_df,
                    expected_id_column="entity_id",
                )
                result.accuracy_by_mode[mode] = eval_metrics.get("overall_accuracy", 0.0)

                # Convert metrics to legacy format for compatibility
                eval_result = {
                    "overall_accuracy": eval_metrics.get("overall_accuracy", 0.0),
                    "total_comparisons": eval_metrics.get("total_evaluations", 0),
                    "correct_comparisons": eval_metrics.get("total_correct", 0),
                    "per_attribute_accuracy": {},
                }
                # Extract per-attribute accuracy from metrics
                for key, value in eval_metrics.items():
                    if key.endswith("_accuracy") and key != "overall_accuracy" and key != "macro_accuracy":
                        attr = key.replace("_accuracy", "")
                        count_key = f"{attr}_count"
                        eval_result["per_attribute_accuracy"][attr] = {
                            "accuracy": value,
                            "total": eval_metrics.get(count_key, 0),
                        }

                # Save evaluation
                eval_path = eval_dir / f"eval_{mode}.json"
                eval_path.write_text(json.dumps(eval_result, indent=2, default=str))
                print(f"  {mode}: {eval_result['overall_accuracy']:.1%} accuracy "
                      f"({eval_result['correct_comparisons']}/{eval_result['total_comparisons']})")

            # NOTE: Test set evaluation is deferred to after best case selection
            # to avoid any possibility of data leakage during optimization

            results[result_key] = result

    # Create validation summary (one row per case x optimization_mode x eval_mode)
    val_rows = []
    test_rows = []
    for result_key, result in results.items():
        # Determine case_dir based on result_key (which may include __opt_mode suffix)
        if "__opt_" in result_key:
            case_dir_name = f"case_{result_key}"
        else:
            case_dir_name = f"case_{result_key}"
        case_dir = output_dir / case_dir_name
        eval_dir = case_dir / "evaluation"

        for mode, acc in result.accuracy_by_mode.items():
            row = {
                "case": result.case_name,
                "optimization_mode": result.optimization_mode or "none",
                "eval_mode": mode,
                "accuracy": acc,
            }

            # Load per-attribute accuracy from evaluation file
            eval_path = eval_dir / f"eval_{mode}.json"
            if eval_path.exists():
                eval_data = json.loads(eval_path.read_text())
                row["correct"] = eval_data.get("correct_comparisons", 0)
                row["total"] = eval_data.get("total_comparisons", 0)

                # Add per-attribute accuracy columns
                for attr, attr_stats in eval_data.get("per_attribute_accuracy", {}).items():
                    row[f"acc_{attr}"] = attr_stats.get("accuracy", 0)

            # Separate test set results from validation results
            if mode == "test":
                test_rows.append(row)
            else:
                val_rows.append(row)

    # Save validation summary
    if val_rows:
        val_df = pd.DataFrame(val_rows)
        val_path = output_dir / "comparison_summary_validation.csv"
        val_df.to_csv(val_path, index=False)
        print(f"\nValidation summary saved to {val_path}")

    # Save test set summary separately
    if test_rows:
        test_df = pd.DataFrame(test_rows)
        test_path = output_dir / "comparison_summary_test.csv"
        test_df.to_csv(test_path, index=False)
        print(f"Test set summary saved to {test_path}")

    # Also save fusion rules comparison
    rules_rows = []
    for result_key, result in results.items():
        for attr, spec in result.rules.items():
            rules_rows.append({
                "case": result.case_name,
                "optimization_mode": result.optimization_mode or "none",
                "attribute": attr,
                "resolver": spec.get("resolver"),
                "kwargs": json.dumps(spec.get("kwargs", {})),
            })

    if rules_rows:
        rules_df = pd.DataFrame(rules_rows)
        rules_path = output_dir / "fusion_rules_comparison.csv"
        rules_df.to_csv(rules_path, index=False)
        print(f"Fusion rules comparison saved to {rules_path}")

    # Save source accuracy stats (only for cases that computed them)
    source_rows = []
    for result_key, result in results.items():
        if result.source_stats:
            for attr, sources in result.source_stats.items():
                for src_name, stats in sources.items():
                    source_rows.append({
                        "case": result.case_name,
                        "optimization_mode": result.optimization_mode or "none",
                        "attribute": attr,
                        "source": src_name,
                        "correct": stats.correct,
                        "total": stats.total,
                        "accuracy": stats.accuracy,
                    })

    if source_rows:
        source_df = pd.DataFrame(source_rows)
        source_path = output_dir / "source_accuracy_stats.csv"
        source_df.to_csv(source_path, index=False)
        print(f"Source accuracy stats saved to {source_path}")

    # Find the best case based on primary validation mode accuracy
    # NOTE: Never use test set for selection - only for final evaluation
    best_case_key = None
    best_accuracy = -1.0
    for result_key, result in results.items():
        # Use primary validation mode for selection (never test set)
        if primary_validation_mode in result.accuracy_by_mode:
            acc = result.accuracy_by_mode[primary_validation_mode]
        else:
            # Fall back to average of all validation modes (excluding test)
            accs = [a for m, a in result.accuracy_by_mode.items() if m != "test"]
            acc = sum(accs) / len(accs) if accs else 0.0

        if acc > best_accuracy:
            best_accuracy = acc
            best_case_key = result_key

    # Determine the directory for the best case
    best_case_dir = None
    if best_case_key:
        if "__opt_" in best_case_key:
            best_case_dir = output_dir / f"case_{best_case_key}"
        else:
            best_case_dir = output_dir / f"case_{best_case_key}"

        print(f"\n{'='*60}")
        print(f"BEST CASE: {best_case_key} (validation accuracy: {best_accuracy:.1%})")
        print(f"{'='*60}")

        # Evaluate best case on test set (if provided) - final evaluation only
        test_accuracy = None
        if test_set is not None and not test_set.empty:
            print(f"\nEvaluating best case on held-out test set...")
            best_fused_path = best_case_dir / "fused.csv"
            if best_fused_path.exists():
                best_fused_df = pd.read_csv(best_fused_path)
                best_result = results[best_case_key]

                # Rebuild strategy from rules for evaluation
                from PyDI.fusion import DataFusionStrategy, DataFusionEvaluator
                strategy = DataFusionStrategy(f"{best_case_key}_strategy")
                registry = _resolver_registry()
                for attr, spec in best_result.rules.items():
                    resolver = registry.get(spec.get("resolver"))
                    if resolver:
                        kwargs = spec.get("kwargs") or {}
                        strategy.add_attribute_fuser(attr, resolver, **kwargs)

                # Add type-aware evaluation functions
                add_evaluation_functions_to_strategy(
                    strategy=strategy,
                    datasets=datasets,
                    config=config,
                    numeric_tolerance=0.2,
                )

                # Create evaluator with debug logging
                eval_dir = best_case_dir / "evaluation"
                eval_dir.mkdir(parents=True, exist_ok=True)
                test_evaluator = DataFusionEvaluator(
                    strategy,
                    debug=True,
                    debug_file=eval_dir / "eval_test_debug.jsonl",
                )

                # Convert list columns in test_set to strings to match fused CSV format
                test_set_eval = test_set.copy()
                for col in test_set_eval.columns:
                    if test_set_eval[col].apply(lambda x: isinstance(x, list)).any():
                        test_set_eval[col] = test_set_eval[col].apply(
                            lambda x: str(x) if isinstance(x, list) else x
                        )

                test_metrics = test_evaluator.evaluate(
                    fused_df=best_fused_df,
                    fused_id_column=config.id_column,
                    expected_df=test_set_eval,
                    expected_id_column="id",
                )
                test_accuracy = test_metrics.get("overall_accuracy", 0.0)
                results[best_case_key].accuracy_by_mode["test"] = test_accuracy

                # Save test evaluation results
                test_eval = {
                    "overall_accuracy": test_accuracy,
                    "total_comparisons": test_metrics.get("total_evaluations", 0),
                    "correct_comparisons": test_metrics.get("total_correct", 0),
                    "per_attribute_accuracy": {},
                }
                for key, value in test_metrics.items():
                    if key.endswith("_accuracy") and key not in ("overall_accuracy", "macro_accuracy"):
                        attr = key.replace("_accuracy", "")
                        count_key = f"{attr}_count"
                        test_eval["per_attribute_accuracy"][attr] = {
                            "accuracy": value,
                            "total": test_metrics.get(count_key, 0),
                        }

                test_eval_path = eval_dir / "eval_test.json"
                test_eval_path.write_text(json.dumps(test_eval, indent=2, default=str))
                print(f"  TEST SET: {test_accuracy:.1%} accuracy "
                      f"({test_eval['correct_comparisons']}/{test_eval['total_comparisons']})")

                # Copy debug file to main output dir
                import shutil
                dest_debug = output_dir / "best_eval_test_debug.jsonl"
                shutil.copy(eval_dir / "eval_test_debug.jsonl", dest_debug)

        # Save best case info
        best_info = {
            "best_case_key": best_case_key,
            "best_case_name": results[best_case_key].case_name,
            "best_validation_accuracy": best_accuracy,
            "best_test_accuracy": test_accuracy,
            "best_case_dir": str(best_case_dir),
            "rules": results[best_case_key].rules,
        }
        best_info_path = output_dir / "best_case.json"
        best_info_path.write_text(json.dumps(best_info, indent=2, default=str))

    return results, best_case_key, best_case_dir
