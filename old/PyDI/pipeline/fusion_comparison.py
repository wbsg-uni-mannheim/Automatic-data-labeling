"""
Fusion comparison report generation.

Compares auto-generated fusion rules vs provided validation set optimized rules,
similar to training_comparison for entity matching.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def load_eval_results(case_dir: Path, eval_mode: str) -> Dict[str, Any]:
    """Load evaluation results for a specific mode from case directory."""
    eval_path = case_dir / "evaluation" / f"eval_{eval_mode}.json"
    if eval_path.exists():
        with open(eval_path) as f:
            return json.load(f)
    return {}


def generate_fusion_comparison_report(
    output_dir: Path,
    auto_dir: Optional[Path] = None,
    provided_dir: Optional[Path] = None,
    auto_best_info: Optional[Dict[str, Any]] = None,
    provided_best_info: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """
    Generate fusion comparison report with per-attribute accuracy.

    Parameters
    ----------
    output_dir : Path
        Directory to save the report
    auto_dir : Path
        Directory containing auto-generated best case results
    provided_dir : Path
        Directory containing provided validation best case results
    auto_best_info : Dict
        Auto-generated best case info
    provided_best_info : Dict
        Provided validation best case info

    Returns
    -------
    Path or None
        Path to saved CSV report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    # Collect all attributes from both evaluations
    all_attributes = set()

    # Load auto evaluation results
    auto_eval = {}
    if auto_dir and auto_dir.exists():
        auto_eval = load_eval_results(auto_dir, "test")
        per_attr = auto_eval.get("per_attribute_accuracy", {})
        all_attributes.update(per_attr.keys())

    # Load provided evaluation results
    provided_eval = {}
    if provided_dir and provided_dir.exists():
        provided_eval = load_eval_results(provided_dir, "test")
        per_attr = provided_eval.get("per_attribute_accuracy", {})
        all_attributes.update(per_attr.keys())

    # Build auto row
    if auto_best_info:
        row = {
            "variant": "auto",
            "case": auto_best_info.get("best_case_key", "unknown"),
            "validation_accuracy": auto_best_info.get("best_validation_accuracy"),
            "test_accuracy": auto_best_info.get("best_test_accuracy"),
            "test_correct": auto_eval.get("correct_comparisons"),
            "test_total": auto_eval.get("total_comparisons"),
        }
        # Add per-attribute accuracy
        per_attr = auto_eval.get("per_attribute_accuracy", {})
        for attr in sorted(all_attributes):
            attr_data = per_attr.get(attr, {})
            row[f"acc_{attr}"] = attr_data.get("accuracy")
        rows.append(row)

    # Build provided row
    if provided_best_info:
        row = {
            "variant": "provided",
            "case": provided_best_info.get("best_case_key", "unknown"),
            "validation_accuracy": provided_best_info.get("best_validation_accuracy"),
            "test_accuracy": provided_best_info.get("best_test_accuracy"),
            "test_correct": provided_eval.get("correct_comparisons"),
            "test_total": provided_eval.get("total_comparisons"),
        }
        # Add per-attribute accuracy
        per_attr = provided_eval.get("per_attribute_accuracy", {})
        for attr in sorted(all_attributes):
            attr_data = per_attr.get(attr, {})
            row[f"acc_{attr}"] = attr_data.get("accuracy")
        rows.append(row)

    if not rows:
        return None

    df = pd.DataFrame(rows)

    # Save summary CSV
    report_path = output_dir / "fusion_comparison_summary.csv"
    df.to_csv(report_path, index=False)

    # Also save detailed JSON with rules comparison
    details = {
        "auto": {
            "best_case_key": auto_best_info.get("best_case_key") if auto_best_info else None,
            "validation_accuracy": auto_best_info.get("best_validation_accuracy") if auto_best_info else None,
            "test_accuracy": auto_best_info.get("best_test_accuracy") if auto_best_info else None,
            "per_attribute_accuracy": auto_eval.get("per_attribute_accuracy", {}),
            "rules": auto_best_info.get("rules") if auto_best_info else None,
        },
        "provided": {
            "best_case_key": provided_best_info.get("best_case_key") if provided_best_info else None,
            "validation_accuracy": provided_best_info.get("best_validation_accuracy") if provided_best_info else None,
            "test_accuracy": provided_best_info.get("best_test_accuracy") if provided_best_info else None,
            "per_attribute_accuracy": provided_eval.get("per_attribute_accuracy", {}),
            "rules": provided_best_info.get("rules") if provided_best_info else None,
        },
    }

    json_path = output_dir / "fusion_comparison_details.json"
    with open(json_path, "w") as f:
        json.dump(details, f, indent=2, default=str)

    return report_path
