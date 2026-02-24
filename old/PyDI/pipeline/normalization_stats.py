"""
Normalization Statistics Module

Reports per-dataset normalization counts:
- LLM taxonomy normalizations (from taxonomy mapping caches)
- Built-in normalizations (from pipeline_stats.json)
"""

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def load_json(path: Path) -> Dict:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def generate_normalization_stats(output_dir: Path) -> pd.DataFrame:
    """Generate per-dataset normalization statistics.

    LLM stats come from taxonomy cache files (source_value_counts_by_dataset).
    Built-in stats come from pipeline_stats.json (transformed/failed counts).
    """
    output_dir = Path(output_dir)
    cache_dir = output_dir / "schema_matching"

    # Load pipeline stats for built-in normalization counts
    pipeline_stats_path = cache_dir / "pipeline_stats.json"
    pipeline_stats = load_json(pipeline_stats_path) if pipeline_stats_path.exists() else {}

    # Get datasets from pipeline stats
    datasets = set(pipeline_stats.keys())

    # Collect LLM counts per dataset from all taxonomy caches
    llm_counts_by_dataset: Dict[str, Dict[str, int]] = {}  # {dataset: {normalized: X, total: Y}}
    llm_columns_by_dataset: Dict[str, int] = {}  # {dataset: num_columns}

    if cache_dir.exists():
        for cache_path in cache_dir.glob("*_taxonomy_mapping.json"):
            cache = load_json(cache_path)
            mappings = cache.get("mappings", {})
            counts_by_dataset = cache.get("source_value_counts_by_dataset", {})

            for dataset_name, source_counts in counts_by_dataset.items():
                if dataset_name not in llm_counts_by_dataset:
                    llm_counts_by_dataset[dataset_name] = {"normalized": 0, "total": 0}
                    llm_columns_by_dataset[dataset_name] = 0

                # Each taxonomy cache file = one column normalized for this dataset
                llm_columns_by_dataset[dataset_name] += 1

                for source_val, count in source_counts.items():
                    llm_counts_by_dataset[dataset_name]["total"] += count
                    if mappings.get(source_val) is not None:
                        llm_counts_by_dataset[dataset_name]["normalized"] += count

                # Also add this dataset to our list
                datasets.add(dataset_name)

    rows = []
    for dataset_name in sorted(datasets):
        # Get LLM counts for this dataset
        llm_stats = llm_counts_by_dataset.get(dataset_name, {"normalized": 0, "total": 0})

        # Get built-in normalization counts from pipeline_stats
        stats = pipeline_stats.get(dataset_name, {})
        transformed = stats.get("transformed", 0)
        failed = stats.get("failed", 0)
        builtin_columns = stats.get("columns_normalized", 0)

        rows.append({
            "dataset": dataset_name,
            "llm_columns": llm_columns_by_dataset.get(dataset_name, 0),
            "llm_normalized": llm_stats["normalized"],
            "llm_total": llm_stats["total"],
            "builtin_columns": builtin_columns,
            "builtin_normalized": transformed,
            "builtin_total": transformed + failed,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        totals = pd.DataFrame([{
            "dataset": "TOTAL",
            "llm_columns": df["llm_columns"].sum(),
            "llm_normalized": df["llm_normalized"].sum(),
            "llm_total": df["llm_total"].sum(),
            "builtin_columns": df["builtin_columns"].sum(),
            "builtin_normalized": df["builtin_normalized"].sum(),
            "builtin_total": df["builtin_total"].sum(),
        }])
        df = pd.concat([df, totals], ignore_index=True)

    return df


def save_normalization_stats(
    output_dir: Path,
    filename: str = "normalization_stats",
) -> Optional[Path]:
    """Generate and save normalization statistics."""
    df = generate_normalization_stats(output_dir)

    if df.empty:
        return None

    reporting_dir = Path(output_dir) / "reporting"
    reporting_dir.mkdir(parents=True, exist_ok=True)

    output_path = reporting_dir / f"{filename}.csv"
    df.to_csv(output_path, index=False)

    return output_path
