"""End-to-end metrics for data integration pipeline evaluation.

This module provides metrics to measure the structural improvement
achieved by the data integration pipeline from input sources to fused output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def calculate_density(df: pd.DataFrame, exclude_prefixes: tuple = ("_fusion_",)) -> float:
    """Calculate the density (non-null ratio) of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.
    exclude_prefixes : tuple
        Column prefixes to exclude from calculation (e.g., metadata columns).

    Returns
    -------
    float
        Density as a ratio between 0 and 1.
    """
    # Filter out metadata columns
    data_cols = [c for c in df.columns if not any(c.startswith(p) for p in exclude_prefixes)]

    if not data_cols or len(df) == 0:
        return 0.0

    subset = df[data_cols]
    total_cells = subset.size
    non_null_cells = subset.notna().sum().sum()

    return non_null_cells / total_cells if total_cells > 0 else 0.0


@dataclass
class SourceStats:
    """Statistics for a single input source."""

    name: str
    rows: int
    columns: int
    density: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "rows": self.rows,
            "columns": self.columns,
            "density": round(self.density, 4),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceStats":
        return cls(
            name=data["name"],
            rows=data["rows"],
            columns=data["columns"],
            density=data["density"],
        )


@dataclass
class EndToEndMetrics:
    """End-to-end metrics for data integration pipeline.

    Captures the key structural metrics measuring improvement from
    input sources to fused output.
    """

    # Input statistics
    num_sources: int = 0
    total_input_rows: int = 0
    total_input_columns: int = 0  # columns in target schema
    avg_input_density: float = 0.0
    per_source_stats: List[SourceStats] = field(default_factory=list)

    # Output statistics
    fused_rows: int = 0
    fused_columns: int = 0
    fused_density: float = 0.0

    # Row gain (vs largest source)
    max_source_rows: int = 0
    row_gain_over_largest: int = 0
    row_gain_pct: float = 0.0
    largest_source_name: str = ""
    largest_source_density: float = 0.0

    # Merge statistics
    merged_records: int = 0

    # Density change
    density_change: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics to dictionary."""
        fusion_ratio = 1 - (self.fused_rows / self.total_input_rows) if self.total_input_rows > 0 else 0.0
        density_change_vs_largest = self.fused_density - self.largest_source_density

        return {
            "input_sources": self.num_sources,
            "total_input_records": self.total_input_rows,
            "fused_output_records": self.fused_rows,
            "fusion_ratio": round(fusion_ratio, 4),
            "row_gain_over_largest": self.row_gain_over_largest,
            "row_gain_percentage": round(self.row_gain_pct, 2),
            "merged_records": self.merged_records,
            "input_columns_target_schema": self.total_input_columns,
            "fused_output_columns": self.fused_columns,
            "average_input_density": round(self.avg_input_density, 4),
            "fused_data_density": round(self.fused_density, 4),
            "data_density_change": round(self.density_change, 4),
            "largest_source_name": self.largest_source_name,
            "largest_source_density": round(self.largest_source_density, 4),
            "density_change_vs_largest": round(density_change_vs_largest, 4),
            "per_source_stats": [s.to_dict() for s in self.per_source_stats],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EndToEndMetrics":
        """Deserialize metrics from dictionary."""
        per_source = [SourceStats.from_dict(s) for s in data.get("per_source_stats", [])]

        return cls(
            num_sources=data.get("input_sources", data.get("num_sources", 0)),
            total_input_rows=data.get("total_input_records", data.get("total_input_rows", 0)),
            total_input_columns=data.get("input_columns_target_schema", data.get("total_input_columns", 0)),
            avg_input_density=data.get("average_input_density", data.get("avg_input_density", 0.0)),
            per_source_stats=per_source,
            fused_rows=data.get("fused_output_records", data.get("fused_rows", 0)),
            fused_columns=data.get("fused_output_columns", data.get("fused_columns", 0)),
            fused_density=data.get("fused_data_density", data.get("fused_density", 0.0)),
            max_source_rows=data.get("max_source_rows", 0),
            row_gain_over_largest=data.get("row_gain_over_largest", 0),
            row_gain_pct=data.get("row_gain_percentage", data.get("row_gain_pct", 0.0)),
            largest_source_name=data.get("largest_source_name", ""),
            largest_source_density=data.get("largest_source_density", 0.0),
            merged_records=data.get("merged_records", 0),
            density_change=data.get("data_density_change", data.get("density_change", 0.0)),
        )

    def save(self, path: Path) -> None:
        """Save metrics to JSON file."""
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> Optional["EndToEndMetrics"]:
        """Load metrics from JSON file."""
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return cls.from_dict(data)
        except Exception:
            return None

    def summary_table(self) -> pd.DataFrame:
        """Create a summary table of key metrics."""
        fusion_ratio = 1 - (self.fused_rows / self.total_input_rows) if self.total_input_rows > 0 else 0.0
        density_change_vs_largest = (self.fused_density - self.largest_source_density) * 100

        rows = [
            {"Metric": "Input Sources", "Value": str(self.num_sources)},
            {"Metric": "Total Input Records", "Value": f"{self.total_input_rows:,}"},
            {"Metric": "Fused Output Records", "Value": f"{self.fused_rows:,}"},
            {"Metric": "Fusion Ratio", "Value": f"{fusion_ratio:.1%}"},
            {"Metric": "Row Gain Over Largest Source", "Value": f"{self.row_gain_over_largest:+,}"},
            {"Metric": "Row Gain Percentage", "Value": f"{self.row_gain_pct:+.1f}%"},
            {"Metric": "Merged Records", "Value": f"{self.merged_records:,}"},
            {"Metric": "Input Columns (Target Schema)", "Value": str(self.total_input_columns)},
            {"Metric": "Fused Output Columns", "Value": str(self.fused_columns)},
            {"Metric": "Average Input Data Density", "Value": f"{self.avg_input_density * 100:.1f}%"},
            {"Metric": "Fused Data Density", "Value": f"{self.fused_density * 100:.1f}%"},
            {"Metric": "Data Density Change", "Value": f"{self.density_change * 100:+.1f}%"},
            {"Metric": f"Largest Source Density ({self.largest_source_name})", "Value": f"{self.largest_source_density * 100:.1f}%"},
            {"Metric": "Density Change vs Largest", "Value": f"{density_change_vs_largest:+.1f}%"},
        ]

        return pd.DataFrame(rows)

    def per_source_table(self) -> pd.DataFrame:
        """Create a table showing per-source statistics."""
        if not self.per_source_stats:
            return pd.DataFrame()

        rows = []
        for stat in self.per_source_stats:
            rows.append({
                "Source": stat.name,
                "Rows": stat.rows,
                "Columns": stat.columns,
                "Density": f"{stat.density * 100:.1f}%",
            })

        return pd.DataFrame(rows)


def calculate_merge_count(fused_df: pd.DataFrame) -> int:
    """Count records that came from 2+ sources.

    Parameters
    ----------
    fused_df : pd.DataFrame
        The fused output DataFrame.

    Returns
    -------
    int
        Number of records merged from multiple sources.
    """
    # Check for _fusion_source_datasets column
    col_name = None
    if "_fusion_source_datasets" in fused_df.columns:
        col_name = "_fusion_source_datasets"
    elif "_fusion_sources" in fused_df.columns:
        col_name = "_fusion_sources"
    else:
        return 0

    merged_count = 0
    for idx, row in fused_df.iterrows():
        val = row[col_name]
        if pd.isna(val) or val == "" or val == "[]":
            continue

        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                count = len(parsed) if isinstance(parsed, list) else 1
            except (json.JSONDecodeError, TypeError):
                count = len([s.strip() for s in val.split(",") if s.strip()])
        elif isinstance(val, list):
            count = len(val)
        else:
            count = 1

        if count > 1:
            merged_count += 1

    return merged_count


def calculate_structural_metrics(
    datasets: Dict[str, pd.DataFrame],
    fused_df: pd.DataFrame,
    debug_file: Optional[Path] = None,
) -> EndToEndMetrics:
    """Calculate end-to-end structural metrics.

    Parameters
    ----------
    datasets : Dict[str, pd.DataFrame]
        Dictionary mapping source names to their DataFrames.
    fused_df : pd.DataFrame
        The fused output DataFrame.
    debug_file : Optional[Path]
        Path to fusion_debug.jsonl file (not used, kept for compatibility).

    Returns
    -------
    EndToEndMetrics
        Computed metrics.
    """
    # Get unified schema (all columns across all sources, excluding metadata)
    all_columns = set()
    for df in datasets.values():
        all_columns.update(c for c in df.columns if not c.startswith("_"))

    # Calculate per-source statistics
    per_source_stats = []
    total_non_null_unified = 0
    total_cells_unified = 0

    for name, df in datasets.items():
        # Density against source's own columns
        own_density = calculate_density(df)

        # Density against unified schema
        source_cols = [c for c in df.columns if not c.startswith("_")]
        non_null_unified = df[source_cols].notna().sum().sum()
        cells_unified = len(df) * len(all_columns)
        total_non_null_unified += non_null_unified
        total_cells_unified += cells_unified

        stats = SourceStats(
            name=name,
            rows=len(df),
            columns=len(source_cols),
            density=own_density,
        )
        per_source_stats.append(stats)

    # Input aggregates
    num_sources = len(datasets)
    total_input_rows = sum(s.rows for s in per_source_stats)
    total_input_columns = len(all_columns)
    avg_input_density = total_non_null_unified / total_cells_unified if total_cells_unified > 0 else 0.0

    # Output statistics
    fused_density = calculate_density(fused_df)
    fused_data_cols = [c for c in fused_df.columns if not c.startswith("_fusion_")]
    fused_rows = len(fused_df)

    # Row gain metrics (vs largest source)
    max_source_rows = max((s.rows for s in per_source_stats), default=0)
    row_gain_over_largest = fused_rows - max_source_rows
    row_gain_pct = (row_gain_over_largest / max_source_rows * 100) if max_source_rows > 0 else 0.0

    # Find largest source and its density
    largest_source_name = ""
    largest_source_density = 0.0
    for stat in per_source_stats:
        if stat.rows == max_source_rows:
            largest_source_name = stat.name
            largest_source_density = stat.density
            break

    # Merged records count
    merged_records = calculate_merge_count(fused_df)

    # Density change
    density_change = fused_density - avg_input_density

    return EndToEndMetrics(
        num_sources=num_sources,
        total_input_rows=total_input_rows,
        total_input_columns=total_input_columns,
        avg_input_density=avg_input_density,
        per_source_stats=per_source_stats,
        fused_rows=fused_rows,
        fused_columns=len(fused_data_cols),
        fused_density=fused_density,
        max_source_rows=max_source_rows,
        row_gain_over_largest=row_gain_over_largest,
        row_gain_pct=row_gain_pct,
        largest_source_name=largest_source_name,
        largest_source_density=largest_source_density,
        merged_records=merged_records,
        density_change=density_change,
    )


def generate_end_to_end_report(metrics: EndToEndMetrics) -> str:
    """Generate a formatted end-to-end metrics report.

    Parameters
    ----------
    metrics : EndToEndMetrics
        The computed end-to-end metrics.

    Returns
    -------
    str
        Formatted report string.
    """
    fusion_ratio = 1 - (metrics.fused_rows / metrics.total_input_rows) if metrics.total_input_rows > 0 else 0.0
    density_vs_largest = (metrics.fused_density - metrics.largest_source_density) * 100

    lines = []
    lines.append("=" * 60)
    lines.append("END-TO-END INTEGRATION METRICS")
    lines.append("=" * 60)

    lines.append("")
    lines.append(f"{'Input Sources:':<35} {metrics.num_sources}")
    lines.append(f"{'Total Input Records:':<35} {metrics.total_input_rows:,}")
    lines.append(f"{'Fused Output Records:':<35} {metrics.fused_rows:,}")
    lines.append(f"{'Fusion Ratio:':<35} {fusion_ratio:.1%}")

    lines.append("")
    lines.append(f"{'Row Gain Over Largest Source:':<35} {metrics.row_gain_over_largest:+,}")
    lines.append(f"{'Row Gain Percentage:':<35} {metrics.row_gain_pct:+.1f}%")
    lines.append(f"{'Merged Records:':<35} {metrics.merged_records:,}")

    lines.append("")
    lines.append(f"{'Input Columns (Target Schema):':<35} {metrics.total_input_columns}")
    lines.append(f"{'Fused Output Columns:':<35} {metrics.fused_columns}")

    lines.append("")
    lines.append(f"{'Average Input Data Density:':<35} {metrics.avg_input_density * 100:.1f}%")
    lines.append(f"{'Fused Data Density:':<35} {metrics.fused_density * 100:.1f}%")
    lines.append(f"{'Data Density Change:':<35} {metrics.density_change * 100:+.1f}%")

    lines.append("")
    lines.append(f"{'Largest Source:':<35} {metrics.largest_source_name}")
    lines.append(f"{'Largest Source Density:':<35} {metrics.largest_source_density * 100:.1f}%")
    lines.append(f"{'Density Change vs Largest:':<35} {density_vs_largest:+.1f}%")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def save_end_to_end_report(
    metrics: EndToEndMetrics,
    output_dir: Path,
    *,
    filename: str = "end_to_end_report",
) -> Tuple[Path, Path]:
    """Save end-to-end metrics report to files.

    Parameters
    ----------
    metrics : EndToEndMetrics
        The computed end-to-end metrics.
    output_dir : Path
        Directory to save reports.
    filename : str
        Base filename (without extension).

    Returns
    -------
    Tuple[Path, Path]
        Paths to the saved text report and CSV table.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save text report
    report_text = generate_end_to_end_report(metrics)
    txt_path = output_dir / f"{filename}.txt"
    txt_path.write_text(report_text)

    # Save CSV table
    report_df = metrics.summary_table()
    csv_path = output_dir / f"{filename}.csv"
    report_df.to_csv(csv_path, index=False)

    return txt_path, csv_path
