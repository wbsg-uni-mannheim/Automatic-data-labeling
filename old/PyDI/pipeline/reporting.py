"""Runtime and token usage tracking for pipeline steps."""

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import pandas as pd
from langchain_community.callbacks import get_openai_callback


@dataclass
class StepMetrics:
    """Metrics collected for a single pipeline step."""

    name: str
    runtime_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    embedding_tokens: int = 0
    total_cost_usd: float = 0.0
    successful_requests: int = 0
    web_search_count: int = 0
    from_cache: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "runtime_seconds": self.runtime_seconds,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "embedding_tokens": self.embedding_tokens,
            "total_cost_usd": self.total_cost_usd,
            "successful_requests": self.successful_requests,
            "web_search_count": self.web_search_count,
            "from_cache": self.from_cache,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StepMetrics":
        return cls(
            name=data["name"],
            runtime_seconds=data.get("runtime_seconds", 0.0),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            embedding_tokens=data.get("embedding_tokens", 0),
            total_cost_usd=data.get("total_cost_usd", 0.0),
            successful_requests=data.get("successful_requests", 0),
            web_search_count=data.get("web_search_count", 0),
            from_cache=data.get("from_cache", False),
        )


@dataclass
class PipelineMetrics:
    """Aggregated metrics for the entire pipeline run."""

    steps: list[StepMetrics] = field(default_factory=list)
    _metrics_path: Path | None = None

    def set_metrics_path(self, path: Path) -> None:
        self._metrics_path = path

    def add(self, step: StepMetrics, accumulate: bool = True) -> None:
        """Add or update step metrics.

        Args:
            step: The step metrics to add
            accumulate: If True, add new values to existing ones (for reruns).
                       If False, replace existing step entirely.
        """
        for i, s in enumerate(self.steps):
            if s.name == step.name:
                if accumulate:
                    # Accumulate metrics from rerun
                    s.runtime_seconds += step.runtime_seconds
                    s.prompt_tokens += step.prompt_tokens
                    s.completion_tokens += step.completion_tokens
                    s.total_tokens += step.total_tokens
                    s.embedding_tokens += step.embedding_tokens
                    s.total_cost_usd += step.total_cost_usd
                    s.successful_requests += step.successful_requests
                    s.web_search_count += step.web_search_count
                    s.from_cache = False  # Mark as fresh since we just ran it
                else:
                    self.steps[i] = step
                self._save()
                return
        self.steps.append(step)
        self._save()

    def get_step(self, name: str) -> StepMetrics | None:
        for s in self.steps:
            if s.name == name:
                return s
        return None

    def total_runtime(self) -> float:
        return sum(s.runtime_seconds for s in self.steps)

    def total_tokens(self) -> int:
        return sum(s.total_tokens for s in self.steps)

    def total_cost(self) -> float:
        return sum(s.total_cost_usd for s in self.steps)

    def total_embedding_tokens(self) -> int:
        return sum(s.embedding_tokens for s in self.steps)

    def total_web_searches(self) -> int:
        return sum(s.web_search_count for s in self.steps)

    def summary_table(self) -> pd.DataFrame:
        """Return a DataFrame summarizing all steps."""
        rows = []
        for s in self.steps:
            rows.append(
                {
                    "step": s.name + (" (cached)" if s.from_cache else ""),
                    "runtime_sec": round(s.runtime_seconds, 2),
                    "prompt_tokens": s.prompt_tokens,
                    "completion_tokens": s.completion_tokens,
                    "embedding_tokens": s.embedding_tokens,
                    "total_tokens": s.total_tokens,
                    "cost_usd": round(s.total_cost_usd, 4),
                    "requests": s.successful_requests,
                    "web_searches": s.web_search_count,
                }
            )
        # Add totals row
        rows.append(
            {
                "step": "TOTAL",
                "runtime_sec": round(self.total_runtime(), 2),
                "prompt_tokens": sum(s.prompt_tokens for s in self.steps),
                "completion_tokens": sum(s.completion_tokens for s in self.steps),
                "embedding_tokens": self.total_embedding_tokens(),
                "total_tokens": self.total_tokens(),
                "cost_usd": round(self.total_cost(), 4),
                "requests": sum(s.successful_requests for s in self.steps),
                "web_searches": self.total_web_searches(),
            }
        )
        return pd.DataFrame(rows)

    def to_dict(self) -> dict:
        return {"steps": [s.to_dict() for s in self.steps]}

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineMetrics":
        metrics = cls()
        for step_data in data.get("steps", []):
            metrics.steps.append(StepMetrics.from_dict(step_data))
        return metrics

    def _save(self) -> None:
        if self._metrics_path:
            try:
                self._metrics_path.write_text(json.dumps(self.to_dict(), indent=2))
            except Exception as e:
                print(f"Warning: Failed to save metrics: {e}")

    @classmethod
    def load(cls, path: Path) -> "PipelineMetrics":
        """Load metrics from JSON file, or return empty metrics if not found."""
        if path.exists():
            try:
                data = json.loads(path.read_text())
                metrics = cls.from_dict(data)
                metrics._metrics_path = path
                # Mark all loaded steps as from_cache
                for step in metrics.steps:
                    step.from_cache = True
                return metrics
            except Exception as e:
                print(f"Warning: Failed to load metrics from {path}: {e}")
        metrics = cls()
        metrics._metrics_path = path
        return metrics


def _print_step_metrics(step: StepMetrics) -> None:
    """Print metrics for a step."""
    cache_note = " (from cache)" if step.from_cache else ""
    embedding_note = f", embedding: {step.embedding_tokens:,}" if step.embedding_tokens > 0 else ""
    web_search_note = f" | Web searches: {step.web_search_count}" if step.web_search_count > 0 else ""
    print(
        f"  [{step.name}]{cache_note} Runtime: {step.runtime_seconds:.1f}s | "
        f"Tokens: {step.total_tokens:,} (prompt: {step.prompt_tokens:,}, "
        f"completion: {step.completion_tokens:,}{embedding_note}) | Cost: ${step.total_cost_usd:.4f}{web_search_note}"
    )


def create_step_tracker(pipeline_metrics: PipelineMetrics):
    """Create step tracking functions bound to a specific PipelineMetrics instance.

    Returns:
        Tuple of (use_cached_step_metrics, track_step) functions.
    """

    def use_cached_step_metrics(name: str) -> None:
        """When skipping a step, show metrics from previous run if available."""
        cached = pipeline_metrics.get_step(name)
        if cached:
            cached.from_cache = True
            _print_step_metrics(cached)
        else:
            print(f"  [{name}] (skipped, no previous metrics available)")

    @contextmanager
    def track_step(name: str, cached: bool = False) -> Generator[None, None, None]:
        """Context manager to track runtime and token usage for a pipeline step.

        Args:
            name: Step name for display and metrics tracking
            cached: If True, preserve previous metrics instead of recording new ones
        """
        # Reset embedding token tracker at start
        embedding_tracker = EmbeddingTokenTracker.get_instance()
        embedding_tracker.get_and_reset()

        # Reset direct OpenAI token tracker at start
        try:
            from PyDI.pipeline.fusion_validation_generation import OpenAITokenTracker
            openai_tracker = OpenAITokenTracker.get_instance()
            openai_tracker.get_and_reset()
        except ImportError:
            openai_tracker = None

        start_time = time.time()
        with get_openai_callback() as cb:
            yield
        elapsed = time.time() - start_time

        # Collect embedding tokens
        embedding_tokens = embedding_tracker.get_and_reset()

        # Collect direct OpenAI API tokens (not captured by LangChain callback)
        direct_prompt_tokens = 0
        direct_completion_tokens = 0
        direct_total_tokens = 0
        web_search_count = 0
        if openai_tracker:
            direct_prompt_tokens, direct_completion_tokens, direct_total_tokens, web_search_count = openai_tracker.get_and_reset()

        # Combine LangChain callback tokens with direct OpenAI API tokens
        total_prompt_tokens = cb.prompt_tokens + direct_prompt_tokens
        total_completion_tokens = cb.completion_tokens + direct_completion_tokens
        combined_total_tokens = cb.total_tokens + direct_total_tokens

        # If cache was used, still accumulate any runtime/tokens from this run
        # (e.g., time spent loading cache, or any minor API calls)
        if cached:
            prev = pipeline_metrics.get_step(name)
            if prev and (prev.total_tokens > 0 or prev.total_cost_usd > 0 or prev.embedding_tokens > 0):
                # Accumulate this run's metrics (even if minimal) to the cached step
                prev.runtime_seconds += elapsed
                prev.prompt_tokens += total_prompt_tokens
                prev.completion_tokens += total_completion_tokens
                prev.total_tokens += combined_total_tokens
                prev.embedding_tokens += embedding_tokens
                prev.total_cost_usd += cb.total_cost
                prev.successful_requests += cb.successful_requests
                prev.web_search_count += web_search_count
                prev.from_cache = True  # Still mark as cached since main work was from cache
                pipeline_metrics._save()
                _print_step_metrics(prev)
                return

        # Record new metrics (will be accumulated with any existing metrics)
        new_step = StepMetrics(
            name=name,
            runtime_seconds=elapsed,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=combined_total_tokens,
            embedding_tokens=embedding_tokens,
            total_cost_usd=cb.total_cost,
            successful_requests=cb.successful_requests,
            web_search_count=web_search_count,
            from_cache=False,
        )
        pipeline_metrics.add(new_step)

        # Print the accumulated totals (not just this run's values)
        accumulated_step = pipeline_metrics.get_step(name)
        _print_step_metrics(accumulated_step)

    return use_cached_step_metrics, track_step


class EmbeddingTokenTracker:
    """Thread-safe tracker for embedding token usage.

    Use as a context manager or call add_tokens() directly.
    The track_step context manager will automatically collect
    tokens from the global instance.
    """

    _instance: "EmbeddingTokenTracker | None" = None

    def __init__(self):
        self._tokens = 0
        self._lock = __import__("threading").Lock()

    @classmethod
    def get_instance(cls) -> "EmbeddingTokenTracker":
        """Get or create the global singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def add_tokens(self, tokens: int) -> None:
        """Add tokens to the current count."""
        with self._lock:
            self._tokens += tokens

    def get_and_reset(self) -> int:
        """Get current token count and reset to zero."""
        with self._lock:
            tokens = self._tokens
            self._tokens = 0
            return tokens

    def __enter__(self) -> "EmbeddingTokenTracker":
        # Reset at start of context
        self.get_and_reset()
        return self

    def __exit__(self, *args) -> None:
        pass


def get_embedding_token_tracker() -> EmbeddingTokenTracker:
    """Get the global embedding token tracker instance."""
    return EmbeddingTokenTracker.get_instance()


# =============================================================================
# Data Quality and Source Statistics Reporting
# =============================================================================

@dataclass
class SourceStats:
    """Statistics for a single data source."""
    name: str
    rows: int
    columns: int
    data_density: float  # Percentage of non-null values
    target_schema_density: float  # Adjusted for target schema coverage
    target_columns: int = 9  # Default target schema size

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "rows": self.rows,
            "columns": self.columns,
            "data_density": self.data_density,
            "target_schema_density": self.target_schema_density,
            "target_columns": self.target_columns,
        }


def calculate_data_density(df: pd.DataFrame, exclude_prefixes: list[str] = None) -> float:
    """Calculate data density (percentage of non-null, non-empty values).

    Args:
        df: DataFrame to analyze
        exclude_prefixes: Column prefixes to exclude (e.g., ['_'] for metadata columns)

    Returns:
        Data density as a percentage (0-100)
    """
    if exclude_prefixes is None:
        exclude_prefixes = ['_']

    cols = [c for c in df.columns if not any(c.startswith(p) for p in exclude_prefixes)]

    if not cols or len(df) == 0:
        return 0.0

    total_cells = len(df) * len(cols)
    filled_cells = 0

    for col in cols:
        if df[col].dtype == 'object':
            filled = (df[col].notna() & (df[col].astype(str).str.strip() != '')).sum()
        else:
            filled = df[col].notna().sum()
        filled_cells += filled

    return (filled_cells / total_cells) * 100 if total_cells > 0 else 0.0


def calculate_source_stats(
    df: pd.DataFrame,
    name: str,
    target_columns: int = 9,
) -> SourceStats:
    """Calculate statistics for a single data source.

    Args:
        df: Source DataFrame
        name: Name of the source
        target_columns: Number of columns in target schema

    Returns:
        SourceStats object with calculated metrics
    """
    cols = [c for c in df.columns if not c.startswith('_')]
    num_cols = len(cols)
    density = calculate_data_density(df)
    target_density = (num_cols / target_columns) * density

    return SourceStats(
        name=name,
        rows=len(df),
        columns=num_cols,
        data_density=density,
        target_schema_density=target_density,
        target_columns=target_columns,
    )


def generate_source_overview_table(
    sources: dict[str, pd.DataFrame],
    target_columns: int = 9,
) -> pd.DataFrame:
    """Generate source overview table with data density metrics.

    Args:
        sources: Dictionary mapping source names to DataFrames
        target_columns: Number of columns in target schema

    Returns:
        DataFrame with source statistics
    """
    rows = []
    total_rows = 0
    weighted_density_sum = 0

    for name, df in sources.items():
        stats = calculate_source_stats(df, name, target_columns)
        rows.append({
            "Source": stats.name.title(),
            "Rows": stats.rows,
            "Columns": stats.columns,
            "Data Density": f"{stats.data_density:.2f}%",
            f"Density in Target Schema ({target_columns} cols)":
                f"{stats.target_schema_density:.2f}% ({stats.columns}/{target_columns} × {stats.data_density:.2f}%)",
        })
        total_rows += stats.rows
        weighted_density_sum += stats.target_schema_density * stats.rows

    # Calculate averages
    avg_density = sum(calculate_data_density(df) for df in sources.values()) / len(sources)
    weighted_avg_target = weighted_density_sum / total_rows if total_rows > 0 else 0

    # Add total row
    rows.append({
        "Source": "Total Input",
        "Rows": total_rows,
        "Columns": f"{target_columns} (target)",
        "Data Density": f"{avg_density:.2f}% avg",
        f"Density in Target Schema ({target_columns} cols)": f"{weighted_avg_target:.2f}% weighted avg",
    })

    return pd.DataFrame(rows)


def save_source_overview(
    sources: dict[str, pd.DataFrame],
    output_dir: Path,
    target_columns: int = 9,
    filename: str = "source_overview.csv",
) -> Path:
    """Generate and save source overview table.

    Args:
        sources: Dictionary mapping source names to DataFrames
        output_dir: Directory to save the report
        target_columns: Number of columns in target schema
        filename: Output filename

    Returns:
        Path to the saved file
    """
    df = generate_source_overview_table(sources, target_columns)
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def generate_column_mapping_table(
    mappings: dict[str, pd.DataFrame],
    target_schema: dict,
    raw_sources: dict[str, pd.DataFrame] = None,
) -> pd.DataFrame:
    """Generate column mapping overview table.

    Args:
        mappings: Dictionary mapping source names to mapping DataFrames
                  (with 'source_column' and 'target_column' columns)
        target_schema: Target schema dictionary with 'properties' key
        raw_sources: Optional dictionary of raw source DataFrames (not used,
                     kept for API compatibility)

    Returns:
        DataFrame showing how each source maps to target columns
    """
    target_columns = list(target_schema.get("properties", {}).keys())
    # Sort source names alphabetically for consistent ordering
    source_names = sorted(mappings.keys())

    rows = []
    mapped_counts = {name: 0 for name in source_names}

    for target_col in target_columns:
        row = {"Target Column": target_col}
        for source_name in source_names:
            mapping_df = mappings[source_name]
            # Find source column that maps to this target
            match = mapping_df[mapping_df["target_column"] == target_col]
            if not match.empty:
                # source_column contains the original column name
                source_col = match.iloc[0]["source_column"]
                row[source_name.title()] = source_col
                mapped_counts[source_name] += 1
            else:
                row[source_name.title()] = "-"
        rows.append(row)

    # Add mapped count row
    mapped_row = {"Target Column": "Mapped"}
    for source_name in source_names:
        count = mapped_counts[source_name]
        total = len(target_columns)
        mapped_row[source_name.title()] = f"{count}/{total}"
    rows.append(mapped_row)

    return pd.DataFrame(rows)


def find_example_records(
    normalized_sources: dict[str, pd.DataFrame],
    correspondences: list[tuple[str, str]] = None,
    max_examples: int = 1,
) -> pd.DataFrame:
    """Find example records that appear in multiple sources.

    Args:
        normalized_sources: Dictionary of normalized DataFrames
        correspondences: List of (id1, id2) tuples showing matching records
        max_examples: Maximum number of example records to return

    Returns:
        DataFrame with example records from each source
    """
    if not correspondences:
        # Return first record from each source as fallback
        rows = []
        for name, df in normalized_sources.items():
            if not df.empty:
                row = df.iloc[0].to_dict()
                row["Source"] = name.title()
                rows.append(row)
        return pd.DataFrame(rows)

    # Find a cluster of matching IDs
    from collections import defaultdict

    # Build graph of matching IDs
    id_graph = defaultdict(set)
    for id1, id2 in correspondences:
        id_graph[id1].add(id2)
        id_graph[id2].add(id1)

    # Find connected components (clusters)
    visited = set()
    clusters = []

    for start_id in id_graph:
        if start_id in visited:
            continue
        cluster = set()
        stack = [start_id]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            cluster.add(node)
            stack.extend(id_graph[node] - visited)
        if len(cluster) >= 2:
            clusters.append(cluster)

    # Sort by cluster size (prefer clusters with more sources)
    clusters.sort(key=len, reverse=True)

    rows = []
    examples_found = 0

    for cluster in clusters:
        if examples_found >= max_examples:
            break

        cluster_rows = []
        sources_in_cluster = set()

        for name, df in normalized_sources.items():
            if "id" not in df.columns:
                continue
            matching = df[df["id"].isin(cluster)]
            if not matching.empty:
                row = matching.iloc[0].to_dict()
                row["Source"] = name.title()
                cluster_rows.append(row)
                sources_in_cluster.add(name)

        # Only use clusters that span multiple sources
        if len(sources_in_cluster) >= 2:
            rows.extend(cluster_rows)
            examples_found += 1

    return pd.DataFrame(rows)


def generate_schema_matching_report(
    mappings: dict[str, pd.DataFrame],
    target_schema: dict,
    normalized_sources: dict[str, pd.DataFrame] = None,
    correspondences: list[tuple[str, str]] = None,
    raw_sources: dict[str, pd.DataFrame] = None,
) -> str:
    """Generate full schema matching report as text.

    Args:
        mappings: Dictionary mapping source names to mapping DataFrames
        target_schema: Target schema dictionary
        normalized_sources: Optional normalized DataFrames for examples
        correspondences: Optional list of matching record IDs
        raw_sources: Optional raw source DataFrames to show original column names

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("SCHEMA MATCHING REPORT")
    lines.append("=" * 80)

    # Column Mapping Overview
    lines.append("\n## Column Mapping Overview\n")
    mapping_df = generate_column_mapping_table(mappings, target_schema, raw_sources)

    # Format as table
    col_widths = {col: max(len(str(col)), mapping_df[col].astype(str).str.len().max()) + 2
                  for col in mapping_df.columns}

    header = " | ".join(f"{col:<{col_widths[col]}}" for col in mapping_df.columns)
    lines.append(header)
    lines.append("-" * len(header))

    for _, row in mapping_df.iterrows():
        row_str = " | ".join(f"{str(row[col]):<{col_widths[col]}}" for col in mapping_df.columns)
        lines.append(row_str)

    # Example Records
    if normalized_sources:
        lines.append("\n\n## Example Records (same release across sources)\n")
        example_df = find_example_records(normalized_sources, correspondences)

        if not example_df.empty:
            # Select key columns for display
            display_cols = ["Source", "id", "name", "artist", "release-date",
                           "release-country", "label", "tracks", "duration"]
            display_cols = [c for c in display_cols if c in example_df.columns]

            example_display = example_df[display_cols].copy()

            # Truncate long values
            for col in example_display.columns:
                example_display[col] = example_display[col].astype(str).str[:50]

            col_widths = {col: max(len(str(col)), example_display[col].str.len().max()) + 2
                          for col in example_display.columns}

            header = " | ".join(f"{col:<{col_widths[col]}}" for col in example_display.columns)
            lines.append(header)
            lines.append("-" * len(header))

            for _, row in example_display.iterrows():
                row_str = " | ".join(f"{str(row[col]):<{col_widths[col]}}" for col in example_display.columns)
                lines.append(row_str)

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)


def save_schema_matching_report(
    mappings: dict[str, pd.DataFrame],
    target_schema: dict,
    output_dir: Path,
    normalized_sources: dict[str, pd.DataFrame] = None,
    correspondences: list[tuple[str, str]] = None,
    raw_sources: dict[str, pd.DataFrame] = None,
) -> tuple[Path, None]:
    """Save schema matching report to files.

    Args:
        mappings: Dictionary mapping source names to mapping DataFrames
        target_schema: Target schema dictionary
        output_dir: Directory to save reports
        normalized_sources: Optional normalized DataFrames for examples
        correspondences: Optional list of matching record IDs
        raw_sources: Optional raw source DataFrames to show original column names

    Returns:
        Tuple of (mapping_csv_path, None)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save column mapping as CSV
    mapping_df = generate_column_mapping_table(mappings, target_schema, raw_sources)
    mapping_csv_path = output_dir / "column_mapping.csv"
    mapping_df.to_csv(mapping_csv_path, index=False)

    # Save example records as CSV (if available)
    if normalized_sources:
        example_df = find_example_records(normalized_sources, correspondences)
        if not example_df.empty:
            example_csv_path = output_dir / "example_records.csv"
            example_df.to_csv(example_csv_path, index=False)

    return mapping_csv_path, None


def print_source_overview(
    sources: dict[str, pd.DataFrame],
    target_columns: int = 9,
) -> None:
    """Print source overview table to console.

    Args:
        sources: Dictionary mapping source names to DataFrames
        target_columns: Number of columns in target schema
    """
    df = generate_source_overview_table(sources, target_columns)

    print("\n" + "=" * 80)
    print("SOURCE OVERVIEW")
    print("=" * 80)

    # Calculate column widths
    col_widths = {}
    for col in df.columns:
        max_width = max(len(str(col)), df[col].astype(str).str.len().max())
        col_widths[col] = min(max_width + 2, 45)  # Cap at 45 chars

    # Print header
    header = " | ".join(f"{col:<{col_widths[col]}}" for col in df.columns)
    print(header)
    print("-" * len(header))

    # Print rows
    for _, row in df.iterrows():
        row_str = " | ".join(
            f"{str(row[col]):<{col_widths[col]}}" for col in df.columns
        )
        print(row_str)

    print("=" * 80 + "\n")


def generate_schema_matching_report_for_pipeline(
    schema_path: Path,
    mappings_dir: Path,
    output_dir: Path,
    normalized_results: dict[str, pd.DataFrame],
    raw_data_dir: Path = None,
) -> None:
    """Generate and save schema matching report after step 1.

    Args:
        schema_path: Path to target schema JSON file
        mappings_dir: Directory containing mapping CSV files
        output_dir: Output directory for reports
        normalized_results: Normalized DataFrames from step 1
        raw_data_dir: Optional directory containing raw source files
    """
    import json
    from PyDI.io.loaders import load_xml

    # Load target schema
    schema_path = Path(schema_path)
    with open(schema_path) as f:
        target_schema = json.load(f)

    # Load mappings from the mappings directory
    mappings_dir = Path(mappings_dir)
    mappings = {}

    if mappings_dir.exists():
        for mapping_file in mappings_dir.glob("*_mapping.csv"):
            source_name = mapping_file.stem.replace("_mapping", "")
            mappings[source_name] = pd.read_csv(mapping_file)

    if not mappings:
        print("  No schema mappings found, skipping schema matching report")
        return

    # Load raw sources if directory provided
    raw_sources = None
    if raw_data_dir:
        raw_data_dir = Path(raw_data_dir)
        if raw_data_dir.exists():
            raw_sources = {}
            for data_path in list(raw_data_dir.glob("*.csv")) + list(raw_data_dir.glob("*.xml")):
                name = data_path.stem
                if name in mappings:  # Only load sources we have mappings for
                    if data_path.suffix == ".csv":
                        raw_sources[name] = pd.read_csv(data_path)
                    elif data_path.suffix == ".xml":
                        raw_sources[name] = load_xml(data_path, nested_handling="aggregate")

    # Create reporting/schema_matching directory
    schema_matching_dir = Path(output_dir) / "reporting" / "schema_matching"
    schema_matching_dir.mkdir(parents=True, exist_ok=True)

    # Generate and save report
    mapping_csv, _ = save_schema_matching_report(
        mappings=mappings,
        target_schema=target_schema,
        output_dir=schema_matching_dir,
        normalized_sources=normalized_results,
        correspondences=None,  # Not available yet at this point
        raw_sources=raw_sources,
    )

    # Print report to console
    report_text = generate_schema_matching_report(
        mappings=mappings,
        target_schema=target_schema,
        normalized_sources=normalized_results,
        raw_sources=raw_sources,
    )
    print(report_text)
    print(f"  Saved column mapping to: {mapping_csv}")


# =============================================================================
# Training Data Comparison Summary Report
# =============================================================================


def generate_training_comparison_summary(
    output_dir: Path,
) -> pd.DataFrame | None:
    """Generate a summary comparing best Auto vs Provided training data performance.

    Scans the training_comparison directories and for each dataset pair finds:
    - Best Auto F1 (and which variant/model achieved it)
    - Best Provided F1 (and which variant/model achieved it)
    - Delta between them

    Parameters
    ----------
    output_dir : Path
        The pipeline output directory containing entity_resolution/training/training_comparison/

    Returns
    -------
    pd.DataFrame | None
        Summary DataFrame with one row per dataset pair, or None if no data found.
    """
    comparison_dir = Path(output_dir) / "entity_resolution" / "training" / "training_comparison"

    if not comparison_dir.exists():
        return None

    rows = []

    for pair_dir in sorted(comparison_dir.iterdir()):
        if not pair_dir.is_dir():
            continue

        summary_file = pair_dir / "comparison_summary.csv"
        if not summary_file.exists():
            continue

        df = pd.read_csv(summary_file)

        # Support both old ("f1") and new ("val_f1") column names
        val_f1_col = "val_f1" if "val_f1" in df.columns else "f1"
        has_test = "test_f1" in df.columns and df["test_f1"].notna().any()

        # Split into auto and provided variants
        auto_df = df[~df["variant"].str.startswith("provided_")]
        provided_df = df[df["variant"].str.startswith("provided_")]

        if auto_df.empty or provided_df.empty:
            continue

        # When test scores are available, report the variant with the best test F1.
        # Otherwise fall back to validation F1.
        if has_test:
            auto_with_test = auto_df[auto_df["test_f1"].notna()]
            provided_with_test = provided_df[provided_df["test_f1"].notna()]
            rank_col = "test_f1"
        else:
            auto_with_test = pd.DataFrame()
            provided_with_test = pd.DataFrame()
            rank_col = None

        if not auto_with_test.empty and not provided_with_test.empty:
            best_auto_idx = auto_with_test["test_f1"].idxmax()
            best_provided_idx = provided_with_test["test_f1"].idxmax()
        else:
            best_auto_idx = auto_df[val_f1_col].idxmax()
            best_provided_idx = provided_df[val_f1_col].idxmax()

        best_auto = df.loc[best_auto_idx]
        best_provided = df.loc[best_provided_idx]

        best_auto_variant = best_auto["variant"]
        best_auto_model = best_auto["model"]
        best_auto_size = int(best_auto["train_total"])
        best_auto_val_f1 = best_auto[val_f1_col]

        best_provided_variant = best_provided["variant"].replace("provided_", "")
        best_provided_model = best_provided["model"]
        best_provided_size = int(best_provided["train_total"])
        best_provided_val_f1 = best_provided[val_f1_col]

        # Format dataset pair name
        pair_name = pair_dir.name.replace("_", "-").title()

        row = {
            "Dataset Pair": pair_name,
            "Auto Variant": f"{best_auto_variant} ({best_auto_model})",
            "Auto Size": best_auto_size,
            "Provided Variant": f"{best_provided_variant} ({best_provided_model})",
            "Provided Size": best_provided_size,
        }

        if has_test and pd.notna(best_auto.get("test_f1")) and pd.notna(best_provided.get("test_f1")):
            best_auto_test_f1 = best_auto["test_f1"]
            best_provided_test_f1 = best_provided["test_f1"]
            row["Best Auto Test F1"] = f"{best_auto_test_f1:.3f}"
            row["Best Provided Test F1"] = f"{best_provided_test_f1:.3f}"
            row["Test Delta"] = f"{(best_auto_test_f1 - best_provided_test_f1) * 100:+.1f}%"
        else:
            row["Best Auto Val F1"] = f"{best_auto_val_f1:.3f}"
            row["Best Provided Val F1"] = f"{best_provided_val_f1:.3f}"
            row["Val Delta"] = f"{(best_auto_val_f1 - best_provided_val_f1) * 100:+.1f}%"

        rows.append(row)

    if not rows:
        return None

    return pd.DataFrame(rows)


def save_training_comparison_summary(
    output_dir: Path,
    *,
    filename: str = "training_comparison_summary",
) -> Path | None:
    """Generate and save training comparison summary report.

    Parameters
    ----------
    output_dir : Path
        The pipeline output directory.
    filename : str
        Base filename (without extension).

    Returns
    -------
    Path | None
        Path to the saved CSV file, or None if no data found.
    """
    df = generate_training_comparison_summary(output_dir)

    if df is None:
        return None

    # Save to reporting directory
    report_dir = Path(output_dir) / "reporting"
    report_dir.mkdir(parents=True, exist_ok=True)

    csv_path = report_dir / f"{filename}.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


# =============================================================================
# Cluster Size Distribution Report
# =============================================================================


def save_cluster_size_report(
    correspondences: list[pd.DataFrame],
    num_datasets: int,
    output_dir: Path,
) -> Path | None:
    """Save cluster size distribution from entity matching correspondences.

    Uses the same connected-component grouping as the fusion engine
    (``build_record_groups_from_correspondences``) so the distribution
    matches what fusion actually operates on.

    Parameters
    ----------
    correspondences : list[pd.DataFrame]
        Per-pair correspondence DataFrames (id1, id2, score).
    num_datasets : int
        Number of input datasets (expected max cluster size).
    output_dir : Path
        Pipeline output directory.  Report is written to
        ``reporting/entity_resolution/cluster_size_distribution.csv``.

    Returns
    -------
    Path | None
        Path to the saved CSV, or None if no correspondences.
    """
    from ..entitymatching.evaluation import EntityMatchingEvaluator

    if not correspondences:
        return None

    all_corr = pd.concat(correspondences, ignore_index=True)
    all_corr = all_corr.drop_duplicates(subset=["id1", "id2"]).reset_index(drop=True)
    if all_corr.empty:
        return None

    report_dir = Path(output_dir) / "reporting" / "entity_resolution"
    report_dir.mkdir(parents=True, exist_ok=True)

    dist_df = EntityMatchingEvaluator.create_cluster_size_distribution(
        all_corr, out_dir=str(report_dir),
    )

    # Flag oversized clusters
    oversized = dist_df[dist_df["cluster_size"] > num_datasets]
    total_clusters = int(dist_df["frequency"].sum())
    target_range = dist_df[
        (dist_df["cluster_size"] >= 2) & (dist_df["cluster_size"] <= num_datasets)
    ]["frequency"].sum()

    print(f"\n--- Cluster Size Distribution (saved to {report_dir}) ---")
    print(f"  Total clusters: {total_clusters}")
    print(f"  Clusters in target range (2-{num_datasets}): {int(target_range)}")
    if not oversized.empty:
        n_oversized = int(oversized["frequency"].sum())
        print(f"  WARNING: {n_oversized} oversized clusters (>{num_datasets} records)")
    else:
        print(f"  No oversized clusters")

    return report_dir / "cluster_size_distribution.csv"
