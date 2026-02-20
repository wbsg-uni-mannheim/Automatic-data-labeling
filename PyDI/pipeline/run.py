"""
Pipeline runner - discovers and processes data files in a directory.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..io import load_xml, load_csv
from .schema_matching import auto_match_schema
from .normalization import auto_normalize
from .entity_resolution import select_dataset_pairs, generate_validation_set

logger = logging.getLogger(__name__)

def _ensure_stable_id(df: pd.DataFrame, *, dataset_name: str, id_column: str = "id") -> pd.DataFrame:
    """Ensure a stable, unique record id column exists (required by entity resolution)."""
    if id_column not in df.columns:
        df = df.copy()
        row_count = len(df)
        pad_width = max(4, len(str(max(row_count - 1, 0))))
        df.insert(0, id_column, [f"{dataset_name}-{i:0{pad_width}d}" for i in range(row_count)])
        df.attrs["dataset_name"] = dataset_name
        return df

    series = df[id_column].astype("string")
    missing = series.isna() | series.str.strip().eq("") | series.str.lower().eq("nan")
    dupes = series.duplicated().any()
    if not missing.any() and not dupes:
        # Ensure dataset_name is set even when no repairs needed
        df.attrs["dataset_name"] = dataset_name
        return df

    # Repair in-place: fill missing and make duplicates unique deterministically.
    df = df.copy()
    row_count = len(df)
    pad_width = max(4, len(str(max(row_count - 1, 0))))
    fallback = pd.Series(
        [f"{dataset_name}-{i:0{pad_width}d}" for i in range(row_count)],
        index=df.index,
        dtype="string",
    )
    repaired = series.copy()
    repaired[missing] = fallback[missing]

    seen_counts: dict[str, int] = {}
    out_vals: list[str] = []
    for val in repaired.tolist():
        key = "" if val is None else str(val)
        count = seen_counts.get(key, 0)
        out_vals.append(key if count == 0 else f"{key}__{count}")
        seen_counts[key] = count + 1

    df[id_column] = pd.Series(out_vals, index=df.index, dtype="string")
    df.attrs["dataset_name"] = dataset_name
    return df


def discover_files(
    data_dir: Path,
    schema_path: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Discover data files in a directory.

    Parameters
    ----------
    data_dir : Path
        Directory containing data files (*.xml, *.csv)
    schema_path : Path, optional
        Path to target_schema.json. If provided, included in result.

    Returns
    -------
    dict
        {"data": [Path, ...], "schema": Path or None}
    """
    data_dir = Path(data_dir)

    files: Dict[str, object] = {
        "data": [],
        "schema": None,
    }

    # Find data files
    if data_dir.exists():
        files["data"] = list(data_dir.glob("*.xml")) + list(data_dir.glob("*.csv"))

    # Include schema if provided
    if schema_path is not None:
        schema_path = Path(schema_path)
        if schema_path.exists():
            files["schema"] = schema_path

    return files


def load_data_file(path: Path, preserve_original_ids: bool = True) -> pd.DataFrame:
    """Load a data file based on extension.

    Parameters
    ----------
    path : Path
        Path to the data file.
    preserve_original_ids : bool, default True
        If True, use the first column as the ID column (preserving original IDs).
        If False, generate new sequential IDs.
    """
    suffix = path.suffix.lower()
    name = path.stem

    if suffix == ".xml":
        # For XML, use built-in index generation
        return load_xml(
            path,
            name=name,
            nested_handling="aggregate",
            add_index=True,
            index_column_name="id",
            id_prefix=name,
        )
    elif suffix == ".csv":
        if preserve_original_ids:
            # Load without adding a new index - use first column as ID
            df = load_csv(
                path,
                name=name,
                add_index=False,
            )
            # Check if 'id' column already exists
            if "id" not in df.columns:
                # Use the first column as the ID column
                first_col = df.columns[0]
                df = df.rename(columns={first_col: "id"})
            df.attrs["dataset_name"] = name
            return df
        else:
            # Generate new sequential IDs
            return load_csv(
                path,
                name=name,
                add_index=True,
                index_column_name="id",
                id_prefix=name,
            )
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def run_pipeline(
    data_dir: str | Path,
    schema_path: str | Path,
    chat_model,
    num_rows: int = 30,
    output_dir: str | Path | None = None,
    force_rematch: bool = False,
    track_step=None,
) -> Dict[str, pd.DataFrame]:
    """
    Run schema matching + normalization on all data files in a directory.

    Results are cached to disk:
    - Schema mappings: {output_dir}/mappings/{name}_mapping.csv
    - Normalized data: {output_dir}/{name}.csv

    On subsequent runs, cached results are loaded unless force_rematch=True.

    Parameters
    ----------
    data_dir : str | Path
        Directory containing data files (*.xml, *.csv)
    schema_path : str | Path
        Path to target_schema.json file
    chat_model : BaseChatModel
        LangChain chat model for schema matching
    num_rows : int
        Sample rows for LLM
    output_dir : str | Path | None
        Directory to write normalized files. If None, writes to data_dir/normalized/
    force_rematch : bool
        If True, regenerate schema mappings even if cached. Default False.
    track_step : callable
        Context manager factory for tracking token usage per step.
        Schema matching and normalization are tracked separately
        as "Step 1a: Schema Matching" and "Step 1b: Normalization".

    Returns
    -------
    dict
        {"dataset_name": normalized_dataframe, ...}
    """
    data_dir = Path(data_dir)
    schema_path = Path(schema_path)

    # No-op fallback when no tracker is provided
    if track_step is None:
        @contextmanager
        def track_step(name):
            yield

    # Find data files
    data_files = list(data_dir.glob("*.xml")) + list(data_dir.glob("*.csv"))

    if not data_files:
        raise ValueError(f"No data files found in {data_dir}")
    if not schema_path.exists():
        raise ValueError(f"Schema file not found: {schema_path}")

    # Set up output directory
    if output_dir is None:
        output_dir = data_dir / "normalized"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up mappings cache directory
    mappings_dir = output_dir / "mappings"
    mappings_dir.mkdir(parents=True, exist_ok=True)

    # Load target schema
    with open(schema_path) as f:
        target_schema = json.load(f)

    logger.info(f"Target schema: {schema_path}")
    logger.info(f"Data files: {[p.name for p in data_files]}")
    logger.info(f"Output directory: {output_dir}")

    # Process each data file
    results = {}
    stats = {}
    for data_path in data_files:
        name = data_path.stem
        mapping_path = mappings_dir / f"{name}_mapping.csv"
        normalized_path = output_dir / f"{name}.csv"

        # Check if we can load from cache
        if not force_rematch and mapping_path.exists() and normalized_path.exists():
            logger.info(f"Loading cached results for {name}...")

            # Load cached mapping (stored as CSV since it's a DataFrame)
            mapping = pd.read_csv(mapping_path)

            # Check if cached mapping is empty (indicates previous schema matching failure)
            if mapping.empty:
                raise ValueError(
                    f"Schema matching failed for dataset '{name}': cached mapping file is empty. "
                    f"This usually means schema matching previously failed. "
                    f"Delete the output directory and re-run with a correct schema, or use --force-rematch."
                )

            # Load cached normalized data
            normalized = pd.read_csv(normalized_path)
            normalized = _ensure_stable_id(normalized, dataset_name=name, id_column="id")

            logger.info(f"  {name}: loaded {len(normalized)} rows from cache")
            results[name] = normalized

            # Load stats if available
            stats_path = output_dir / "pipeline_stats.json"
            if stats_path.exists():
                with open(stats_path) as f:
                    cached_stats = json.load(f)
                if name in cached_stats:
                    stats[name] = cached_stats[name]
            continue

        logger.info(f"Processing {name}...")

        # Load data
        df = load_data_file(data_path)

        # Schema matching
        with track_step("Step 1a: Schema Matching"):
            mapping = auto_match_schema(df, target_schema, chat_model, num_rows=num_rows)

        # Check if schema matching found any mappings
        if mapping.empty:
            raise ValueError(
                f"Schema matching failed for dataset '{name}': no column mappings found. "
                f"Please check that the source data columns can be matched to the target schema. "
                f"Source columns: {list(df.columns)}"
            )

        # Save mapping to cache (SchemaMapping is a DataFrame)
        mapping.to_csv(mapping_path, index=False)
        logger.info(f"  Saved mapping to {mapping_path}")

        # Normalization
        # Use on_failure="null" to set failed normalizations to null
        # TODO: Add proper error resolution for normalization failures
        with track_step("Step 1b: Normalization"):
            normalized, transform_result = auto_normalize(
                df, mapping, target_schema,
                on_failure="null",
                chat_model=chat_model,
                schema_base_path=str(schema_path.parent),
                taxonomy_cache_dir=str(output_dir),
            )
        normalized = _ensure_stable_id(normalized, dataset_name=name, id_column="id")

        # Track stats
        stats[name] = {
            "rows": len(df),
            "mappings": len(mapping),
            "columns_normalized": transform_result.columns_normalized,
            "transformed": transform_result.total_transformed,
            "failed": transform_result.total_failed,
        }

        # Write to disk
        normalized.to_csv(normalized_path, index=False)
        logger.info(
            f"  {name}: {len(df)} rows, {len(mapping)} mappings, "
            f"{transform_result.total_transformed} transformed, "
            f"{transform_result.total_failed} failed -> {normalized_path}"
        )

        results[name] = normalized

    # Write stats summary
    stats_path = output_dir / "pipeline_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats written to {stats_path}")

    return results
