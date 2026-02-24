"""
DataFrame transformation based on normalization specifications.

This module applies transformations to DataFrames according to
NormalizationSpec configurations. It handles data transformation only -
validation is handled separately by the validators module.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .spec import NormalizationSpec, ColumnSpec
from .profile import profile_dataframe
from .units import normalize_quantity
from .scale import parse_scaled_number
from .integrations import (
    normalize_country,
    normalize_currency,
    format_phone,
    parse_phone,
    normalize_email as email_normalize,
    format_stdnum,
    detect_stdnum_type,
)

try:
    from langchain_core.language_models.chat_models import BaseChatModel
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseChatModel = None

logger = logging.getLogger(__name__)


@dataclass
class TransformResult:
    """Result of transforming a single column."""

    column_name: str
    original_dtype: str
    new_dtype: str
    values_transformed: int
    values_failed: int
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "column_name": self.column_name,
            "original_dtype": self.original_dtype,
            "new_dtype": self.new_dtype,
            "values_transformed": self.values_transformed,
            "values_failed": self.values_failed,
            "errors": self.errors,
        }


@dataclass
class DataFrameTransformResult:
    """Result of transforming an entire DataFrame."""

    dataframe: pd.DataFrame
    columns: dict[str, TransformResult]
    total_transformed: int
    total_failed: int

    @property
    def columns_normalized(self) -> int:
        """Number of columns where at least one value was transformed."""
        return sum(1 for r in self.columns.values() if r.values_transformed > 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": {name: col.to_dict() for name, col in self.columns.items()},
            "columns_normalized": self.columns_normalized,
            "total_transformed": self.total_transformed,
            "total_failed": self.total_failed,
        }


def _transform_unit_quantity(
    value: Any,
    spec: ColumnSpec,
) -> tuple[Any, bool]:
    """Transform a value with units."""
    if pd.isna(value):
        return value, False

    text = str(value).strip()
    if not text:
        return None, False

    result = normalize_quantity(
        text,
        target_unit=spec.target_unit,
        expand_scales=spec.expand_scale_modifiers,
    )

    if result:
        return result[0], True

    return value, False


def _transform_scaled_number(
    value: Any,
    spec: ColumnSpec,
) -> tuple[Any, bool]:
    """Transform a scaled number (e.g., '5 MEO' → 5000000)."""
    if pd.isna(value):
        return value, False

    text = str(value).strip()
    if not text:
        return None, False

    result = parse_scaled_number(text)

    if result:
        return result[0], True

    # Try parsing as plain number
    try:
        return float(text.replace(",", "")), True
    except ValueError:
        pass

    return value, False


def _transform_percentage(
    value: Any,
    spec: ColumnSpec,
) -> tuple[Any, bool]:
    """Transform a percentage value.

    to_decimal: '50%' → 0.5, 50 → 0.5
    to_percent: 0.5 → 50, '50%' → 50
    """
    if pd.isna(value):
        return value, False

    if not spec.convert_percentage or spec.convert_percentage == "keep":
        return value, False

    text = str(value).strip()
    if not text:
        return None, False

    # Check if value has % symbol
    has_percent_symbol = text.endswith('%')
    if has_percent_symbol:
        text = text[:-1].strip()

    # Parse the numeric value
    try:
        # Handle comma as decimal separator
        num = float(text.replace(',', '.'))
    except ValueError:
        return value, False

    if spec.convert_percentage == "to_decimal":
        # '50%' → 0.5, or 50 → 0.5 (if it had % symbol)
        if has_percent_symbol:
            return num / 100.0, True
        # If no % symbol and value > 1, assume it's a percentage to convert
        elif num > 1:
            return num / 100.0, True
        else:
            # Already in decimal form (0.5)
            return num, True

    elif spec.convert_percentage == "to_percent":
        # 0.5 → 50, or '50%' → 50
        if has_percent_symbol:
            # Already a percentage, just return the number
            return num, True
        elif 0 <= num <= 1:
            # Looks like a decimal, convert to percentage
            return num * 100.0, True
        else:
            # Already looks like a percentage
            return num, True

    return value, False


def _transform_country(
    value: Any,
    spec: ColumnSpec,
) -> tuple[Any, bool]:
    """Transform a country code/name."""
    if pd.isna(value):
        return value, False

    text = str(value).strip()
    if not text:
        return None, False

    output_format = spec.country_format or "alpha_2"
    if output_format == "keep":
        return value, False

    result = normalize_country(text, output_format=output_format)

    if result:
        return result, True

    return value, False


def _transform_currency(
    value: Any,
    spec: ColumnSpec,
) -> tuple[Any, bool]:
    """Transform a currency code/name."""
    if pd.isna(value):
        return value, False

    text = str(value).strip()
    if not text:
        return None, False

    output_format = spec.currency_format or "alpha_3"
    if output_format == "keep":
        return value, False

    result = normalize_currency(text, output_format=output_format)

    if result:
        return result, True

    return value, False


def _transform_phone(
    value: Any,
    spec: ColumnSpec,
) -> tuple[Any, bool]:
    """Transform a phone number."""
    if pd.isna(value):
        return value, False

    text = str(value).strip()
    if not text:
        return None, False

    phone_format = spec.phone_format or "e164"
    if phone_format == "keep":
        return value, False

    parsed = parse_phone(text, default_region=spec.phone_default_region)

    if parsed:
        formatted = format_phone(text, phone_format, spec.phone_default_region)
        if formatted:
            return formatted, True

    return value, False


def _transform_email(
    value: Any,
    spec: ColumnSpec,
) -> tuple[Any, bool]:
    """Transform an email address."""
    if pd.isna(value):
        return value, False

    text = str(value).strip()
    if not text:
        return None, False

    if not spec.normalize_email:
        return value, False

    result = email_normalize(text)

    if result:
        return result, True

    return value, False


def _transform_stdnum(
    value: Any,
    spec: ColumnSpec,
) -> tuple[Any, bool]:
    """Transform a standard number (ISBN, IBAN, etc.)."""
    if pd.isna(value):
        return value, False

    text = str(value).strip()
    if not text:
        return None, False

    if not spec.stdnum_format:
        return value, False

    # Detect the type first
    stdnum_type = detect_stdnum_type(text)
    if stdnum_type:
        formatted = format_stdnum(text, stdnum_type)
        if formatted:
            return formatted, True

    return value, False


def _transform_datetime(
    value: Any,
    spec: ColumnSpec,
) -> tuple[Any, bool]:
    """Transform a value to datetime."""
    if pd.isna(value):
        return value, False

    try:
        if spec.date_format:
            return pd.to_datetime(str(value), format=spec.date_format), True
        return pd.to_datetime(value), True
    except (ValueError, TypeError):
        return value, False


def _transform_taxonomy(
    value: Any,
    mapping: dict[str, str | None],
) -> tuple[Any, bool]:
    """Transform a value using a taxonomy mapping.

    Args:
        value: Value to transform.
        mapping: Pre-computed mapping from source values to taxonomy values.
            Values mapped to None keep their original value.

    Returns:
        Tuple of (transformed_value, was_transformed).
    """
    if pd.isna(value):
        return value, False

    text = str(value).strip()
    if not text:
        return value, False

    mapped = mapping.get(text)
    if mapped is not None:
        return mapped, True

    # No mapping found or mapped to None - keep original
    return value, False


def _transform_text(
    value: Any,
    spec: ColumnSpec,
) -> tuple[Any, bool]:
    """Apply text transformations."""
    if pd.isna(value):
        return value, False

    text = str(value)
    transformed = False

    if spec.strip_whitespace:
        new_text = text.strip()
        if new_text != text:
            text = new_text
            transformed = True

    if spec.case:
        if spec.case == "lower":
            new_text = text.lower()
        elif spec.case == "upper":
            new_text = text.upper()
        elif spec.case == "title":
            new_text = text.title()
        else:
            new_text = text

        if new_text != text:
            text = new_text
            transformed = True

    return text, transformed


def _apply_output_type(value: Any, output_type: str, date_format: str | None = None) -> Any:
    """Convert value to specified output type."""
    if pd.isna(value):
        return value

    if output_type == "keep":
        return value

    if output_type == "string":
        return str(value)

    if output_type == "float":
        try:
            return float(value)
        except (ValueError, TypeError):
            return value

    if output_type == "int":
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return value

    if output_type == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "yes", "1", "t", "y")
        return bool(value)

    if output_type == "datetime":
        # Skip if already a datetime (handled by _transform_datetime)
        if isinstance(value, (pd.Timestamp, pd.DatetimeTZDtype)) or pd.api.types.is_datetime64_any_dtype(type(value)):
            return value
        try:
            if date_format:
                return pd.to_datetime(str(value), format=date_format)
            return pd.to_datetime(value)
        except (ValueError, TypeError):
            return value

    return value


def transform_column(
    series: pd.Series,
    spec: ColumnSpec,
    detected_type: str | None = None,
    *,
    taxonomy_mapping: dict[str, str | None] | None = None,
) -> tuple[pd.Series, TransformResult]:
    """
    Transform a single column according to specification.

    Args:
        series: Pandas Series to transform
        spec: Column specification
        detected_type: Pre-detected type (optional optimization)
        taxonomy_mapping: Pre-computed taxonomy mapping for taxonomy normalization.
            Required if spec.taxonomy_path is set.

    Returns:
        Tuple of (transformed Series, TransformResult)
    """
    column_name = str(series.name)
    original_dtype = str(series.dtype)
    values_transformed = 0
    values_failed = 0
    values_skipped = 0  # Null/empty values that don't need transformation
    errors: list[str] = []

    result_values = []

    # Determine if a transformation was requested
    transformation_requested = (
        spec.target_unit is not None
        or spec.expand_scale_modifiers
        or (spec.country_format and spec.country_format != "keep")
        or (spec.currency_format and spec.currency_format != "keep")
        or (spec.phone_format and spec.phone_format != "keep")
        or spec.normalize_email
        or spec.stdnum_format
        or (spec.convert_percentage and spec.convert_percentage != "keep")
        or spec.output_type == "datetime"
        or spec.taxonomy_path is not None
    )

    for idx, value in series.items():
        transformed_value = value
        was_transformed = False
        attempted_transform = False

        try:
            # Skip null/empty values - they don't count as failures
            if pd.isna(value):
                values_skipped += 1
                result_values.append(transformed_value)
                continue

            # Apply type-specific transformation
            if spec.target_unit or spec.expand_scale_modifiers:
                attempted_transform = True
                if spec.target_unit:
                    transformed_value, was_transformed = _transform_unit_quantity(
                        value, spec
                    )
                elif spec.expand_scale_modifiers:
                    transformed_value, was_transformed = _transform_scaled_number(
                        value, spec
                    )

            elif spec.country_format and spec.country_format != "keep":
                attempted_transform = True
                transformed_value, was_transformed = _transform_country(value, spec)

            elif spec.currency_format and spec.currency_format != "keep":
                attempted_transform = True
                transformed_value, was_transformed = _transform_currency(value, spec)

            elif spec.phone_format and spec.phone_format != "keep":
                attempted_transform = True
                transformed_value, was_transformed = _transform_phone(value, spec)

            elif spec.normalize_email:
                attempted_transform = True
                transformed_value, was_transformed = _transform_email(value, spec)

            elif spec.stdnum_format:
                attempted_transform = True
                transformed_value, was_transformed = _transform_stdnum(value, spec)

            elif spec.convert_percentage and spec.convert_percentage != "keep":
                attempted_transform = True
                transformed_value, was_transformed = _transform_percentage(value, spec)

            elif spec.output_type == "datetime":
                attempted_transform = True
                transformed_value, was_transformed = _transform_datetime(value, spec)

            elif spec.taxonomy_path is not None and taxonomy_mapping is not None:
                attempted_transform = True
                transformed_value, was_transformed = _transform_taxonomy(
                    value, taxonomy_mapping
                )

            # Apply text transformations
            if spec.case or spec.strip_whitespace:
                transformed_value, text_transformed = _transform_text(
                    transformed_value, spec
                )
                was_transformed = was_transformed or text_transformed

            # Apply output type conversion
            transformed_value = _apply_output_type(
                transformed_value, spec.output_type, spec.date_format
            )

            if was_transformed:
                values_transformed += 1
            elif attempted_transform:
                # Transformation was attempted but failed (value unchanged)
                values_failed += 1
                if len(errors) < 10:  # Limit error messages
                    errors.append(f"Row {idx}: Could not transform '{value}'")

                # Handle failure according to spec
                if spec.on_failure == "null":
                    transformed_value = None
                elif spec.on_failure == "raise":
                    raise ValueError(f"Transformation failed for value '{value}' in column '{column_name}'")

        except Exception as e:
            # Re-raise if it's our own ValueError from on_failure="raise"
            if spec.on_failure == "raise" and isinstance(e, ValueError):
                raise

            values_failed += 1
            if len(errors) < 10:  # Limit error messages
                errors.append(f"Row {idx}: {e!s}")
            logger.debug("Transform error at row %s: %s", idx, e)

            # Handle failure according to spec
            if spec.on_failure == "null":
                transformed_value = None

        result_values.append(transformed_value)

    result_series = pd.Series(result_values, index=series.index, name=series.name)

    transform_result = TransformResult(
        column_name=column_name,
        original_dtype=original_dtype,
        new_dtype=str(result_series.dtype),
        values_transformed=values_transformed,
        values_failed=values_failed,
        errors=errors,
    )

    return result_series, transform_result


def transform_dataframe(
    df: pd.DataFrame,
    spec: NormalizationSpec,
    *,
    chat_model: "BaseChatModel | None" = None,
    taxonomy_cache_dir: str | None = None,
    schema_base_path: str | None = None,
) -> DataFrameTransformResult:
    """
    Transform a DataFrame according to specification.

    Args:
        df: DataFrame to transform
        spec: Normalization specification
        chat_model: LangChain chat model for taxonomy mapping. Required if any
            column has taxonomy_path set and no cached mapping exists.
        taxonomy_cache_dir: Directory for taxonomy mapping cache files. If not
            specified, uses the taxonomy_mapping_path from column spec.
        schema_base_path: Base path for resolving relative taxonomy file paths.

    Returns:
        DataFrameTransformResult with transformed DataFrame and metadata

    Examples:
        >>> spec = NormalizationSpec()
        >>> spec.set_column("revenue", expand_scale_modifiers=True, output_type="float")
        >>> spec.set_column("country", country_format="alpha_2")
        >>> result = transform_dataframe(df, spec)
        >>> normalized_df = result.dataframe

        # With taxonomy normalization
        >>> from langchain_openai import ChatOpenAI
        >>> chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        >>> spec.set_column("industry", taxonomy_path="taxonomies/gics.csv", taxonomy_column="Industry Name")
        >>> result = transform_dataframe(df, spec, chat_model=chat)
    """
    from pathlib import Path

    result_df = df.copy()
    column_results: dict[str, TransformResult] = {}
    total_transformed = 0
    total_failed = 0

    # Prepare taxonomy mappings for columns that need them
    taxonomy_mappings: dict[str, dict[str, str | None]] = {}

    # Get dataset name from DataFrame attrs if available
    dataset_name = df.attrs.get("dataset_name")

    for col_name, col_spec in spec.columns.items():
        if col_spec.taxonomy_path and col_name in df.columns:
            mapping = _prepare_taxonomy_mapping(
                df[col_name],
                col_spec,
                chat_model=chat_model,
                cache_dir=taxonomy_cache_dir,
                base_path=schema_base_path,
                dataset_name=dataset_name,
            )
            taxonomy_mappings[col_name] = mapping

    for col_name, col_spec in spec.columns.items():
        if col_name not in df.columns:
            logger.warning("Column '%s' not found in DataFrame", col_name)
            continue

        # Get taxonomy mapping if available for this column
        taxonomy_mapping = taxonomy_mappings.get(col_name)

        transformed_series, col_result = transform_column(
            df[col_name],
            col_spec,
            taxonomy_mapping=taxonomy_mapping,
        )

        result_df[col_name] = transformed_series
        column_results[col_name] = col_result
        total_transformed += col_result.values_transformed
        total_failed += col_result.values_failed

    return DataFrameTransformResult(
        dataframe=result_df,
        columns=column_results,
        total_transformed=total_transformed,
        total_failed=total_failed,
    )


def _prepare_taxonomy_mapping(
    series: pd.Series,
    spec: ColumnSpec,
    *,
    chat_model: "BaseChatModel | None",
    cache_dir: str | None,
    base_path: str | None,
    dataset_name: str | None = None,
) -> dict[str, str | None]:
    """Prepare taxonomy mapping for a column.

    1. Load cached mapping if available
    2. Check if current dataset has values not in the cache
    3. Map only new values via LLM and merge into existing cache
    4. Save updated mapping to cache

    Args:
        series: Source data column.
        spec: Column specification with taxonomy settings.
        chat_model: LangChain chat model for creating mappings.
        cache_dir: Directory for cache files.
        base_path: Base path for resolving relative taxonomy paths.
        dataset_name: Name of the dataset being processed (for logging).

    Returns:
        Dictionary mapping source values to taxonomy values.

    Raises:
        ValueError: If taxonomy_path is set but no chat_model provided and
            no cached mapping exists.
    """
    from pathlib import Path
    from .taxonomy import (
        TaxonomyLoader,
        TaxonomyMapper,
        TaxonomyMappingResult,
        load_mapping_cache,
        save_mapping_cache,
    )

    column_name = str(series.name)

    # Determine cache path
    cache_path = spec.taxonomy_mapping_path
    if cache_path is None and cache_dir is not None:
        cache_path = str(Path(cache_dir) / f"{column_name}_taxonomy_mapping.json")

    # If still no cache path, default to saving next to the taxonomy file
    if cache_path is None and base_path is not None:
        cache_path = str(Path(base_path) / f"{column_name}_taxonomy_mapping.json")

    # Try to load existing cache (we'll check if it covers all our values)
    cached_mapping: dict[str, str | None] | None = None
    if cache_path:
        cached_mapping = load_mapping_cache(cache_path)

    # Load taxonomy first to check if LLM is actually needed
    loader = TaxonomyLoader()
    taxonomy_path = Path(spec.taxonomy_path) if spec.taxonomy_path else None
    if taxonomy_path is None:
        raise ValueError(f"taxonomy_path is required for column '{column_name}'")

    # Load taxonomy values and full CSV content
    taxonomy_values = loader.load(taxonomy_path, spec.taxonomy_column, base_path=base_path)
    taxonomy_csv_content = loader.load_full_csv(taxonomy_path, base_path=base_path)

    # Determine taxonomy column (for the prompt)
    taxonomy_column = spec.taxonomy_column
    if taxonomy_column is None:
        # Use first column from the CSV
        from io import StringIO
        temp_df = pd.read_csv(StringIO(taxonomy_csv_content), nrows=0)
        taxonomy_column = temp_df.columns[0]

    # Extract unique source values
    source_values = series.dropna().astype(str).str.strip().unique().tolist()
    source_values = [v for v in source_values if v]  # Remove empty strings

    # Pre-filter: values already in taxonomy don't need LLM mapping
    taxonomy_set = set(taxonomy_values)

    # Also consider values already in the cache as "mapped"
    cached_set = set(cached_mapping.keys()) if cached_mapping else set()

    # Values that need LLM mapping: not in taxonomy AND not in cache
    needs_mapping = [v for v in source_values if v not in taxonomy_set and v not in cached_set]

    # Pre-mapped: values that exactly match taxonomy
    pre_mapped = {v: v for v in source_values if v in taxonomy_set}

    # Values we can get from cache
    from_cache = {v: cached_mapping[v] for v in source_values if v in cached_set and v not in taxonomy_set}

    # Show progress information
    total_unique = len(source_values)
    already_in_taxonomy = len(pre_mapped)
    already_in_cache = len(from_cache)
    to_map = len(needs_mapping)

    # Build header with dataset name if available
    if dataset_name:
        header = f"\n  Taxonomy mapping for '{dataset_name}.{column_name}':"
    else:
        header = f"\n  Taxonomy mapping for '{column_name}':"

    print(header)
    print(f"    Total unique values: {total_unique}")
    print(f"    Already in taxonomy: {already_in_taxonomy}")
    if cached_mapping:
        print(f"    Found in cache:      {already_in_cache}")
    print(f"    Need LLM mapping:    {to_map}")

    # Start with pre-mapped and cached values
    full_mapping = {**pre_mapped, **from_cache}
    llm_model_used = None

    # Create mapping for remaining values via LLM
    if needs_mapping:
        # Now we need a chat_model
        if chat_model is None:
            raise ValueError(
                f"Column '{column_name}' has {len(needs_mapping)} values that don't match "
                f"the taxonomy and no chat_model was provided. Either provide a chat_model "
                f"or ensure a cached mapping exists at {cache_path or 'taxonomy_mapping_path'}."
            )

        mapper = TaxonomyMapper(chat_model)
        result = mapper.create_mapping(
            needs_mapping,
            taxonomy_csv_content,
            taxonomy_column,
            column_name=column_name,
        )

        # Add LLM-mapped values
        full_mapping.update(result.mapping)
        llm_model_used = result.llm_model

        # Show summary
        mapped_count = sum(1 for v in result.mapping.values() if v is not None)
        unmapped_count = len(result.unmapped)
        print(f"    Successfully mapped: {mapped_count}")
        print(f"    Could not map:       {unmapped_count}")

    elif not needs_mapping and (pre_mapped or from_cache):
        print(f"    All values already mapped!")

    # Build the complete mapping for saving (merge with existing cache)
    save_mapping = full_mapping.copy()
    if cached_mapping:
        # Merge: keep existing cache entries, add new ones
        save_mapping = {**cached_mapping, **full_mapping}

    # Determine all unmapped values
    all_unmapped = [v for v, m in save_mapping.items() if m is None]

    # Compute source value counts for THIS dataset
    current_counts = (
        series.dropna().astype(str).str.strip()
        .value_counts()
        .to_dict()
    )
    # Filter to only include values that are in the mapping
    current_counts = {k: v for k, v in current_counts.items() if k in save_mapping}

    # Load existing per-dataset counts from cache and merge
    existing_counts_by_dataset = {}
    if cache_path:
        try:
            with open(cache_path) as f:
                cache_data = json.load(f)
                existing_counts_by_dataset = cache_data.get("source_value_counts_by_dataset", {})
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    # Add/update counts for this dataset
    ds_key = dataset_name or "unknown"
    existing_counts_by_dataset[ds_key] = current_counts

    # Build result for saving
    save_result = TaxonomyMappingResult(
        mapping=save_mapping,
        unmapped=all_unmapped,
        taxonomy_values=taxonomy_values,
        taxonomy_column=taxonomy_column,
        llm_model=llm_model_used,
        source_value_counts_by_dataset=existing_counts_by_dataset,
    )

    # Save to cache (always save to update with new mappings)
    if cache_path and (needs_mapping or not cached_mapping):
        save_mapping_cache(cache_path, save_result)
        print(f"    Saved mapping to: {cache_path}")

    return full_mapping


def normalize_dataframe(
    df: pd.DataFrame,
    spec: NormalizationSpec | None = None,
    auto: bool = False,
) -> pd.DataFrame:
    """
    Normalize a DataFrame with optional auto-detection.

    This is the main entry point for DataFrame normalization.

    Args:
        df: DataFrame to normalize
        spec: Normalization specification (optional if auto=True)
        auto: If True, auto-detect normalizations from profile

    Returns:
        Normalized DataFrame

    Examples:
        # Manual specification
        >>> spec = NormalizationSpec()
        >>> spec.set_column("revenue", expand_scale_modifiers=True)
        >>> normalized = normalize_dataframe(df, spec)

        # Auto-detection
        >>> normalized = normalize_dataframe(df, auto=True)
    """
    if spec is None:
        if auto:
            profile = profile_dataframe(df)
            spec = NormalizationSpec.from_profile(profile)
        else:
            # No spec and no auto - return unchanged
            return df.copy()

    result = transform_dataframe(df, spec)
    return result.dataframe


__all__ = [
    "TransformResult",
    "DataFrameTransformResult",
    "transform_column",
    "transform_dataframe",
    "normalize_dataframe",
]
