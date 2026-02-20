"""
Normalization for the integration pipeline.
"""

from __future__ import annotations

import pandas as pd

from ..schemamatching import SchemaTranslator, SchemaMapping
from ..normalization import load_normalization_spec, NormalizationSpec


def auto_normalize(
    source_df: pd.DataFrame,
    mapping: SchemaMapping,
    target_schema: dict,
    on_failure: str = "keep",
    chat_model=None,
    schema_base_path: str | None = None,
    taxonomy_cache_dir: str | None = None,
):
    """
    Translate and normalize source data to target schema.

    Parameters
    ----------
    source_df : pd.DataFrame
        Source dataset
    mapping : SchemaMapping
        Column mappings from auto_match_schema()
    target_schema : dict
        JSON Schema defining target types/formats
    on_failure : str
        How to handle normalization failures: "keep", "null", "raise"
    chat_model : BaseChatModel, optional
        LangChain chat model for taxonomy-based normalization. Required if
        the schema includes columns with x-pydi-taxonomy set and no cached
        mapping exists.
    schema_base_path : str, optional
        Base path for resolving relative taxonomy file paths in the schema.
    taxonomy_cache_dir : str, optional
        Directory for taxonomy mapping cache files.

    Returns
    -------
    tuple
        (normalized_df, transform_result) where transform_result has
        total_transformed and total_failed counts
    """
    # Load normalization spec from schema
    spec = load_normalization_spec(target_schema)

    # Translate columns + normalize values
    translator = SchemaTranslator()
    return translator.translate(
        source_df,
        mapping,
        normalize=spec,
        on_failure=on_failure,
        return_result=True,
        chat_model=chat_model,
        schema_base_path=schema_base_path,
        taxonomy_cache_dir=taxonomy_cache_dir,
    )
