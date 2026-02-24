"""
Schema matching for the integration pipeline.
"""

from __future__ import annotations

import pandas as pd

from ..schemamatching import LLMBasedSchemaMatcher, SchemaMapping


def auto_match_schema(
    source_df: pd.DataFrame,
    target_schema: dict,
    chat_model,
    num_rows: int = 10,
    debug: bool = False,
    out_dir: str = "output/schemamatching",
) -> SchemaMapping:
    """
    Match source columns to target schema using LLM.

    Parameters
    ----------
    source_df : pd.DataFrame
        Source dataset
    target_schema : dict
        JSON Schema with "properties" defining target columns
    chat_model : BaseChatModel
        LangChain chat model (e.g., ChatOpenAI)
    num_rows : int
        Sample rows to show LLM
    debug : bool
        Enable debug mode to save prompts and responses
    out_dir : str
        Output directory for debug artifacts

    Returns
    -------
    SchemaMapping
        DataFrame with source_column -> target_column mappings
    """
    # Get ALL target columns from schema properties (including arrays/objects)
    # Note: load_normalization_spec excludes array/object types, but we need them
    # for schema matching. The normalization step will skip them anyway.
    target_columns = list(target_schema.get("properties", {}).keys())

    # Create empty target DataFrame (LLM matcher needs this)
    df_target = pd.DataFrame(columns=target_columns)
    df_target.attrs["dataset_name"] = target_schema.get("title", "target")

    # Run LLM matching
    matcher = LLMBasedSchemaMatcher(
        chat_model=chat_model,
        num_rows=num_rows,
        target_schema=target_schema,
        debug=debug,
        out_dir=out_dir,
    )

    return matcher.match(source_df, df_target)
