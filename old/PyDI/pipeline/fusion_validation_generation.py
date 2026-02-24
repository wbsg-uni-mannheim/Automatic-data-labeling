"""
Fusion validation set generation.

This module generates validation sets for data fusion optimization by:
1. Building entity groups from pairwise matches using transitive closure
2. Selecting well-known entities using LLM (easier to verify correctness)
3. Generating ground truth values using LLM knowledge for conflicting attributes
4. Outputting a validation set focused on conflict resolution

The validation set helps optimize fusion strategies by providing ground truth
for which source values are correct when sources disagree.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import numpy as np

from ..fusion.engine import build_record_groups_from_correspondences
from ..fusion.base import RecordGroup

logger = logging.getLogger(__name__)


# =============================================================================
# Entity Group Building (Using Existing Fusion Module)
# =============================================================================


def build_entity_groups(
    correspondences: pd.DataFrame,
    datasets: Dict[str, pd.DataFrame],
) -> List[RecordGroup]:
    """
    Build entity groups from pairwise correspondences using transitive closure.

    Uses the existing fusion module's connected components implementation.

    Parameters
    ----------
    correspondences : pd.DataFrame
        All pairwise correspondences with columns [id1, id2, score]
    datasets : Dict[str, pd.DataFrame]
        Dictionary mapping dataset names to DataFrames

    Returns
    -------
    List[RecordGroup]
        List of RecordGroup objects representing entity groups
    """
    # Convert datasets dict to list with attrs set
    dataset_list = []
    for dataset_name, df in datasets.items():
        df_copy = df.copy()
        df_copy.attrs["dataset_name"] = dataset_name
        dataset_list.append(df_copy)

    # Use the existing fusion module's connected components implementation
    record_groups, _ = build_record_groups_from_correspondences(
        datasets=dataset_list,
        correspondences=correspondences,
        id_column="id",
    )

    # Filter to only groups spanning multiple datasets
    multi_source_groups = [
        group for group in record_groups
        if len(set(group.source_datasets.values())) > 1
    ]

    logger.info(
        f"Built {len(multi_source_groups)} entity groups spanning multiple datasets "
        f"(from {len(record_groups)} total groups)"
    )

    return multi_source_groups


# =============================================================================
# Well-Known Entity Selection
# =============================================================================


def _format_entity_for_selection(
    group: RecordGroup,
    display_columns: Optional[List[str]] = None,
) -> str:
    """Format an entity group for LLM selection prompt."""
    lines = []
    for record in group.records:
        record_id = record.get("_id", record.get("id", "unknown"))
        dataset_name = group.source_datasets.get(str(record_id), "unknown")

        if display_columns:
            cols = [c for c in display_columns if c in record.index]
        else:
            # Use first few non-id columns
            cols = [c for c in record.index if c not in ("id", "_id")][:5]

        values = ", ".join(f"{c}={record[c]}" for c in cols if pd.notna(record[c]))
        lines.append(f"  {dataset_name}: {values}")

    return "\n".join(lines)


def select_well_known_entities(
    entity_groups: List[RecordGroup],
    chat_model: Any,
    n_select: int = 30,
    sample_size: int = 100,
    display_columns: Optional[List[str]] = None,
) -> List[RecordGroup]:
    """
    Select well-known entities from entity groups using LLM.

    Well-known entities are easier to verify because the LLM has reliable
    knowledge about them (e.g., famous movies, popular artists, major companies).

    Parameters
    ----------
    entity_groups : List[RecordGroup]
        Entity groups from build_entity_groups()
    chat_model : Any
        LLM chat model for selection
    n_select : int
        Number of well-known entities to select
    sample_size : int
        Number of entities to show LLM for selection (should be > n_select)
    display_columns : List[str], optional
        Columns to show when displaying entities to LLM

    Returns
    -------
    List[RecordGroup]
        Selected well-known entity groups
    """
    if len(entity_groups) <= n_select:
        logger.info(f"Only {len(entity_groups)} groups available, returning all")
        return entity_groups

    # Sample entities to show LLM
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(
        len(entity_groups),
        size=min(sample_size, len(entity_groups)),
        replace=False
    )
    sampled_groups = [entity_groups[i] for i in sample_indices]

    # Format entities for prompt
    entity_descriptions = []
    for i, group in enumerate(sampled_groups):
        desc = _format_entity_for_selection(group, display_columns)
        entity_descriptions.append(f"[{i}]\n{desc}")

    entities_text = "\n\n".join(entity_descriptions)

    prompt = f"""You are helping create a validation dataset for data fusion.

Below are {len(sampled_groups)} entities that appear across multiple data sources.
Each entity shows values from different sources.

Your task: Select the {n_select} most WELL-KNOWN entities from this list.
Well-known means: famous, recognizable, entities you have reliable knowledge about.

For example:
- Famous movies (Titanic, The Matrix) over obscure films
- Popular artists (The Beatles, Taylor Swift) over unknown bands
- Major companies (Apple, Google) over small businesses

ENTITIES:
{entities_text}

Return ONLY a comma-separated list of indices (e.g., "0, 5, 12, 23, ...").
Select exactly {n_select} indices of the most well-known entities.
"""

    response = chat_model.invoke(prompt)
    response_text = response.content if hasattr(response, "content") else str(response)

    # Parse indices from response
    indices = []
    for match in re.findall(r'\d+', response_text):
        idx = int(match)
        if 0 <= idx < len(sampled_groups) and idx not in indices:
            indices.append(idx)
        if len(indices) >= n_select:
            break

    # If LLM didn't return enough, fill with random
    if len(indices) < n_select:
        remaining = [i for i in range(len(sampled_groups)) if i not in indices]
        rng.shuffle(remaining)
        indices.extend(remaining[: n_select - len(indices)])

    selected = [sampled_groups[i] for i in indices[:n_select]]
    logger.info(f"Selected {len(selected)} well-known entities for validation")
    return selected


# =============================================================================
# Identifying Columns Detection
# =============================================================================


def identify_identifying_columns(
    attributes: List[str],
    chat_model: Any,
) -> Set[str]:
    """
    Use LLM to identify which columns are 'identifying' attributes.

    Identifying attributes are those that help identify an entity (like name,
    title, ID-like fields) vs descriptive attributes (like duration, revenue,
    population).

    Parameters
    ----------
    attributes : List[str]
        List of attribute/column names
    chat_model : Any
        LLM chat model

    Returns
    -------
    Set[str]
        Set of attribute names considered 'identifying'
    """
    if not attributes:
        return set()

    attrs_str = ", ".join(attributes)

    prompt = f"""Given these attribute/column names from a database:
{attrs_str}

Which of these are "identifying" attributes - ones that help identify what the entity IS (like name, title, identifier)?

The rest are "descriptive" attributes - ones that describe properties of the entity (like duration, revenue, population, date).

Return ONLY the identifying attribute names, comma-separated. If none are identifying, return "NONE".

Examples:
- "name, title, id, artist_name" are identifying
- "duration, revenue, founded, population, assets" are descriptive
"""

    response = chat_model.invoke(prompt)
    response_text = response.content if hasattr(response, "content") else str(response)

    if response_text.strip().upper() == "NONE":
        return set()

    # Parse comma-separated list
    identifying = set()
    for attr in response_text.split(","):
        attr = attr.strip().lower()
        # Match against original attributes (case-insensitive)
        for orig_attr in attributes:
            if orig_attr.lower() == attr:
                identifying.add(orig_attr)
                break

    return identifying


# =============================================================================
# Ground Truth Generation
# =============================================================================


def _get_entity_attributes(
    group: RecordGroup,
) -> Dict[str, Dict[str, Any]]:
    """
    Get all attribute values for an entity group across datasets.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Mapping of attribute_name -> {dataset_name: value}
    """
    attributes: Dict[str, Dict[str, Any]] = {}

    for record in group.records:
        record_id = record.get("_id", record.get("id", "unknown"))
        dataset_name = group.source_datasets.get(str(record_id), "unknown")

        for col in record.index:
            if col in ("id", "_id"):
                continue
            if col not in attributes:
                attributes[col] = {}
            val = record[col]
            if pd.notna(val):
                attributes[col][dataset_name] = val

    return attributes


def _find_conflicting_attributes(
    attributes: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Find attributes where sources have different values.

    Returns only attributes where at least 2 sources disagree.
    """
    conflicts = {}
    for attr_name, source_values in attributes.items():
        if len(source_values) < 2:
            continue

        # Check if values differ (normalize strings for comparison)
        unique_values = set()
        for val in source_values.values():
            if isinstance(val, str):
                unique_values.add(val.strip().lower())
            else:
                unique_values.add(str(val))

        if len(unique_values) > 1:
            conflicts[attr_name] = source_values

    return conflicts


def _get_entity_context(
    group: RecordGroup,
    omit_attribute: Optional[str] = None,
) -> str:
    """Get a context string showing ALL records from all matched sources.

    Parameters
    ----------
    group : RecordGroup
        The entity group
    omit_attribute : str, optional
        If provided, omit this attribute's values from the context
        (to prevent bias when verifying that attribute)
    """
    if not group.records:
        return ""

    lines = []
    for record in group.records:
        record_id = record.get("_id", record.get("id", "unknown"))
        dataset_name = group.source_datasets.get(str(record_id), "unknown")

        # Get all non-id columns with values, optionally omitting one
        cols = [c for c in record.index if c not in ("id", "_id")]
        if omit_attribute:
            cols = [c for c in cols if c != omit_attribute]
        values = ", ".join(f"{c}={record[c]}" for c in cols if pd.notna(record[c]))
        lines.append(f"[{dataset_name}] {values}")

    return "\n".join(lines)


def _get_dataset_ids(group: RecordGroup) -> Dict[str, str]:
    """Get mapping of dataset_name -> record_id for a group."""
    result = {}
    for record in group.records:
        record_id = record.get("_id", record.get("id", "unknown"))
        dataset_name = group.source_datasets.get(str(record_id), "unknown")
        result[dataset_name] = str(record_id)
    return result


class OpenAITokenTracker:
    """Track token usage and web search count from direct OpenAI API calls."""

    _instance = None
    _lock = None

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.web_search_count = 0

    @classmethod
    def get_instance(cls) -> "OpenAITokenTracker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def add(self, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens

    def add_web_search(self) -> None:
        """Increment web search count."""
        self.web_search_count += 1

    def get_and_reset(self) -> tuple:
        """Return (prompt_tokens, completion_tokens, total_tokens, web_search_count) and reset."""
        result = (self.prompt_tokens, self.completion_tokens, self.total_tokens, self.web_search_count)
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.web_search_count = 0
        return result


def _call_openai_with_web_search(
    prompt: str,
    model: str = "gpt-5.2",
) -> str:
    """
    Call OpenAI Responses API with web search enabled.

    Parameters
    ----------
    prompt : str
        The prompt to send
    model : str
        OpenAI model to use (default: gpt-4o)

    Returns
    -------
    str
        The response text
    """
    import os
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.responses.create(
        model=model,
        tools=[{"type": "web_search"}],
        input=prompt,
    )

    # Track token usage and web search count
    tracker = OpenAITokenTracker.get_instance()
    tracker.add_web_search()
    if hasattr(response, "usage") and response.usage:
        tracker.add(
            prompt_tokens=getattr(response.usage, "input_tokens", 0),
            completion_tokens=getattr(response.usage, "output_tokens", 0),
            total_tokens=getattr(response.usage, "total_tokens", 0),
        )

    # Extract text from response
    if hasattr(response, "output_text"):
        return response.output_text
    elif hasattr(response, "output"):
        # Handle list of output items
        for item in response.output:
            if hasattr(item, "content"):
                for content in item.content:
                    if hasattr(content, "text"):
                        return content.text
    return str(response)


def _query_single_attribute(
    entity_context: str,
    attr_name: str,
    source_values: Dict[str, Any],
    chat_model: Any,
    use_web_search: bool = False,
    debug_file: Optional[Path] = None,
    omit_values: bool = False,
) -> Tuple[str, str]:
    """
    Query LLM for the correct value of a single attribute.

    Parameters
    ----------
    entity_context : str
        Full context showing all records from all matched sources
    attr_name : str
        Name of the attribute to resolve
    source_values : Dict[str, Any]
        Mapping of source_name -> value for this attribute
    chat_model : Any
        LLM chat model (used when use_web_search=False)
    use_web_search : bool
        If True, use OpenAI Responses API with web search
    debug_file : Path, optional
        If provided, append prompt and response to this file for debugging
    omit_values : bool
        If True, omit the "VALUES FROM OUR DATABASES" section entirely
        (for unbiased verification of non-identifying attributes)

    Returns
    -------
    Tuple[str, str]
        (correct_value, reasoning)
    """
    values_str = "\n".join(f"  - {src}: '{val}'" for src, val in source_values.items())

    # Build the values section (or omit it entirely for unbiased verification)
    if omit_values:
        values_section = ""
    else:
        values_section = f"\nVALUES FROM OUR DATABASES:\n{values_str}\n"

    if use_web_search:
        prompt = f"""You are a fact-checker verifying data about an entity from multiple databases.

ENTITY CONTEXT:
{entity_context}

ATTRIBUTE TO VERIFY: "{attr_name}"
{values_section}
INSTRUCTIONS:
1. Search the web to find the correct value from authoritative external sources
2. Do NOT trust any database values - independently verify the correct value
3. You MUST provide a definitive answer - only respond "UNKNOWN" if the entity cannot be identified at all
4. For values that change over time (e.g., financial figures), use the most recent reliable data
5. Prefer primary sources (official sites, filings) over secondary sources

Respond in this exact format (two lines only):
CORRECT_VALUE: <the verified value>
REASONING: <brief explanation with source citation>
"""
        response_text = _call_openai_with_web_search(prompt)
    else:
        prompt = f"""You are a fact-checker verifying data about an entity from multiple databases.

ENTITY CONTEXT:
{entity_context}

ATTRIBUTE TO VERIFY: "{attr_name}"
{values_section}
INSTRUCTIONS:
1. Use your knowledge to determine the correct value for this attribute
2. Do NOT simply pick from database values - verify based on what you know to be factually correct
3. You MUST provide a definitive answer - only respond "UNKNOWN" if the entity cannot be identified at all
4. For values that change over time, use the most recent data in your knowledge
5. Prefer authoritative knowledge (official facts) over uncertain information

Respond in this exact format (two lines only):
CORRECT_VALUE: <the correct value>
REASONING: <brief explanation>
"""
        response = chat_model.invoke(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)

    # Debug logging
    if debug_file:
        with open(debug_file, "a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"ATTRIBUTE: {attr_name}\n")
            f.write("=" * 80 + "\n\n")
            f.write(">>> PROMPT:\n")
            f.write(prompt)
            f.write("\n\n>>> RESPONSE:\n")
            f.write(response_text)
            f.write("\n\n")

    # Parse response
    correct_value = "UNKNOWN"
    reasoning = ""

    for line in response_text.split("\n"):
        line = line.strip()
        if line.startswith("CORRECT_VALUE:"):
            correct_value = line.replace("CORRECT_VALUE:", "").strip()
        elif line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()

    return correct_value, reasoning


def generate_ground_truth_for_entity(
    group: RecordGroup,
    chat_model: Any,
    use_web_search: bool = False,
    omit_target_attribute: bool = False,
    identifying_columns: Optional[Set[str]] = None,
    debug_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Generate ground truth values for a single entity using LLM knowledge.

    Makes one LLM call per attribute for better accuracy.
    Queries ALL attributes that have at least one value (including single-source attributes).

    Parameters
    ----------
    group : RecordGroup
        Entity group from build_entity_groups()
    chat_model : Any
        LLM chat model (used when use_web_search=False)
    use_web_search : bool
        If True, use OpenAI Responses API with web search for verification.
        If False, use the provided chat_model with its built-in knowledge.
    omit_target_attribute : bool
        If True, omit the target attribute's values from the entity context
        when querying non-identifying attributes (to prevent bias).
    identifying_columns : Set[str], optional
        Set of column names that are 'identifying' (e.g., name, title).
        Required when omit_target_attribute=True.
    debug_file : Path, optional
        If provided, write prompts and responses to this file for debugging.

    Returns
    -------
    Dict[str, Any]
        Ground truth record with:
        - 'entity_ids': mapping of dataset_name -> record_id
        - 'entity_context': context string for the entity
        - 'attributes': dict of attribute -> {correct_value, source_values, reasoning}
    """
    attributes = _get_entity_attributes(group)

    # Query ALL attributes that have at least one value (including single-source)
    attrs_to_query = {k: v for k, v in attributes.items() if len(v) >= 1}

    # Get full entity context (used for identifying attributes and as base)
    full_entity_context = _get_entity_context(group)

    if not attrs_to_query:
        return {
            "entity_ids": _get_dataset_ids(group),
            "entity_context": full_entity_context,
            "attributes": {},
        }

    # Query each attribute individually for better accuracy
    result = {
        "entity_ids": _get_dataset_ids(group),
        "entity_context": full_entity_context,
        "attributes": {},
    }

    for attr_name, source_values in attrs_to_query.items():
        # Determine which context to use and whether to omit values
        omit_values = False
        if omit_target_attribute and identifying_columns is not None:
            # For non-identifying attributes, omit the target attribute from context and values
            if attr_name not in identifying_columns:
                entity_context = _get_entity_context(group, omit_attribute=attr_name)
                omit_values = True
            else:
                entity_context = full_entity_context
        else:
            entity_context = full_entity_context

        correct_value, reasoning = _query_single_attribute(
            entity_context=entity_context,
            attr_name=attr_name,
            source_values=source_values,
            chat_model=chat_model,
            debug_file=debug_file,
            use_web_search=use_web_search,
            omit_values=omit_values,
        )

        result["attributes"][attr_name] = {
            "correct_value": correct_value,
            "source_values": source_values,
            "reasoning": reasoning,
        }

    return result


# =============================================================================
# Main Generation Function
# =============================================================================


def generate_fusion_validation_set(
    correspondences: pd.DataFrame,
    datasets: Dict[str, pd.DataFrame],
    chat_model: Any,
    *,
    n_entities: int = 30,
    sample_size: int = 100,
    display_columns: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    use_web_search: bool = False,
    omit_target_attribute: bool = False,
    target_entity_ids: Optional[Set[str]] = None,
    identifying_columns: Optional[Set[str]] = None,
    selected_groups: Optional[List["RecordGroup"]] = None,
) -> pd.DataFrame:
    """
    Generate a fusion validation set for optimizing data fusion strategies.

    This function:
    1. Builds entity groups from pairwise correspondences (transitive closure)
    2. Selects well-known entities (easier to verify) OR filters to target IDs
    3. Generates ground truth values using LLM knowledge (optionally with web search)
    4. Returns a validation set for all attributes with multiple source values

    Parameters
    ----------
    correspondences : pd.DataFrame
        All pairwise correspondences with columns [id1, id2, score]
    datasets : Dict[str, pd.DataFrame]
        Dictionary mapping dataset names to DataFrames
    chat_model : Any
        LLM chat model (used for entity selection and when use_web_search=False)
    n_entities : int
        Number of entities in the validation set (default 30)
    sample_size : int
        Number of entities to sample for well-known selection
    display_columns : List[str], optional
        Columns to display when selecting well-known entities
    output_dir : Path, optional
        Directory to save validation set and debug info
    use_web_search : bool
        If True, use OpenAI Responses API with web search for ground truth verification.
        If False, use the chat_model's built-in knowledge. Default is False.
    omit_target_attribute : bool
        If True, when verifying non-identifying attributes (like revenue, duration),
        omit that attribute's values from the entity context to prevent bias.
        Identifying columns are determined via LLM unless provided. Default is False.
    target_entity_ids : Set[str], optional
        If provided, only generate ground truth for entity groups containing
        these record IDs. Skips well-known selection. Useful for comparing
        against a test set.
    identifying_columns : Set[str], optional
        Pre-computed set of identifying column names (e.g., name, title).
        If provided and omit_target_attribute=True, these will be used instead
        of querying the LLM. Useful for running multiple variants efficiently.
    selected_groups : List[RecordGroup], optional
        Pre-selected entity groups to use for validation. If provided, skips
        both building entity groups and selecting well-known entities. Useful
        for running multiple validation variants on the same set of entities.

    Returns
    -------
    pd.DataFrame
        Validation set with columns:
        - entity_id: unique identifier for the fused entity
        - dataset_ids: JSON mapping of dataset -> record_id
        - attribute: attribute name
        - correct_value: ground truth value from LLM
        - source_values: JSON mapping of dataset -> value
        - reasoning: LLM's reasoning for the correct value
    """
    # Use pre-selected groups if provided, otherwise build and select
    if selected_groups is not None:
        print(f"Using {len(selected_groups)} pre-selected entity groups")
    else:
        print(f"Building entity groups from {len(correspondences)} correspondences...")
        entity_groups = build_entity_groups(correspondences, datasets)
        print(f"  >> Found {len(entity_groups)} entity groups spanning multiple datasets")

        if not entity_groups:
            logger.warning("No entity groups found spanning multiple datasets")
            return pd.DataFrame()

        # Filter to target entities if provided, otherwise select well-known entities
        if target_entity_ids is not None:
            print(f"Filtering to {len(target_entity_ids)} target entity IDs...")
            selected_groups = []
            for group in entity_groups:
                for record in group.records:
                    record_id = str(record.get("_id", record.get("id", "")))
                    if record_id in target_entity_ids:
                        selected_groups.append(group)
                        break
            print(f"  >> Found {len(selected_groups)} groups matching target IDs")
        else:
            print(f"Selecting {n_entities} well-known entities...")
            selected_groups = select_well_known_entities(
                entity_groups,
                chat_model,
                n_select=n_entities,
                sample_size=sample_size,
                display_columns=display_columns,
            )
            print(f"  >> Selected {len(selected_groups)} well-known entities")

    # Identify identifying columns if omit mode is enabled (and not already provided)
    if omit_target_attribute and identifying_columns is None:
        # Collect all attribute names across all datasets
        all_attributes = set()
        for df in datasets.values():
            all_attributes.update(c for c in df.columns if c not in ("id", "_id"))
        print(f"Identifying 'identifying' columns from {len(all_attributes)} attributes...")
        identifying_columns = identify_identifying_columns(list(all_attributes), chat_model)
        print(f"  >> Identified {len(identifying_columns)} identifying columns: {identifying_columns}")
    elif omit_target_attribute and identifying_columns is not None:
        print(f"  >> Using pre-computed identifying columns: {identifying_columns}")

    # Generate ground truth for each entity
    validation_records = []
    mode_parts = []
    if use_web_search:
        mode_parts.append("web search")
    else:
        mode_parts.append("LLM knowledge")
    if omit_target_attribute:
        mode_parts.append("omit target attr")
    search_mode = " + ".join(mode_parts)
    print(f"Generating ground truth for {len(selected_groups)} entities ({search_mode})...")

    # Setup debug file for first entity only (to inspect prompts/responses)
    debug_file: Optional[Path] = None
    if output_dir:
        debug_file = Path(output_dir) / "fusion_validation_debug.txt"
        # Clear existing debug file
        debug_file.parent.mkdir(parents=True, exist_ok=True)
        debug_file.write_text("")
        print(f"  >> Debug output will be written to {debug_file}")

    for i, group in enumerate(selected_groups):
        print(f"  >> Processing entity {i + 1}/{len(selected_groups)}...")
        try:
            # Only write debug for first entity
            entity_debug_file = debug_file if i == 0 else None
            if entity_debug_file:
                with open(entity_debug_file, "a", encoding="utf-8") as f:
                    f.write(f"{'#' * 80}\n")
                    f.write(f"# ENTITY {i}: {_get_dataset_ids(group)}\n")
                    f.write(f"# Identifying columns: {identifying_columns}\n")
                    f.write(f"{'#' * 80}\n\n")

            truth = generate_ground_truth_for_entity(
                group,
                chat_model,
                use_web_search=use_web_search,
                omit_target_attribute=omit_target_attribute,
                identifying_columns=identifying_columns,
                debug_file=entity_debug_file,
            )

            entity_id = f"entity_{i}"

            for attr_name, attr_data in truth["attributes"].items():
                validation_records.append({
                    "entity_id": entity_id,
                    "dataset_ids": json.dumps(truth["entity_ids"]),
                    "entity_context": truth["entity_context"],
                    "attribute": attr_name,
                    "correct_value": attr_data["correct_value"],
                    "source_values": json.dumps(attr_data["source_values"]),
                    "reasoning": attr_data["reasoning"],
                })
        except Exception as e:
            logger.warning(f"Error generating truth for entity {i}: {e}")
            continue

    validation_df = pd.DataFrame(validation_records)
    print(f"  >> Generated {len(validation_df)} validation records for {len(selected_groups)} entities")

    # Save if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "fusion_validation_set.csv"
        validation_df.to_csv(output_path, index=False)
        logger.info(f"Saved fusion validation set to {output_path}")
        print(f"  >> Saved to {output_path}")

    return validation_df


# =============================================================================
# Cache Management
# =============================================================================


def load_fusion_validation_from_cache(
    output_dir: Path,
) -> Optional[pd.DataFrame]:
    """Load fusion validation set from cache if it exists."""
    cache_path = output_dir / "fusion_validation_set.csv"
    if cache_path.exists():
        logger.info(f"Loading cached fusion validation set from {cache_path}")
        return pd.read_csv(cache_path)
    return None


def save_fusion_validation_to_cache(
    validation_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save fusion validation set to cache."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "fusion_validation_set.csv"
    validation_df.to_csv(cache_path, index=False)
    logger.info(f"Saved fusion validation set to {cache_path}")


# =============================================================================
# Tabular Output Format (Dataset-like view)
# =============================================================================


def convert_to_tabular_format(
    validation_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert per-attribute validation set to tabular dataset-like format.

    Produces two tables:
    1. Ground truth table: entity_id, source_ids, and all attributes with correct values
    2. Origin table: entity_id, source_ids, and which source each attribute value came from

    Parameters
    ----------
    validation_df : pd.DataFrame
        Validation set from generate_fusion_validation_set() with columns:
        [entity_id, dataset_ids, entity_context, attribute, correct_value, source_values, reasoning]
    output_dir : Path, optional
        Directory to save output files

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (ground_truth_df, origin_df)
    """
    if validation_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Group by entity
    entities = validation_df.groupby("entity_id")

    ground_truth_rows = []
    origin_rows = []

    for entity_id, group in entities:
        # Get source IDs as a simple list
        dataset_ids_json = group.iloc[0]["dataset_ids"]
        dataset_ids_dict = json.loads(dataset_ids_json) if isinstance(dataset_ids_json, str) else dataset_ids_json
        source_ids = ", ".join(dataset_ids_dict.values())

        # Build ground truth row and origin row
        truth_row = {"entity_id": entity_id, "source_ids": source_ids}
        origin_row = {"entity_id": entity_id, "source_ids": source_ids}

        for _, attr_row in group.iterrows():
            attr_name = attr_row["attribute"]
            correct_value = attr_row["correct_value"]
            source_values_json = attr_row["source_values"]
            source_values = json.loads(source_values_json) if isinstance(source_values_json, str) else source_values_json

            # Add correct value to ground truth
            truth_row[attr_name] = correct_value

            # Determine which source the correct value came from
            origin_source = "LLM"  # Default if value doesn't match any source
            for src_name, src_value in source_values.items():
                # Normalize for comparison
                if str(src_value).strip().lower() == str(correct_value).strip().lower():
                    origin_source = src_name
                    break

            origin_row[attr_name] = origin_source

        ground_truth_rows.append(truth_row)
        origin_rows.append(origin_row)

    ground_truth_df = pd.DataFrame(ground_truth_rows)
    origin_df = pd.DataFrame(origin_rows)

    # Save if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        truth_path = output_dir / "fusion_ground_truth.csv"
        ground_truth_df.to_csv(truth_path, index=False)
        print(f"  >> Saved ground truth to {truth_path}")

        origin_path = output_dir / "fusion_origin.csv"
        origin_df.to_csv(origin_path, index=False)
        print(f"  >> Saved origin table to {origin_path}")

    return ground_truth_df, origin_df
