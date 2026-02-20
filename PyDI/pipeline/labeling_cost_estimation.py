"""
Labeling Cost Estimation Module

Estimates the cost of using an LLM to label all entity resolution candidates.
Uses tiktoken for accurate token counting with OpenAI models.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    import tiktoken
except ImportError:
    tiktoken = None


# Pricing per 1M tokens (input, output) for various models
MODEL_PRICING = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "gpt-5.2": (1.75, 14.00),  # $1.75/1M input, $14.00/1M output
    "gpt-5-mini": (1.00, 3.00),
}

# Default system prompt for entity matching (simplified version)
DEFAULT_SYSTEM_PROMPT = """You are an expert entity resolver. Your task is to decide if two records refer to the same real-world entity.

Analyze the provided records carefully and return your decision as strict JSON in this format:
{"match": true|false}

Guidelines:
- match: true if records refer to the same entity, false otherwise
- Consider variations in naming, formatting, abbreviations, and data quality
- Respond with ONLY the JSON object and nothing else."""

# Average output tokens per response (based on typical matching results)
AVG_OUTPUT_TOKENS = 15  # {"match": true} or {"match": false}


def get_tokenizer(model: str = "gpt-4o"):
    """Get tiktoken encoder for the specified model."""
    if tiktoken is None:
        raise ImportError(
            "tiktoken is required for token counting. Install with: pip install tiktoken"
        )

    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base for unknown models (GPT-4/5 family)
        return tiktoken.get_encoding("cl100k_base")


def serialize_record(record: pd.Series, fields: List[str], max_length: int = 200) -> str:
    """Serialize a record for the LLM prompt."""
    data = {}
    for field in fields:
        if field in record and pd.notna(record[field]):
            value = str(record[field])
            if len(value) > max_length:
                value = value[:max_length] + "..."
            data[field] = value
    return json.dumps(data, ensure_ascii=False)


def build_prompt_for_pair(
    left_record: pd.Series,
    right_record: pd.Series,
    fields: List[str],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Build the full prompt for a candidate pair."""
    left_data = serialize_record(left_record, fields)
    right_data = serialize_record(right_record, fields)

    human_message = (
        f"Left record: {left_data}\n"
        f"Right record: {right_data}\n\n"
        "Return JSON matching the schema described above."
    )

    # Combine system + human message (this is what gets tokenized)
    return f"{system_prompt}\n\n{human_message}"


def count_tokens(text: str, encoder) -> int:
    """Count tokens in text using tiktoken encoder."""
    return len(encoder.encode(text))


def estimate_labeling_cost_for_pair(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    candidates: pd.DataFrame,
    id_column: str = "id",
    fields: Optional[List[str]] = None,
    model: str = "gpt-5.2",
) -> Dict:
    """
    Calculate cost of labeling all candidates for a dataset pair.

    Parameters
    ----------
    df_left : pd.DataFrame
        Left dataset
    df_right : pd.DataFrame
        Right dataset
    candidates : pd.DataFrame
        Candidate pairs with id1, id2 columns
    id_column : str
        Column name for record IDs
    fields : List[str], optional
        Fields to include in prompts. If None, auto-selects string columns.
    model : str
        Model name for pricing lookup

    Returns
    -------
    Dict with cost calculation results
    """
    encoder = get_tokenizer(model)

    # Auto-select fields if not provided
    if fields is None:
        common_cols = set(df_left.columns) & set(df_right.columns) - {id_column}
        fields = [c for c in common_cols if df_left[c].dtype == 'object'][:10]

    num_candidates = len(candidates)

    # Index dataframes for fast lookup
    df_left_indexed = df_left.set_index(id_column)
    df_right_indexed = df_right.set_index(id_column)

    # Count tokens for all candidates
    total_input_tokens = 0
    valid_pairs = 0

    for _, row in candidates.iterrows():
        left_id, right_id = row["id1"], row["id2"]

        # Get records using index
        if left_id not in df_left_indexed.index or right_id not in df_right_indexed.index:
            continue

        left_record = df_left_indexed.loc[left_id]
        right_record = df_right_indexed.loc[right_id]

        # Build prompt and count tokens
        prompt = build_prompt_for_pair(left_record, right_record, fields)
        total_input_tokens += count_tokens(prompt, encoder)
        valid_pairs += 1

    # Output is fixed: {"match": true} or {"match": false} = 15 tokens
    total_output_tokens = AVG_OUTPUT_TOKENS * valid_pairs

    # Get pricing
    input_price, output_price = MODEL_PRICING.get(model, (5.00, 15.00))

    # Calculate cost (pricing is per 1M tokens)
    input_cost = (total_input_tokens / 1_000_000) * input_price
    output_cost = (total_output_tokens / 1_000_000) * output_price
    total_cost = input_cost + output_cost

    return {
        "num_candidates": num_candidates,
        "valid_pairs": valid_pairs,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "input_cost_usd": round(input_cost, 4),
        "output_cost_usd": round(output_cost, 4),
        "total_cost_usd": round(total_cost, 4),
        "model": model,
    }


def estimate_labeling_costs(
    results: Dict[str, pd.DataFrame],
    candidates_dict: Dict[Tuple[str, str], pd.DataFrame],
    id_column: str = "id",
    model: str = "gpt-5.2",
) -> pd.DataFrame:
    """
    Estimate labeling costs for all dataset pairs.

    Parameters
    ----------
    results : Dict[str, pd.DataFrame]
        Dictionary of normalized datasets {name: DataFrame}
    candidates_dict : Dict[Tuple[str, str], pd.DataFrame]
        Dictionary mapping (left, right) pairs to candidate DataFrames
    id_column : str
        Column name for record IDs
    model : str
        Model name for pricing

    Returns
    -------
    pd.DataFrame with cost estimates per dataset pair
    """
    estimates = []

    for (left_name, right_name), candidates in candidates_dict.items():
        if candidates is None or candidates.empty:
            continue

        df_left = results.get(left_name)
        df_right = results.get(right_name)

        if df_left is None or df_right is None:
            continue

        # Estimate cost
        estimate = estimate_labeling_cost_for_pair(
            df_left, df_right, candidates,
            id_column=id_column,
            model=model,
        )

        estimates.append({
            "dataset_pair": f"{left_name.capitalize()}-{right_name.capitalize()}",
            "num_candidates": estimate["num_candidates"],
            "input_tokens": estimate["total_input_tokens"],
            "output_tokens": estimate["total_output_tokens"],
            "estimated_cost_usd": estimate["total_cost_usd"],
        })

    if not estimates:
        return pd.DataFrame(columns=[
            "dataset_pair", "num_candidates", "input_tokens",
            "output_tokens", "estimated_cost_usd"
        ])

    df = pd.DataFrame(estimates)

    # Add totals row
    totals = pd.DataFrame([{
        "dataset_pair": "TOTAL",
        "num_candidates": df["num_candidates"].sum(),
        "input_tokens": df["input_tokens"].sum(),
        "output_tokens": df["output_tokens"].sum(),
        "estimated_cost_usd": df["estimated_cost_usd"].sum(),
    }])

    df = pd.concat([df, totals], ignore_index=True)

    return df


def generate_cost_report(
    results: Dict[str, pd.DataFrame],
    candidates_dict: Dict[Tuple[str, str], pd.DataFrame],
    model: str = "gpt-5.2",
    id_column: str = "id",
) -> Optional[pd.DataFrame]:
    """
    Generate labeling cost estimation report.

    Parameters
    ----------
    results : Dict[str, pd.DataFrame]
        Dictionary of normalized datasets
    candidates_dict : Dict[Tuple[str, str], pd.DataFrame]
        Dictionary mapping (left, right) pairs to candidate DataFrames
    model : str
        Model for cost estimation
    id_column : str
        ID column name

    Returns
    -------
    DataFrame with cost estimates or None if no candidates found
    """
    if not candidates_dict:
        return None

    return estimate_labeling_costs(
        results, candidates_dict,
        id_column=id_column,
        model=model,
    )


def save_cost_report(
    results: Dict[str, pd.DataFrame],
    candidates_dict: Dict[Tuple[str, str], pd.DataFrame],
    output_dir: Path,
    model: str = "gpt-5.2",
    id_column: str = "id",
    filename: str = "estimated_labeling_cost",
) -> Optional[Path]:
    """
    Generate and save labeling cost estimation report.

    Parameters
    ----------
    results : Dict[str, pd.DataFrame]
        Dictionary of normalized datasets
    candidates_dict : Dict[Tuple[str, str], pd.DataFrame]
        Dictionary mapping (left, right) pairs to candidate DataFrames
    output_dir : Path
        Pipeline output directory
    model : str
        Model for cost estimation
    id_column : str
        ID column name
    filename : str
        Output filename (without extension)

    Returns
    -------
    Path to saved report or None if no candidates found
    """
    output_dir = Path(output_dir)

    df = generate_cost_report(results, candidates_dict, model, id_column)

    if df is None or df.empty:
        return None

    # Save to reporting directory
    reporting_dir = output_dir / "reporting"
    reporting_dir.mkdir(parents=True, exist_ok=True)

    output_path = reporting_dir / f"{filename}.csv"
    df.to_csv(output_path, index=False)

    return output_path
