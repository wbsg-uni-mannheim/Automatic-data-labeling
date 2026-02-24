"""
Validate Entity Matching Sets

This script loads entity matching train/test/val sets and uses an LLM to re-label them.
It outputs files containing only the pairs where the LLM disagrees with the ground truth,
along with full entity attributes for manual review.

Supports two modes:
- Synchronous: Uses LLMBasedMatcher for real-time processing (default)
- Batch: Uses OpenAI Batch API for faster, cheaper processing (--batch flag)

Batch mode supports resume:
- Submits all batches and saves state to batch_state.json
- On restart, checks for pending batches and retrieves completed results

Usage:
    python validate_entity_matching_sets.py --data-dir usecases/input/companies/data \
        --matching-dir usecases/input/companies/entitymatching \
        --output-dir scripts/output/companies_validation

    # Use OpenAI Batch API for faster processing:
    python validate_entity_matching_sets.py --data-dir usecases/input/music/data \
        --matching-dir usecases/input/music/entitymatching \
        --output-dir scripts/output/music_validation --batch
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from PyDI.io.loaders import load_xml

load_dotenv()

# Configuration
LLM_MODEL = "gpt-5.2"
BATCH_POLL_INTERVAL = 30  # seconds between polling batch status
BATCH_STATE_FILE = "batch_state.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate entity matching sets using LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Path to directory containing input XML data files",
    )
    parser.add_argument(
        "--matching-dir", type=str, required=True,
        help="Path to directory containing entity matching CSV files",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Path to output directory for disagreement files",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Use OpenAI Batch API for faster processing",
    )
    parser.add_argument(
        "--model", type=str, default=LLM_MODEL,
        help=f"OpenAI model to use (default: {LLM_MODEL})",
    )
    parser.add_argument(
        "--reassign", action="store_true",
        help="Reprocess existing batch output files using saved id_mapping files",
    )
    return parser.parse_args()


def load_datasets(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all XML datasets from the data directory."""
    datasets = {}
    for xml_file in data_dir.glob("*.xml"):
        name = xml_file.stem
        print(f"Loading {name}...")
        df = load_xml(xml_file, nested_handling="aggregate")
        datasets[name] = df
        print(f"  {len(df)} records, columns: {list(df.columns)}")
    return datasets


def parse_matching_filename(filename: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse entity matching filename to extract dataset names and split type.

    Examples:
        forbes_2_dbpedia_train.csv -> ('forbes', 'dbpedia', 'train')
        forbes_2_fullcontact_val.csv -> ('forbes', 'fullcontact', 'val')
    """
    # Pattern: {dataset1}_2_{dataset2}_{split}.csv
    match = re.match(r"(.+)_2_(.+)_(train|test|val|all)\.csv", filename)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None


def load_matching_file(filepath: Path) -> pd.DataFrame:
    """
    Load an entity matching CSV file (no header).
    Format: id1,id2,label
    """
    df = pd.read_csv(filepath, header=None, names=["id1", "id2", "label"])
    # Normalize labels to uppercase strings
    df["label"] = df["label"].astype(str).str.strip().str.upper()
    return df


def get_common_fields(df1: pd.DataFrame, df2: pd.DataFrame) -> List[str]:
    """Get common non-ID fields between two dataframes."""
    common = set(df1.columns) & set(df2.columns)
    # Exclude ID columns
    excluded = {"id", "_id", "ID"}
    return sorted([c for c in common if c not in excluded])


def build_pairs_with_attributes(
    matching_df: pd.DataFrame,
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a DataFrame with all attributes from both entities for each pair.

    Returns DataFrame with columns:
        id1, id2, ground_truth, {attr}_entity1, {attr}_entity2, ...
    """
    # Flatten list values first to avoid parquet serialization issues
    df_left = flatten_list_values(df_left)
    df_right = flatten_list_values(df_right)

    # Create lookup dicts by ID (drop duplicates, keep first)
    left_dedup = df_left.drop_duplicates(subset=["id"], keep="first")
    right_dedup = df_right.drop_duplicates(subset=["id"], keep="first")
    left_lookup = left_dedup.set_index("id").to_dict("index")
    right_lookup = right_dedup.set_index("id").to_dict("index")

    # Get all columns from both datasets (excluding id)
    left_cols = [c for c in df_left.columns if c != "id"]
    right_cols = [c for c in df_right.columns if c != "id"]
    all_cols = sorted(set(left_cols) | set(right_cols))

    rows = []
    for _, row in matching_df.iterrows():
        id1, id2, label = row["id1"], row["id2"], row["label"]

        # Get entity attributes
        left_entity = left_lookup.get(id1, {})
        right_entity = right_lookup.get(id2, {})

        new_row = {
            "id1": id1,
            "id2": id2,
            "ground_truth": label,
        }

        # Add attributes from both entities
        for col in all_cols:
            new_row[f"{col}_entity1"] = left_entity.get(col, "")
            new_row[f"{col}_entity2"] = right_entity.get(col, "")

        rows.append(new_row)

    return pd.DataFrame(rows)


def flatten_list_values(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten any list/array values in dataframe columns to strings."""
    df = df.copy()
    for col in df.columns:
        # Check if any values are lists
        mask = df[col].apply(lambda x: isinstance(x, (list, tuple)))
        if mask.any():
            # Convert lists to comma-separated strings
            df.loc[mask, col] = df.loc[mask, col].apply(
                lambda x: ", ".join(str(v) for v in x) if isinstance(x, (list, tuple)) else x
            )
    return df


def get_system_prompt() -> str:
    """Get the system prompt for entity matching."""
    return """You are an expert entity resolver. Your task is to decide if two records refer to the same real-world entity.

Analyze the provided records carefully and return your decision as strict JSON in this format:
{"match": true|false, "explanation": "<brief explanation>"}

Guidelines:
- match: true if records refer to the same entity, false otherwise
- explanation should be concise (1-2 sentences)
- Consider variations in naming, formatting, abbreviations, and data quality
- Respond with ONLY the JSON object and nothing else."""


def serialize_record(record: Dict[str, Any], max_length: int = 200) -> str:
    """Serialize a record for the LLM prompt."""
    data = {}
    for field, value in record.items():
        if pd.notna(value) and value != "":
            value_str = str(value)
            if len(value_str) > max_length:
                value_str = value_str[:max_length] + "..."
            data[field] = value_str
    return json.dumps(data, ensure_ascii=False)


def create_batch_request(
    custom_id: str,
    left_record: Dict[str, Any],
    right_record: Dict[str, Any],
    model: str,
) -> Dict[str, Any]:
    """Create a single batch API request."""
    left_data = serialize_record(left_record)
    right_data = serialize_record(right_record)

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": f"Left record: {left_data}\nRight record: {right_data}\n\nReturn JSON matching the schema described above."}
            ],
        }
    }


def prepare_batch_requests(
    pairs_df: pd.DataFrame,
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    model: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Tuple[str, str]]]:
    """Prepare batch requests for all pairs.

    Returns (requests, id_mapping) where id_mapping maps custom_id -> (id1, id2)
    """
    # Flatten list values
    df_left = flatten_list_values(df_left)
    df_right = flatten_list_values(df_right)

    # Create lookups by ID
    left_lookup = df_left.drop_duplicates(subset=["id"], keep="first").set_index("id").to_dict("index")
    right_lookup = df_right.drop_duplicates(subset=["id"], keep="first").set_index("id").to_dict("index")

    requests = []
    id_mapping = {}  # custom_id -> (original_idx, id1, id2)

    for i, (idx, row) in enumerate(pairs_df.iterrows()):
        id1, id2 = str(row["id1"]), str(row["id2"])

        left_entity = left_lookup.get(id1, {})
        right_entity = right_lookup.get(id2, {})

        # Use simple numeric custom_id to avoid special character issues
        custom_id = f"req_{i}"
        id_mapping[custom_id] = (idx, id1, id2)

        request = create_batch_request(custom_id, left_entity, right_entity, model)
        requests.append(request)

    return requests, id_mapping


def write_batch_file(requests: List[Dict[str, Any]], output_path: Path) -> Path:
    """Write batch requests to JSONL file."""
    with open(output_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    return output_path


def submit_batch(client: OpenAI, batch_file_path: Path, description: str) -> str:
    """Submit batch to OpenAI API and return batch ID."""
    # Upload the file
    with open(batch_file_path, "rb") as f:
        file_response = client.files.create(file=f, purpose="batch")

    print(f"  Uploaded batch file: {file_response.id}")

    # Create the batch
    batch = client.batches.create(
        input_file_id=file_response.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description}
    )

    print(f"  Created batch: {batch.id}")
    return batch.id


def check_batch_status(client: OpenAI, batch_id: str) -> Tuple[str, Optional[str]]:
    """Check batch status. Returns (status, output_file_id or None)."""
    batch = client.batches.retrieve(batch_id)
    completed = batch.request_counts.completed if batch.request_counts else 0
    total = batch.request_counts.total if batch.request_counts else 0
    print(f"    Batch {batch_id}: {batch.status} ({completed}/{total})")

    if batch.status == "completed":
        return "completed", batch.output_file_id
    elif batch.status in ["failed", "expired", "cancelled"]:
        return batch.status, None
    else:
        return "pending", None


def download_batch_results(client: OpenAI, output_file_id: str, output_path: Path) -> Path:
    """Download batch results to a file."""
    content = client.files.content(output_file_id)
    with open(output_path, "wb") as f:
        f.write(content.read())
    return output_path


def parse_batch_results(results_path: Path) -> Dict[str, Tuple[str, str]]:
    """Parse batch results file and return dict of custom_id -> (llm_label, explanation)."""
    results = {}

    with open(results_path, "r") as f:
        for line in f:
            result = json.loads(line)
            custom_id = result["custom_id"]

            if result.get("error"):
                results[custom_id] = ("ERROR", str(result["error"]))
                continue

            # Extract the response
            try:
                response_body = result["response"]["body"]
                content = response_body["choices"][0]["message"]["content"]

                # Parse the JSON response
                # Find JSON in content
                start = content.find('{')
                end = content.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = content[start:end]
                    data = json.loads(json_str)
                    match = data.get("match", False)
                    explanation = data.get("explanation", "")
                    llm_label = "TRUE" if match else "FALSE"
                    results[custom_id] = (llm_label, explanation)
                else:
                    results[custom_id] = ("ERROR", "No JSON found in response")
            except Exception as e:
                results[custom_id] = ("ERROR", str(e))

    return results


def load_batch_state(output_dir: Path) -> Dict[str, Any]:
    """Load batch state from file."""
    state_path = output_dir / BATCH_STATE_FILE
    if state_path.exists():
        with open(state_path, "r") as f:
            return json.load(f)
    return {"batches": {}, "completed": []}


def save_batch_state(output_dir: Path, state: Dict[str, Any]):
    """Save batch state to file."""
    state_path = output_dir / BATCH_STATE_FILE
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def prepare_file_for_batch(
    matching_path: Path,
    datasets: Dict[str, pd.DataFrame],
    output_dir: Path,
    model: str,
) -> Optional[Tuple[Path, pd.DataFrame]]:
    """Prepare a matching file for batch processing. Returns (batch_input_path, pairs_df) or None."""
    parsed = parse_matching_filename(matching_path.name)
    if not parsed:
        print(f"Skipping {matching_path.name}: could not parse filename")
        return None

    left_name, right_name, split_type = parsed
    print(f"\nPreparing: {matching_path.name}")
    print(f"  Left dataset: {left_name}, Right dataset: {right_name}")

    # Check if datasets exist
    if left_name not in datasets:
        print(f"  ERROR: Dataset '{left_name}' not found")
        return None
    if right_name not in datasets:
        print(f"  ERROR: Dataset '{right_name}' not found")
        return None

    import html

    df_left = datasets[left_name].copy()
    df_right = datasets[right_name].copy()

    # Normalize IDs
    df_left["id"] = df_left["id"].astype(str).apply(html.unescape)
    df_right["id"] = df_right["id"].astype(str).apply(html.unescape)

    df_left = df_left.drop_duplicates(subset=["id"], keep="first")
    df_right = df_right.drop_duplicates(subset=["id"], keep="first")

    # Load matching file
    matching_df = load_matching_file(matching_path)
    print(f"  Loaded {len(matching_df)} pairs")

    # Normalize IDs in matching file
    matching_df["id1"] = matching_df["id1"].astype(str).apply(html.unescape)
    matching_df["id2"] = matching_df["id2"].astype(str).apply(html.unescape)

    # Filter valid pairs
    left_ids = set(df_left["id"].astype(str))
    right_ids = set(df_right["id"].astype(str))

    valid_mask = (
        matching_df["id1"].astype(str).isin(left_ids) &
        matching_df["id2"].astype(str).isin(right_ids)
    )
    valid_matching_df = matching_df[valid_mask].copy()

    if len(valid_matching_df) == 0:
        print("  Error: No valid pairs to process!")
        return None

    print(f"  Processing {len(valid_matching_df)} valid pairs")

    # Build pairs with attributes
    pairs_df = build_pairs_with_attributes(valid_matching_df, df_left, df_right)

    # Prepare batch requests
    requests, id_mapping = prepare_batch_requests(pairs_df, df_left, df_right, model)

    # Write batch file
    batch_input_path = output_dir / f"{matching_path.stem}_batch_input.jsonl"
    write_batch_file(requests, batch_input_path)
    print(f"  Wrote batch input to: {batch_input_path}")

    # Save pairs_df for later processing
    pairs_path = output_dir / f"{matching_path.stem}_pairs.parquet"
    pairs_df.to_parquet(pairs_path)

    # Save id_mapping for result processing
    mapping_path = output_dir / f"{matching_path.stem}_id_mapping.json"
    # Convert tuple values to lists for JSON serialization
    serializable_mapping = {k: list(v) for k, v in id_mapping.items()}
    with open(mapping_path, "w") as f:
        json.dump(serializable_mapping, f)

    return batch_input_path, pairs_df


def process_completed_batch(
    file_prefix: str,
    output_dir: Path,
    results: Dict[str, Tuple[str, str]],
) -> Optional[pd.DataFrame]:
    """Process completed batch results and generate disagreements."""
    # Load pairs_df
    pairs_path = output_dir / f"{file_prefix}_pairs.parquet"
    if not pairs_path.exists():
        print(f"  ERROR: Pairs file not found: {pairs_path}")
        return None

    pairs_df = pd.read_parquet(pairs_path)

    # Load id_mapping
    mapping_path = output_dir / f"{file_prefix}_id_mapping.json"
    if not mapping_path.exists():
        print(f"  ERROR: ID mapping file not found: {mapping_path}")
        return None

    with open(mapping_path, "r") as f:
        id_mapping = json.load(f)

    # Create reverse mapping: original_idx -> custom_id
    idx_to_custom_id = {}
    for custom_id, (orig_idx, id1, id2) in id_mapping.items():
        idx_to_custom_id[orig_idx] = custom_id

    # Map results back to pairs using id_mapping
    llm_labels = []
    explanations = []
    for i, (idx, row) in enumerate(pairs_df.iterrows()):
        # The custom_id was created with the enumeration index, not the DataFrame index
        custom_id = f"req_{i}"

        if custom_id in results:
            label, explanation = results[custom_id]
            llm_labels.append(label)
            explanations.append(explanation)
        else:
            llm_labels.append("UNKNOWN")
            explanations.append("No result from batch API")

    pairs_df = pairs_df.copy()
    pairs_df["llm_label"] = llm_labels
    pairs_df["llm_explanation"] = explanations

    # Count results
    n_true = sum(1 for l in llm_labels if l == "TRUE")
    n_false = sum(1 for l in llm_labels if l == "FALSE")
    n_error = sum(1 for l in llm_labels if l in ["ERROR", "UNKNOWN"])
    print(f"  Results: {n_true} TRUE, {n_false} FALSE, {n_error} errors")

    # Find disagreements
    disagreements = pairs_df[pairs_df["ground_truth"] != pairs_df["llm_label"]].copy()
    n_disagreements = len(disagreements)
    agreement_rate = 1 - (n_disagreements / len(pairs_df)) if len(pairs_df) > 0 else 1

    print(f"  LLM agreement rate: {agreement_rate:.1%}")
    print(f"  Disagreements: {n_disagreements}")

    # Save disagreements
    if n_disagreements > 0:
        # Reorder columns
        id_cols = ["id1", "id2"]
        label_cols = ["ground_truth", "llm_label", "llm_explanation"]
        attr_cols = [c for c in disagreements.columns if c not in id_cols + label_cols]

        sorted_attrs = []
        attr_bases = sorted(set(c.replace("_entity1", "").replace("_entity2", "")
                               for c in attr_cols))
        for base in attr_bases:
            if f"{base}_entity1" in attr_cols:
                sorted_attrs.append(f"{base}_entity1")
            if f"{base}_entity2" in attr_cols:
                sorted_attrs.append(f"{base}_entity2")

        final_cols = id_cols + label_cols + sorted_attrs
        disagreements = disagreements[final_cols]

        output_path = output_dir / f"{file_prefix}_disagreements.csv"
        disagreements.to_csv(output_path, index=False)
        print(f"  Saved disagreements to: {output_path}")

    # Save full labeled set
    full_output_path = output_dir / f"{file_prefix}_llm_labeled.csv"
    pairs_df.to_csv(full_output_path, index=False)
    print(f"  Saved full labeled set to: {full_output_path}")

    return disagreements


def label_pairs_with_llm_sync(
    pairs_df: pd.DataFrame,
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    model: str,
) -> pd.DataFrame:
    """
    Label entity pairs using synchronous OpenAI API calls.

    Returns the input DataFrame with an additional 'llm_label' column.
    """
    from langchain_openai import ChatOpenAI
    from PyDI.entitymatching import LLMBasedMatcher

    chat_model = ChatOpenAI(model=model, temperature=0)
    matcher = LLMBasedMatcher()

    # Flatten any list values in the dataframes (from XML nested elements)
    df_left = flatten_list_values(df_left)
    df_right = flatten_list_values(df_right)

    # Get ALL columns from both datasets (not just common ones)
    left_cols = [c for c in df_left.columns if c != "id"]
    right_cols = [c for c in df_right.columns if c != "id"]
    all_fields = sorted(set(left_cols) | set(right_cols))

    print(f"  Using ALL fields for LLM comparison: {all_fields}")

    # Create candidates DataFrame for the matcher
    candidates = pairs_df[["id1", "id2"]].copy()

    # Run LLM matching with ALL fields
    results = matcher.match(
        df_left=df_left,
        df_right=df_right,
        candidates=candidates,
        id_column="id",
        chat_model=chat_model,
        fields=all_fields,
        generate_explanations=True,
        debug=False,
    )

    # Merge results back
    result_lookup = {}
    for _, row in results.iterrows():
        key = (str(row["id1"]), str(row["id2"]))
        match_val = row.get("match", False)
        llm_label = "TRUE" if match_val else "FALSE"
        explanation = row.get("explanation", "")
        result_lookup[key] = (llm_label, explanation)

    # Add LLM labels to pairs
    llm_labels = []
    explanations = []
    for _, row in pairs_df.iterrows():
        key = (str(row["id1"]), str(row["id2"]))
        if key in result_lookup:
            llm_labels.append(result_lookup[key][0])
            explanations.append(result_lookup[key][1])
        else:
            llm_labels.append("UNKNOWN")
            explanations.append("LLM did not return result")

    pairs_df = pairs_df.copy()
    pairs_df["llm_label"] = llm_labels
    pairs_df["llm_explanation"] = explanations

    return pairs_df


def find_disagreements(labeled_df: pd.DataFrame) -> pd.DataFrame:
    """Find pairs where ground truth and LLM label disagree."""
    disagreements = labeled_df[
        labeled_df["ground_truth"] != labeled_df["llm_label"]
    ].copy()
    return disagreements


def process_matching_file(
    matching_path: Path,
    datasets: Dict[str, pd.DataFrame],
    output_dir: Path,
    model: str,
    use_batch: bool = False,
) -> Optional[pd.DataFrame]:
    """Process a single matching file and output disagreements (sync mode only)."""
    parsed = parse_matching_filename(matching_path.name)
    if not parsed:
        print(f"Skipping {matching_path.name}: could not parse filename")
        return None

    left_name, right_name, split_type = parsed
    print(f"\n{'='*60}")
    print(f"Processing: {matching_path.name}")
    print(f"  Left dataset: {left_name}")
    print(f"  Right dataset: {right_name}")
    print(f"  Split type: {split_type}")

    # Check if datasets exist
    if left_name not in datasets:
        print(f"  ERROR: Dataset '{left_name}' not found")
        return None
    if right_name not in datasets:
        print(f"  ERROR: Dataset '{right_name}' not found")
        return None

    import html

    df_left = datasets[left_name].copy()
    df_right = datasets[right_name].copy()

    # Normalize IDs first: decode XML entities like &amp; -> &
    df_left["id"] = df_left["id"].astype(str).apply(html.unescape)
    df_right["id"] = df_right["id"].astype(str).apply(html.unescape)

    # Log and drop duplicate IDs (keep first) to avoid ambiguous lookups
    left_dups = df_left[df_left.duplicated(subset=["id"], keep=False)]
    right_dups = df_right[df_right.duplicated(subset=["id"], keep=False)]

    if len(left_dups) > 0:
        print(f"  Warning: {len(left_dups)} duplicate IDs in {left_name} dataset")
        dup_output_path = output_dir / f"{left_name}_duplicate_ids.csv"
        left_dups.to_csv(dup_output_path, index=False)
        print(f"  Saved duplicates to: {dup_output_path}")

    if len(right_dups) > 0:
        print(f"  Warning: {len(right_dups)} duplicate IDs in {right_name} dataset")
        dup_output_path = output_dir / f"{right_name}_duplicate_ids.csv"
        right_dups.to_csv(dup_output_path, index=False)
        print(f"  Saved duplicates to: {dup_output_path}")

    df_left = df_left.drop_duplicates(subset=["id"], keep="first")
    df_right = df_right.drop_duplicates(subset=["id"], keep="first")

    # Load matching file
    matching_df = load_matching_file(matching_path)
    print(f"  Loaded {len(matching_df)} pairs")

    n_true = (matching_df["label"] == "TRUE").sum()
    n_false = (matching_df["label"] == "FALSE").sum()
    print(f"  Ground truth distribution: {n_true} TRUE, {n_false} FALSE")

    # Normalize IDs in matching file too
    matching_df["id1"] = matching_df["id1"].astype(str).apply(html.unescape)
    matching_df["id2"] = matching_df["id2"].astype(str).apply(html.unescape)

    # Filter out pairs where IDs don't exist in the datasets
    left_ids = set(df_left["id"].astype(str))
    right_ids = set(df_right["id"].astype(str))

    valid_mask = (
        matching_df["id1"].astype(str).isin(left_ids) &
        matching_df["id2"].astype(str).isin(right_ids)
    )
    valid_matching_df = matching_df[valid_mask].copy()
    invalid_matching_df = matching_df[~valid_mask].copy()

    if len(invalid_matching_df) > 0:
        print(f"  Warning: {len(invalid_matching_df)} pairs have IDs not found in datasets")
        invalid_output_name = matching_path.stem + "_ids_not_found.csv"
        invalid_output_path = output_dir / invalid_output_name
        invalid_matching_df.to_csv(invalid_output_path, index=False)
        print(f"  Saved invalid pairs to: {invalid_output_path}")

    if len(valid_matching_df) == 0:
        print("  Error: No valid pairs to process!")
        return None

    print(f"  Processing {len(valid_matching_df)} valid pairs")

    # Build pairs with attributes
    pairs_df = build_pairs_with_attributes(valid_matching_df, df_left, df_right)

    # Label with LLM (sync only in this function)
    print(f"  Labeling {len(pairs_df)} pairs with LLM...")
    labeled_df = label_pairs_with_llm_sync(pairs_df, df_left, df_right, model)

    # Find disagreements
    disagreements = find_disagreements(labeled_df)
    n_disagreements = len(disagreements)
    agreement_rate = 1 - (n_disagreements / len(labeled_df)) if len(labeled_df) > 0 else 1

    print(f"  LLM agreement rate: {agreement_rate:.1%}")
    print(f"  Disagreements: {n_disagreements}")

    # Analyze disagreement types
    if n_disagreements > 0:
        false_positives = ((disagreements["ground_truth"] == "FALSE") &
                          (disagreements["llm_label"] == "TRUE")).sum()
        false_negatives = ((disagreements["ground_truth"] == "TRUE") &
                          (disagreements["llm_label"] == "FALSE")).sum()
        print(f"    - Ground truth FALSE, LLM TRUE (potential label errors): {false_positives}")
        print(f"    - Ground truth TRUE, LLM FALSE (potential LLM errors): {false_negatives}")

    # Save disagreements
    output_name = matching_path.stem + "_disagreements.csv"
    output_path = output_dir / output_name

    if n_disagreements > 0:
        id_cols = ["id1", "id2"]
        label_cols = ["ground_truth", "llm_label", "llm_explanation"]
        attr_cols = [c for c in disagreements.columns if c not in id_cols + label_cols]

        sorted_attrs = []
        attr_bases = sorted(set(c.replace("_entity1", "").replace("_entity2", "")
                               for c in attr_cols))
        for base in attr_bases:
            if f"{base}_entity1" in attr_cols:
                sorted_attrs.append(f"{base}_entity1")
            if f"{base}_entity2" in attr_cols:
                sorted_attrs.append(f"{base}_entity2")

        final_cols = id_cols + label_cols + sorted_attrs
        disagreements = disagreements[final_cols]

        disagreements.to_csv(output_path, index=False)
        print(f"  Saved disagreements to: {output_path}")
    else:
        print(f"  No disagreements found - skipping file creation")

    # Also save full labeled set for reference
    full_output_name = matching_path.stem + "_llm_labeled.csv"
    full_output_path = output_dir / full_output_name
    labeled_df.to_csv(full_output_path, index=False)
    print(f"  Saved full labeled set to: {full_output_path}")

    return disagreements


def reassign_batch_results(output_dir: Path):
    """Reprocess all existing batch output files using saved id_mapping files."""
    print("\n--- Reassigning batch results ---")

    # Find all batch output files
    batch_output_files = sorted(output_dir.glob("*_batch_output.jsonl"))

    if not batch_output_files:
        print("No batch output files found.")
        return

    print(f"Found {len(batch_output_files)} batch output files")

    summary = []
    for batch_output_path in batch_output_files:
        # Extract file prefix (e.g., "forbes_2_dbpedia_train" from "forbes_2_dbpedia_train_batch_output.jsonl")
        file_prefix = batch_output_path.stem.replace("_batch_output", "")
        print(f"\nProcessing: {file_prefix}")

        # Check for required files
        pairs_path = output_dir / f"{file_prefix}_pairs.parquet"
        mapping_path = output_dir / f"{file_prefix}_id_mapping.json"

        if not pairs_path.exists():
            print(f"  Skipping: pairs file not found ({pairs_path.name})")
            continue

        if not mapping_path.exists():
            print(f"  Skipping: id_mapping file not found ({mapping_path.name})")
            continue

        # Parse batch results
        results = parse_batch_results(batch_output_path)
        print(f"  Parsed {len(results)} results from batch output")

        # Process completed batch
        disagreements = process_completed_batch(file_prefix, output_dir, results)

        if disagreements is not None:
            summary.append({
                "file": f"{file_prefix}.csv",
                "disagreements": len(disagreements),
            })

    # Print summary
    if summary:
        print("\n" + "="*60)
        print("REASSIGNMENT SUMMARY")
        print("="*60)
        summary_df = pd.DataFrame(summary)
        print(summary_df.to_string(index=False))
        summary_df.to_csv(output_dir / "validation_summary.csv", index=False)

    print("\n=== Reassignment Complete ===")


def main_batch_mode(args, matching_dir: Path, output_dir: Path, datasets: Dict[str, pd.DataFrame]):
    """Main function for batch mode with resume support."""
    client = OpenAI()

    # Load existing state
    state = load_batch_state(output_dir)
    print(f"\n--- Batch State ---")
    print(f"Pending batches: {len(state['batches'])}")
    print(f"Completed files: {len(state['completed'])}")

    # Find all matching files
    matching_files = sorted(matching_dir.glob("*.csv"))
    print(f"\n--- Found {len(matching_files)} matching files ---")

    # Check status of existing batches
    if state["batches"]:
        print("\n--- Checking pending batches ---")
        completed_batches = []
        still_pending = []

        for file_prefix, batch_info in state["batches"].items():
            batch_id = batch_info["batch_id"]
            status, output_file_id = check_batch_status(client, batch_id)

            if status == "completed":
                # Download and process results
                print(f"\n  Processing completed batch for {file_prefix}...")
                batch_output_path = output_dir / f"{file_prefix}_batch_output.jsonl"
                download_batch_results(client, output_file_id, batch_output_path)
                print(f"  Downloaded results to: {batch_output_path}")

                results = parse_batch_results(batch_output_path)
                disagreements = process_completed_batch(file_prefix, output_dir, results)

                completed_batches.append(file_prefix)
                state["completed"].append(file_prefix)

            elif status in ["failed", "expired", "cancelled"]:
                print(f"  Batch {batch_id} for {file_prefix} {status}!")
                # Remove from pending so it can be retried
                completed_batches.append(file_prefix)
            else:
                still_pending.append(file_prefix)

        # Update state
        for prefix in completed_batches:
            if prefix in state["batches"]:
                del state["batches"][prefix]

        save_batch_state(output_dir, state)

        if still_pending:
            print(f"\n--- {len(still_pending)} batches still pending ---")
            print("Run the script again later to check their status.")
            print(f"Pending: {', '.join(still_pending)}")
            return

    # Submit new batches for files not yet processed
    files_to_process = []
    for matching_path in matching_files:
        file_prefix = matching_path.stem
        if file_prefix not in state["completed"] and file_prefix not in state["batches"]:
            files_to_process.append(matching_path)

    if not files_to_process:
        print("\n--- All files have been processed! ---")
        # Generate summary
        summary = []
        for file_prefix in state["completed"]:
            disagreements_path = output_dir / f"{file_prefix}_disagreements.csv"
            if disagreements_path.exists():
                df = pd.read_csv(disagreements_path)
                summary.append({"file": f"{file_prefix}.csv", "disagreements": len(df)})
            else:
                summary.append({"file": f"{file_prefix}.csv", "disagreements": 0})

        if summary:
            print("\n" + "="*60)
            print("SUMMARY")
            print("="*60)
            summary_df = pd.DataFrame(summary)
            print(summary_df.to_string(index=False))
            summary_df.to_csv(output_dir / "validation_summary.csv", index=False)

        print("\n=== Validation Complete ===")
        return

    print(f"\n--- Submitting {len(files_to_process)} new batches ---")

    for matching_path in files_to_process:
        file_prefix = matching_path.stem
        result = prepare_file_for_batch(matching_path, datasets, output_dir, args.model)

        if result is None:
            continue

        batch_input_path, pairs_df = result

        # Submit batch
        print(f"  Submitting batch to OpenAI...")
        batch_id = submit_batch(client, batch_input_path, f"Entity matching validation: {file_prefix}")

        # Save to state
        state["batches"][file_prefix] = {
            "batch_id": batch_id,
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    save_batch_state(output_dir, state)

    print(f"\n--- Submitted {len(state['batches'])} batches ---")
    print("Batches are processing in the background.")
    print("Run the script again to check status and download results.")
    print(f"\nBatch state saved to: {output_dir / BATCH_STATE_FILE}")


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    matching_dir = Path(args.matching_dir)
    output_dir = Path(args.output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Entity Matching Set Validation")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"Matching directory: {matching_dir}")
    print(f"Output directory: {output_dir}")
    print(f"LLM Model: {args.model}")

    # Handle reassign mode (doesn't need datasets)
    if args.reassign:
        print(f"Mode: Reassign batch results")
        reassign_batch_results(output_dir)
        return

    print(f"Mode: {'Batch API' if args.batch else 'Synchronous'}")

    # Load datasets
    print("\n--- Loading Datasets ---")
    datasets = load_datasets(data_dir)
    print(f"Loaded {len(datasets)} datasets: {list(datasets.keys())}")

    if args.batch:
        # Use batch mode with resume support
        main_batch_mode(args, matching_dir, output_dir, datasets)
    else:
        # Use synchronous mode
        matching_files = sorted(matching_dir.glob("*.csv"))
        print(f"\n--- Found {len(matching_files)} matching files ---")

        summary = []
        for matching_path in matching_files:
            result = process_matching_file(
                matching_path, datasets, output_dir, args.model, args.batch
            )
            if result is not None:
                summary.append({
                    "file": matching_path.name,
                    "disagreements": len(result),
                })

        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        if summary:
            summary_df = pd.DataFrame(summary)
            print(summary_df.to_string(index=False))
            summary_df.to_csv(output_dir / "validation_summary.csv", index=False)
        else:
            print("No files processed.")

        print("\n=== Validation Complete ===")


if __name__ == "__main__":
    main()
