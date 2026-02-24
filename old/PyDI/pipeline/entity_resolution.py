"""
Entity resolution for the integration pipeline.

Generates validation sets for entity matching by:
1. Selecting dataset pairs based on size (largest to others)
2. Using LLM to select good blocking columns (content columns, not IDs)
3. Using multiple blockers to generate diverse candidates
4. Labeling candidates with LLM to create training data
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple, Optional

import pandas as pd

from ..entitymatching.blocking import TokenBlocker, SortedNeighbourhoodBlocker
from ..entitymatching import LLMBasedMatcher

logger = logging.getLogger(__name__)




def select_dataset_pairs(
    datasets: Dict[str, pd.DataFrame],
) -> List[Tuple[str, str]]:
    """
    Select which dataset pairs need entity resolution.

    Strategy: Connect largest dataset to all others.
    Remaining connections can be inferred transitively.

    For 3 datasets A (largest), B, C:
      - A <-> B
      - A <-> C
      - (B <-> C inferred through A)

    Parameters
    ----------
    datasets : dict
        {"name": dataframe, ...}

    Returns
    -------
    list of tuples
        [(larger_name, smaller_name), ...]
    """
    # Sort by size descending
    sorted_names = sorted(datasets.keys(), key=lambda k: len(datasets[k]), reverse=True)

    if len(sorted_names) < 2:
        return []

    # Connect largest to all others
    largest = sorted_names[0]
    pairs = [(largest, name) for name in sorted_names[1:]]

    logger.info(f"Selected {len(pairs)} dataset pairs for entity resolution:")
    for left, right in pairs:
        logger.info(f"  {left} ({len(datasets[left])}) <-> {right} ({len(datasets[right])})")

    return pairs


def select_blocking_columns(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    chat_model,
    id_column: str = "id",
) -> List[str]:
    """
    Use LLM to select blocking strategies (column combinations to test).

    Parameters
    ----------
    df_left, df_right : pd.DataFrame
        The two datasets to block
    chat_model : BaseChatModel
        LangChain chat model
    id_column : str
        Record ID column name to exclude from blocking

    Returns
    -------
    list of str
        Blocking strategies to test. Each entry is either a single column name
        (e.g., "title") or a combination (e.g., "title+year").

    Raises
    ------
    ValueError
        If no common columns found or LLM fails to return valid strategies
    """
    import json
    from langchain_core.messages import HumanMessage

    # Find common text columns
    common_cols = set(df_left.columns) & set(df_right.columns) - {id_column}
    candidate_cols = [
        col for col in sorted(common_cols)
        if _is_text_column(df_left[col]) or _is_text_column(df_right[col])
        and df_left[col].notna().any() and df_right[col].notna().any()
    ]

    if not candidate_cols:
        raise ValueError("No common text columns found for blocking")

    # Compute simple stats for each column
    stats = []
    for col in candidate_cols:
        left_cov = df_left[col].notna().mean()
        right_cov = df_right[col].notna().mean()
        stats.append({
            "column": col,
            "coverage_left": round(left_cov, 2),
            "coverage_right": round(right_cov, 2),
        })

    # Get sample values
    samples = {
        col: {
            "left": df_left[col].dropna().head(3).tolist(),
            "right": df_right[col].dropna().head(3).tolist(),
        }
        for col in candidate_cols
    }

    prompt = f"""Select blocking strategies for entity matching between two datasets.

Available columns with coverage:
{pd.DataFrame(stats).to_string(index=False)}

Sample values:
{_format_samples(samples)}

Guidelines:
- Return 1-3 blocking strategies as a JSON list
- Each strategy can be a single column ("name") or columns joined with "+" ("name+year")
- Combined strategies are MORE SELECTIVE than single columns - prefer these when it makes sense
- Choose columns that help UNIQUELY IDENTIFY the entity:
  * Primary identifiers: name, title, product_name (high priority)
  * Secondary qualifiers: year, version, edition (good for combining)
- AVOID overly broad columns that match too many unrelated records:
  * Pure location columns alone (city, country) - too generic
  * Pure category columns alone (genre, type) - too many records share these
- A good strategy combines a primary identifier with a qualifier, e.g. "title+year" or "name+developer"
- DO NOT use the internal ID column '{id_column}'

Return ONLY a JSON list, e.g.: ["name+developer", "title", "name+year"]"""

    response = chat_model.invoke([HumanMessage(content=prompt)])

    # Parse and validate
    try:
        strategies = json.loads(response.content.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}")

    valid = [
        s for s in strategies
        if isinstance(s, str) and _validate_strategy(s, candidate_cols, id_column)
    ]

    if not valid:
        raise ValueError(f"LLM returned no valid strategies. Response: {response.content}")

    logger.info(f"LLM selected blocking strategies: {valid}")
    return valid


def _is_text_column(series: pd.Series) -> bool:
    """Check if a column contains text data."""
    return (
        series.dtype == object
        or pd.api.types.is_string_dtype(series.dtype)
    )


def _validate_strategy(strategy: str, candidate_cols: List[str], id_column: str) -> bool:
    """Check that all columns in a strategy are valid."""
    cols = [c.strip() for c in strategy.split("+")]
    return all(c in candidate_cols and c != id_column for c in cols)


def parse_blocking_strategy(strategy: str) -> List[str]:
    """
    Parse a blocking strategy string into a list of column names.

    Parameters
    ----------
    strategy : str
        Either a single column name (e.g., "title") or multiple columns
        joined with '+' (e.g., "title+year").

    Returns
    -------
    list of str
        List of column names.
    """
    return [c.strip() for c in strategy.split("+")]


def _format_samples(samples: Dict[str, Dict]) -> str:
    """Format column samples for LLM prompt."""
    lines = []
    for col, vals in samples.items():
        lines.append(f"\n{col}:")
        lines.append(f"  Dataset 1: {vals['left'][:3]}")
        lines.append(f"  Dataset 2: {vals['right'][:3]}")
    return "\n".join(lines)


def generate_candidates_multi_blocker(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    blocking_columns: List[str],
    id_column: str = "id",
    max_candidates_per_blocker: int = 1000,
    similarity_threshold: float = 0.3,
    *,
    use_embedding_blocker: bool = False,
    embedding_text_cols: Optional[List[str]] = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_top_k: int = 30,
    embedding_threshold: Optional[float] = None,
    embedding_backend: str = "sklearn",
    embedding_device: Optional[str] = None,
    blocker_timeout: Optional[float] = None,
) -> pd.DataFrame:
    """
    Generate candidate pairs using multiple blocking strategies.

    Uses different blockers on different content columns to get diverse candidates.
    Does not apply additional similarity filtering; blocking strategies themselves
    are responsible for constraining the candidate space.

    Parameters
    ----------
    df_left, df_right : pd.DataFrame
        Datasets to block
    blocking_columns : list of str
        Content columns to use for blocking (not IDs)
    id_column : str
        ID column name (used to identify records, not for blocking)
    max_candidates_per_blocker : int
        Max candidates to take from each blocker
    similarity_threshold : float
        Used as the default threshold for the optional EmbeddingBlocker when
        embedding_threshold is not provided. Default 0.3.
    blocker_timeout : float, optional
        Maximum time in seconds for each individual blocker. If exceeded, the
        blocker is skipped. Default None (no timeout).

    Returns
    -------
    pd.DataFrame
        Candidate pairs with columns [id1, id2, blocker, similarity]
    """
    # Each entry: (candidates_df, col_for_jaccard, skip_jaccard_filtering)
    all_candidates: List[Tuple[pd.DataFrame, Optional[str], bool]] = []

    # Build lookup dicts for similarity calculation
    left_lookup = df_left.set_index(id_column)
    right_lookup = df_right.set_index(id_column)

    # Log blocking configuration
    logger.info("=" * 60)
    logger.info("BLOCKING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"  Blocking columns: {blocking_columns}")
    logger.info(f"  Threshold: {similarity_threshold} (used as default EmbeddingBlocker threshold)")
    logger.info(f"  Max candidates per blocker: {max_candidates_per_blocker}")
    logger.info(f"  Left dataset: {len(df_left)} rows")
    logger.info(f"  Right dataset: {len(df_right)} rows")
    logger.info(f"  Use EmbeddingBlocker: {use_embedding_blocker}")
    logger.info("")

    total_strategies = len(blocking_columns)
    for i, strategy in enumerate(blocking_columns):
        print(f"    Strategy {i+1}/{total_strategies}: '{strategy}'")
        logger.info(f"--- Blocking on strategy: '{strategy}' ---")

        # Parse the strategy to get individual columns
        # For composite strategies like 'name+releaseYear', use the first column for blocking
        cols = parse_blocking_strategy(strategy)
        col = cols[0]
        df_left_blocking = df_left
        df_right_blocking = df_right

        # Show sample values from both datasets
        left_samples = df_left_blocking[col].dropna().head(3).tolist()
        right_samples = df_right_blocking[col].dropna().head(3).tolist()
        logger.info(f"  Sample values (left):  {left_samples}")
        logger.info(f"  Sample values (right): {right_samples}")

        # Strategy 1: Token blocking (default tokenizer)
        try:
            start_time = time.time()
            logger.info(f"  [1] TokenBlocker(column='{col}', tokenizer=default)")
            blocker = TokenBlocker(
                df_left_blocking, df_right_blocking,
                column=col,
                id_column=id_column,
            )
            candidates = blocker.materialize()
            candidates = candidates.head(max_candidates_per_blocker * 3)
            candidates["blocker"] = f"token_{col}"
            all_candidates.append((candidates[["id1", "id2", "blocker"]], col, False))
            elapsed = time.time() - start_time
            logger.info(f"      {len(candidates)} candidates ({elapsed:.1f}s)")
            print(f"      TokenBlocker: {len(candidates)} candidates ({elapsed:.1f}s)")
        except Exception as e:
            logger.warning(f"      FAILED: {e}")
            print(f"      TokenBlocker: FAILED")

        # Strategy 2: Token blocking with character ngrams (use larger ngrams for stricter matching)
        try:
            start_time = time.time()
            logger.info(f"  [2] TokenBlocker(column='{col}', ngram_size=4, ngram_type='character')")
            blocker = TokenBlocker(
                df_left_blocking, df_right_blocking,
                column=col,
                id_column=id_column,
                ngram_size=4,
                ngram_type="character",
            )
            candidates = blocker.materialize()
            candidates = candidates.head(max_candidates_per_blocker * 3)
            candidates["blocker"] = f"ngram_{col}"
            all_candidates.append((candidates[["id1", "id2", "blocker"]], col, False))
            elapsed = time.time() - start_time
            logger.info(f"      {len(candidates)} candidates ({elapsed:.1f}s)")
            print(f"      NgramBlocker: {len(candidates)} candidates ({elapsed:.1f}s)")
        except Exception as e:
            logger.warning(f"      FAILED: {e}")
            print(f"      NgramBlocker: FAILED")

        # Strategy 3: Sorted neighbourhood (only for first column, smaller window)
        if i == 0:
            try:
                start_time = time.time()
                logger.info(f"  [3] SortedNeighbourhoodBlocker(key='{col}', window=3)")
                blocker = SortedNeighbourhoodBlocker(
                    df_left_blocking, df_right_blocking,
                    key=col,
                    id_column=id_column,
                    window=3,
                )
                candidates = blocker.materialize()
                candidates = candidates.head(max_candidates_per_blocker * 3)
                candidates["blocker"] = f"snb_{col}"
                all_candidates.append((candidates[["id1", "id2", "blocker"]], col, False))
                elapsed = time.time() - start_time
                logger.info(f"      {len(candidates)} candidates ({elapsed:.1f}s)")
                print(f"      SortedNeighbourhoodBlocker: {len(candidates)} candidates ({elapsed:.1f}s)")
            except Exception as e:
                logger.warning(f"      FAILED: {e}")
                print(f"      SortedNeighbourhoodBlocker: FAILED")

    # Optional: semantic blocking via embeddings. Treat as already thresholded; do not apply Jaccard filtering.
    if use_embedding_blocker:
        try:
            from ..entitymatching.blocking import EmbeddingBlocker

            text_cols = embedding_text_cols or blocking_columns
            thresh = similarity_threshold if embedding_threshold is None else float(embedding_threshold)

            logger.info(f"--- Embedding blocking on columns: {text_cols} ---")
            logger.info(f"  [E] EmbeddingBlocker(text_cols={text_cols}, top_k={embedding_top_k}, threshold={thresh})")

            blocker = EmbeddingBlocker(
                df_left,
                df_right,
                text_cols=text_cols,
                id_column=id_column,
                model=embedding_model,
                top_k=embedding_top_k,
                threshold=thresh,
                index_backend=embedding_backend,
                device=embedding_device,
                batch_size=25_000,
            )
            candidates = blocker.materialize()

            # Keep the similarity if provided by the blocker; otherwise we'll treat as 1.0 downstream.
            candidates["blocker"] = "embedding"

            keep_cols = ["id1", "id2", "blocker"]
            if "similarity" in candidates.columns:
                keep_cols.append("similarity")
            all_candidates.append((candidates[keep_cols], None, True))

            logger.info(f"      Raw candidates (capped): {len(candidates)}")
        except Exception as e:
            logger.warning(f"      EmbeddingBlocker FAILED: {e}")

    if not all_candidates:
        return pd.DataFrame(columns=["id1", "id2", "blocker", "similarity"])

    # Cap each candidate set and normalize to a common schema.
    per_blocker: List[pd.DataFrame] = []
    for candidates_df, _col, _skip_jaccard_filtering in all_candidates:
        if candidates_df.empty:
            continue

        out = candidates_df.copy()
        if "similarity" not in out.columns:
            # Similarity is not available for most blockers; keep as NaN to avoid
            # misleading logs that suggest a meaningful score exists.
            out["similarity"] = float("nan")
        else:
            out["similarity"] = pd.to_numeric(out["similarity"], errors="coerce")

        if len(out) > max_candidates_per_blocker:
            # Prefer higher similarity where available (e.g., EmbeddingBlocker), otherwise keep head().
            if out["similarity"].nunique(dropna=True) > 1:
                out = out.nlargest(max_candidates_per_blocker, "similarity")
            else:
                out = out.head(max_candidates_per_blocker)

        per_blocker.append(out[["id1", "id2", "blocker", "similarity"]])

    if not per_blocker:
        logger.warning("No candidates generated by blockers")
        return pd.DataFrame(columns=["id1", "id2", "blocker", "similarity"])

    # Combine and dedupe; keep max similarity and union of blocker names per pair.
    combined = pd.concat(per_blocker, ignore_index=True)

    def _join_blockers(vals: pd.Series) -> str:
        uniq = sorted({str(v) for v in vals.dropna().tolist() if str(v)})
        return "+".join(uniq) if uniq else "unknown"

    combined = (
        combined.groupby(["id1", "id2"], as_index=False)
        .agg({"similarity": "max", "blocker": _join_blockers})
        .sort_values("similarity", ascending=False)
        .reset_index(drop=True)
    )

    logger.info("=" * 60)
    logger.info(f"FINAL: {len(combined)} unique candidates after capping and deduplication")
    logger.info("=" * 60)

    # Log example candidates (avoid over-emphasizing similarity when unavailable).
    if len(combined) > 0:
        similarity_informative = (
            combined["similarity"].notna().any()
            and combined["similarity"].nunique(dropna=True) > 1
        )
        if similarity_informative:
            logger.info("Top 5 candidates (by blocker-provided similarity):")
            preview = combined.sort_values("similarity", ascending=False).head(5)
        else:
            logger.info("Example candidates (no similarity scores available):")
            preview = combined.head(5)

        for _, row in preview.iterrows():
            try:
                left_val = str(left_lookup.loc[row["id1"], blocking_columns[0]])[:40]
                right_val = str(right_lookup.loc[row["id2"], blocking_columns[0]])[:40]
            except KeyError:
                left_val = str(row["id1"])
                right_val = str(row["id2"])

            if pd.notna(row["similarity"]):
                logger.info(
                    f"  sim={float(row['similarity']):.3f} blocker={row['blocker']}: '{left_val}' <-> '{right_val}'"
                )
            else:
                logger.info(
                    f"  blocker={row['blocker']}: '{left_val}' <-> '{right_val}'"
                )

    return combined


def _sample_middle_out(df: pd.DataFrame, n: int, sort_col: str = "similarity") -> pd.DataFrame:
    """
    Sample from a sorted dataframe using middle-out strategy.

    Sorts by sort_col descending, then alternately picks from high and low
    similarity ends, starting from the middle. This ensures we get a mix of
    high-similarity pairs (likely matches) and low-similarity pairs (likely
    non-matches) for a balanced validation set.

    Example with 10 items sorted by similarity [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
    - Start at middle (index 5, sim=0.4)
    - Pick alternating: idx 4 (0.5), idx 5 (0.4), idx 3 (0.6), idx 6 (0.3), ...
    - Result: mix of high and low similarity pairs

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to sample from
    n : int
        Number of samples to take
    sort_col : str
        Column to sort by (descending)

    Returns
    -------
    pd.DataFrame
        Sampled dataframe with mix of high and low similarity pairs
    """
    if len(df) <= n:
        return df

    # Sort by similarity descending
    sorted_df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    # Start from middle, alternate up and down
    middle = len(sorted_df) // 2
    selected_indices = []

    up = middle - 1  # Go towards high similarity
    down = middle    # Go towards low similarity

    while len(selected_indices) < n:
        # Pick from high similarity side
        if up >= 0 and len(selected_indices) < n:
            selected_indices.append(up)
            up -= 1
        # Pick from low similarity side
        if down < len(sorted_df) and len(selected_indices) < n:
            selected_indices.append(down)
            down += 1

    return sorted_df.iloc[selected_indices].copy()


def label_candidates_with_llm(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    candidates: pd.DataFrame,
    chat_model,
    id_column: str = "id",
    fields: Optional[List[str]] = None,
    sample_size: int = 100,
) -> pd.DataFrame:
    """
    Label candidate pairs using LLM.

    Uses a middle-out sampling strategy: candidates are sorted by similarity
    (descending), then sampled alternately from high and low similarity ends.
    This ensures the validation set contains both likely matches (high similarity)
    and likely non-matches (low similarity).

    Parameters
    ----------
    df_left, df_right : pd.DataFrame
        Source datasets
    candidates : pd.DataFrame
        Candidate pairs with id1, id2, and optionally similarity score
    chat_model : BaseChatModel
        LangChain chat model
    id_column : str
        ID column name
    fields : list of str, optional
        Content fields to show LLM. If None, uses all common non-ID columns.
    sample_size : int
        Number of candidates to label

    Returns
    -------
    pd.DataFrame
        Labeled pairs with columns [id1, id2, label] where label is TRUE/FALSE
    """
    # Sample candidates using middle-out strategy only when similarity is informative.
    if (
        "similarity" in candidates.columns
        and len(candidates) > sample_size
        and candidates["similarity"].notna().any()
        and candidates["similarity"].nunique(dropna=True) > 1
    ):
        sampled = _sample_middle_out(candidates, sample_size, sort_col="similarity")
        logger.info(f"Sampling {sample_size} candidates using middle-out strategy")
        sim_min = float(sampled["similarity"].min())
        sim_max = float(sampled["similarity"].max())
        logger.info(f"  Similarity range in sample: {sim_min:.3f} - {sim_max:.3f}")
    elif len(candidates) > sample_size:
        sampled = candidates.sample(n=sample_size, random_state=42)
    else:
        sampled = candidates

    logger.info(f"Labeling {len(sampled)} candidate pairs with LLM")

    # Determine fields to use (exclude ID columns)
    if fields is None:
        left_cols = set(df_left.columns) - {id_column}
        right_cols = set(df_right.columns) - {id_column}
        common = left_cols & right_cols
        # Filter out ID-like columns
        fields = [c for c in common if "id" not in c.lower()]

    # Use LLM matcher without explanations for speed
    matcher = LLMBasedMatcher()
    matches = matcher.match(
        df_left=df_left,
        df_right=df_right,
        candidates=sampled[["id1", "id2"]],
        id_column=id_column,
        chat_model=chat_model,
        fields=fields,
        generate_explanations=False,  # Skip explanations for speed
    )

    # Convert to validation set format
    # matches has columns: id1, id2, match (boolean), notes
    match_results = {(row["id1"], row["id2"]): row["match"] for _, row in matches.iterrows()}

    results = []
    for _, row in sampled.iterrows():
        pair = (row["id1"], row["id2"])
        is_match = match_results.get(pair, False)
        label = "TRUE" if is_match else "FALSE"
        results.append({"id1": row["id1"], "id2": row["id2"], "label": label})

    result_df = pd.DataFrame(results)

    n_positive = (result_df["label"] == "TRUE").sum()
    n_negative = (result_df["label"] == "FALSE").sum()
    logger.info(f"Labeled {n_positive} positive and {n_negative} negative pairs")

    return result_df


def generate_validation_set(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    chat_model,
    id_column: str = "id",
    target_size: int = 100,
    target_positives: int = 30,
    similarity_threshold: float = 0.3,
    blocker_timeout: Optional[float] = None,
    batch_size: int = 50,
) -> pd.DataFrame:
    """
    Generate a validation set for entity matching.

    High-level function that:
    1. Selects blocking columns using LLM (content columns, not IDs)
    2. Generates candidates using multiple blockers
    3. Labels candidates iteratively in batches until targets are met
    4. Balances positive/negative examples

    Uses an iterative approach with early stopping: labels candidates in batches
    and stops once we have enough positives and negatives. Candidates are sampled
    with mild stratification by similarity to increase chances of finding matches
    without introducing heavy bias.

    Parameters
    ----------
    df_left, df_right : pd.DataFrame
        Datasets to match
    chat_model : BaseChatModel
        LangChain chat model
    id_column : str
        ID column name
    target_size : int
        Target number of pairs in validation set
    target_positives : int
        Target number of positive (matching) pairs
    similarity_threshold : float
        Used as the default threshold for the optional EmbeddingBlocker when
        embedding_threshold is not provided. Default 0.3.
    blocker_timeout : float, optional
        Maximum time in seconds for each individual blocker. If exceeded, the
        blocker is skipped. Default None (no timeout).
    batch_size : int
        Number of candidates to label per batch. Default 50.

    Returns
    -------
    pd.DataFrame
        Validation set with columns [id1, id2, label]
    """
    # Step 1: Select blocking columns (content columns, not IDs)
    print("  Selecting blocking keys...")
    blocking_cols = select_blocking_columns(df_left, df_right, chat_model, id_column)

    if not blocking_cols:
        raise ValueError("No suitable blocking columns found")
    print(f"  Selected blocking keys: {blocking_cols}")

    # Step 2: Generate candidates with multiple blockers
    # Get more candidates than needed to ensure we find enough positives
    print("  Starting blocking...")
    candidates = generate_candidates_multi_blocker(
        df_left, df_right,
        blocking_columns=blocking_cols,
        id_column=id_column,
        max_candidates_per_blocker=target_size * 10,  # Get more candidates for iterative sampling
        similarity_threshold=similarity_threshold,
        blocker_timeout=blocker_timeout,
    )

    if len(candidates) == 0:
        raise ValueError("No candidates generated by blockers")
    print(f"  Finished blocking: {len(candidates)} candidate pairs generated")

    # Step 3: Iteratively label candidates until we have enough positives and negatives
    target_negatives = target_size - target_positives
    all_positives = []
    all_negatives = []
    labeled_ids = set()  # Track already-labeled pairs to avoid duplicates

    # Max labels is 5x target size to bound LLM costs
    max_labels = target_size * 5

    # Prepare candidates with mild stratified sampling
    remaining_candidates = _prepare_stratified_candidates(candidates)

    print(f"  Starting iterative labeling (target: {target_positives} positives, {target_negatives} negatives, max labels: {max_labels})...")

    total_labeled = 0
    batch_num = 0

    while total_labeled < max_labels:
        # Check stopping criteria
        have_enough_positives = len(all_positives) >= target_positives
        have_enough_negatives = len(all_negatives) >= target_negatives

        if have_enough_positives and have_enough_negatives:
            print(f"  Early stopping at batch {batch_num + 1}: targets reached")
            break

        # Select next batch of candidates (not yet labeled)
        batch_candidates = []
        for _, row in remaining_candidates.iterrows():
            pair_id = (row["id1"], row["id2"])
            if pair_id not in labeled_ids:
                batch_candidates.append(row)
                if len(batch_candidates) >= batch_size:
                    break

        if not batch_candidates:
            print(f"  No more candidates to label after batch {batch_num + 1}")
            break

        batch_df = pd.DataFrame(batch_candidates)

        # Label this batch
        labeled = label_candidates_with_llm(
            df_left, df_right, batch_df, chat_model,
            id_column=id_column,
            sample_size=len(batch_df),  # Label all candidates in batch
        )

        # Track labeled pairs
        for _, row in labeled.iterrows():
            labeled_ids.add((row["id1"], row["id2"]))

        total_labeled += len(labeled)
        batch_num += 1

        # Collect positives and negatives
        batch_positives = labeled[labeled["label"] == "TRUE"]
        batch_negatives = labeled[labeled["label"] == "FALSE"]

        all_positives.extend(batch_positives.to_dict("records"))
        all_negatives.extend(batch_negatives.to_dict("records"))

        print(f"  Batch {batch_num}: +{len(batch_positives)} positives, +{len(batch_negatives)} negatives "
              f"(total: {len(all_positives)}/{target_positives} pos, {len(all_negatives)}/{target_negatives} neg)")

    # Step 4: Combine all labeled pairs (keep everything, don't discard)
    positives_df = pd.DataFrame(all_positives) if all_positives else pd.DataFrame()
    negatives_df = pd.DataFrame(all_negatives) if all_negatives else pd.DataFrame()

    result = pd.concat([positives_df, negatives_df], ignore_index=True)

    # Shuffle
    result = result.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(
        f"Generated validation set: {len(result)} pairs "
        f"({(result['label'] == 'TRUE').sum()} positive, "
        f"{(result['label'] == 'FALSE').sum()} negative)"
    )

    return result


def _prepare_stratified_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare candidates with mild stratified sampling.

    If similarity scores are available, orders candidates to sample from
    different similarity strata in a round-robin fashion. This mildly
    prioritizes higher-similarity candidates (more likely matches) without
    heavily biasing the sample.

    The strategy: divide candidates into 3 strata (high/medium/low similarity)
    and interleave them, giving a slight preference to higher strata.

    Parameters
    ----------
    candidates : pd.DataFrame
        Candidate pairs, optionally with a 'similarity' column

    Returns
    -------
    pd.DataFrame
        Reordered candidates for stratified sampling
    """
    if "similarity" not in candidates.columns or candidates["similarity"].isna().all():
        # No similarity scores - just shuffle randomly
        return candidates.sample(frac=1, random_state=42).reset_index(drop=True)

    # Sort by similarity descending
    sorted_cands = candidates.sort_values("similarity", ascending=False).reset_index(drop=True)
    n = len(sorted_cands)

    if n < 10:
        return sorted_cands

    # Divide into 3 strata: top 33%, middle 33%, bottom 33%
    third = n // 3
    high_sim = sorted_cands.iloc[:third]
    mid_sim = sorted_cands.iloc[third:2*third]
    low_sim = sorted_cands.iloc[2*third:]

    # Interleave with mild preference for higher similarity
    # Pattern: 2 from high, 2 from mid, 1 from low (repeat)
    # This gives ~40% high, ~40% mid, ~20% low - a mild bias toward likely matches
    result_rows = []
    h_idx, m_idx, l_idx = 0, 0, 0

    while h_idx < len(high_sim) or m_idx < len(mid_sim) or l_idx < len(low_sim):
        # Take 2 from high stratum
        for _ in range(2):
            if h_idx < len(high_sim):
                result_rows.append(high_sim.iloc[h_idx])
                h_idx += 1
        # Take 2 from mid stratum
        for _ in range(2):
            if m_idx < len(mid_sim):
                result_rows.append(mid_sim.iloc[m_idx])
                m_idx += 1
        # Take 1 from low stratum
        if l_idx < len(low_sim):
            result_rows.append(low_sim.iloc[l_idx])
            l_idx += 1

    return pd.DataFrame(result_rows).reset_index(drop=True)
