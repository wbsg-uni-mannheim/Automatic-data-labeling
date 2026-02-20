"""
Similarity-based training and validation set generation.

This module generates labeled sets using embedding similarity search:
1. Embeds the larger dataset using OpenAI embeddings
2. Uses FAISS for fast nearest neighbor search
3. Queries with records from the smaller dataset
4. Labels pairs using an LLM (GPT-5.2)
5. Generates triplets (1 match + 2 hard negatives) for training

Key differences from labeled_set_generation.py:
- Uses asymmetric embedding (only index larger dataset)
- Uses OpenAI embeddings instead of sentence-transformers
- Prioritizes FAISS for similarity search
- Generates structured triplets for training
"""

from __future__ import annotations

# Set OpenMP threads to 1 BEFORE any other imports to prevent segfaults on macOS ARM64.
# FAISS uses OpenMP internally, and conflicting OpenMP runtimes can cause crashes.
import os
os.environ["OMP_NUM_THREADS"] = "1"

import logging
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .entity_resolution import select_blocking_columns, parse_blocking_strategy
from .labeled_set_generation import (
    _config_hash,
    load_labeled_set_from_cache,
    save_labeled_set_to_cache,
    drop_overlapping_pairs,
    _select_balanced_set,
)
from ..entitymatching import LLMBasedMatcher

logger = logging.getLogger(__name__)


# =============================================================================
# Embedding Cache Functions
# =============================================================================


def _get_embedding_cache_path(
    output_dir: Path,
    dataset_name: str,
) -> Path:
    """Generate a cache path for embeddings.

    Embeddings are stored in a dedicated 'embeddings' subfolder within the
    entity_resolution output directory, so they can be reused across
    validation and training set generation.
    """
    # Store in entity_resolution/embeddings/ folder
    embeddings_dir = output_dir.parent / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    return embeddings_dir / f"embeddings_{dataset_name}.npy"


def _load_embeddings_from_cache(
    cache_path: Path,
    expected_rows: int,
) -> Optional[np.ndarray]:
    """Load embeddings from cache if valid."""
    if not cache_path.exists():
        return None

    try:
        embeddings = np.load(cache_path)
        if embeddings.shape[0] == expected_rows:
            logger.info(f"Loaded cached embeddings from {cache_path}")
            return embeddings
        else:
            logger.warning(
                f"Cached embeddings have {embeddings.shape[0]} rows, "
                f"expected {expected_rows}. Regenerating."
            )
            return None
    except Exception as e:
        logger.warning(f"Error loading cached embeddings: {e}")
        return None


def _save_embeddings_to_cache(
    embeddings: np.ndarray,
    cache_path: Path,
) -> None:
    """Save embeddings to cache."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        logger.info(f"Saved embeddings to {cache_path}")
    except Exception as e:
        logger.warning(f"Error saving embeddings to cache: {e}")


# =============================================================================
# OpenAI Embedding Functions
# =============================================================================


def _compute_openai_embeddings(
    texts: List[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
) -> np.ndarray:
    """
    Compute embeddings using OpenAI API.

    Parameters
    ----------
    texts : List[str]
        Texts to embed
    model : str
        OpenAI embedding model name
    batch_size : int
        Number of texts to embed per API call

    Returns
    -------
    np.ndarray
        Embeddings array of shape (len(texts), embedding_dim)
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai is required for OpenAI embeddings. "
            "Install with: pip install openai"
        )

    from .reporting import get_embedding_token_tracker

    client = OpenAI()
    embeddings = []
    token_tracker = get_embedding_token_tracker()

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Handle empty strings
        batch = [t if t.strip() else " " for t in batch]

        response = client.embeddings.create(input=batch, model=model)
        batch_embeddings = [e.embedding for e in response.data]
        embeddings.extend(batch_embeddings)

        # Track embedding token usage
        if hasattr(response, "usage") and response.usage:
            token_tracker.add_tokens(response.usage.total_tokens)

        if i > 0 and i % (batch_size * 10) == 0:
            logger.info(f"Embedded {i}/{len(texts)} texts")

    return np.array(embeddings, dtype=np.float32)


def _combine_text_columns(
    df: pd.DataFrame,
    text_cols: List[str],
) -> List[str]:
    """Combine text columns into single strings for embedding."""
    combined_texts = []

    for _, row in df.iterrows():
        text_parts = []
        for col in text_cols:
            value = row.get(col, "")
            # Handle None, NaN, and empty values
            # pd.isna() fails on arrays, so check type first
            if value is None:
                value = ""
            elif isinstance(value, (list, tuple, np.ndarray)):
                # Join list values into a single string
                value = " ".join(str(v) for v in value if v is not None and not (isinstance(v, float) and np.isnan(v)))
            elif isinstance(value, float) and np.isnan(value):
                value = ""
            text_parts.append(str(value))
        combined_texts.append(" ".join(text_parts))

    return combined_texts


# =============================================================================
# FAISS Index Functions
# =============================================================================


def _build_faiss_index(embeddings: np.ndarray) -> Any:
    """
    Build FAISS inner product index for cosine similarity search.

    Parameters
    ----------
    embeddings : np.ndarray
        Embeddings of shape (n_records, embedding_dim)

    Returns
    -------
    faiss.IndexFlatIP
        FAISS index for inner product search (cosine similarity on normalized vectors)
    """
    try:
        import faiss
    except ImportError:
        raise ImportError(
            "faiss-cpu is required for FAISS backend. "
            "Install with: pip install faiss-cpu"
        )

    # Limit FAISS internal threads to prevent OpenMP conflicts
    faiss.omp_set_num_threads(1)

    logger.info(f"Building FAISS index with {embeddings.shape[0]} vectors, dim={embeddings.shape[1]}")

    # Handle NaN/Inf values
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        logger.warning("Embeddings contain NaN or Inf values, replacing with zeros")
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure float32 and C-contiguous (required by FAISS)
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    # L2 normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    # Create and populate index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    logger.info(f"FAISS index built: {index.ntotal} vectors indexed")
    return index


def _query_faiss_index(
    query_embeddings: np.ndarray,
    index: Any,
    k: int = 20,
    bottom_k: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Query FAISS index for nearest neighbors plus random bottom (least similar) examples.

    Parameters
    ----------
    query_embeddings : np.ndarray
        Query embeddings of shape (n_queries, embedding_dim)
    index : faiss.IndexFlatIP
        FAISS index
    k : int
        Total number of neighbors to retrieve (including bottom_k)
    bottom_k : int
        Number of least similar examples to include per query (default 2)
    random_state : int
        Random seed for selecting bottom examples

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (indices, similarities) - both of shape (n_queries, k)
        Each row contains (k - bottom_k) most similar + bottom_k randomly selected
        from the bottom half of results
    """
    import faiss
    faiss.omp_set_num_threads(1)

    rng = np.random.RandomState(random_state)
    n_queries = query_embeddings.shape[0]
    logger.info(f"Querying FAISS index: {n_queries} queries, k={k} (including {bottom_k} random bottom)")

    # Handle NaN/Inf in query embeddings
    if np.isnan(query_embeddings).any() or np.isinf(query_embeddings).any():
        logger.warning("Query embeddings contain NaN or Inf values, replacing with zeros")
        query_embeddings = np.nan_to_num(query_embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure float32 and C-contiguous (required by FAISS)
    query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)

    # L2 normalize for cosine similarity
    faiss.normalize_L2(query_embeddings)

    # We need to fetch more results to get bottom examples
    # Fetch all items if index is small, otherwise fetch enough to get diverse results
    top_k = k - bottom_k
    fetch_k = min(index.ntotal, max(k + 50, index.ntotal))  # Fetch more to pick bottom

    # Search the index
    all_similarities, all_indices = index.search(query_embeddings, fetch_k)

    # For each query, take top (k - bottom_k) and randomly select bottom_k from bottom half
    final_indices = np.zeros((n_queries, k), dtype=np.int64)
    final_similarities = np.zeros((n_queries, k), dtype=np.float32)

    for i in range(n_queries):
        row_idx = all_indices[i]
        row_sim = all_similarities[i]

        # Filter out -1 (no result) entries
        valid_mask = row_idx >= 0
        row_idx = row_idx[valid_mask]
        row_sim = row_sim[valid_mask]

        n_valid = len(row_idx)
        if n_valid == 0:
            # No valid results for this query
            final_indices[i, :] = -1
            final_similarities[i, :] = 0.0
            continue

        if n_valid >= k and n_valid > top_k + bottom_k:
            # Enough results: take top (k - bottom_k) and randomly select bottom_k from bottom half
            top_indices = row_idx[:top_k]
            top_sims = row_sim[:top_k]

            # Select from bottom half of results (excluding top_k already taken)
            bottom_half_start = max(top_k, n_valid // 2)
            bottom_pool_idx = row_idx[bottom_half_start:]
            bottom_pool_sim = row_sim[bottom_half_start:]

            if len(bottom_pool_idx) >= bottom_k:
                # Randomly select bottom_k from the bottom pool
                selected = rng.choice(len(bottom_pool_idx), size=bottom_k, replace=False)
                bottom_indices = bottom_pool_idx[selected]
                bottom_sims = bottom_pool_sim[selected]
            else:
                # Not enough in bottom pool, take all
                bottom_indices = bottom_pool_idx
                bottom_sims = bottom_pool_sim
                # Pad if needed
                if len(bottom_indices) < bottom_k:
                    pad_size = bottom_k - len(bottom_indices)
                    bottom_indices = np.concatenate([bottom_indices, np.full(pad_size, -1)])
                    bottom_sims = np.concatenate([bottom_sims, np.zeros(pad_size)])

            final_indices[i, :] = np.concatenate([top_indices, bottom_indices])
            final_similarities[i, :] = np.concatenate([top_sims, bottom_sims])
        else:
            # Not enough results, return all we have (padded with -1 if needed)
            actual_k = min(n_valid, k)
            final_indices[i, :actual_k] = row_idx[:actual_k]
            final_similarities[i, :actual_k] = row_sim[:actual_k]
            if actual_k < k:
                final_indices[i, actual_k:] = -1
                final_similarities[i, actual_k:] = 0.0

    logger.info(f"FAISS search complete: {n_queries} queries, {top_k} top + {bottom_k} random bottom per query")
    return final_indices, final_similarities


# =============================================================================
# Similarity-Based Set Generator Class
# =============================================================================


class SimilarityBasedSetGenerator:
    """
    Generate training/validation sets using embedding similarity search.

    Uses the smaller dataset to query the larger dataset via FAISS,
    then labels pairs using an LLM.

    Parameters
    ----------
    df_small : pd.DataFrame
        The smaller dataset (used as queries)
    df_large : pd.DataFrame
        The larger dataset (indexed for similarity search)
    id_column : str
        Column name for record identifiers
    embedding_columns : List[str]
        Columns to use for embedding
    embedding_model : str
        OpenAI embedding model name
    k : int
        Number of nearest neighbors to retrieve per query
    neighbors_per_query : Tuple[int, int]
        Range of neighbors to select per query (min, max)
    output_dir : Path, optional
        Directory for caching embeddings. If provided, embeddings are saved/loaded.
    small_name : str, optional
        Name for small dataset (used in cache filename)
    large_name : str, optional
        Name for large dataset (used in cache filename)
    input_embeddings_dir : Path, optional
        Directory containing pre-computed embeddings. If provided, embeddings are
        loaded from here first before checking output_dir cache.
    """

    def __init__(
        self,
        df_small: pd.DataFrame,
        df_large: pd.DataFrame,
        id_column: str,
        embedding_columns: List[str],
        *,
        embedding_model: str = "text-embedding-3-small",
        k: int = 20,
        neighbors_per_query: Tuple[int, int] = (3, 5),
        output_dir: Optional[Path] = None,
        small_name: str = "small",
        large_name: str = "large",
        input_embeddings_dir: Optional[Path] = None,
    ):
        self.df_small = df_small.reset_index(drop=True)
        self.df_large = df_large.reset_index(drop=True)
        # Preserve attrs (reset_index doesn't copy them)
        self.df_small.attrs = df_small.attrs.copy()
        self.df_large.attrs = df_large.attrs.copy()
        # Also set dataset_name in attrs if not already set (use small_name/large_name)
        if "dataset_name" not in self.df_small.attrs:
            self.df_small.attrs["dataset_name"] = small_name
        if "dataset_name" not in self.df_large.attrs:
            self.df_large.attrs["dataset_name"] = large_name

        self.id_column = id_column
        self.embedding_columns = embedding_columns
        self.embedding_model = embedding_model
        self.k = k
        self.neighbors_per_query = neighbors_per_query
        self.output_dir = Path(output_dir) if output_dir else None
        self.small_name = small_name
        self.large_name = large_name
        self.input_embeddings_dir = Path(input_embeddings_dir) if input_embeddings_dir else None

        # Lazy-loaded components
        self._large_embeddings: Optional[np.ndarray] = None
        self._small_embeddings: Optional[np.ndarray] = None
        self._index: Optional[Any] = None

        # ID lookups
        self._small_ids = self.df_small[id_column].tolist()
        self._large_ids = self.df_large[id_column].tolist()

        logger.info(
            f"SimilarityBasedSetGenerator initialized: "
            f"small={len(df_small)}, large={len(df_large)}, "
            f"columns={embedding_columns}, k={k}"
        )

    def _get_cache_path(self, dataset_name: str) -> Optional[Path]:
        """Get cache path for a dataset's embeddings."""
        if self.output_dir is None:
            return None
        return _get_embedding_cache_path(self.output_dir, dataset_name)

    def _get_input_embedding_path(self, dataset_name: str) -> Optional[Path]:
        """Get path for pre-computed embeddings in input directory."""
        if self.input_embeddings_dir is None:
            return None
        path = self.input_embeddings_dir / f"embeddings_{dataset_name}.npy"
        return path if path.exists() else None

    def _ensure_embeddings(self) -> None:
        """Compute embeddings if not already done, using cache if available."""
        if self._large_embeddings is None:
            # Try loading from input directory first (pre-computed embeddings)
            input_path = self._get_input_embedding_path(self.large_name)
            if input_path:
                self._large_embeddings = _load_embeddings_from_cache(
                    input_path, len(self.df_large)
                )
                if self._large_embeddings is not None:
                    logger.info(f"Loaded pre-computed embeddings from {input_path}")

            # Try loading from output cache
            cache_path = self._get_cache_path(self.large_name)
            if self._large_embeddings is None and cache_path:
                self._large_embeddings = _load_embeddings_from_cache(
                    cache_path, len(self.df_large)
                )

            # Compute if not cached
            if self._large_embeddings is None:
                logger.info(f"Computing embeddings for large dataset ({len(self.df_large)} records)...")
                texts = _combine_text_columns(self.df_large, self.embedding_columns)
                self._large_embeddings = _compute_openai_embeddings(
                    texts, model=self.embedding_model
                )
                logger.info(f"Large dataset embeddings: {self._large_embeddings.shape}")

                # Save to cache
                if cache_path:
                    _save_embeddings_to_cache(self._large_embeddings, cache_path)

        if self._small_embeddings is None:
            # Try loading from input directory first (pre-computed embeddings)
            input_path = self._get_input_embedding_path(self.small_name)
            if input_path:
                self._small_embeddings = _load_embeddings_from_cache(
                    input_path, len(self.df_small)
                )
                if self._small_embeddings is not None:
                    logger.info(f"Loaded pre-computed embeddings from {input_path}")

            # Try loading from output cache
            cache_path = self._get_cache_path(self.small_name)
            if self._small_embeddings is None and cache_path:
                self._small_embeddings = _load_embeddings_from_cache(
                    cache_path, len(self.df_small)
                )

            # Compute if not cached
            if self._small_embeddings is None:
                logger.info(f"Computing embeddings for small dataset ({len(self.df_small)} records)...")
                texts = _combine_text_columns(self.df_small, self.embedding_columns)
                self._small_embeddings = _compute_openai_embeddings(
                    texts, model=self.embedding_model
                )
                logger.info(f"Small dataset embeddings: {self._small_embeddings.shape}")

                # Save to cache
                if cache_path:
                    _save_embeddings_to_cache(self._small_embeddings, cache_path)

    def _ensure_index(self) -> None:
        """Build FAISS index if not already done."""
        self._ensure_embeddings()

        if self._index is None:
            logger.info("Building FAISS index on large dataset...")
            self._index = _build_faiss_index(self._large_embeddings)

    def find_all_neighbors(
        self,
        random_state: Optional[int] = None,
        bottom_k: int = 2,
    ) -> pd.DataFrame:
        """
        Find k nearest neighbors for all records in small dataset.

        Parameters
        ----------
        random_state : int, optional
            If provided, shuffle the small dataset before querying.
            Different random states produce different query orderings.
        bottom_k : int
            Number of least similar examples to include per query (default 2)

        Returns
        -------
        pd.DataFrame
            All neighbors with columns:
            - query_id: ID from smaller dataset
            - neighbor_id: ID from larger dataset
            - similarity: Cosine similarity score
            - rank: 1-based rank (1 = closest)
        """
        self._ensure_index()

        # Shuffle small dataset if random_state provided
        if random_state is not None:
            rng = np.random.RandomState(random_state)
            shuffle_indices = rng.permutation(len(self._small_ids))
            small_embeddings = self._small_embeddings[shuffle_indices]
            small_ids = [self._small_ids[i] for i in shuffle_indices]
            logger.info(f"Shuffled small dataset with random_state={random_state}")
        else:
            small_embeddings = self._small_embeddings
            small_ids = self._small_ids

        indices, similarities = _query_faiss_index(
            small_embeddings, self._index, k=self.k, bottom_k=bottom_k,
            random_state=random_state if random_state is not None else 42
        )

        results = []
        for q_idx, (neighbors, sims) in enumerate(zip(indices, similarities)):
            query_id = small_ids[q_idx]

            for rank, (neighbor_idx, sim) in enumerate(zip(neighbors, sims), start=1):
                if 0 <= neighbor_idx < len(self._large_ids):
                    results.append({
                        "query_id": query_id,
                        "neighbor_id": self._large_ids[neighbor_idx],
                        "similarity": float(sim),
                        "rank": rank,
                    })

        return pd.DataFrame(results)

    def select_candidates(
        self,
        all_neighbors: Optional[pd.DataFrame] = None,
        *,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Select candidate pairs for labeling.

        For each query, selects the top neighbors_per_query[0] to neighbors_per_query[1]
        neighbors sorted by similarity (highest first).

        Parameters
        ----------
        all_neighbors : pd.DataFrame, optional
            Pre-computed neighbors. If None, calls find_all_neighbors().
        random_state : int
            Random seed for reproducibility (used to vary number of neighbors per query)

        Returns
        -------
        pd.DataFrame
            Candidate pairs with columns [id1, id2, similarity]
            where id1 is from small dataset, id2 is from large dataset
            Sorted by similarity descending (highest first)
        """
        if all_neighbors is None:
            all_neighbors = self.find_all_neighbors()

        rng = np.random.default_rng(random_state)
        min_neighbors, max_neighbors = self.neighbors_per_query

        results = []
        for query_id, group in all_neighbors.groupby("query_id"):
            # Sort by similarity descending (highest first)
            sorted_group = group.sort_values("similarity", ascending=False)

            # Randomly select how many neighbors to take (between min and max)
            n_select = rng.integers(min_neighbors, max_neighbors + 1)
            n_select = min(n_select, len(sorted_group))

            # Take top N by similarity
            selected = sorted_group.head(n_select)

            for _, row in selected.iterrows():
                results.append({
                    "id1": row["query_id"],
                    "id2": row["neighbor_id"],
                    "similarity": row["similarity"],
                })

        candidates = pd.DataFrame(results)

        # Sort final candidates by similarity descending for labeling order
        candidates = candidates.sort_values("similarity", ascending=False).reset_index(drop=True)

        logger.info(f"Selected {len(candidates)} candidate pairs from {len(all_neighbors)} total neighbors (sorted by similarity)")
        return candidates


# =============================================================================
# Entity Uniqueness Guidelines
# =============================================================================


def generate_entity_uniqueness_guidelines(
    df_small: pd.DataFrame,
    df_large: pd.DataFrame,
    chat_model,
    id_column: str,
    n_samples: int = 10,
) -> str:
    """
    Generate entity uniqueness guidelines by analyzing sample records from both datasets.

    This function calls GPT-5.2 to analyze sample records and generate guidelines
    for the labeling LLM about what makes entities unique in this specific domain.

    For example, for video games, it might generate guidelines like:
    "A video game with the same name is NOT the same entity if they are on different platforms."

    Parameters
    ----------
    df_small : pd.DataFrame
        The smaller dataset (query dataset).
    df_large : pd.DataFrame
        The larger dataset (index dataset).
    chat_model : BaseChatModel
        LangChain chat model instance.
    id_column : str
        Name of the ID column.
    n_samples : int
        Number of sample rows to show from each dataset. Default is 10.

    Returns
    -------
    str
        Generated guidelines for the labeling LLM.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    # Get sample rows from each dataset (excluding ID column for cleaner display)
    display_cols_small = [c for c in df_small.columns if c != id_column and not c.startswith("_")]
    display_cols_large = [c for c in df_large.columns if c != id_column and not c.startswith("_")]

    samples_small = df_small[display_cols_small].head(n_samples).to_string(index=False)
    samples_large = df_large[display_cols_large].head(n_samples).to_string(index=False)

    # Get dataset names if available
    small_name = df_small.attrs.get("dataset_name", "Dataset A")
    large_name = df_large.attrs.get("dataset_name", "Dataset B")

    system_prompt = """You are an expert in entity resolution. Provide brief, actionable tips."""

    user_prompt = f"""Analyze these two datasets and give 2-3 short tips for matching entities. What attributes are highly relevant and which are of lower importance.

=== {small_name} (Sample) ===
{samples_small}

=== {large_name} (Sample) ===
{samples_large}

In 2-3 sentences, what key attributes identify a match and what differentiates similar entities?"""

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = chat_model.invoke(messages)
        guidelines = response.content

        # Log the generated guidelines
        logger.info("=" * 60)
        logger.info("ENTITY UNIQUENESS GUIDELINES (Generated by LLM)")
        logger.info("=" * 60)
        logger.info(f"\n{guidelines}")
        logger.info("=" * 60)

        return guidelines

    except Exception as e:
        logger.warning(f"Failed to generate entity uniqueness guidelines: {e}")
        return ""


# =============================================================================
# Labeling Functions
# =============================================================================


def _label_batch(
    df_small: pd.DataFrame,
    df_large: pd.DataFrame,
    pairs: pd.DataFrame,
    chat_model,
    id_column: str,
    entity_guidelines: str = "",
) -> pd.DataFrame:
    """Label a batch of pairs using LLM.

    Parameters
    ----------
    entity_guidelines : str, optional
        Domain-specific guidelines for entity matching generated by
        generate_entity_uniqueness_guidelines(). If provided, these are
        incorporated into the system prompt to help the LLM make better
        matching decisions.
    """
    matcher = LLMBasedMatcher()

    # Build custom system prompt if guidelines are provided
    system_prompt = None
    if entity_guidelines:
        system_prompt = f"""You are an expert entity resolver. Your task is to decide if two records refer to the same real-world entity.

IMPORTANT DOMAIN-SPECIFIC GUIDELINES:
{entity_guidelines}

Analyze the provided records carefully and return your decision as strict JSON in this format:
{{{{"match": true|false}}}}

Additional guidelines:
- match: true if records refer to the same entity, false otherwise
- Consider variations in naming, formatting, abbreviations, and data quality
- Pay close attention to the domain-specific guidelines above
- Respond with ONLY the JSON object and nothing else."""

    try:
        pred = matcher.match(
            df_left=df_small,
            df_right=df_large,
            candidates=pairs[["id1", "id2"]],
            id_column=id_column,
            chat_model=chat_model,
            generate_explanations=False,
            parse_strictness="skip",
            system_prompt=system_prompt,
        )

        if pred is not None and not pred.empty:
            pred["label"] = pred["match"].apply(
                lambda x: "TRUE" if bool(x) else "FALSE"
            )
            # Merge similarity back
            pred = pred.merge(
                pairs[["id1", "id2", "similarity"]],
                on=["id1", "id2"],
                how="left",
            )
            return pred[["id1", "id2", "label", "similarity"]]
    except Exception as e:
        logger.warning(f"Error labeling batch: {e}")

    return pd.DataFrame(columns=["id1", "id2", "label", "similarity"])


def _label_query_until_satisfied(
    df_small: pd.DataFrame,
    df_large: pd.DataFrame,
    query_neighbors: pd.DataFrame,
    chat_model,
    id_column: str,
    *,
    target_positives: int = 1,
    target_negatives: int = 2,
    batch_size: int = 10,
    entity_guidelines: str = "",
    bottom_k: int = 2,
) -> pd.DataFrame:
    """
    Label neighbors for a single query until we find target_positives matches.

    Goes through ALL k neighbors from highest to lowest similarity until we find
    at least target_positives matches. Also labels the bottom_k candidates (lowest
    similarity) to get likely non-matches for diversity.

    If 2+ matches are found, labels ALL k candidates and keeps everything.
    Otherwise, randomly selects target_negatives from all labeled negatives.

    Parameters
    ----------
    query_neighbors : pd.DataFrame
        Neighbors for this query, sorted by similarity descending
        Columns: [query_id, neighbor_id, similarity, rank]
    target_positives : int
        Keep labeling until we find this many positives (or exhaust all neighbors)
    target_negatives : int
        Number of negatives to include (randomly sampled from all labeled negatives)
        Only applies if fewer than 2 matches found.
    batch_size : int
        How many pairs to label at once
    entity_guidelines : str, optional
        Domain-specific guidelines for entity matching.
    bottom_k : int
        Number of bottom (lowest similarity) candidates to also label (default 2)

    Returns
    -------
    pd.DataFrame
        Labeled pairs with columns [id1, id2, label, similarity]
    """
    # Ensure sorted by similarity descending
    sorted_neighbors = query_neighbors.sort_values("similarity", ascending=False)

    labeled_results = []
    n_positives = 0
    n_negatives_from_top = 0  # Track negatives found while labeling top candidates

    # Convert to candidate format
    candidates = pd.DataFrame({
        "id1": sorted_neighbors["query_id"],
        "id2": sorted_neighbors["neighbor_id"],
        "similarity": sorted_neighbors["similarity"],
    }).reset_index(drop=True)

    # Track which indices we've labeled (to avoid double-labeling bottom candidates)
    labeled_indices = set()

    # Label from top down until we find target_positives + enough negatives to sample from
    # We need at least target_negatives to sample from, so keep going until we have that many
    for i in range(0, len(candidates), batch_size):
        # Only stop if we found enough positives AND have enough negatives to sample from
        if n_positives >= target_positives and n_negatives_from_top >= target_negatives:
            break

        batch = candidates.iloc[i : i + batch_size]
        batch_indices = list(range(i, min(i + batch_size, len(candidates))))
        labeled = _label_batch(df_small, df_large, batch, chat_model, id_column, entity_guidelines)

        if not labeled.empty:
            labeled_results.append(labeled)
            labeled_indices.update(batch_indices)

            # Count what we found
            batch_pos = (labeled["label"] == "TRUE").sum()
            batch_neg = (labeled["label"] == "FALSE").sum()
            n_positives += batch_pos
            n_negatives_from_top += batch_neg

            logger.debug(
                f"Query batch {i//batch_size + 1}: +{batch_pos}/-{batch_neg}, "
                f"total: {n_positives}/{target_positives} pos, {n_negatives_from_top} neg"
            )

    # If we found 2+ matches, label ALL remaining candidates and keep everything
    if n_positives >= 2:
        logger.debug(f"Found {n_positives} matches - labeling all remaining candidates")
        # Label any remaining unlabeled candidates
        unlabeled_indices = [i for i in range(len(candidates)) if i not in labeled_indices]
        if unlabeled_indices:
            remaining_batch = candidates.iloc[unlabeled_indices]
            remaining_labeled = _label_batch(
                df_small, df_large, remaining_batch, chat_model, id_column, entity_guidelines
            )
            if not remaining_labeled.empty:
                labeled_results.append(remaining_labeled)
                labeled_indices.update(unlabeled_indices)

        # Return ALL labeled results (no sampling)
        if not labeled_results:
            return pd.DataFrame(columns=["id1", "id2", "label", "similarity"])
        return pd.concat(labeled_results, ignore_index=True)

    # Otherwise (0-1 matches), also label the bottom_k candidates for diversity
    # These are likely non-matches and add variety to our negative samples
    if len(candidates) > bottom_k:
        bottom_indices = list(range(len(candidates) - bottom_k, len(candidates)))
        # Only label bottom candidates that weren't already labeled
        unlabeled_bottom_indices = [idx for idx in bottom_indices if idx not in labeled_indices]

        if unlabeled_bottom_indices:
            bottom_batch = candidates.iloc[unlabeled_bottom_indices]
            bottom_labeled = _label_batch(
                df_small, df_large, bottom_batch, chat_model, id_column, entity_guidelines
            )
            if not bottom_labeled.empty:
                labeled_results.append(bottom_labeled)
                labeled_indices.update(unlabeled_bottom_indices)
                bottom_neg = (bottom_labeled["label"] == "FALSE").sum()
                logger.debug(f"Labeled {len(bottom_labeled)} bottom candidates ({bottom_neg} negatives)")

    if not labeled_results:
        return pd.DataFrame(columns=["id1", "id2", "label", "similarity"])

    # Combine all labeled results
    all_labeled = pd.concat(labeled_results, ignore_index=True)

    # Separate positives and negatives
    positives = all_labeled[all_labeled["label"] == "TRUE"]
    negatives = all_labeled[all_labeled["label"] == "FALSE"]

    # Randomly sample target_negatives from all labeled negatives (both top and bottom)
    if len(negatives) > target_negatives:
        sampled_negatives = negatives.sample(n=target_negatives, random_state=42)
        logger.debug(
            f"Randomly sampled {target_negatives} negatives from {len(negatives)} total"
        )
    else:
        sampled_negatives = negatives

    # Combine positives with sampled negatives
    result = pd.concat([positives, sampled_negatives], ignore_index=True)
    return result


def _label_iteratively_per_query(
    df_small: pd.DataFrame,
    df_large: pd.DataFrame,
    all_neighbors: pd.DataFrame,
    chat_model,
    id_column: str,
    *,
    target_positives_per_query: int = 1,
    target_negatives_per_query: int = 2,
    total_target_positives: int = 30,
    total_target_size: int = 100,
    max_labels: Optional[int] = None,
    batch_size: int = 5,
    query_order: str = "random",
    generate_guidelines: bool = True,
    bottom_k: int = 2,
) -> pd.DataFrame:
    """
    Label neighbors query by query, stopping early when targets are met.

    For each query:
    1. Get k=20 neighbors sorted by similarity (highest first)
    2. Label from top down until we find target_positives + target_negatives
    3. Only include this query's results if we found at least 1 positive
    4. Stop processing queries once we have enough total positives

    Parameters
    ----------
    all_neighbors : pd.DataFrame
        All neighbors from FAISS with columns [query_id, neighbor_id, similarity, rank]
    target_positives_per_query : int
        Target positives per query before moving to next query
    target_negatives_per_query : int
        Target negatives (hard negatives) per query
    total_target_positives : int
        Stop processing queries once we have this many total positives
    total_target_size : int
        Max total labeled pairs (soft limit - won't stop if we need more positives)
    max_labels : int, optional
        Maximum total labels to generate. Hard cap on LLM calls.
        If None, no limit is applied.
    batch_size : int
        LLM batch size per query
    query_order : str
        Order to process queries: "random" (shuffle) or "similarity" (highest max similarity first)
    generate_guidelines : bool
        Whether to generate entity uniqueness guidelines before labeling.
        Default is True.
    bottom_k : int
        Number of bottom (lowest similarity) candidates to also label per query
        for diversity in negative samples. Default is 2.

    Returns
    -------
    pd.DataFrame
        Labeled pairs with columns [id1, id2, label, similarity]
    """
    # Generate entity uniqueness guidelines before starting labeling
    entity_guidelines = ""
    if generate_guidelines:
        logger.info("Generating entity uniqueness guidelines...")
        entity_guidelines = generate_entity_uniqueness_guidelines(
            df_small=df_small,
            df_large=df_large,
            chat_model=chat_model,
            id_column=id_column,
            n_samples=10,
        )

    all_labeled = []
    total_positives = 0
    total_negatives = 0
    total_labels = 0
    queries_processed = 0
    queries_with_matches = 0

    # Group neighbors by query_id for efficient lookup
    query_groups = {qid: group for qid, group in all_neighbors.groupby("query_id")}

    # Order queries based on query_order parameter
    if query_order == "similarity":
        # Sort queries by their best neighbor's similarity (highest first)
        # This prioritizes queries most likely to have matches
        query_max_sim = all_neighbors.groupby("query_id")["similarity"].max()
        query_ids = query_max_sim.sort_values(ascending=False).index.to_numpy()
        logger.info(f"Processing {len(query_ids)} queries ordered by max similarity (highest first)")
    else:
        # Random shuffle (default) - avoids bias from ordering
        rng = np.random.default_rng(42)
        query_ids = all_neighbors["query_id"].unique()
        rng.shuffle(query_ids)
        logger.info(f"Processing {len(query_ids)} queries in random order")

    if max_labels:
        logger.info(f"Max labels cap: {max_labels}")

    n_queries = len(query_ids)

    for query_id in query_ids:
        group = query_groups[query_id]
        # Check if we have enough total positives
        if total_positives >= total_target_positives:
            logger.info(
                f"Reached target positives ({total_positives}/{total_target_positives}), "
                f"stopping after {queries_processed}/{n_queries} queries "
                f"({queries_with_matches} had matches)"
            )
            break

        # Check if we hit the max labels cap
        if max_labels and total_labels >= max_labels:
            logger.info(
                f"Reached max labels cap ({total_labels}/{max_labels}), "
                f"stopping after {queries_processed}/{n_queries} queries "
                f"({total_positives} positives, {total_negatives} negatives)"
            )
            break

        # Label this query's neighbors
        query_labeled = _label_query_until_satisfied(
            df_small=df_small,
            df_large=df_large,
            query_neighbors=group,
            chat_model=chat_model,
            id_column=id_column,
            target_positives=target_positives_per_query,
            target_negatives=target_negatives_per_query,
            batch_size=batch_size,
            entity_guidelines=entity_guidelines,
            bottom_k=bottom_k,
        )

        queries_processed += 1

        if not query_labeled.empty:
            query_pos = (query_labeled["label"] == "TRUE").sum()
            query_neg = (query_labeled["label"] == "FALSE").sum()
            query_total = len(query_labeled)

            # Count all LLM calls toward max_labels limit (even queries without matches)
            total_labels += query_total

            # Only include this query in results if it found at least 1 match
            if query_pos > 0:
                all_labeled.append(query_labeled)
                total_positives += query_pos
                total_negatives += query_neg
                queries_with_matches += 1

                max_labels_str = f"/{max_labels}" if max_labels else ""
                logger.info(
                    f"[+{query_pos}] {total_positives} matches, {total_negatives} non-matches | "
                    f"{total_labels}{max_labels_str} labels"
                )
                # Print overall progress towards target
                print(
                    f"  >> Overall: {total_positives}/{total_target_positives} positives, "
                    f"{total_negatives} negatives ({queries_with_matches} queries with matches)"
                )

    if not all_labeled:
        return pd.DataFrame(columns=["id1", "id2", "label", "similarity"])

    result = pd.concat(all_labeled, ignore_index=True)
    logger.info(
        f"Labeling complete: {len(result)} pairs "
        f"({total_positives} positive, {total_negatives} negative) "
        f"from {queries_with_matches} queries with matches "
        f"(processed {queries_processed} total)"
    )
    return result


# =============================================================================
# High-Level Generation Functions
# =============================================================================


def _determine_datasets(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    """
    Determine which dataset is smaller (query) and larger (index).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, bool]
        (df_small, df_large, is_left_small)
    """
    if len(df_left) <= len(df_right):
        return df_left, df_right, True
    else:
        return df_right, df_left, False


def _get_embedding_columns(
    df_small: pd.DataFrame,
    df_large: pd.DataFrame,
    chat_model,
    id_column: str,
) -> List[str]:
    """Select columns for embedding using LLM."""
    # Use existing LLM-based column selection
    blocking_strategies = select_blocking_columns(
        df_small, df_large, chat_model, id_column
    )

    if not blocking_strategies:
        # Fallback: use all common string columns
        common_cols = set(df_small.columns) & set(df_large.columns) - {id_column}
        string_cols = [
            c for c in common_cols
            if df_small[c].dtype == object or df_large[c].dtype == object
        ]
        if string_cols:
            logger.warning(f"LLM column selection failed, using: {string_cols[:3]}")
            return string_cols[:3]
        raise ValueError("No suitable columns for embedding")

    # Parse strategies like "title+year" into individual columns
    all_cols = []
    for strategy in blocking_strategies:
        all_cols.extend(parse_blocking_strategy(strategy))

    # Remove duplicates while preserving order
    unique_cols = list(dict.fromkeys(all_cols))

    # Use up to 5 columns
    result = unique_cols[:5]
    logger.info(f"Selected embedding columns: {result}")
    return result


def _generate_with_faiss(
    df_small: pd.DataFrame,
    df_large: pd.DataFrame,
    chat_model,
    embedding_columns: List[str],
    *,
    id_column: str = "id",
    target_size: int = 100,
    target_positives: int = 30,
    embedding_model: str = "text-embedding-3-small",
    k: int = 20,
    max_labels: Optional[int] = None,
    neighbors_per_query: Tuple[int, int] = (1, 2),
    batch_size: int = 5,
    query_order: str = "random",
    output_dir: Optional[Path] = None,
    small_name: str = "small",
    large_name: str = "large",
    generate_guidelines: bool = True,
    exclude_pairs: Optional[Set[Tuple[str, str]]] = None,
    exclude_query_ids: Optional[Set[str]] = None,
    shuffle_random_state: Optional[int] = 42,
    bottom_k: int = 2,
    input_embeddings_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generate labeled pairs using FAISS (embedding-based) retrieval.

    Parameters
    ----------
    df_small : pd.DataFrame
        Smaller dataset (used as queries)
    df_large : pd.DataFrame
        Larger dataset (indexed)
    chat_model : BaseChatModel
        LangChain chat model for labeling
    embedding_columns : List[str]
        Columns to use for embedding
    id_column : str
        ID column name
    target_size : int
        Target number of pairs
    target_positives : int
        Target positive pairs
    embedding_model : str
        OpenAI embedding model name
    k : int
        Neighbors per query
    neighbors_per_query : Tuple[int, int]
        (target_pos_per_query, target_neg_per_query)
    batch_size : int
        LLM batch size
    query_order : str
        "random" or "similarity"
    output_dir : Path, optional
        For embedding cache
    small_name, large_name : str
        Dataset names for caching
    generate_guidelines : bool
        Whether to generate entity guidelines
    exclude_pairs : Set[Tuple[str, str]], optional
        Set of (id1, id2) pairs to exclude from labeling
    exclude_query_ids : Set[str], optional
        Query IDs to exclude entirely
    shuffle_random_state : int, optional
        Random state for shuffling the small dataset before querying.
        Pass different values to get different query orderings.
    bottom_k : int
        Number of least similar examples to include per query (default 2)

    Returns
    -------
    pd.DataFrame
        Labeled pairs with columns [id1, id2, label, similarity]
    """
    logger.info(f"Generating FAISS-based labeled pairs (target: {target_positives} positives)...")

    # Create generator with caching
    generator = SimilarityBasedSetGenerator(
        df_small=df_small,
        df_large=df_large,
        id_column=id_column,
        embedding_columns=embedding_columns,
        embedding_model=embedding_model,
        k=k,
        neighbors_per_query=neighbors_per_query,
        output_dir=output_dir,
        small_name=small_name,
        large_name=large_name,
        input_embeddings_dir=input_embeddings_dir,
    )

    # Find all k neighbors for each query (with shuffle and bottom examples)
    all_neighbors = generator.find_all_neighbors(
        random_state=shuffle_random_state,
        bottom_k=bottom_k,
    )

    if all_neighbors.empty:
        logger.warning("No neighbors found from FAISS search")
        return pd.DataFrame(columns=["id1", "id2", "label", "similarity"])

    # Exclude query IDs if specified
    if exclude_query_ids:
        before = len(all_neighbors)
        all_neighbors = all_neighbors[~all_neighbors["query_id"].isin(exclude_query_ids)]
        logger.info(f"Excluded {before - len(all_neighbors)} neighbors from {len(exclude_query_ids)} excluded queries")

    # Exclude specific pairs if specified
    if exclude_pairs:
        before = len(all_neighbors)
        all_neighbors = all_neighbors[
            ~all_neighbors.apply(
                lambda r: (str(r["query_id"]), str(r["neighbor_id"])) in exclude_pairs
                or (str(r["neighbor_id"]), str(r["query_id"])) in exclude_pairs,
                axis=1
            )
        ]
        logger.info(f"Excluded {before - len(all_neighbors)} already-labeled pairs")

    if all_neighbors.empty:
        logger.warning("No neighbors remaining after exclusions")
        return pd.DataFrame(columns=["id1", "id2", "label", "similarity"])

    # Interpret neighbors_per_query as (target_positives_per_query, target_negatives_per_query)
    target_pos_per_query, target_neg_per_query = neighbors_per_query

    # Label iteratively per query
    labeled = _label_iteratively_per_query(
        df_small=df_small,
        df_large=df_large,
        all_neighbors=all_neighbors,
        chat_model=chat_model,
        id_column=id_column,
        target_positives_per_query=target_pos_per_query,
        target_negatives_per_query=target_neg_per_query,
        total_target_positives=target_positives,
        total_target_size=target_size,
        max_labels=max_labels,
        batch_size=batch_size,
        query_order=query_order,
        generate_guidelines=generate_guidelines,
    )

    n_pos = (labeled["label"].astype(str).str.upper() == "TRUE").sum() if not labeled.empty else 0
    n_neg = (labeled["label"].astype(str).str.upper() == "FALSE").sum() if not labeled.empty else 0
    logger.info(f"FAISS labeling complete: {len(labeled)} pairs ({n_pos} positive, {n_neg} negative)")

    return labeled


def generate_similarity_based_labeled_set(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    chat_model,
    *,
    id_column: str = "id",
    target_size: int = 100,
    target_positives: int = 30,
    embedding_model: str = "text-embedding-3-small",
    k: int = 20,
    max_labels: Optional[int] = None,
    neighbors_per_query: Tuple[int, int] = (3, 5),
    exclude_pairs: Optional[pd.DataFrame] = None,
    batch_size: int = 5,
    query_order: str = "random",
    output_dir: Optional[Path] = None,
    left_name: str = "left",
    right_name: str = "right",
    generate_guidelines: bool = True,
    retrieval_method: str = "faiss",  # Kept for API compatibility, only "faiss" supported
    input_embeddings_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generate a labeled set using FAISS similarity search.

    For each query (record in smaller dataset):
    1. Get k neighbors from larger dataset, sorted by similarity
    2. Label from highest similarity down
    3. Stop labeling that query once we find enough positives + negatives
    4. Stop overall once we have target_positives total

    Parameters
    ----------
    df_left, df_right : pd.DataFrame
        Datasets to match
    chat_model : BaseChatModel
        LangChain chat model for labeling
    id_column : str
        ID column name
    target_size : int
        Target number of pairs
    target_positives : int
        Target number of positive pairs
    embedding_model : str
        OpenAI embedding model name
    k : int
        Number of neighbors to retrieve per query
    neighbors_per_query : Tuple[int, int]
        (target_positives_per_query, target_negatives_per_query)
        e.g., (1, 2) means find 1 positive + 2 hard negatives per query
    exclude_pairs : pd.DataFrame, optional
        Pairs to exclude (e.g., validation set). Query IDs from these pairs
        will be skipped during labeling.
    batch_size : int
        LLM labeling batch size per query
    query_order : str
        Order to process queries: "random" (shuffle) or "similarity" (highest max similarity first)
    generate_guidelines : bool
        Whether to generate entity uniqueness guidelines before labeling.
        Default is True.
    retrieval_method : str
        Kept for API compatibility. Only "faiss" is supported.

    Returns
    -------
    pd.DataFrame
        Labeled set with columns [id1, id2, label]
    """
    # Log warning if non-faiss retrieval method requested
    if retrieval_method != "faiss":
        logger.warning(f"retrieval_method='{retrieval_method}' is deprecated. Using 'faiss' instead.")

    logger.info(f"Generating similarity-based labeled set (target: {target_size} pairs, {target_positives} positives)...")

    # Determine smaller/larger datasets
    df_small, df_large, is_left_small = _determine_datasets(df_left, df_right)
    logger.info(f"Small dataset: {len(df_small)} rows, Large dataset: {len(df_large)} rows")

    # Get embedding columns
    embedding_columns = _get_embedding_columns(
        df_small, df_large, chat_model, id_column
    )

    # Determine dataset names for caching
    small_name = left_name if is_left_small else right_name
    large_name = right_name if is_left_small else left_name

    # Build set of query IDs to exclude (from exclude_pairs)
    exclude_query_ids: Optional[Set[str]] = None
    if exclude_pairs is not None and not exclude_pairs.empty:
        # Get the query IDs to exclude (id1 if left is small, id2 if right is small)
        if is_left_small:
            exclude_query_ids = set(str(x) for x in exclude_pairs["id1"].unique())
        else:
            exclude_query_ids = set(str(x) for x in exclude_pairs["id2"].unique())
        logger.info(f"Excluding {len(exclude_query_ids)} queries from {len(exclude_pairs)} exclude_pairs")

    # Generate labeled pairs using FAISS
    labeled = _generate_with_faiss(
        df_small=df_small,
        df_large=df_large,
        chat_model=chat_model,
        embedding_columns=embedding_columns,
        id_column=id_column,
        target_size=target_size,
        target_positives=target_positives,
        embedding_model=embedding_model,
        k=k,
        max_labels=max_labels,
        neighbors_per_query=neighbors_per_query,
        batch_size=batch_size,
        query_order=query_order,
        output_dir=output_dir,
        small_name=small_name,
        large_name=large_name,
        generate_guidelines=generate_guidelines,
        exclude_query_ids=exclude_query_ids,
        input_embeddings_dir=input_embeddings_dir,
    )

    if labeled.empty:
        raise ValueError("No pairs labeled")

    # Balance the set if needed
    result = _select_balanced_set(labeled, target_size, target_positives)

    # Save unused labeled pairs for potential future use
    if output_dir is not None and len(labeled) > len(result):
        # Find pairs that were labeled but not included in the final set
        result_keys = set(zip(result["id1"].astype(str), result["id2"].astype(str)))
        unused_mask = ~labeled.apply(
            lambda r: (str(r["id1"]), str(r["id2"])) in result_keys, axis=1
        )
        unused = labeled[unused_mask][["id1", "id2", "label"]].copy()

        if not unused.empty:
            # Swap id1/id2 for unused pairs too if needed
            if not is_left_small:
                unused = unused.rename(columns={"id1": "id2", "id2": "id1"})

            unused_path = Path(output_dir) / f"{left_name}_{right_name}_unused_labels.csv"
            unused.to_csv(unused_path, index=False)
            n_unused_pos = (unused["label"].astype(str).str.upper() == "TRUE").sum()
            n_unused_neg = (unused["label"].astype(str).str.upper() == "FALSE").sum()
            logger.info(f"Saved {len(unused)} unused labeled pairs to {unused_path} ({n_unused_pos} positive, {n_unused_neg} negative)")

    # Swap id1/id2 back if needed (so id1 is always from left dataset)
    if not is_left_small:
        result = result.rename(columns={"id1": "id2", "id2": "id1"})

    n_pos = (result["label"].astype(str).str.upper() == "TRUE").sum()
    n_neg = (result["label"].astype(str).str.upper() == "FALSE").sum()
    logger.info(f"Labeled set complete: {len(result)} pairs ({n_pos} positive, {n_neg} negative)")

    return result[["id1", "id2", "label"]]


# Backward-compatible aliases
def generate_similarity_based_validation_set(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    chat_model,
    *,
    target_size: int = 100,
    target_positives: int = 30,
    neighbors_per_query: Tuple[int, int] = (3, 5),
    **kwargs,
) -> pd.DataFrame:
    """Generate validation set. See generate_similarity_based_labeled_set for full docs."""
    return generate_similarity_based_labeled_set(
        df_left=df_left,
        df_right=df_right,
        chat_model=chat_model,
        target_size=target_size,
        target_positives=target_positives,
        neighbors_per_query=neighbors_per_query,
        **kwargs,
    )


def generate_similarity_based_training_set(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    chat_model,
    *,
    target_size: int = 500,
    target_positives: int = 150,
    neighbors_per_query: Tuple[int, int] = (1, 2),
        
    **kwargs,
) -> pd.DataFrame:
    """Generate training set. See generate_similarity_based_labeled_set for full docs."""
    return generate_similarity_based_labeled_set(
        df_left=df_left,
        df_right=df_right,
        chat_model=chat_model,
        target_size=target_size,
        target_positives=target_positives,
        neighbors_per_query=neighbors_per_query,
        **kwargs,
    )


# =============================================================================
# Caching Wrappers
# =============================================================================


def load_or_generate_similarity_labeled_set(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    left_name: str,
    right_name: str,
    chat_model,
    output_dir: Path,
    *,
    set_type: str = "validation",
    id_column: str = "id",
    target_size: int = 100,
    target_positives: int = 30,
    exclude_pairs: Optional[pd.DataFrame] = None,
    force_regenerate: bool = False,
    generate_guidelines: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Load labeled set from cache or generate using FAISS similarity search.

    Parameters
    ----------
    df_left, df_right : pd.DataFrame
        Datasets to match
    left_name, right_name : str
        Names for the datasets (used in filename)
    chat_model : BaseChatModel
        LangChain chat model for labeling
    output_dir : Path
        Directory to store/load labeled sets
    set_type : str
        Type of set: "validation" or "training" (used in cache filename)
    id_column : str
        ID column name
    target_size : int
        Target number of pairs
    target_positives : int
        Target positive pairs
    exclude_pairs : pd.DataFrame, optional
        Pairs to exclude (e.g., validation set for training)
    force_regenerate : bool
        If True, regenerate even if cache exists
    generate_guidelines : bool
        Whether to generate entity uniqueness guidelines before labeling.
    **kwargs
        Additional arguments passed to generate_similarity_based_labeled_set

    Returns
    -------
    pd.DataFrame
        Labeled set with columns [id1, id2, label]
    """
    output_dir = Path(output_dir)

    # Build cache key
    cache_set_type = f"similarity_{set_type}_faiss"

    # Try cache first
    if not force_regenerate:
        cached = load_labeled_set_from_cache(
            output_dir, left_name, right_name, set_type=cache_set_type
        )
        if cached is not None:
            # Apply exclusions to cached data if provided
            if exclude_pairs is not None and not exclude_pairs.empty:
                cached = drop_overlapping_pairs(cached, exclude_pairs=exclude_pairs)
            n_pos = (cached["label"].astype(str).str.upper() == "TRUE").sum()
            n_neg = (cached["label"].astype(str).str.upper() == "FALSE").sum()
            logger.info(f"Loaded cached {set_type} set: {len(cached)} pairs ({n_pos} positive, {n_neg} negative)")
            return cached

    # Generate new labeled set
    logger.info(f"Generating new similarity-based {set_type} set for {left_name} <-> {right_name}")

    labeled_set = generate_similarity_based_labeled_set(
        df_left=df_left,
        df_right=df_right,
        chat_model=chat_model,
        id_column=id_column,
        target_size=target_size,
        target_positives=target_positives,
        exclude_pairs=exclude_pairs,
        output_dir=output_dir,
        left_name=left_name,
        right_name=right_name,
        generate_guidelines=generate_guidelines,
        **kwargs,
    )

    # Save to cache
    save_labeled_set_to_cache(
        labeled_set, output_dir, left_name, right_name, set_type=cache_set_type
    )

    return labeled_set


# Backward-compatible aliases
def load_or_generate_similarity_validation_set(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    left_name: str,
    right_name: str,
    chat_model,
    output_dir: Path,
    *,
    target_size: int = 100,
    target_positives: int = 30,
    retrieval_method: str = "faiss",  # Kept for API compatibility
    **kwargs,
) -> pd.DataFrame:
    """Load or generate validation set. See load_or_generate_similarity_labeled_set for full docs."""
    if retrieval_method != "faiss":
        logger.warning(f"retrieval_method='{retrieval_method}' is deprecated. Using 'faiss' instead.")
    return load_or_generate_similarity_labeled_set(
        df_left=df_left,
        df_right=df_right,
        left_name=left_name,
        right_name=right_name,
        chat_model=chat_model,
        output_dir=output_dir,
        set_type="validation",
        target_size=target_size,
        target_positives=target_positives,
        **kwargs,
    )


def load_or_generate_similarity_training_set(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    left_name: str,
    right_name: str,
    chat_model,
    output_dir: Path,
    *,
    target_size: int = 500,
    target_positives: int = 150,
    retrieval_method: str = "faiss",  # Kept for API compatibility
    **kwargs,
) -> pd.DataFrame:
    """Load or generate training set. See load_or_generate_similarity_labeled_set for full docs."""
    if retrieval_method != "faiss":
        logger.warning(f"retrieval_method='{retrieval_method}' is deprecated. Using 'faiss' instead.")
    return load_or_generate_similarity_labeled_set(
        df_left=df_left,
        df_right=df_right,
        left_name=left_name,
        right_name=right_name,
        chat_model=chat_model,
        output_dir=output_dir,
        set_type="training",
        target_size=target_size,
        target_positives=target_positives,
        **kwargs,
    )


# =============================================================================
# FAISS Candidate Generation (for matching without blocking)
# =============================================================================


def generate_faiss_candidates(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    left_name: str,
    right_name: str,
    *,
    id_column: str = "id",
    k: int = 20,
    embedding_model: str = "text-embedding-3-small",
    embedding_columns: Optional[List[str]] = None,
    chat_model: Any = None,
    output_dir: Optional[Path] = None,
    input_embeddings_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generate candidate pairs using FAISS similarity search.

    This function reuses cached embeddings from validation/training set generation
    (stored in entity_resolution/embeddings/). It queries the larger dataset with
    records from the smaller dataset and returns all k nearest neighbors.

    Parameters
    ----------
    df_left : pd.DataFrame
        Left dataset
    df_right : pd.DataFrame
        Right dataset
    left_name : str
        Name of left dataset (for cache path)
    right_name : str
        Name of right dataset (for cache path)
    id_column : str
        Column containing record IDs
    k : int
        Number of neighbors to retrieve per query (default: 20)
    embedding_model : str
        OpenAI embedding model name
    embedding_columns : List[str], optional
        Columns to use for embedding. If None, uses LLM to select.
    chat_model : Any, optional
        LLM for selecting embedding columns (required if embedding_columns is None)
    output_dir : Path, optional
        Directory for embedding cache (should be validation_dir or training_dir
        to reuse embeddings from set generation)

    Returns
    -------
    pd.DataFrame
        Candidate pairs with columns: id1, id2, similarity
        - id1: ID from left dataset
        - id2: ID from right dataset
        - similarity: Cosine similarity score (0-1)
    """
    logger.info(f"Generating FAISS candidates for {left_name} <-> {right_name}")

    # Determine smaller/larger datasets
    df_small, df_large, is_left_small = _determine_datasets(df_left, df_right)
    small_name = left_name if is_left_small else right_name
    large_name = right_name if is_left_small else left_name

    logger.info(
        f"  Small dataset: {small_name} ({len(df_small)} rows), "
        f"Large dataset: {large_name} ({len(df_large)} rows)"
    )

    # Select embedding columns if not provided
    if embedding_columns is None:
        if chat_model is None:
            raise ValueError("chat_model is required when embedding_columns is not provided")
        embedding_columns = _get_embedding_columns(df_small, df_large, chat_model, id_column)

    logger.info(f"  Embedding columns: {embedding_columns}")

    # Create generator (reuses cached embeddings if available)
    generator = SimilarityBasedSetGenerator(
        df_small=df_small,
        df_large=df_large,
        small_name=small_name,
        large_name=large_name,
        id_column=id_column,
        embedding_columns=embedding_columns,
        embedding_model=embedding_model,
        k=k,
        output_dir=output_dir,
        input_embeddings_dir=input_embeddings_dir,
    )

    # Find all neighbors (k per query, with bottom examples for likely non-matches)
    all_neighbors = generator.find_all_neighbors(random_state=42, bottom_k=2)

    if all_neighbors.empty:
        logger.warning("No neighbors found from FAISS search")
        return pd.DataFrame(columns=["id1", "id2", "similarity"])

    logger.info(f"  Found {len(all_neighbors)} neighbor pairs from {all_neighbors['query_id'].nunique()} queries")

    # Transform to standard candidate format (id1, id2, similarity)
    # Ensure id1 comes from left dataset, id2 from right dataset
    if is_left_small:
        # query_id is from left (small), neighbor_id is from right (large)
        candidates = pd.DataFrame({
            "id1": all_neighbors["query_id"],
            "id2": all_neighbors["neighbor_id"],
            "similarity": all_neighbors["similarity"],
        })
    else:
        # query_id is from right (small), neighbor_id is from left (large)
        # Swap so id1 is from left
        candidates = pd.DataFrame({
            "id1": all_neighbors["neighbor_id"],
            "id2": all_neighbors["query_id"],
            "similarity": all_neighbors["similarity"],
        })

    # Sort by similarity descending
    candidates = candidates.sort_values("similarity", ascending=False).reset_index(drop=True)

    logger.info(f"  Generated {len(candidates)} candidate pairs")
    return candidates


__all__ = [
    # Core classes
    "SimilarityBasedSetGenerator",
    # Generation functions (unified)
    "generate_similarity_based_labeled_set",
    # Generation functions (backward-compatible aliases)
    "generate_similarity_based_validation_set",
    "generate_similarity_based_training_set",
    # Caching wrappers (unified)
    "load_or_generate_similarity_labeled_set",
    # Caching wrappers (backward-compatible aliases)
    "load_or_generate_similarity_validation_set",
    "load_or_generate_similarity_training_set",
    # Candidate generation
    "generate_faiss_candidates",
    # Entity uniqueness guidelines
    "generate_entity_uniqueness_guidelines",
]
