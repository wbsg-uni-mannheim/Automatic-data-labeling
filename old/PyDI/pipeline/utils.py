"""Utility functions for the pipeline."""

import pandas as pd

from PyDI.entitymatching.evaluation import EntityMatchingEvaluator


def format_record(
    df: pd.DataFrame,
    record_id: object,
    *,
    id_column: str,
    cols: list[str],
    prefix: str,
) -> str:
    """Format a single record for display."""
    row = df[df[id_column] == record_id]
    if row.empty:
        return f"{prefix}: {record_id} (not found)"
    row = row.iloc[0]
    attrs = [f"{c}={row[c]}" for c in cols if c in df.columns and pd.notna(row.get(c))]
    return f"{prefix} [{record_id}]: {', '.join(attrs[:3])}"


def report_correspondences(
    *,
    title: str,
    corr: pd.DataFrame,
    threshold: float,
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    id_column: str,
    display_cols: list[str],
    top_examples: int = 10,
) -> None:
    """Print a summary of correspondence results for diagnostics.

    Args:
        title: Section title for the report.
        corr: DataFrame with columns 'id1', 'id2', 'score'.
        threshold: Score threshold for matches vs non-matches.
        df_left: Left DataFrame containing source records.
        df_right: Right DataFrame containing source records.
        id_column: Column name used as record identifier.
        display_cols: Columns to display in example output.
        top_examples: Number of example matches/non-matches to show.
    """
    if corr is None or corr.empty:
        print(f"\n  --- {title} ---")
        print("  No correspondences produced.")
        return
    corr = corr.copy()
    corr["score"] = pd.to_numeric(corr["score"], errors="coerce")
    matches = corr[corr["score"] >= float(threshold)].copy()
    non_matches = corr[corr["score"] < float(threshold)].copy()

    print(f"\n  --- {title} (threshold={threshold}) ---")
    print(f"  Total scored candidates: {len(corr)}")
    print(f"  Total matches: {len(matches)}")
    print(f"  Total non-matches: {len(non_matches)}")

    if not matches.empty:
        print("\n  --- Cluster Size Distribution ---")
        print(EntityMatchingEvaluator.create_cluster_size_distribution(matches))

    n = max(top_examples, 0)
    if n:
        print(f"\n  --- {n} Example Matches (highest scores) ---")
        top_matches = matches.nlargest(n, "score") if len(matches) else pd.DataFrame()
        for i, (_, row) in enumerate(top_matches.iterrows(), 1):
            print(f"\n  Match {i} (score={row['score']:.3f}):")
            print(f"    {format_record(df_left, row['id1'], id_column=id_column, cols=display_cols, prefix='Left')}")
            print(f"    {format_record(df_right, row['id2'], id_column=id_column, cols=display_cols, prefix='Right')}")

        print(f"\n  --- {n} Example Non-Matches (closest to threshold) ---")
        top_non = non_matches.nlargest(n, "score") if len(non_matches) else pd.DataFrame()
        for i, (_, row) in enumerate(top_non.iterrows(), 1):
            print(f"\n  Non-Match {i} (score={row['score']:.3f}):")
            print(f"    {format_record(df_left, row['id1'], id_column=id_column, cols=display_cols, prefix='Left')}")
            print(f"    {format_record(df_right, row['id2'], id_column=id_column, cols=display_cols, prefix='Right')}")
