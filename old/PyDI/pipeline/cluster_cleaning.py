"""Post-processing cluster cleaning for oversized entity clusters.

When multiple datasets are matched pair-wise, transitive closure can produce
clusters larger than the number of input datasets.  This module applies PyDI's
post-clustering algorithms **per pair** to enforce one-to-one matching, then
picks the strategy that fixes the most oversized clusters with the least
collateral damage (lost correct-sized matches).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from ..entitymatching.post_clustering import (
    GreedyOneToOneMatchingAlgorithm,
    MaximumBipartiteMatching,
    StableMatching,
)
from ..utils.cluster_stats import cluster_size_distribution_from_sizes

logger = logging.getLogger(__name__)

# Strategies to evaluate, in order.
_STRATEGIES: dict[str, type] = {
    "MaximumBipartiteMatching": MaximumBipartiteMatching,
    "GreedyOneToOne": GreedyOneToOneMatchingAlgorithm,
    "StableMatching": StableMatching,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_global_cluster_metrics(
    corr_list: list[pd.DataFrame],
    num_datasets: int,
) -> dict[str, Any]:
    """Compute cluster-level metrics from a list of per-pair correspondences."""
    if not corr_list:
        return {
            "total_correspondences": 0,
            "total_clusters": 0,
            "oversized_clusters": 0,
            "target_range_clusters": 0,
            "max_cluster_size": 0,
            "distribution": pd.DataFrame(
                columns=["cluster_size", "frequency", "percentage"]
            ),
        }

    all_corr = pd.concat(corr_list, ignore_index=True)
    all_corr = all_corr.drop_duplicates(subset=["id1", "id2"]).reset_index(drop=True)

    if all_corr.empty:
        return _compute_global_cluster_metrics([], num_datasets)

    G = nx.Graph()
    for _, row in all_corr.iterrows():
        G.add_edge(row["id1"], row["id2"])

    clusters = list(nx.connected_components(G))
    sizes = [len(c) for c in clusters]
    dist_df = cluster_size_distribution_from_sizes(sizes)

    oversized = int(
        dist_df.loc[dist_df["cluster_size"] > num_datasets, "frequency"].sum()
    )
    target = int(
        dist_df.loc[
            (dist_df["cluster_size"] >= 2)
            & (dist_df["cluster_size"] <= num_datasets),
            "frequency",
        ].sum()
    )

    return {
        "total_correspondences": len(all_corr),
        "total_clusters": len(clusters),
        "oversized_clusters": oversized,
        "target_range_clusters": target,
        "max_cluster_size": max(sizes) if sizes else 0,
        "distribution": dist_df,
    }


def _apply_strategy(
    per_pair_correspondences: list[pd.DataFrame],
    strategy_name: str,
) -> list[pd.DataFrame]:
    """Apply a post-clustering strategy to each pair independently."""
    cls = _STRATEGIES[strategy_name]
    clusterer = cls()
    cleaned: list[pd.DataFrame] = []

    for pair_df in per_pair_correspondences:
        if pair_df.empty:
            cleaned.append(pair_df)
            continue
        result = clusterer.cluster(pair_df)
        cleaned.append(result)

    return cleaned


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def clean_oversized_clusters(
    per_pair_correspondences: list[pd.DataFrame],
    num_datasets: int,
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    """Clean oversized clusters by evaluating post-clustering strategies.

    Each strategy is applied **per pair** to enforce one-to-one matching.
    The strategy that removes the most oversized clusters with the least
    collateral damage is selected automatically.

    Parameters
    ----------
    per_pair_correspondences : list[pd.DataFrame]
        Per-pair correspondence DataFrames (id1, id2, score).
    num_datasets : int
        Number of input datasets (expected max cluster size).

    Returns
    -------
    tuple[list[pd.DataFrame], dict]
        ``(cleaned_correspondences, report)`` where *report* contains
        before/after metrics and per-strategy comparison.
    """
    before = _compute_global_cluster_metrics(per_pair_correspondences, num_datasets)

    if before["oversized_clusters"] == 0:
        logger.info("No oversized clusters found — skipping cleaning.")
        return per_pair_correspondences, {
            "before": before,
            "after": before,
            "best_strategy": None,
            "strategies": [],
        }

    print(
        f"\n--- Cluster Cleaning ---\n"
        f"  {before['oversized_clusters']} oversized clusters detected "
        f"(max size {before['max_cluster_size']}). "
        f"Evaluating {len(_STRATEGIES)} strategies..."
    )

    strategy_results: list[dict[str, Any]] = []

    for name in _STRATEGIES:
        cleaned = _apply_strategy(per_pair_correspondences, name)
        metrics = _compute_global_cluster_metrics(cleaned, num_datasets)

        oversized_fixed = before["oversized_clusters"] - metrics["oversized_clusters"]
        collateral = before["target_range_clusters"] - metrics["target_range_clusters"]

        entry = {
            "strategy": name,
            "correspondences_after": metrics["total_correspondences"],
            "clusters_after": metrics["total_clusters"],
            "oversized_after": metrics["oversized_clusters"],
            "target_range_after": metrics["target_range_clusters"],
            "max_cluster_size_after": metrics["max_cluster_size"],
            "oversized_fixed": oversized_fixed,
            "collateral_damage": collateral,
            "_cleaned": cleaned,
            "_metrics": metrics,
        }
        strategy_results.append(entry)

        print(
            f"  {name:35s} | "
            f"oversized fixed: {oversized_fixed:>5d} | "
            f"collateral: {collateral:>5d} | "
            f"max size: {metrics['max_cluster_size']:>3d}"
        )

    # Pick best: most oversized fixed, then least collateral damage.
    strategy_results.sort(key=lambda r: (-r["oversized_fixed"], r["collateral_damage"]))
    best = strategy_results[0]

    print(f"  → Selected: {best['strategy']}")

    report = {
        "before": before,
        "after": best["_metrics"],
        "best_strategy": best["strategy"],
        "strategies": [
            {k: v for k, v in s.items() if not k.startswith("_")}
            for s in strategy_results
        ],
    }

    return best["_cleaned"], report


def save_cluster_cleaning_report(
    report: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Save cluster cleaning artefacts (CSVs + JSON summary).

    Writes to ``reporting/entity_resolution/``:
    - ``cluster_size_distribution_before_cleaning.csv``
    - ``cluster_size_distribution_after_cleaning.csv``
    - ``cluster_cleaning_strategies.csv``
    - ``cluster_cleaning_summary.json``

    Returns the report directory path.
    """
    report_dir = Path(output_dir) / "reporting" / "entity_resolution"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Before / after distributions
    before_dist: pd.DataFrame = report["before"]["distribution"]
    after_dist: pd.DataFrame = report["after"]["distribution"]

    before_dist.to_csv(
        report_dir / "cluster_size_distribution_before_cleaning.csv", index=False
    )
    after_dist.to_csv(
        report_dir / "cluster_size_distribution_after_cleaning.csv", index=False
    )

    # Strategy comparison table
    strat_df = pd.DataFrame(report["strategies"])
    strat_df.to_csv(report_dir / "cluster_cleaning_strategies.csv", index=False)

    # JSON summary (no DataFrames)
    summary = {
        "best_strategy": report["best_strategy"],
        "before": {
            k: v for k, v in report["before"].items() if k != "distribution"
        },
        "after": {
            k: v for k, v in report["after"].items() if k != "distribution"
        },
        "strategies": report["strategies"],
    }
    with open(report_dir / "cluster_cleaning_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return report_dir


def save_cluster_cleaning_latex(
    report: dict[str, Any],
    num_datasets: int,
    output_dir: Path,
) -> Path:
    """Generate a LaTeX table summarising cluster cleaning results.

    Writes ``reporting/entity_resolution/cluster_cleaning.tex``.
    """
    report_dir = Path(output_dir) / "reporting" / "entity_resolution"
    report_dir.mkdir(parents=True, exist_ok=True)

    before = report["before"]
    strategies = report["strategies"]

    lines = [
        r"% Cluster cleaning strategy comparison (auto-generated)",
        r"\begin{table}[t]",
        r"\caption{Cluster cleaning strategies applied per dataset pair. "
        r'"Oversized fixed" counts clusters reduced from $>' + str(num_datasets)
        + r"$ to $\leq " + str(num_datasets) + r"$ entities.}",
        r"\label{tab:cluster_cleaning}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"\textbf{Strategy} & \textbf{Corr.} & \textbf{Clusters} "
        r"& \textbf{Oversized} & \textbf{Fixed} & \textbf{Collateral} \\",
        r"\midrule",
        # Baseline row
        f"Baseline (none) & {before['total_correspondences']} "
        f"& {before['total_clusters']} & {before['oversized_clusters']} "
        r"& --- & --- \\",
        r"\midrule",
    ]

    for s in strategies:
        name_tex = s["strategy"].replace("_", r"\_")
        marker = r" $\star$" if s["strategy"] == report["best_strategy"] else ""
        lines.append(
            f"{name_tex}{marker} & {s['correspondences_after']} "
            f"& {s['clusters_after']} & {s['oversized_after']} "
            f"& {s['oversized_fixed']} & {s['collateral_damage']} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]

    tex_path = report_dir / "cluster_cleaning.tex"
    tex_path.write_text("\n".join(lines))
    return tex_path
