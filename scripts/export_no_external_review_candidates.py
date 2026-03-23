#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "labeling" / "benchmarks_active.yaml"
SELF_KNN_DIR = ROOT / "reports" / "noisy_autolabels_self_knn" / "candidates"
TRANSITIVE_DIR = ROOT / "reports" / "noisy_autolabels_transitive" / "candidates"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "no_external_review_candidates"


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return payload


def _enrich_with_source_fields(df: pd.DataFrame, benchmark_cfg: Dict[str, Any]) -> pd.DataFrame:
    left_source = pd.read_csv(ROOT / benchmark_cfg["left_csv"])
    right_source = pd.read_csv(ROOT / benchmark_cfg["right_csv"])
    left_source["id"] = left_source["id"].astype(str).str.strip()
    right_source["id"] = right_source["id"].astype(str).str.strip()

    field_map = {str(k): str(v) for k, v in (benchmark_cfg.get("fields") or {}).items()}
    left_cols = ["id"] + [src for src in field_map.values() if src in left_source.columns]
    right_cols = ["id"] + [src for src in field_map.values() if src in right_source.columns]
    left_lookup = left_source[left_cols].drop_duplicates(subset=["id"], keep="first").copy()
    right_lookup = right_source[right_cols].drop_duplicates(subset=["id"], keep="first").copy()
    left_lookup = left_lookup.rename(columns={"id": "id_left"})
    right_lookup = right_lookup.rename(columns={"id": "id_right"})
    for out_name, src_name in field_map.items():
        if src_name in left_lookup.columns:
            left_lookup = left_lookup.rename(columns={src_name: f"{out_name}_left"})
        if src_name in right_lookup.columns:
            right_lookup = right_lookup.rename(columns={src_name: f"{out_name}_right"})

    return df.merge(left_lookup, on="id_left", how="left").merge(right_lookup, on="id_right", how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export high-precision no-external-label review candidates.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--self-knn-rate", type=float, default=0.10)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    config = _load_yaml(Path(args.config))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []

    for self_path in sorted(SELF_KNN_DIR.glob("*_self_knn_candidates.csv.gz")):
        stem = self_path.name.replace("_self_knn_candidates.csv.gz", "")
        trans_path = TRANSITIVE_DIR / f"{stem}_transitive_candidates.csv.gz"
        if not trans_path.exists():
            continue

        benchmark, profile = stem.rsplit("_", 1)
        benchmark_cfg = (config.get("benchmarks") or {}).get(benchmark) or {}

        self_df = pd.read_csv(self_path)
        trans_df = pd.read_csv(trans_path)
        top_n = max(1, int(math.ceil(args.self_knn_rate * len(self_df))))
        self_top = self_df.head(top_n).copy()
        self_top["self_knn_selected"] = True
        trans_flagged = trans_df[trans_df["is_flagged"] == True].copy()
        trans_flagged["transitive_selected"] = True

        merged = self_top.merge(
            trans_flagged[
                [
                    "pair_key",
                    "candidate_type",
                    "closure_component_size",
                    "contradiction_count",
                    "suspicion_score",
                ]
            ].rename(columns={"suspicion_score": "transitive_suspicion_score"}),
            on="pair_key",
            how="inner",
        )
        if merged.empty:
            continue

        merged["benchmark"] = benchmark
        merged["profile"] = profile
        merged["selection_reason"] = (
            "Selected because the pair is in the top self-kNN embedding disagreement slice "
            "and also violates transitive-closure consistency in the generated label graph."
        )
        merged = _enrich_with_source_fields(merged, benchmark_cfg)
        rows.append(merged)

        summary_rows.append(
            {
                "benchmark": benchmark,
                "profile": profile,
                "candidate_pairs": int(len(merged)),
                "cluster_proxy_errors": int(merged["cluster_proxy_error"].sum()),
                "cluster_proxy_precision": float(merged["cluster_proxy_error"].mean()) if len(merged) else None,
            }
        )

    candidates_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows).sort_values(["benchmark", "profile"]).reset_index(drop=True)

    csv_path = output_dir / "review_candidates.csv.gz"
    xlsx_path = output_dir / "review_candidates.xlsx"
    summary_path = output_dir / "summary.xlsx"

    candidates_df.to_csv(csv_path, index=False, compression="gzip")
    with pd.ExcelWriter(xlsx_path) as writer:
        candidates_df.to_excel(writer, sheet_name="candidates", index=False)
    with pd.ExcelWriter(summary_path) as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)

    print(csv_path)
    print(xlsx_path)
    print(summary_path)


if __name__ == "__main__":
    main()
