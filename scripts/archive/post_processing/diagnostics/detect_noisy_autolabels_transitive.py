#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "labeling" / "benchmarks_active.yaml"
AUTO_LABEL_V1_ROOT = ROOT / "data" / "auto_label_v1"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "noisy_autolabels_transitive"

POS_LABELS = {"1", "TRUE", "T", "YES", "Y"}
NEG_LABELS = {"0", "FALSE", "F", "NO", "N"}


@dataclass(frozen=True)
class GeneratedSet:
    benchmark: str
    profile: str
    labels_csv: Path


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}
        self.size: Dict[str, int] = {}

    def add(self, item: str) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.size[item] = 1

    def find(self, item: str) -> str:
        root = self.parent[item]
        while root != self.parent[root]:
            root = self.parent[root]
        while item != root:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, a: str, b: str) -> None:
        self.add(a)
        self.add(b)
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def component_size(self, item: str) -> int:
        return self.size[self.find(item)]


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return payload


def _normalize_label(value: Any) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        iv = int(value)
        if iv in {0, 1}:
            return iv
    text = str(value).strip().upper()
    if text in POS_LABELS:
        return 1
    if text in NEG_LABELS:
        return 0
    return None


def _normalize_pair_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {}
    if "id1" in out.columns and "id_left" not in out.columns:
        rename_map["id1"] = "id_left"
    if "id2" in out.columns and "id_right" not in out.columns:
        rename_map["id2"] = "id_right"
    if rename_map:
        out = out.rename(columns=rename_map)
    if "pair_id" not in out.columns and {"id_left", "id_right"}.issubset(out.columns):
        out["pair_id"] = out["id_left"].astype(str).str.strip() + "#" + out["id_right"].astype(str).str.strip()
    elif "pair_id" in out.columns and not {"id_left", "id_right"}.issubset(out.columns):
        pair_parts = out["pair_id"].astype(str).str.split("#", n=1, expand=True)
        if pair_parts.shape[1] == 2:
            out["id_left"] = pair_parts[0]
            out["id_right"] = pair_parts[1]
    out["label"] = out["label"].map(_normalize_label)
    out["id_left"] = out["id_left"].astype(str).str.strip()
    out["id_right"] = out["id_right"].astype(str).str.strip()
    out["pair_id"] = out["pair_id"].astype(str).str.strip()
    out["pair_key"] = out["id_left"] + "#" + out["id_right"]
    return out.reset_index(drop=True)


def _discover_generated_sets(config: Dict[str, Any]) -> List[GeneratedSet]:
    out: List[GeneratedSet] = []
    config_benchmarks = config.get("benchmarks") or {}
    for manifest_path in sorted(AUTO_LABEL_V1_ROOT.glob("*/profile_manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        benchmark = str(manifest.get("benchmark", "")).strip()
        if benchmark not in config_benchmarks:
            continue
        profiles = manifest.get("profiles") or {}
        if not isinstance(profiles, dict):
            continue
        run_dir = manifest_path.parent
        for profile_name in sorted(profiles.keys()):
            labels_csv = run_dir / "profiles" / profile_name / "labels_final.csv"
            if labels_csv.exists():
                out.append(GeneratedSet(benchmark=benchmark, profile=profile_name, labels_csv=labels_csv))
    return out


def _prepare_df(df: pd.DataFrame, benchmark_cfg: Dict[str, Any]) -> pd.DataFrame:
    left_source = pd.read_csv(ROOT / benchmark_cfg["left_csv"])
    right_source = pd.read_csv(ROOT / benchmark_cfg["right_csv"])
    left_source["id"] = left_source["id"].astype(str).str.strip()
    right_source["id"] = right_source["id"].astype(str).str.strip()
    left_clusters = left_source[["id", "cluster_id"]].rename(columns={"id": "id_left", "cluster_id": "cluster_id_left"})
    right_clusters = right_source[["id", "cluster_id"]].rename(columns={"id": "id_right", "cluster_id": "cluster_id_right"})
    out = _normalize_pair_frame(df)
    out = out.merge(left_clusters, on="id_left", how="left").merge(right_clusters, on="id_right", how="left")
    out["cluster_known"] = out["cluster_id_left"].notna() & out["cluster_id_right"].notna()
    out["cluster_match"] = False
    known = out["cluster_known"]
    out.loc[known, "cluster_match"] = (
        out.loc[known, "cluster_id_left"].astype(str).str.strip() == out.loc[known, "cluster_id_right"].astype(str).str.strip()
    )
    out["cluster_proxy_fp"] = ((out["label"] == 1) & (~out["cluster_match"]) & out["cluster_known"])
    out["cluster_proxy_fn"] = ((out["label"] == 0) & (out["cluster_match"]) & out["cluster_known"])
    out["cluster_proxy_error"] = out["cluster_proxy_fp"] | out["cluster_proxy_fn"]
    return out


def _node_left(left_id: str) -> str:
    return f"L::{left_id}"


def _node_right(right_id: str) -> str:
    return f"R::{right_id}"


def _build_components(df: pd.DataFrame) -> Tuple[UnionFind, Dict[str, set[str]], Dict[str, set[str]]]:
    uf = UnionFind()
    positive = df[df["label"] == 1]
    for row in positive.itertuples(index=False):
        left_node = _node_left(row.id_left)
        right_node = _node_right(row.id_right)
        uf.union(left_node, right_node)

    component_lefts: Dict[str, set[str]] = {}
    component_rights: Dict[str, set[str]] = {}
    for row in positive.itertuples(index=False):
        left_node = _node_left(row.id_left)
        right_node = _node_right(row.id_right)
        comp = uf.find(left_node)
        component_lefts.setdefault(comp, set()).add(left_node)
        component_rights.setdefault(comp, set()).add(right_node)
    return uf, component_lefts, component_rights


def _score_transitive(df: pd.DataFrame) -> pd.DataFrame:
    uf, component_lefts, component_rights = _build_components(df)
    negative_rights_by_left: Dict[str, set[str]] = {}
    negative_lefts_by_right: Dict[str, set[str]] = {}

    for row in df[df["label"] == 0].itertuples(index=False):
        left_node = _node_left(row.id_left)
        right_node = _node_right(row.id_right)
        negative_rights_by_left.setdefault(left_node, set()).add(right_node)
        negative_lefts_by_right.setdefault(right_node, set()).add(left_node)

    scored = df.copy()
    candidate_type: List[str] = []
    suspicion_score: List[float] = []
    closure_component_size: List[int] = []
    contradiction_count: List[int] = []

    for row in scored.itertuples(index=False):
        left_node = _node_left(row.id_left)
        right_node = _node_right(row.id_right)

        left_in_graph = left_node in uf.parent
        right_in_graph = right_node in uf.parent
        same_component = left_in_graph and right_in_graph and uf.find(left_node) == uf.find(right_node)
        comp_left = uf.find(left_node) if left_in_graph else None
        comp_right = uf.find(right_node) if right_in_graph else None

        if row.label == 0 and same_component:
            comp = uf.find(left_node)
            comp_size = len(component_lefts.get(comp, set())) + len(component_rights.get(comp, set()))
            candidate_type.append("transitive_fn")
            contradiction_count.append(1)
            closure_component_size.append(comp_size)
            suspicion_score.append(float(comp_size))
            continue

        if row.label == 1 and left_in_graph and right_in_graph:
            left_comp_lefts = component_lefts.get(comp_left or "", set())
            right_comp_rights = component_rights.get(comp_right or "", set())
            contradictions = 0
            if left_node in negative_rights_by_left:
                contradictions += len(negative_rights_by_left[left_node] & right_comp_rights)
            if right_node in negative_lefts_by_right:
                contradictions += len(negative_lefts_by_right[right_node] & left_comp_lefts)
            if contradictions > 0:
                comp_size = max(len(left_comp_lefts) + len(component_rights.get(comp_left or "", set())), 0)
                candidate_type.append("transitive_fp")
                contradiction_count.append(contradictions)
                closure_component_size.append(comp_size)
                suspicion_score.append(float(contradictions))
                continue

        candidate_type.append("")
        contradiction_count.append(0)
        closure_component_size.append(0)
        suspicion_score.append(0.0)

    scored["candidate_type"] = candidate_type
    scored["closure_component_size"] = closure_component_size
    scored["contradiction_count"] = contradiction_count
    scored["suspicion_score"] = suspicion_score
    scored["is_flagged"] = scored["candidate_type"] != ""
    return scored


def _summary_for_profile(scored: pd.DataFrame, benchmark: str, profile: str, labels_csv: Path) -> Dict[str, Any]:
    flagged = scored[scored["is_flagged"]]
    flagged_fp = flagged[flagged["candidate_type"] == "transitive_fp"]
    flagged_fn = flagged[flagged["candidate_type"] == "transitive_fn"]
    proxy_total = int(scored["cluster_proxy_error"].sum())
    proxy_fp_total = int(scored["cluster_proxy_fp"].sum())
    proxy_fn_total = int(scored["cluster_proxy_fn"].sum())
    found_total = int(flagged["cluster_proxy_error"].sum())
    found_fp = int(flagged_fp["cluster_proxy_fp"].sum())
    found_fn = int(flagged_fn["cluster_proxy_fn"].sum())
    return {
        "benchmark": benchmark,
        "profile": profile,
        "labels_csv": str(labels_csv.relative_to(ROOT)),
        "rows_total": int(len(scored)),
        "rows_flagged": int(len(flagged)),
        "flagged_transitive_fp": int(len(flagged_fp)),
        "flagged_transitive_fn": int(len(flagged_fn)),
        "cluster_proxy_errors": proxy_total,
        "cluster_proxy_fp_total": proxy_fp_total,
        "cluster_proxy_fn_total": proxy_fn_total,
        "proxy_errors_found": found_total,
        "proxy_fp_found": found_fp,
        "proxy_fn_found": found_fn,
        "proxy_precision": float(found_total / len(flagged)) if len(flagged) else None,
        "proxy_recall": float(found_total / proxy_total) if proxy_total else None,
        "proxy_fp_recall": float(found_fp / proxy_fp_total) if proxy_fp_total else None,
        "proxy_fn_recall": float(found_fn / proxy_fn_total) if proxy_fn_total else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect noisy auto-labels via transitive closure contradictions.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    config = _load_yaml(Path(args.config))
    generated_sets = _discover_generated_sets(config)
    if not generated_sets:
        raise RuntimeError(f"No generated labels found under {AUTO_LABEL_V1_ROOT}")

    output_dir = Path(args.output_dir)
    candidates_dir = output_dir / "candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    top_candidates: List[pd.DataFrame] = []

    for generated in generated_sets:
        benchmark_cfg = (config.get("benchmarks") or {}).get(generated.benchmark) or {}
        df = pd.read_csv(generated.labels_csv)
        scored = _score_transitive(_prepare_df(df, benchmark_cfg))
        summary_rows.append(_summary_for_profile(scored, generated.benchmark, generated.profile, generated.labels_csv))

        scored = scored.sort_values(
            ["is_flagged", "suspicion_score", "closure_component_size"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        out_path = candidates_dir / f"{generated.benchmark}_{generated.profile}_transitive_candidates.csv.gz"
        scored.to_csv(out_path, index=False, compression="gzip")
        top_candidates.append(scored[scored["is_flagged"]].head(200).assign(source_file=out_path.name))

    summary_df = pd.DataFrame(summary_rows).sort_values(["benchmark", "profile"]).reset_index(drop=True)
    top_candidates_df = pd.concat(top_candidates, ignore_index=True) if top_candidates else pd.DataFrame()

    summary_json = output_dir / "summary.json"
    summary_xlsx = output_dir / "summary.xlsx"
    summary_json.write_text(json.dumps({"method": "transitive_closure", "summary": summary_rows}, indent=2))
    with pd.ExcelWriter(summary_xlsx) as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        top_candidates_df.to_excel(writer, sheet_name="top_candidates", index=False)

    print(summary_xlsx)
    print(summary_json)
    print(candidates_dir)


if __name__ == "__main__":
    main()
