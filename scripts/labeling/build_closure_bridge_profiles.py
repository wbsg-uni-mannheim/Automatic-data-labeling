#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_INPUT_ROOT = ROOT / "output" / "three_phase_labeling_ditto_only_v2"
DEFAULT_RELABELED_ROOT = ROOT / "output" / "three_phase_labeling_ditto_only_v2_relabel_batch_gpt-5-mini_agent_precision"
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "three_phase_labeling_ditto_only_v2_closure_bridge_drop"
PAIR_KEY_COLUMNS = ("id1", "id2", "rid1", "rid2")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drop positive bridge edges from three-phase generated label profiles."
    )
    parser.add_argument(
        "--input-root",
        default=str(DEFAULT_INPUT_ROOT),
        help="Root with source three_phase_labeling_ditto_only_v2 runs. Default: %(default)s",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root for generated closure-bridge-drop runs. Default: %(default)s",
    )
    parser.add_argument(
        "--relabeled-root",
        default=str(DEFAULT_RELABELED_ROOT),
        help=(
            "Root with relabeled runs. Used with --require-relabel-changed or "
            "--include-relabel-changed. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--require-relabel-changed",
        action="store_true",
        help="Only drop positive bridge edges whose label also changed in the relabeled run.",
    )
    parser.add_argument(
        "--include-relabel-changed",
        action="store_true",
        help=(
            "Drop positive bridge edges OR pairs whose label changed in the relabeled run. "
            "This is the union/OR variant."
        ),
    )
    parser.add_argument(
        "--datasets",
        default="",
        help="Comma-separated dataset directories to process. Default: all runs with profile_manifest.json.",
    )
    parser.add_argument(
        "--profiles",
        default="",
        help="Comma-separated profiles to materialize. Default: all profiles in the manifest.",
    )
    parser.add_argument(
        "--master-profile",
        default="",
        help="Profile used to detect bridges. Default: all_plus20random if present, else all.",
    )
    parser.add_argument(
        "--min-component-nodes",
        type=int,
        default=3,
        help=(
            "Only drop bridge edges from positive components with at least this many nodes. "
            "Default 3 keeps isolated 1:1 positive pairs."
        ),
    )
    parser.add_argument(
        "--no-export-ditto",
        action="store_true",
        help="Write CSV profiles only and skip filtered *train.json.gz exports.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_jsonl_gz(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl_gz(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _to_repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _dataset_dirs(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Input root does not exist: {root}")
    return sorted(path for path in root.iterdir() if path.is_dir() and (path / "profile_manifest.json").exists())


def _select_dataset_dirs(root: Path, requested: Sequence[str]) -> List[Path]:
    available = {path.name: path for path in _dataset_dirs(root)}
    if not requested:
        return [available[name] for name in sorted(available)]
    missing = [name for name in requested if name not in available]
    if missing:
        raise FileNotFoundError(f"Dataset directories not found under {root}: {missing}")
    return [available[name] for name in requested]


def _select_profiles(manifest: Dict[str, Any], requested: Sequence[str]) -> List[str]:
    available = list((manifest.get("profiles") or {}).keys())
    if not requested:
        return available
    missing = [name for name in requested if name not in available]
    if missing:
        raise KeyError(f"Requested profiles not in manifest: {missing}. Available: {available}")
    return [name for name in available if name in requested]


def _source_profile_dir(dataset_dir: Path, profile: str) -> Path:
    candidates = [
        dataset_dir / profile,
        dataset_dir / "profiles" / profile,
    ]
    for candidate in candidates:
        if (candidate / "active_labels_latest.csv").exists():
            return candidate
    return candidates[0]


def _choose_master_profile(dataset_dir: Path, manifest: Dict[str, Any], explicit: str) -> str:
    profiles = set((manifest.get("profiles") or {}).keys())
    candidates = [explicit] if explicit else ["all_plus20random", "all"]
    for profile in candidates:
        if profile and profile in profiles and (_source_profile_dir(dataset_dir, profile) / "active_labels_latest.csv").exists():
            return profile
    raise FileNotFoundError(
        f"Could not resolve master profile for {dataset_dir}. Tried: {candidates}; available: {sorted(profiles)}"
    )


def _normalize_label(value: Any) -> int:
    text = str(value).strip().upper()
    if text in {"TRUE", "1", "YES", "Y", "T"}:
        return 1
    if text in {"FALSE", "0", "NO", "N", "F"}:
        return 0
    raise ValueError(f"Unsupported label value: {value!r}")


def _pair_key(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return tuple(str(row.get(col, "")) for col in PAIR_KEY_COLUMNS)


def _node_ids(row: Dict[str, Any]) -> Tuple[str, str]:
    rid1 = str(row.get("rid1", "")).strip()
    rid2 = str(row.get("rid2", "")).strip()
    if rid1 and rid2:
        left = rid1 if rid1.startswith("L:") else f"L:{rid1}"
        right = rid2 if rid2.startswith("R:") else f"R:{rid2}"
        return left, right
    return f"L:{row.get('id1', '')}", f"R:{row.get('id2', '')}"


def _positive_edges(df: pd.DataFrame) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []
    for idx, row in enumerate(df.to_dict("records")):
        if _normalize_label(row.get("label")) != 1:
            continue
        left, right = _node_ids(row)
        edges.append(
            {
                "edge_id": len(edges),
                "row_index": idx,
                "left_node": left,
                "right_node": right,
                "pair_key": _pair_key(row),
                "row": row,
            }
        )
    return edges


def _component_sizes(edges: List[Dict[str, Any]]) -> Dict[str, int]:
    parent: Dict[str, str] = {}
    size: Dict[str, int] = {}

    def find(x: str) -> str:
        if x not in parent:
            parent[x] = x
            size[x] = 1
            return x
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    for edge in edges:
        union(edge["left_node"], edge["right_node"])

    component_size_by_node: Dict[str, int] = {}
    for node in list(parent):
        root = find(node)
        component_size_by_node[node] = size[root]
    return component_size_by_node


def _bridge_edge_ids(edges: List[Dict[str, Any]]) -> set[int]:
    adjacency: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for edge in edges:
        edge_id = int(edge["edge_id"])
        left = str(edge["left_node"])
        right = str(edge["right_node"])
        adjacency[left].append((right, edge_id))
        adjacency[right].append((left, edge_id))

    tin: Dict[str, int] = {}
    low: Dict[str, int] = {}
    bridges: set[int] = set()
    timer = 0

    def dfs(node: str, parent_edge: int | None) -> None:
        nonlocal timer
        tin[node] = low[node] = timer
        timer += 1
        for nxt, edge_id in adjacency[node]:
            if edge_id == parent_edge:
                continue
            if nxt in tin:
                low[node] = min(low[node], tin[nxt])
                continue
            dfs(nxt, edge_id)
            low[node] = min(low[node], low[nxt])
            if low[nxt] > tin[node]:
                bridges.add(edge_id)

    for node in sorted(adjacency):
        if node not in tin:
            dfs(node, None)

    return bridges


def _detect_positive_bridge_edges(
    master_df: pd.DataFrame,
    *,
    min_component_nodes: int,
) -> Tuple[pd.DataFrame, set[Tuple[str, str, str, str]]]:
    edges = _positive_edges(master_df)
    component_size_by_node = _component_sizes(edges)
    bridge_ids = _bridge_edge_ids(edges)

    rows: List[Dict[str, Any]] = []
    drop_keys: set[Tuple[str, str, str, str]] = set()
    for edge in edges:
        component_nodes = int(component_size_by_node.get(str(edge["left_node"]), 1))
        if int(edge["edge_id"]) not in bridge_ids:
            continue
        if component_nodes < int(min_component_nodes):
            continue
        row = dict(edge["row"])
        row.update(
            {
                "edge_id": int(edge["edge_id"]),
                "row_index": int(edge["row_index"]),
                "left_node": edge["left_node"],
                "right_node": edge["right_node"],
                "component_nodes": component_nodes,
                "closure_filter_reason": "positive_bridge_edge",
            }
        )
        rows.append(row)
        drop_keys.add(edge["pair_key"])

    return pd.DataFrame(rows), drop_keys


def _changed_label_keys(
    original_df: pd.DataFrame,
    relabeled_df: pd.DataFrame,
) -> set[Tuple[str, str, str, str]]:
    missing = [
        col
        for col in PAIR_KEY_COLUMNS
        if col not in original_df.columns or col not in relabeled_df.columns
    ]
    if missing:
        raise ValueError(f"Cannot compare relabeled labels; missing key columns: {missing}")

    merged = (
        original_df[list(PAIR_KEY_COLUMNS) + ["label"]]
        .rename(columns={"label": "label_original"})
        .merge(
            relabeled_df[list(PAIR_KEY_COLUMNS) + ["label"]].rename(columns={"label": "label_relabeled"}),
            on=list(PAIR_KEY_COLUMNS),
            how="inner",
        )
    )
    changed: set[Tuple[str, str, str, str]] = set()
    for row in merged.to_dict("records"):
        if _normalize_label(row["label_original"]) != _normalize_label(row["label_relabeled"]):
            changed.add(_pair_key(row))
    return changed


def _pair_keys_frame(
    pair_keys: set[Tuple[str, str, str, str]],
    *,
    bridge_keys: set[Tuple[str, str, str, str]],
    relabel_changed_keys: set[Tuple[str, str, str, str]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for key in sorted(pair_keys):
        row = dict(zip(PAIR_KEY_COLUMNS, key))
        is_bridge = key in bridge_keys
        is_relabel_changed = key in relabel_changed_keys
        if is_bridge and is_relabel_changed:
            reason = "positive_bridge_edge_and_relabel_changed"
        elif is_bridge:
            reason = "positive_bridge_edge"
        else:
            reason = "relabel_changed"
        row.update(
            {
                "positive_bridge_edge": is_bridge,
                "relabel_changed": is_relabel_changed,
                "closure_filter_reason": reason,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _profile_train_json_gz(profile_dir: Path) -> Path | None:
    matches = sorted(profile_dir.glob("*train.json.gz"))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Expected one *train.json.gz in {profile_dir}, found {len(matches)}")
    return matches[0]


def _load_profile(profile_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]], str | None]:
    active_path = profile_dir / "active_labels_latest.csv"
    final_path = profile_dir / "labels_final.csv"
    if not active_path.exists():
        raise FileNotFoundError(f"Missing {active_path}")
    active = pd.read_csv(active_path).reset_index(drop=True)
    final = pd.read_csv(final_path).reset_index(drop=True) if final_path.exists() else active.copy()
    train_path = _profile_train_json_gz(profile_dir)
    train_rows: List[Dict[str, Any]] = []
    train_name: str | None = None
    if train_path:
        train_rows = _read_jsonl_gz(train_path)
        train_name = train_path.name
    return active, final, train_rows, train_name


def _filter_profile(
    active: pd.DataFrame,
    final: pd.DataFrame,
    train_rows: List[Dict[str, Any]],
    drop_keys: set[Tuple[str, str, str, str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]], Dict[str, int]]:
    if len(active) != len(final):
        raise ValueError("active_labels_latest.csv and labels_final.csv row counts differ")
    if train_rows and len(train_rows) != len(active):
        raise ValueError("train json.gz row count does not match active_labels_latest.csv")

    keep_mask: List[bool] = []
    dropped_pos = 0
    dropped_neg = 0
    for row in active.to_dict("records"):
        should_drop = _pair_key(row) in drop_keys
        keep_mask.append(not should_drop)
        if should_drop:
            if _normalize_label(row.get("label")) == 1:
                dropped_pos += 1
            else:
                dropped_neg += 1

    keep = pd.Series(keep_mask, index=active.index, dtype="bool")
    filtered_active = active.loc[keep].reset_index(drop=True)
    filtered_final = final.loc[keep].reset_index(drop=True)
    filtered_train = [row for row, keep_value in zip(train_rows, keep_mask) if keep_value] if train_rows else []

    return (
        filtered_active,
        filtered_final,
        filtered_train,
        {
            "rows_before": int(len(active)),
            "rows_after": int(len(filtered_active)),
            "dropped_rows": int(len(active) - len(filtered_active)),
            "dropped_positive_rows": int(dropped_pos),
            "dropped_negative_rows": int(dropped_neg),
        },
    )


def _count_labels(df: pd.DataFrame) -> Tuple[int, int, int]:
    labels = df["label"].map(_normalize_label)
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    return int(len(df)), pos, neg


def main() -> None:
    args = _parse_args()
    if args.require_relabel_changed and args.include_relabel_changed:
        raise ValueError("Use either --require-relabel-changed (AND) or --include-relabel-changed (OR), not both.")

    if args.require_relabel_changed:
        relabel_change_mode = "and"
        drop_rule = "positive_bridge_edge_and_relabel_changed"
    elif args.include_relabel_changed:
        relabel_change_mode = "or"
        drop_rule = "positive_bridge_edge_or_relabel_changed"
    else:
        relabel_change_mode = "bridge_only"
        drop_rule = "positive_bridge_edge"

    input_root = Path(args.input_root).expanduser().resolve()
    relabeled_root = Path(args.relabeled_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    requested_datasets = [item.strip() for item in str(args.datasets).split(",") if item.strip()]
    requested_profiles = [item.strip() for item in str(args.profiles).split(",") if item.strip()]

    output_root.mkdir(parents=True, exist_ok=True)
    run_summaries: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for dataset_dir in _select_dataset_dirs(input_root, requested_datasets):
        manifest = _load_json(dataset_dir / "profile_manifest.json")
        benchmark = str(manifest.get("benchmark") or dataset_dir.name)
        profiles = _select_profiles(manifest, requested_profiles)
        master_profile = _choose_master_profile(dataset_dir, manifest, str(args.master_profile).strip())
        master_profile_dir = _source_profile_dir(dataset_dir, master_profile)
        master_df = pd.read_csv(master_profile_dir / "active_labels_latest.csv").reset_index(drop=True)
        bridge_df, bridge_keys = _detect_positive_bridge_edges(
            master_df,
            min_component_nodes=int(args.min_component_nodes),
        )
        candidate_bridge_count = int(len(bridge_keys))
        candidate_bridge_keys = set(bridge_keys)
        drop_keys = set(candidate_bridge_keys)
        relabel_changed_keys: set[Tuple[str, str, str, str]] = set()
        relabel_changed_count = None
        if args.require_relabel_changed or args.include_relabel_changed:
            relabeled_dataset_dir = relabeled_root / dataset_dir.name
            if not relabeled_dataset_dir.exists():
                raise FileNotFoundError(f"Relabeled dataset directory not found: {relabeled_dataset_dir}")
            relabeled_manifest = _load_json(relabeled_dataset_dir / "profile_manifest.json")
            relabeled_master_profile = _choose_master_profile(
                relabeled_dataset_dir,
                relabeled_manifest,
                master_profile,
            )
            relabeled_master_profile_dir = _source_profile_dir(relabeled_dataset_dir, relabeled_master_profile)
            relabeled_master_df = pd.read_csv(
                relabeled_master_profile_dir / "active_labels_latest.csv"
            ).reset_index(drop=True)
            relabel_changed_keys = _changed_label_keys(master_df, relabeled_master_df)
            relabel_changed_count = int(len(relabel_changed_keys))
            if args.require_relabel_changed:
                drop_keys = candidate_bridge_keys & relabel_changed_keys
            else:
                drop_keys = candidate_bridge_keys | relabel_changed_keys
            if not bridge_df.empty:
                bridge_df["relabel_changed"] = bridge_df.apply(
                    lambda row: _pair_key(row.to_dict()) in relabel_changed_keys,
                    axis=1,
                )
                if args.require_relabel_changed:
                    bridge_df = bridge_df[bridge_df["relabel_changed"]].reset_index(drop=True)

        out_dataset_dir = output_root / dataset_dir.name
        out_dataset_dir.mkdir(parents=True, exist_ok=True)
        bridge_df.to_csv(out_dataset_dir / "positive_bridge_edges.csv", index=False)
        dropped_pairs_df = _pair_keys_frame(
            drop_keys,
            bridge_keys=candidate_bridge_keys,
            relabel_changed_keys=relabel_changed_keys,
        )
        dropped_pairs_df.to_csv(out_dataset_dir / "dropped_pairs.csv", index=False)

        updated_manifest = deepcopy(manifest)
        updated_manifest["run_dir"] = _to_repo_relative(out_dataset_dir)
        updated_manifest["runner_script"] = "scripts/labeling/build_closure_bridge_profiles.py"
        updated_manifest["run_summary_json"] = _to_repo_relative(out_dataset_dir / "closure_bridge_summary.json")
        updated_manifest["source_profile_manifest"] = _to_repo_relative(dataset_dir / "profile_manifest.json")
        updated_manifest["source_train_file"] = _to_repo_relative(out_dataset_dir / "labels_final.csv")
        updated_manifest["source_all_examples_file"] = _to_repo_relative(out_dataset_dir / "active_labels_latest.csv")
        updated_manifest["closure_bridge_filter"] = {
            "applied": True,
            "master_profile": master_profile,
            "min_component_nodes": int(args.min_component_nodes),
            "require_relabel_changed": bool(args.require_relabel_changed),
            "include_relabel_changed": bool(args.include_relabel_changed),
            "relabel_change_mode": relabel_change_mode,
            "drop_rule": drop_rule,
            "candidate_positive_bridge_edges": candidate_bridge_count,
            "relabel_changed_pairs": relabel_changed_count,
            "pairs_dropped_from_master": int(len(drop_keys)),
            "positive_bridge_edges": candidate_bridge_count,
            "positive_bridge_edges_dropped": int(len(candidate_bridge_keys & drop_keys)),
            "positive_bridge_edges_csv": _to_repo_relative(out_dataset_dir / "positive_bridge_edges.csv"),
            "dropped_pairs_csv": _to_repo_relative(out_dataset_dir / "dropped_pairs.csv"),
        }

        profile_summaries: Dict[str, Any] = {}
        new_profiles: Dict[str, Any] = {}
        root_active_written = False
        for profile_name in profiles:
            profile_dir = _source_profile_dir(dataset_dir, profile_name)
            active, final, train_rows, train_name = _load_profile(profile_dir)
            filtered_active, filtered_final, filtered_train, summary = _filter_profile(
                active,
                final,
                train_rows,
                drop_keys,
            )

            out_profile_dir = out_dataset_dir / profile_name
            out_profile_dir.mkdir(parents=True, exist_ok=True)
            filtered_active.to_csv(out_profile_dir / "active_labels_latest.csv", index=False)
            filtered_final.to_csv(out_profile_dir / "labels_final.csv", index=False)
            if train_name and not args.no_export_ditto:
                _write_jsonl_gz(out_profile_dir / train_name, filtered_train)

            if profile_name == master_profile:
                filtered_active.to_csv(out_dataset_dir / "active_labels_latest.csv", index=False)
                filtered_final.to_csv(out_dataset_dir / "labels_final.csv", index=False)
                root_active_written = True

            total, pos, neg = _count_labels(filtered_active)
            profile_meta = deepcopy((manifest.get("profiles") or {}).get(profile_name) or {})
            profile_meta["actual_total"] = total
            profile_meta["actual_pos"] = pos
            profile_meta["actual_neg"] = neg
            profile_meta["labels_csv"] = _to_repo_relative(out_profile_dir / "active_labels_latest.csv")
            if train_name and not args.no_export_ditto:
                profile_meta["ditto_train_json_gz"] = _to_repo_relative(out_profile_dir / train_name)
            new_profiles[profile_name] = profile_meta

            profile_summaries[profile_name] = {
                **summary,
                "active_labels_latest": _to_repo_relative(out_profile_dir / "active_labels_latest.csv"),
                "labels_final": _to_repo_relative(out_profile_dir / "labels_final.csv"),
                "train_json_gz": _to_repo_relative(out_profile_dir / train_name) if train_name and not args.no_export_ditto else "",
            }
            summary_rows.append(
                {
                    "benchmark": benchmark,
                    "dataset_dir": dataset_dir.name,
                    "profile": profile_name,
                    "master_profile": master_profile,
                    "require_relabel_changed": bool(args.require_relabel_changed),
                    "include_relabel_changed": bool(args.include_relabel_changed),
                    "relabel_change_mode": relabel_change_mode,
                    "drop_rule": drop_rule,
                    "candidate_positive_bridge_edges_in_master": candidate_bridge_count,
                    "relabel_changed_pairs_in_master": relabel_changed_count,
                    "pairs_dropped_from_master": int(len(drop_keys)),
                    "positive_bridge_edges_in_master": candidate_bridge_count,
                    "positive_bridge_edges_dropped_from_master": int(len(candidate_bridge_keys & drop_keys)),
                    "rows_before": int(summary["rows_before"]),
                    "rows_after": int(summary["rows_after"]),
                    "removed_pairs": int(summary["dropped_rows"]),
                    "removed_positive_pairs": int(summary["dropped_positive_rows"]),
                    "removed_negative_pairs": int(summary["dropped_negative_rows"]),
                    "output_profile_dir": _to_repo_relative(out_profile_dir),
                }
            )

        if not root_active_written and profiles:
            first_profile = profiles[0]
            source_dir = out_dataset_dir / first_profile
            (out_dataset_dir / "active_labels_latest.csv").write_bytes((source_dir / "active_labels_latest.csv").read_bytes())
            (out_dataset_dir / "labels_final.csv").write_bytes((source_dir / "labels_final.csv").read_bytes())

        updated_manifest["profiles"] = new_profiles
        _write_json(out_dataset_dir / "profile_manifest.json", updated_manifest)

        summary_payload = {
            "benchmark": benchmark,
            "source_dataset_dir": _to_repo_relative(dataset_dir),
            "output_dataset_dir": _to_repo_relative(out_dataset_dir),
            "master_profile": master_profile,
            "min_component_nodes": int(args.min_component_nodes),
            "require_relabel_changed": bool(args.require_relabel_changed),
            "include_relabel_changed": bool(args.include_relabel_changed),
            "relabel_change_mode": relabel_change_mode,
            "drop_rule": drop_rule,
            "candidate_positive_bridge_edges": candidate_bridge_count,
            "relabel_changed_pairs": relabel_changed_count,
            "pairs_dropped_from_master": int(len(drop_keys)),
            "positive_bridge_edges": candidate_bridge_count,
            "positive_bridge_edges_dropped": int(len(candidate_bridge_keys & drop_keys)),
            "profiles": profile_summaries,
            "export_ditto": not args.no_export_ditto,
        }
        _write_json(out_dataset_dir / "closure_bridge_summary.json", summary_payload)
        run_summaries.append(summary_payload)
        removed_by_profile = ", ".join(
            f"{profile}={profile_summaries[profile]['dropped_rows']}"
            for profile in profiles
            if profile in profile_summaries
        )
        print(
            f"[{benchmark}] candidate_bridge_edges={candidate_bridge_count} "
            f"relabel_changed_pairs={relabel_changed_count if relabel_changed_count is not None else 'n/a'} "
            f"drop_rule={drop_rule} "
            f"pairs_dropped_from_master={len(drop_keys)} "
            f"removed_pairs_by_profile: {removed_by_profile} "
            f"output={out_dataset_dir}"
        )

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(output_root / "closure_bridge_profile_summary.csv", index=False)
    _write_json(
        output_root / "closure_bridge_run_summary.json",
        {
            "input_root": _to_repo_relative(input_root),
            "relabeled_root": (
                _to_repo_relative(relabeled_root)
                if args.require_relabel_changed or args.include_relabel_changed
                else ""
            ),
            "output_root": _to_repo_relative(output_root),
            "min_component_nodes": int(args.min_component_nodes),
            "require_relabel_changed": bool(args.require_relabel_changed),
            "include_relabel_changed": bool(args.include_relabel_changed),
            "relabel_change_mode": relabel_change_mode,
            "drop_rule": drop_rule,
            "datasets": run_summaries,
            "profile_summary_csv": _to_repo_relative(output_root / "closure_bridge_profile_summary.csv"),
            "export_ditto": not args.no_export_ditto,
        },
    )
    print(output_root)


if __name__ == "__main__":
    main()
