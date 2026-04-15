#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_ORIGINAL_ROOT = ROOT / "output" / "three_phase_labeling_ditto_only_v2"
DEFAULT_RELABELED_ROOT = ROOT / "output" / "three_phase_labeling_ditto_only_v2_relabel_batch_gpt-5-mini_agent_precision"
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "three_phase_labeling_ditto_only_v2_drop_changed"
KEY_COLUMNS = ("id1", "id2", "rid1", "rid2")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build drop-changed profile exports by removing every pair whose label changed between "
            "the original three-phase run and the relabeled run."
        )
    )
    parser.add_argument(
        "--original-root",
        default=str(DEFAULT_ORIGINAL_ROOT),
        help="Directory with the original three_phase_labeling_ditto_only_v2 runs. Default: %(default)s",
    )
    parser.add_argument(
        "--relabeled-root",
        default=str(DEFAULT_RELABELED_ROOT),
        help="Directory with the relabeled runs. Default: %(default)s",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where the drop_changed runs should be written. Default: %(default)s",
    )
    parser.add_argument(
        "--datasets",
        default="",
        help=(
            "Comma-separated dataset run directory names to process, e.g. "
            "benchmark_abt-buy_20260323_202820. Default: every dataset found under --relabeled-root."
        ),
    )
    parser.add_argument(
        "--profiles",
        default="",
        help="Optional comma-separated profile names to keep. Default: all profiles from the relabeled manifest.",
    )
    parser.add_argument(
        "--no-export-ditto",
        action="store_true",
        help="Do not write filtered *train.json.gz files. CSV outputs are still written.",
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


def _resolve_existing_path(raw: Any, *, dataset_dir: Path, extra_dirs: Sequence[Path] = ()) -> Path | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    path = Path(text)
    candidates: List[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                path,
                ROOT / path,
                dataset_dir / path,
                dataset_dir / path.name,
                dataset_dir.parent / path,
                dataset_dir.parent / path.name,
            ]
        )
        for base in extra_dirs:
            candidates.append(base / path)
            candidates.append(base / path.name)
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists():
            return candidate
    return None


def _dataset_dirs(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")
    return sorted(path for path in root.iterdir() if path.is_dir() and (path / "profile_manifest.json").exists())


def _select_dataset_dirs(root: Path, requested: Sequence[str]) -> List[Path]:
    all_dirs = {path.name: path for path in _dataset_dirs(root)}
    if not requested:
        return [all_dirs[name] for name in sorted(all_dirs)]
    missing = [name for name in requested if name not in all_dirs]
    if missing:
        raise FileNotFoundError(f"Dataset directories not found under {root}: {missing}")
    return [all_dirs[name] for name in requested]


def _normalize_label_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .map({"TRUE": 1, "FALSE": 0, "1": 1, "0": 0, "YES": 1, "NO": 0, "T": 1, "F": 0, "Y": 1, "N": 0})
    )


def _normalize_label_value(value: Any) -> int:
    text = str(value).strip().upper()
    if text in {"TRUE", "1", "YES", "T", "Y"}:
        return 1
    if text in {"FALSE", "0", "NO", "F", "N"}:
        return 0
    raise ValueError(f"Unsupported label value: {value!r}")


def _count_labels(df: pd.DataFrame) -> Tuple[int, int, int]:
    labels = _normalize_label_series(df["label"])
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    return int(len(df)), pos, neg


def _profile_active_csv(profile_dir: Path) -> Path:
    path = profile_dir / "active_labels_latest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing profile labels file: {path}")
    return path


def _profile_final_csv(profile_dir: Path) -> Path:
    path = profile_dir / "labels_final.csv"
    return path if path.exists() else _profile_active_csv(profile_dir)


def _profile_train_json_gz(profile_dir: Path) -> Path | None:
    matches = sorted(profile_dir.glob("*train.json.gz"))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Expected exactly one *train.json.gz in {profile_dir}, found {len(matches)}")
    return matches[0]


def _master_profile_name(original_manifest: Dict[str, Any], relabeled_manifest: Dict[str, Any]) -> str | None:
    original_profiles = set((original_manifest.get("profiles") or {}).keys())
    relabeled_profiles = set((relabeled_manifest.get("profiles") or {}).keys())
    for candidate in ("all_plus20random", "all"):
        if candidate in original_profiles and candidate in relabeled_profiles:
            return candidate
    return None


def _master_active_csv(dataset_dir: Path, manifest: Dict[str, Any], master_profile_name: str | None) -> Path:
    if master_profile_name:
        candidate = dataset_dir / master_profile_name / "active_labels_latest.csv"
        if candidate.exists():
            return candidate
    direct = dataset_dir / "active_labels_latest.csv"
    if direct.exists():
        return direct
    resolved = _resolve_existing_path(
        manifest.get("source_all_examples_file"),
        dataset_dir=dataset_dir,
        extra_dirs=[dataset_dir],
    )
    if resolved and resolved.exists():
        return resolved
    resolved = _resolve_existing_path(
        manifest.get("source_train_file"),
        dataset_dir=dataset_dir,
        extra_dirs=[dataset_dir],
    )
    if resolved and resolved.exists():
        return resolved
    raise FileNotFoundError(f"Could not resolve master labels csv for {dataset_dir}")


def _changed_pairs(original_csv: Path, relabeled_csv: Path) -> Tuple[pd.DataFrame, set[Tuple[str, str, str, str]]]:
    original_df = pd.read_csv(original_csv)
    relabeled_df = pd.read_csv(relabeled_csv)
    missing = [col for col in KEY_COLUMNS if col not in original_df.columns or col not in relabeled_df.columns]
    if missing:
        raise ValueError(f"Missing key columns {missing} when comparing {original_csv} and {relabeled_csv}")

    merged = (
        original_df[list(KEY_COLUMNS) + ["label"]]
        .rename(columns={"label": "label_before"})
        .merge(
            relabeled_df[list(KEY_COLUMNS) + ["label"]].rename(columns={"label": "label_after"}),
            on=list(KEY_COLUMNS),
            how="inner",
        )
    )
    merged["label_before_norm"] = _normalize_label_series(merged["label_before"])
    merged["label_after_norm"] = _normalize_label_series(merged["label_after"])
    bad = merged["label_before_norm"].isna() | merged["label_after_norm"].isna()
    if bad.any():
        sample = merged.loc[bad, list(KEY_COLUMNS) + ["label_before", "label_after"]].head(5).to_dict("records")
        raise ValueError(f"Could not normalize labels for some rows: {sample}")

    changed = merged[merged["label_before_norm"] != merged["label_after_norm"]].copy()
    changed_keys = {
        tuple(str(row[col]) for col in KEY_COLUMNS)
        for row in changed[list(KEY_COLUMNS)].to_dict("records")
    }
    return changed.reset_index(drop=True), changed_keys


def _load_profile_bundle(profile_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]], str | None]:
    active_df = pd.read_csv(_profile_active_csv(profile_dir)).reset_index(drop=True)
    final_df = pd.read_csv(_profile_final_csv(profile_dir)).reset_index(drop=True)
    train_path = _profile_train_json_gz(profile_dir)
    train_rows: List[Dict[str, Any]] = []
    train_name: str | None = None
    if train_path is not None:
        train_rows = _read_jsonl_gz(train_path)
        train_name = train_path.name
    return active_df, final_df, train_rows, train_name


def _filter_profile_rows(
    active_df: pd.DataFrame,
    final_df: pd.DataFrame,
    train_rows: List[Dict[str, Any]],
    changed_keys: set[Tuple[str, str, str, str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]], Dict[str, int]]:
    if len(final_df) != len(active_df):
        raise ValueError("active_labels_latest.csv and labels_final.csv row counts differ")
    if train_rows and len(train_rows) != len(active_df):
        raise ValueError("Train json.gz row count does not match active_labels_latest.csv")

    keep_mask: List[bool] = []
    for row in active_df.to_dict("records"):
        key = tuple(str(row.get(col, "")) for col in KEY_COLUMNS)
        keep_mask.append(key not in changed_keys)

    keep_series = pd.Series(keep_mask, index=active_df.index, dtype="bool")
    filtered_active = active_df.loc[keep_series].reset_index(drop=True)
    filtered_final = final_df.loc[keep_series].reset_index(drop=True)
    filtered_train = [row for row, keep in zip(train_rows, keep_mask) if keep] if train_rows else []

    total_before, pos_before, neg_before = _count_labels(active_df)
    total_after, pos_after, neg_after = _count_labels(filtered_active)
    summary = {
        "rows_before": total_before,
        "rows_after": total_after,
        "dropped_rows": total_before - total_after,
        "positive_before": pos_before,
        "negative_before": neg_before,
        "positive_after": pos_after,
        "negative_after": neg_after,
    }
    return filtered_active, filtered_final, filtered_train, summary


def _copy_optional(src: Path | None, dst: Path) -> None:
    if src is None or not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def _selected_profiles(manifest: Dict[str, Any], requested: Sequence[str]) -> List[str]:
    available = list((manifest.get("profiles") or {}).keys())
    if not requested:
        return available
    missing = [name for name in requested if name not in available]
    if missing:
        raise KeyError(f"Requested profiles not present in manifest: {missing}. Available: {available}")
    return [name for name in available if name in requested]


def _relabel_direction_counts(changed_df: pd.DataFrame) -> Dict[str, int]:
    before = changed_df["label_before_norm"].astype(int)
    after = changed_df["label_after_norm"].astype(int)
    return {
        "changed_total": int(len(changed_df)),
        "match_to_non_match": int(((before == 1) & (after == 0)).sum()),
        "non_match_to_match": int(((before == 0) & (after == 1)).sum()),
    }


def main() -> None:
    args = _parse_args()
    original_root = Path(args.original_root).expanduser().resolve()
    relabeled_root = Path(args.relabeled_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    requested_datasets = [item.strip() for item in str(args.datasets).split(",") if item.strip()]
    requested_profiles = [item.strip() for item in str(args.profiles).split(",") if item.strip()]

    relabeled_dirs = _select_dataset_dirs(relabeled_root, requested_datasets)
    output_root.mkdir(parents=True, exist_ok=True)

    top_summary: List[Dict[str, Any]] = []
    for relabeled_dir in relabeled_dirs:
        dataset_name = relabeled_dir.name
        original_dir = original_root / dataset_name
        if not original_dir.exists():
            raise FileNotFoundError(f"Original dataset run not found: {original_dir}")

        relabeled_manifest = _load_json(relabeled_dir / "profile_manifest.json")
        original_manifest = _load_json(original_dir / "profile_manifest.json")
        profiles = _selected_profiles(relabeled_manifest, requested_profiles)
        master_profile_name = _master_profile_name(original_manifest, relabeled_manifest)

        original_master_csv = _master_active_csv(original_dir, original_manifest, master_profile_name)
        relabeled_master_csv = _master_active_csv(relabeled_dir, relabeled_manifest, master_profile_name)
        changed_df, changed_keys = _changed_pairs(original_master_csv, relabeled_master_csv)
        changed_counts = _relabel_direction_counts(changed_df)

        out_dataset_dir = output_root / dataset_name
        out_dataset_dir.mkdir(parents=True, exist_ok=True)

        root_active_df = pd.read_csv(relabeled_master_csv).reset_index(drop=True)
        root_final_path = relabeled_dir / "labels_final.csv"
        root_final_df = (
            pd.read_csv(root_final_path).reset_index(drop=True)
            if root_final_path.exists()
            else root_active_df.copy()
        )
        filtered_root_active, filtered_root_final, _, root_summary = _filter_profile_rows(
            root_active_df,
            root_final_df,
            [],
            changed_keys,
        )
        filtered_root_active.to_csv(out_dataset_dir / "active_labels_latest.csv", index=False)
        filtered_root_final.to_csv(out_dataset_dir / "labels_final.csv", index=False)
        changed_df.to_csv(out_dataset_dir / "changed_pairs.csv", index=False)

        updated_manifest = deepcopy(relabeled_manifest)
        updated_manifest["run_dir"] = _to_repo_relative(out_dataset_dir)
        updated_manifest["runner_script"] = "scripts/labeling/build_drop_changed_profiles.py"
        updated_manifest["run_summary_json"] = _to_repo_relative(out_dataset_dir / "drop_changed_summary.json")
        updated_manifest["source_original_manifest"] = _to_repo_relative(original_dir / "profile_manifest.json")
        updated_manifest["source_relabeled_manifest"] = _to_repo_relative(relabeled_dir / "profile_manifest.json")
        updated_manifest["source_train_file"] = _to_repo_relative(out_dataset_dir / "labels_final.csv")
        updated_manifest["source_all_examples_file"] = _to_repo_relative(out_dataset_dir / "active_labels_latest.csv")
        updated_manifest["drop_changed"] = {
            "applied": True,
            "master_profile": master_profile_name or "root",
            "changed_pairs_csv": _to_repo_relative(out_dataset_dir / "changed_pairs.csv"),
            **changed_counts,
            "rows_compared": int(len(root_active_df)),
            "rows_after_drop": int(len(filtered_root_active)),
        }

        profile_summaries: Dict[str, Any] = {}
        new_profiles: Dict[str, Any] = {}
        for profile_name in profiles:
            source_profile_dir = relabeled_dir / profile_name
            active_df, final_df, train_rows, train_name = _load_profile_bundle(source_profile_dir)
            filtered_active, filtered_final, filtered_train, profile_summary = _filter_profile_rows(
                active_df,
                final_df,
                train_rows,
                changed_keys,
            )

            out_profile_dir = out_dataset_dir / profile_name
            out_profile_dir.mkdir(parents=True, exist_ok=True)
            filtered_active.to_csv(out_profile_dir / "active_labels_latest.csv", index=False)
            filtered_final.to_csv(out_profile_dir / "labels_final.csv", index=False)
            if train_name and not args.no_export_ditto:
                _write_jsonl_gz(out_profile_dir / train_name, filtered_train)

            profile_meta = deepcopy((relabeled_manifest.get("profiles") or {}).get(profile_name) or {})
            profile_meta["actual_total"] = int(profile_summary["rows_after"])
            profile_meta["actual_pos"] = int(profile_summary["positive_after"])
            profile_meta["actual_neg"] = int(profile_summary["negative_after"])
            profile_meta["labels_csv"] = _to_repo_relative(out_profile_dir / "active_labels_latest.csv")
            if train_name and not args.no_export_ditto:
                profile_meta["ditto_train_json_gz"] = _to_repo_relative(out_profile_dir / train_name)
            new_profiles[profile_name] = profile_meta

            profile_summaries[profile_name] = {
                **profile_summary,
                "active_labels_latest": _to_repo_relative(out_profile_dir / "active_labels_latest.csv"),
                "labels_final": _to_repo_relative(out_profile_dir / "labels_final.csv"),
                "train_json_gz": _to_repo_relative(out_profile_dir / train_name) if train_name and not args.no_export_ditto else "",
            }

        updated_manifest["profiles"] = new_profiles
        _write_json(out_dataset_dir / "profile_manifest.json", updated_manifest)
        _copy_optional(relabeled_dir / "source_profile_manifest.json", out_dataset_dir / "source_profile_manifest.json")
        _copy_optional(relabeled_dir / "dataset_manifest.json", out_dataset_dir / "dataset_manifest.json")
        _copy_optional(relabeled_dir / "batch_info.json", out_dataset_dir / "batch_info.json")

        summary_payload = {
            "dataset": relabeled_manifest.get("benchmark", dataset_name),
            "dataset_dir": _to_repo_relative(out_dataset_dir),
            "original_dataset_dir": _to_repo_relative(original_dir),
            "relabeled_dataset_dir": _to_repo_relative(relabeled_dir),
            "master_profile": master_profile_name or "root",
            "root_summary": root_summary,
            "changed_pairs": changed_counts,
            "profiles": profile_summaries,
            "export_ditto": not args.no_export_ditto,
        }
        _write_json(out_dataset_dir / "drop_changed_summary.json", summary_payload)

        top_summary.append(
            {
                "dataset": relabeled_manifest.get("benchmark", dataset_name),
                "dataset_dir": _to_repo_relative(out_dataset_dir),
                "master_profile": master_profile_name or "root",
                "changed_total": int(changed_counts["changed_total"]),
                "match_to_non_match": int(changed_counts["match_to_non_match"]),
                "non_match_to_match": int(changed_counts["non_match_to_match"]),
                "rows_after_drop": int(root_summary["rows_after"]),
                "rows_before_drop": int(root_summary["rows_before"]),
            }
        )

        print(
            f"[{relabeled_manifest.get('benchmark', dataset_name)}] "
            f"changed={changed_counts['changed_total']} "
            f"m2n={changed_counts['match_to_non_match']} "
            f"n2m={changed_counts['non_match_to_match']} "
            f"output={out_dataset_dir}"
        )

    _write_json(
        output_root / "drop_changed_run_summary.json",
        {
            "original_root": _to_repo_relative(original_root),
            "relabeled_root": _to_repo_relative(relabeled_root),
            "output_root": _to_repo_relative(output_root),
            "datasets": top_summary,
            "export_ditto": not args.no_export_ditto,
        },
    )
    print(output_root)


if __name__ == "__main__":
    main()
