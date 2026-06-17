#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_CANDIDATES = [
    ROOT / "configs" / "labeling" / "benchmarks_active.yaml",
    ROOT / "configs" / "labeling" / "benchmarks.yaml",
]


def _load_benchmark_module():
    module_path = ROOT / "scripts" / "archive" / "labeling_helpers" / "run_benchmark_labeling.py"
    spec = importlib.util.spec_from_file_location("_run_benchmark_labeling_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark_module = _load_benchmark_module()

_load_yaml = benchmark_module._load_yaml
_coerce_mapping = benchmark_module._coerce_mapping
_coerce_field_list = benchmark_module._coerce_field_list
_coerce_str_mapping = benchmark_module._coerce_str_mapping
_normalize_benchmark_config = benchmark_module._normalize_benchmark_config
_normalize_field_mapping = benchmark_module._normalize_field_mapping
_resolve_train_fields = benchmark_module._resolve_train_fields
_materialize_source_csv = benchmark_module._materialize_source_csv
_select_profiles = benchmark_module._select_profiles
_normalize_label = benchmark_module._normalize_label
_subset_from_master = benchmark_module._subset_from_master
_merge_base_with_random_labels = benchmark_module._merge_base_with_random_labels
_write_profile_outputs = benchmark_module._write_profile_outputs
_resolve_random_profile_settings = benchmark_module._resolve_random_profile_settings
_build_random_profile_name = benchmark_module._build_random_profile_name
_read_json_if_exists = benchmark_module._read_json_if_exists


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build profile exports from an existing labeling run without doing any additional labeling."
        )
    )
    parser.add_argument(
        "run_dir",
        help="Path to an existing labeling run directory that already contains labels_final.csv or active_labels_latest.csv.",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Benchmark key. If omitted, infer it from the run directory name.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Labeling benchmark config. If omitted, try benchmarks_active.yaml then benchmarks.yaml.",
    )
    parser.add_argument(
        "--profiles",
        default="",
        help="Comma-separated base profile names to build. Default: all configured base profiles.",
    )
    parser.add_argument(
        "--no-export-ditto",
        action="store_true",
        help="Skip conversion to Ditto train json.gz files.",
    )
    parser.add_argument(
        "--skip-random-profiles",
        action="store_true",
        help="Do not build *_plus20random profiles even if random_profile_labels.csv exists.",
    )
    parser.add_argument(
        "--rebuild-canonical",
        action="store_true",
        help="Rebuild canonical/left.csv and canonical/right.csv even if they already exist.",
    )
    return parser.parse_args()


def _infer_benchmark(run_dir: Path) -> str | None:
    manifest_path = run_dir / "profile_manifest.json"
    if manifest_path.exists():
        payload = _read_json_if_exists(manifest_path)
        benchmark = str(payload.get("benchmark", "")).strip()
        if benchmark:
            return benchmark

    name = run_dir.name
    if name.startswith("benchmark_"):
        suffix = name[len("benchmark_") :]
        parts = suffix.split("_")
        if len(parts) >= 3:
            return "_".join(parts[:-2]).replace("_", "-")
    return None


def _resolve_config(
    benchmark: str,
    config_path_raw: str | None,
) -> tuple[Path, Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    candidates = [Path(config_path_raw).expanduser().resolve()] if config_path_raw else DEFAULT_CONFIG_CANDIDATES
    for path in candidates:
        if not path.exists():
            continue
        payload = _load_yaml(path)
        defaults = _coerce_mapping(payload.get("defaults"), "defaults")
        benchmarks = _coerce_mapping(payload.get("benchmarks"), "benchmarks")
        if benchmark not in benchmarks:
            continue
        profiles_cfg = _coerce_mapping(payload.get("profiles"), "profiles")
        benchmark_cfg = _normalize_benchmark_config(
            _coerce_mapping(benchmarks[benchmark], f"benchmarks.{benchmark}")
        )
        return path, payload, defaults, profiles_cfg, benchmark_cfg
    raise FileNotFoundError(
        f"Could not resolve benchmark {benchmark!r} from config. Tried: {[str(p) for p in candidates]}"
    )


def _load_master_labels(run_dir: Path) -> tuple[pd.DataFrame, Path]:
    candidates = [
        run_dir / "labels_final.csv",
        run_dir / "active_labels_latest.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path).reset_index(drop=True)
            if "label" not in df.columns:
                raise ValueError(f"Missing label column in {path}")
            return df, path
    raise FileNotFoundError(
        f"Expected one of {run_dir / 'labels_final.csv'} or {run_dir / 'active_labels_latest.csv'}"
    )


def _load_all_examples_labels(run_dir: Path, fallback_path: Path) -> tuple[pd.DataFrame, Path]:
    active_path = run_dir / "active_labels_latest.csv"
    if active_path.exists():
        df = pd.read_csv(active_path).reset_index(drop=True)
        if "label" not in df.columns:
            raise ValueError(f"Missing label column in {active_path}")
        return df, active_path
    df = pd.read_csv(fallback_path).reset_index(drop=True)
    if "label" not in df.columns:
        raise ValueError(f"Missing label column in {fallback_path}")
    return df, fallback_path


def _count_binary_labels(df: pd.DataFrame) -> tuple[int, int, int]:
    labels = df["label"].astype(str).str.upper()
    pos = int((labels == "TRUE").sum())
    neg = int((labels == "FALSE").sum())
    return int(len(df)), pos, neg


def _write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _resolve_repo_path(raw: str) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    benchmark = args.benchmark or _infer_benchmark(run_dir)
    if not benchmark:
        raise ValueError("Could not infer benchmark from run directory. Please pass --benchmark explicitly.")

    config_path, _payload, defaults, profiles_cfg, benchmark_cfg = _resolve_config(benchmark, args.config)
    benchmark_profile_overrides = _coerce_mapping(benchmark_cfg.get("profiles"), "benchmarks.*.profiles")
    effective_profiles_cfg = dict(profiles_cfg)
    effective_profiles_cfg.update(benchmark_profile_overrides)
    requested_profiles = [x.strip() for x in args.profiles.split(",") if x.strip()]
    profiles = _select_profiles(effective_profiles_cfg, requested_profiles)

    left_fields = _normalize_field_mapping(_coerce_str_mapping(benchmark_cfg.get("left_fields"), "left_fields"))
    right_fields = _normalize_field_mapping(_coerce_str_mapping(benchmark_cfg.get("right_fields"), "right_fields"))
    train_fields_raw = _coerce_field_list(benchmark_cfg.get("train_fields"), "train_fields")
    train_fields = _resolve_train_fields(train_fields_raw, left_fields=left_fields, right_fields=right_fields)
    random_profile_settings = _resolve_random_profile_settings(defaults, benchmark_cfg)

    canonical_dir = run_dir / "canonical"
    left_canonical = canonical_dir / "left.csv"
    right_canonical = canonical_dir / "right.csv"
    if args.rebuild_canonical or not left_canonical.exists():
        _materialize_source_csv(
            _resolve_repo_path(str(benchmark_cfg["left_csv"])),
            str(benchmark_cfg.get("left_id_col", "id")),
            left_fields,
            left_canonical,
            ensure_fields=train_fields,
        )
    if args.rebuild_canonical or not right_canonical.exists():
        _materialize_source_csv(
            _resolve_repo_path(str(benchmark_cfg["right_csv"])),
            str(benchmark_cfg.get("right_id_col", "id")),
            right_fields,
            right_canonical,
            ensure_fields=train_fields,
        )

    master_df, master_path = _load_master_labels(run_dir)
    all_examples_df, all_examples_path = _load_all_examples_labels(run_dir, master_path)
    master_df["label"] = master_df["label"].apply(_normalize_label)
    all_examples_df["label"] = all_examples_df["label"].apply(_normalize_label)
    profiles_root = run_dir / "profiles"
    profiles_root.mkdir(parents=True, exist_ok=True)

    run_summary = _read_json_if_exists(run_dir / "summary.json")
    runner_script = str(
        benchmark_cfg.get(
            "runner_script",
            defaults.get("runner_script", "scripts/labeling/active_learning_ml.py"),
        )
    ).strip() or "scripts/labeling/active_learning_ml.py"

    manifest: Dict[str, Any] = {
        "benchmark": benchmark,
        "run_dir": str(run_dir),
        "runner_script": runner_script,
        "run_summary_json": str(run_dir / "summary.json"),
        "labeling_cost": run_summary.get("labeling_cost"),
        "combined_labeling_cost_usd": run_summary.get("combined_labeling_cost_usd"),
        "source_train_file": str(master_path),
        "source_all_examples_file": str(all_examples_path),
        "canonical_left_csv": str(left_canonical),
        "canonical_right_csv": str(right_canonical),
        "left_fields": left_fields,
        "right_fields": right_fields,
        "ditto_fields": train_fields,
        "random_profile_settings": random_profile_settings,
        "profiles": {},
    }

    random_labels_path = run_dir / "random_profile_labels.csv"
    random_usage_path = run_dir / "random_profile_usage.json"
    random_labels_df = pd.DataFrame()
    if not args.skip_random_profiles and random_labels_path.exists():
        random_labels_df = pd.read_csv(random_labels_path).reset_index(drop=True)
        if "label" not in random_labels_df.columns:
            raise ValueError(f"Missing label column in {random_labels_path}")
        manifest["random_profile_usage"] = _read_json_if_exists(random_usage_path)
        if manifest["random_profile_usage"]:
            try:
                manifest["random_profile_cost"] = benchmark_module._estimate_usage_costs(
                    str(random_profile_settings["model"]),
                    manifest["random_profile_usage"],
                )
            except Exception:
                pass
        manifest["random_profile_meta"] = {
            "candidate_count": int(len(random_labels_df)),
            "labeled_count": int(len(random_labels_df)),
            "source": str(random_labels_path),
        }
    elif not args.skip_random_profiles and random_profile_settings.get("enabled"):
        manifest["random_profile_meta"] = {
            "available": False,
            "reason": "random_profile_labels.csv not found; skipping *_plus20random profiles without additional labeling.",
        }

    export_ditto = not args.no_export_ditto
    built_profile_names: List[str] = []
    skipped_profiles: List[Dict[str, str]] = []

    for spec in profiles:
        try:
            if spec.all_examples:
                subset = all_examples_df.copy().reset_index(drop=True)
                target_total, target_pos, target_neg = _count_binary_labels(subset)
            else:
                subset = _subset_from_master(master_df, target_pos=spec.target_pos, target_neg=spec.target_neg)
                target_total = int(spec.target_size)
                target_pos = int(spec.target_pos)
                target_neg = int(spec.target_neg)
        except RuntimeError as exc:
            skipped_profiles.append(
                {
                    "profile": spec.name,
                    "reason": str(exc),
                }
            )
            print(f"Skipping profile {spec.name}: {exc}", flush=True)
            continue

        output_meta = _write_profile_outputs(
            benchmark=benchmark,
            profile_name=spec.name,
            subset=subset,
            profile_dir=profiles_root / spec.name,
            left_canonical=left_canonical,
            right_canonical=right_canonical,
            train_fields=train_fields,
            export_ditto=export_ditto,
        )
        profile_meta: Dict[str, Any] = {
            "all_examples": bool(spec.all_examples),
            "target_total": target_total,
            "target_pos": target_pos,
            "target_neg": target_neg,
        }
        profile_meta.update(output_meta)
        manifest["profiles"][spec.name] = profile_meta
        built_profile_names.append(spec.name)
        print(
            f"Profile {spec.name}: total={profile_meta['actual_total']} "
            f"pos={profile_meta['actual_pos']} neg={profile_meta['actual_neg']}"
        )

        if (
            not spec.all_examples
            and not args.skip_random_profiles
            and random_profile_settings.get("enabled")
            and not random_labels_df.empty
        ):
            augmented_name = _build_random_profile_name(spec.name, float(random_profile_settings["fraction"]))
            extra_n = max(1, int(round(spec.target_size * float(random_profile_settings["fraction"]))))
            augmented_subset = _merge_base_with_random_labels(subset, random_labels_df, extra_n=extra_n)
            augmented_output_meta = _write_profile_outputs(
                benchmark=benchmark,
                profile_name=augmented_name,
                subset=augmented_subset,
                profile_dir=profiles_root / augmented_name,
                left_canonical=left_canonical,
                right_canonical=right_canonical,
                train_fields=train_fields,
                export_ditto=export_ditto,
            )
            augmented_meta: Dict[str, Any] = {
                "all_examples": False,
                "base_profile": spec.name,
                "random_fraction": float(random_profile_settings["fraction"]),
                "random_additions": int(extra_n),
                "target_total": int(len(augmented_subset)),
                "shared_random_pool": True,
                "shared_random_model": str(random_profile_settings["model"]),
            }
            augmented_meta["target_pos"] = int(augmented_output_meta["actual_pos"])
            augmented_meta["target_neg"] = int(augmented_output_meta["actual_neg"])
            augmented_meta.update(augmented_output_meta)
            manifest["profiles"][augmented_name] = augmented_meta
            built_profile_names.append(augmented_name)
            print(
                f"Profile {augmented_name}: total={augmented_meta['actual_total']} "
                f"pos={augmented_meta['actual_pos']} neg={augmented_meta['actual_neg']} "
                f"(base={spec.name}, random_additions={extra_n})"
            )

    if (
        not args.skip_random_profiles
        and random_profile_settings.get("enabled")
        and not random_labels_df.empty
    ):
        all_augmented_name = _build_random_profile_name("all", float(random_profile_settings["fraction"]))
        all_augmented_subset = _merge_base_with_random_labels(
            all_examples_df,
            random_labels_df,
            extra_n=int(len(random_labels_df)),
        )
        all_augmented_output_meta = _write_profile_outputs(
            benchmark=benchmark,
            profile_name=all_augmented_name,
            subset=all_augmented_subset,
            profile_dir=profiles_root / all_augmented_name,
            left_canonical=left_canonical,
            right_canonical=right_canonical,
            train_fields=train_fields,
            export_ditto=export_ditto,
        )
        all_augmented_meta: Dict[str, Any] = {
            "all_examples": True,
            "base_profile": "all",
            "random_fraction": float(random_profile_settings["fraction"]),
            "random_additions": int(len(random_labels_df)),
            "target_total": int(len(all_augmented_subset)),
            "target_pos": int(all_augmented_output_meta["actual_pos"]),
            "target_neg": int(all_augmented_output_meta["actual_neg"]),
            "shared_random_pool": True,
            "shared_random_model": str(random_profile_settings["model"]),
        }
        all_augmented_meta.update(all_augmented_output_meta)
        manifest["profiles"][all_augmented_name] = all_augmented_meta
        built_profile_names.append(all_augmented_name)
        print(
            f"Profile {all_augmented_name}: total={all_augmented_meta['actual_total']} "
            f"pos={all_augmented_meta['actual_pos']} neg={all_augmented_meta['actual_neg']} "
            f"(base=all, random_additions={len(random_labels_df)})"
        )

    manifest_path = run_dir / "profile_manifest.json"
    _write_manifest(manifest_path, manifest)

    summary_path = run_dir / "build_profiles_summary.json"
    summary_payload = {
        "run_dir": str(run_dir),
        "benchmark": benchmark,
        "config": str(config_path),
        "source_train_file": str(master_path),
        "source_all_examples_file": str(all_examples_path),
        "export_ditto": bool(export_ditto),
        "built_profiles": built_profile_names,
        "skipped_profiles": skipped_profiles,
        "skipped_random_profiles": bool(args.skip_random_profiles or random_labels_df.empty),
    }
    _write_manifest(summary_path, summary_payload)

    print(f"Saved manifest: {manifest_path}")
    print(f"Saved summary: {summary_path}")
    if skipped_profiles:
        print("Skipped profiles:")
        for row in skipped_profiles:
            print(f"  - {row['profile']}: {row['reason']}")


if __name__ == "__main__":
    main()
