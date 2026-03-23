#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
import yaml

POSITIVE_LABELS = {"TRUE", "1", "YES", "Y", "T"}
NEGATIVE_LABELS = {"FALSE", "0", "NO", "N", "F"}
RESERVED_FEATURE_FIELDS = {"id", "__rid", "pair_id", "label", "is_hard_negative", "rid1", "rid2", "similarity"}


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    all_examples: bool
    target_size: int
    target_pos: int
    target_neg: int


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError("Benchmark config must be a mapping")
    return payload


def _coerce_mapping(value: Any, name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _coerce_field_list(value: Any, name: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [x.strip() for x in value.split(",")]
    elif isinstance(value, (list, tuple)):
        parts = [str(x).strip() for x in value]
    else:
        raise ValueError(f"{name} must be a comma-separated string or list")
    return [p for p in parts if p]


def _coerce_str_mapping(value: Any, name: str) -> Dict[str, str]:
    raw = _coerce_mapping(value, name)
    out: Dict[str, str] = {}
    for k, v in raw.items():
        key = str(k).strip()
        val = str(v).strip() if v is not None else ""
        if not key:
            continue
        if not val:
            continue
        out[key] = val
    return out


def _coerce_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return False
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off", ""}:
        return False
    raise ValueError(f"{name} must be boolean-like, got: {value!r}")


def _normalize_field_mapping(field_map: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in field_map.items():
        key = "priceCurrency" if str(k).strip() == "currency" else str(k).strip()
        val = str(v).strip()
        if not key or not val:
            continue
        out[key] = val
    return out


def _resolve_train_fields(
    train_fields_raw: Sequence[str],
    left_fields: Dict[str, str],
    right_fields: Dict[str, str],
) -> List[str]:
    # Default to all mapped output names, preserving left->right order.
    desired: List[str]
    if train_fields_raw:
        desired = [str(x).strip() for x in train_fields_raw if str(x).strip()]
    else:
        desired = list(left_fields.keys())
        for k in right_fields.keys():
            if k not in desired:
                desired.append(k)

    if not desired:
        return []

    # Backward compatibility: if an old config still passes source column
    # names in train_fields, remap them to the configured output field names.
    source_to_field: Dict[str, str] = {}
    for out_name, src_name in {**left_fields, **right_fields}.items():
        if src_name and src_name not in source_to_field:
            source_to_field[src_name] = out_name

    resolved: List[str] = []
    for f in desired:
        if f in left_fields or f in right_fields:
            resolved.append(f)
            continue
        resolved.append(source_to_field.get(f, f))

    # Remove reserved/meta fields from model features.
    resolved = [f for f in resolved if f not in RESERVED_FEATURE_FIELDS]
    # Deduplicate while preserving order.
    return list(dict.fromkeys(resolved))


def _normalize_benchmark_config(benchmark_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize benchmark config aliases to the keys expected by this runner."""
    cfg = dict(benchmark_cfg)

    # Support nested embeddings config:
    # embeddings: {dir, left_file, right_file}
    embeddings_cfg = _coerce_mapping(cfg.get("embeddings"), "benchmarks.*.embeddings")
    if embeddings_cfg:
        cfg.setdefault("embeddings_dir", embeddings_cfg.get("dir"))
        cfg.setdefault("left_emb", embeddings_cfg.get("left_file"))
        cfg.setdefault("right_emb", embeddings_cfg.get("right_file"))

    id_col = str(cfg.get("id_col", "id")).strip() or "id"
    cfg.setdefault("left_id_col", str(cfg.get("left_id_col", id_col)).strip() or id_col)
    cfg.setdefault("right_id_col", str(cfg.get("right_id_col", id_col)).strip() or id_col)

    fields_cfg = _normalize_field_mapping(_coerce_str_mapping(cfg.get("fields"), "benchmarks.*.fields"))
    left_fields_cfg = _normalize_field_mapping(_coerce_str_mapping(cfg.get("left_fields"), "benchmarks.*.left_fields"))
    right_fields_cfg = _normalize_field_mapping(_coerce_str_mapping(cfg.get("right_fields"), "benchmarks.*.right_fields"))

    if not left_fields_cfg and fields_cfg:
        left_fields_cfg = dict(fields_cfg)
    if not right_fields_cfg and fields_cfg:
        right_fields_cfg = dict(fields_cfg)

    # Backward compatibility with old schema-map config.
    if not left_fields_cfg:
        left_schema_map = _coerce_str_mapping(cfg.get("left_schema_map"), "benchmarks.*.left_schema_map")
        if left_schema_map:
            cfg["left_id_col"] = str(left_schema_map.get("id", cfg["left_id_col"])).strip() or cfg["left_id_col"]
            left_fields_cfg = _normalize_field_mapping({k: v for k, v in left_schema_map.items() if k != "id"})
    if not right_fields_cfg:
        right_schema_map = _coerce_str_mapping(cfg.get("right_schema_map"), "benchmarks.*.right_schema_map")
        if right_schema_map:
            cfg["right_id_col"] = str(right_schema_map.get("id", cfg["right_id_col"])).strip() or cfg["right_id_col"]
            right_fields_cfg = _normalize_field_mapping({k: v for k, v in right_schema_map.items() if k != "id"})

    cfg["left_fields"] = left_fields_cfg
    cfg["right_fields"] = right_fields_cfg

    return cfg


def _materialize_source_csv(
    src_csv: Path,
    id_col: str,
    field_map: Dict[str, str],
    out_csv: Path,
    ensure_fields: Sequence[str] | None = None,
) -> None:
    src = pd.read_csv(src_csv).reset_index(drop=True)
    source_id_col = str(id_col).strip() or "id"
    if source_id_col not in src.columns:
        raise ValueError(
            f"Required id column not found in {src_csv}. "
            f"Expected '{source_id_col}'."
        )
    out = pd.DataFrame(index=src.index)
    out["id"] = src[source_id_col]
    for output_name, source_col in field_map.items():
        if source_col not in src.columns:
            raise ValueError(
                f"Mapped source column '{source_col}' not found in {src_csv} "
                f"for output field '{output_name}'."
            )
        out[output_name] = src[source_col]
    for col in (ensure_fields or []):
        c = str(col).strip()
        if not c or c in out.columns:
            continue
        if c in src.columns:
            out[c] = src[c]
            continue
        out[c] = ""
    out["id"] = out["id"].astype(str)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)


def _to_profile(name: str, payload: Dict[str, Any]) -> ProfileSpec:
    all_examples = _coerce_bool(payload.get("all_examples", payload.get("all")), f"profiles.{name}.all_examples")
    if all_examples:
        return ProfileSpec(name=name, all_examples=True, target_size=0, target_pos=0, target_neg=0)

    target_pos_raw = payload.get("target_positives", payload.get("target_pos"))
    if target_pos_raw is None:
        raise ValueError(f"Profile '{name}' missing target_positives/target_pos")
    target_pos = int(target_pos_raw)
    target_neg_raw = payload.get("target_negatives", payload.get("target_neg"))
    target_size_raw = payload.get("target_size", payload.get("target_total"))

    if target_neg_raw is None and target_size_raw is None:
        raise ValueError(f"Profile '{name}' must define target_size or target_negatives")

    if target_neg_raw is None:
        target_size = int(target_size_raw)
        target_neg = int(target_size - target_pos)
    else:
        target_neg = int(target_neg_raw)
        target_size = int(target_size_raw) if target_size_raw is not None else int(target_pos + target_neg)

    if target_size != target_pos + target_neg:
        raise ValueError(
            f"Profile '{name}' inconsistent targets: target_size={target_size} "
            f"but target_pos+target_neg={target_pos + target_neg}"
        )
    if min(target_size, target_pos, target_neg) < 0:
        raise ValueError(f"Profile '{name}' targets must be non-negative")
    return ProfileSpec(
        name=name,
        all_examples=False,
        target_size=target_size,
        target_pos=target_pos,
        target_neg=target_neg,
    )


def _select_profiles(config_profiles: Dict[str, Any], requested: List[str]) -> List[ProfileSpec]:
    if not requested:
        requested = list(config_profiles.keys())
    selected: List[ProfileSpec] = []
    for name in requested:
        if name not in config_profiles:
            raise KeyError(f"Unknown profile '{name}'. Available: {sorted(config_profiles.keys())}")
        spec = _to_profile(name, _coerce_mapping(config_profiles[name], f"profiles.{name}"))
        selected.append(spec)

    numeric = [p for p in selected if not p.all_examples]
    all_profiles = [p for p in selected if p.all_examples]

    numeric = sorted(numeric, key=lambda p: (p.target_size, p.target_pos, p.target_neg, p.name))
    prev = None
    for spec in numeric:
        if prev is not None:
            if spec.target_size < prev.target_size or spec.target_pos < prev.target_pos or spec.target_neg < prev.target_neg:
                raise ValueError(
                    "Profiles must be monotonic for nesting (size/pos/neg must not decrease): "
                    f"{prev.name} -> {spec.name}"
                )
        prev = spec
    return numeric + all_profiles


def _normalize_label(v: object) -> str:
    s = str(v).strip().upper()
    if s in POSITIVE_LABELS:
        return "TRUE"
    if s in NEGATIVE_LABELS:
        return "FALSE"
    raise ValueError(f"Unsupported label value: {v}")


def _flatten_cli_args(raw: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for key, value in raw.items():
        flag = f"--{str(key).strip().replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                out.append(flag)
            continue
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            for v in value:
                out.extend([flag, str(v)])
            continue
        out.extend([flag, str(value)])
    return out


def _subset_from_master(master: pd.DataFrame, target_pos: int, target_neg: int) -> pd.DataFrame:
    labels = master["label"].apply(_normalize_label)
    pos_idx = labels[labels == "TRUE"].index.to_list()
    neg_idx = labels[labels == "FALSE"].index.to_list()
    if len(pos_idx) < target_pos or len(neg_idx) < target_neg:
        raise RuntimeError(
            f"Master labels do not contain enough examples for target "
            f"({target_pos} pos, {target_neg} neg). Available: {len(pos_idx)} pos, {len(neg_idx)} neg."
        )
    keep = sorted(set(pos_idx[:target_pos]) | set(neg_idx[:target_neg]))
    out = master.iloc[keep].copy().reset_index(drop=True)
    out["label"] = out["label"].apply(_normalize_label)
    return out


def _read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark-aware wrapper around a config-selected labeling runner")
    parser.add_argument("--config", default="configs/labeling/benchmarks.yaml")
    parser.add_argument("--benchmark", required=True, help="Benchmark key from the config file")
    parser.add_argument(
        "--profiles",
        default="",
        help="Comma-separated profile names. Default: all profiles in config.",
    )
    parser.add_argument("--output-root", default=None, help="Override output root")
    parser.add_argument("--run-name", default=None, help="Override run name")
    parser.add_argument("--resume", action="store_true", help="Forward --resume to run_simple_labeling.py")
    parser.add_argument("--no-export-ditto", action="store_true", help="Skip Ditto train file export")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved commands and exit")
    args = parser.parse_args()

    cfg = _load_yaml(Path(args.config))
    defaults = _coerce_mapping(cfg.get("defaults"), "defaults")
    benchmarks = _coerce_mapping(cfg.get("benchmarks"), "benchmarks")
    profiles_cfg = _coerce_mapping(cfg.get("profiles"), "profiles")
    if args.benchmark not in benchmarks:
        raise KeyError(f"Unknown benchmark '{args.benchmark}'. Available: {sorted(benchmarks.keys())}")

    benchmark_cfg = _coerce_mapping(benchmarks[args.benchmark], f"benchmarks.{args.benchmark}")
    benchmark_cfg = _normalize_benchmark_config(benchmark_cfg)
    left_fields = _coerce_str_mapping(benchmark_cfg.get("left_fields"), "benchmarks.*.left_fields")
    right_fields = _coerce_str_mapping(benchmark_cfg.get("right_fields"), "benchmarks.*.right_fields")
    train_fields_raw = _coerce_field_list(benchmark_cfg.get("train_fields"), "benchmarks.*.train_fields")
    train_fields = _resolve_train_fields(train_fields_raw, left_fields=left_fields, right_fields=right_fields)

    benchmark_profile_overrides = _coerce_mapping(benchmark_cfg.get("profiles"), "benchmarks.*.profiles")
    effective_profiles_cfg = dict(profiles_cfg)
    effective_profiles_cfg.update(benchmark_profile_overrides)
    requested_profiles = [x.strip() for x in args.profiles.split(",") if x.strip()]
    profiles = _select_profiles(effective_profiles_cfg, requested_profiles)
    numeric_profiles = [p for p in profiles if not p.all_examples]
    largest = numeric_profiles[-1] if numeric_profiles else None

    output_root = Path(args.output_root or defaults.get("output_root", "output/simple_labeling"))
    run_prefix = str(defaults.get("run_prefix", args.benchmark)).strip() or args.benchmark
    run_name = args.run_name or f"{run_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    canonical_dir = run_dir / "canonical"
    left_src = Path(str(benchmark_cfg["left_csv"]))
    right_src = Path(str(benchmark_cfg["right_csv"]))
    left_canonical = canonical_dir / "left.csv"
    right_canonical = canonical_dir / "right.csv"
    _materialize_source_csv(
        left_src,
        str(benchmark_cfg.get("left_id_col", "id")),
        left_fields,
        left_canonical,
        ensure_fields=train_fields,
    )
    _materialize_source_csv(
        right_src,
        str(benchmark_cfg.get("right_id_col", "id")),
        right_fields,
        right_canonical,
        ensure_fields=train_fields,
    )

    labeling_defaults = _coerce_mapping(defaults.get("labeling_args"), "defaults.labeling_args")
    labeling_override = _coerce_mapping(benchmark_cfg.get("labeling_args"), "benchmarks.*.labeling_args")
    labeling_args = dict(labeling_defaults)
    labeling_args.update(labeling_override)
    for reserved in {"embeddings_dir", "left_csv", "right_csv", "target_size", "target_positives", "output_root", "run_name"}:
        labeling_args.pop(reserved, None)
    runner_script = str(
        benchmark_cfg.get(
            "runner_script",
            defaults.get("runner_script", "scripts/labeling/run_simple_labeling.py"),
        )
    ).strip() or "scripts/labeling/run_simple_labeling.py"
    runner_path = Path(runner_script)
    if not runner_path.exists():
        raise FileNotFoundError(f"Configured runner_script does not exist: {runner_script}")

    run_simple_cmd: List[str] = []
    if largest is not None:
        run_simple_cmd = [
            sys.executable,
            runner_script,
            "--embeddings-dir",
            str(benchmark_cfg["embeddings_dir"]),
            "--left-csv",
            str(left_canonical),
            "--right-csv",
            str(right_canonical),
            "--left-emb",
            str(benchmark_cfg["left_emb"]),
            "--right-emb",
            str(benchmark_cfg["right_emb"]),
            "--target-size",
            str(largest.target_size),
            "--target-positives",
            str(largest.target_pos),
            "--output-root",
            str(output_root),
            "--run-name",
            run_name,
        ]
        if train_fields:
            run_simple_cmd.extend(["--feature-fields", ",".join(train_fields)])
        if args.resume:
            run_simple_cmd.append("--resume")
        run_simple_cmd.extend(_flatten_cli_args(labeling_args))

    print(f"Benchmark: {args.benchmark}")
    print(f"Run directory: {run_dir}")
    if largest is not None:
        print(f"Largest target: total={largest.target_size} pos={largest.target_pos} neg={largest.target_neg}")
    else:
        print("Largest target: <none> (all requested profiles use all_examples=true)")
    if args.dry_run:
        if run_simple_cmd:
            print(f"Dry run: runner command ({runner_script}):")
            print(" ".join(run_simple_cmd))
        else:
            print("Dry run: no run_simple_labeling command (all_examples-only selection)")
        for spec in profiles:
            if spec.all_examples:
                print(f"Profile plan {spec.name}: all_examples=true")
            else:
                print(
                    f"Profile plan {spec.name}: total={spec.target_size} pos={spec.target_pos} neg={spec.target_neg}"
                )
        return
    if run_simple_cmd:
        subprocess.run(run_simple_cmd, check=True)
    else:
        print("No numeric profile selected; skipping run_simple_labeling and exporting from existing outputs.")

    master_path = run_dir / "labels_final.csv"
    active_master_path = run_dir / "active_labels_latest.csv"

    master = pd.DataFrame()
    if numeric_profiles:
        if not master_path.exists():
            raise FileNotFoundError(f"Expected master labels at {master_path}")
        master = pd.read_csv(master_path).reset_index(drop=True)
        if "label" not in master.columns:
            raise ValueError(f"Missing label column in {master_path}")
    elif not active_master_path.exists() and not master_path.exists():
        raise FileNotFoundError(
            "No export source found. Expected one of: "
            f"{active_master_path} or {master_path}"
        )

    profiles_root = run_dir / "profiles"
    profiles_root.mkdir(parents=True, exist_ok=True)
    export_ditto = not args.no_export_ditto
    run_summary = _read_json_if_exists(run_dir / "summary.json")
    manifest: Dict[str, Any] = {
        "benchmark": args.benchmark,
        "run_dir": str(run_dir),
        "runner_script": runner_script,
        "run_summary_json": str(run_dir / "summary.json"),
        "labeling_cost": run_summary.get("labeling_cost"),
        "source_train_file": str(master_path if master_path.exists() else active_master_path),
        "source_all_examples_file": str(active_master_path if active_master_path.exists() else master_path),
        "canonical_left_csv": str(left_canonical),
        "canonical_right_csv": str(right_canonical),
        "left_fields": left_fields,
        "right_fields": right_fields,
        "ditto_fields": train_fields,
        "profiles": {},
    }

    for spec in profiles:
        if spec.all_examples:
            if active_master_path.exists():
                subset = pd.read_csv(active_master_path).reset_index(drop=True)
            elif master_path.exists():
                subset = pd.read_csv(master_path).reset_index(drop=True)
            else:
                raise FileNotFoundError(
                    "Profile requires all examples, but no source labels file exists at "
                    f"{active_master_path} or {master_path}"
                )
            if "label" not in subset.columns:
                raise ValueError(f"Missing label column for all-examples profile in source file")
            subset = subset.copy()
            subset["label"] = subset["label"].apply(_normalize_label)
            target_total = int(len(subset))
            target_pos = int((subset["label"].astype(str).str.upper() == "TRUE").sum())
            target_neg = int((subset["label"].astype(str).str.upper() == "FALSE").sum())
        else:
            subset = _subset_from_master(master, target_pos=spec.target_pos, target_neg=spec.target_neg)
            target_total = int(spec.target_size)
            target_pos = int(spec.target_pos)
            target_neg = int(spec.target_neg)
        profile_dir = profiles_root / spec.name
        profile_dir.mkdir(parents=True, exist_ok=True)
        labels_csv = profile_dir / "active_labels_latest.csv"
        subset.to_csv(labels_csv, index=False)
        # Keep naming consistent with the existing pipeline outputs.
        subset.to_csv(profile_dir / "labels_final.csv", index=False)

        pos = int((subset["label"].astype(str).str.upper() == "TRUE").sum())
        neg = int((subset["label"].astype(str).str.upper() == "FALSE").sum())
        profile_meta: Dict[str, Any] = {
            "all_examples": bool(spec.all_examples),
            "target_total": target_total,
            "target_pos": target_pos,
            "target_neg": target_neg,
            "actual_total": int(len(subset)),
            "actual_pos": int(pos),
            "actual_neg": int(neg),
            "labels_csv": str(labels_csv),
        }

        if export_ditto:
            ditto_json_gz = profile_dir / f"active_labels_latest_{args.benchmark}_{spec.name}_train.json.gz"
            convert_cmd = [
                sys.executable,
                "scripts/ditto/convert_active_labels_to_wdc.py",
                "--labels-csv",
                str(labels_csv),
                "--left-csv",
                str(left_canonical),
                "--right-csv",
                str(right_canonical),
                "--output-json-gz",
                str(ditto_json_gz),
            ]
            if train_fields:
                convert_cmd.extend(["--fields", ",".join(train_fields)])
            subprocess.run(convert_cmd, check=True)
            profile_meta["ditto_train_json_gz"] = str(ditto_json_gz)
            profile_meta["ditto_fields"] = list(train_fields)
        manifest["profiles"][spec.name] = profile_meta
        print(
            f"Profile {spec.name}: total={profile_meta['actual_total']} "
            f"pos={profile_meta['actual_pos']} neg={profile_meta['actual_neg']}"
        )

    manifest_path = run_dir / "profile_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
