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

CANONICAL_FIELDS: Sequence[str] = (
    "id",
    "title",
    "brand",
    "description",
    "price",
    "priceCurrency",
)
POSITIVE_LABELS = {"TRUE", "1", "YES", "Y", "T"}
NEGATIVE_LABELS = {"FALSE", "0", "NO", "N", "F"}


@dataclass(frozen=True)
class ProfileSpec:
    name: str
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


def _normalize_benchmark_config(benchmark_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize benchmark config aliases to the canonical keys expected by this runner."""
    cfg = dict(benchmark_cfg)

    # Support nested embeddings config:
    # embeddings: {dir, left_file, right_file}
    embeddings_cfg = _coerce_mapping(cfg.get("embeddings"), "benchmarks.*.embeddings")
    if embeddings_cfg:
        cfg.setdefault("embeddings_dir", embeddings_cfg.get("dir"))
        cfg.setdefault("left_emb", embeddings_cfg.get("left_file"))
        cfg.setdefault("right_emb", embeddings_cfg.get("right_file"))

    # Support concise schema config:
    # id_col + fields: {title, brand, description, price, currency}
    if "left_schema_map" not in cfg and "right_schema_map" not in cfg:
        fields_cfg = _coerce_mapping(cfg.get("fields"), "benchmarks.*.fields")
        if fields_cfg:
            id_col = str(cfg.get("id_col", "id")).strip() or "id"
            schema_map: Dict[str, str] = {"id": id_col}

            for canonical in ("title", "brand", "description", "price"):
                raw = fields_cfg.get(canonical)
                if raw is None:
                    continue
                val = str(raw).strip()
                if val:
                    schema_map[canonical] = val

            # Accept either `priceCurrency` or `currency` alias.
            price_currency_raw = fields_cfg.get("priceCurrency", fields_cfg.get("currency"))
            if price_currency_raw is not None:
                price_currency = str(price_currency_raw).strip()
                if price_currency:
                    schema_map["priceCurrency"] = price_currency

            cfg.setdefault("left_schema_map", dict(schema_map))
            cfg.setdefault("right_schema_map", dict(schema_map))

    return cfg


def _canonicalize_source_csv(src_csv: Path, schema_map: Dict[str, str], out_csv: Path) -> None:
    src = pd.read_csv(src_csv).reset_index(drop=True)
    out = pd.DataFrame(index=src.index)
    for canonical in CANONICAL_FIELDS:
        source_col = str(schema_map.get(canonical, canonical))
        if source_col in src.columns:
            out[canonical] = src[source_col]
        elif canonical == "id":
            raise ValueError(
                f"Required id column not found in {src_csv}. "
                f"Expected mapped column '{source_col}' from schema_map={schema_map}"
            )
        else:
            out[canonical] = ""
    out["id"] = out["id"].astype(str)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)


def _to_profile(name: str, payload: Dict[str, Any]) -> ProfileSpec:
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
    return ProfileSpec(name=name, target_size=target_size, target_pos=target_pos, target_neg=target_neg)


def _select_profiles(config_profiles: Dict[str, Any], requested: List[str]) -> List[ProfileSpec]:
    if not requested:
        requested = list(config_profiles.keys())
    selected: List[ProfileSpec] = []
    for name in requested:
        if name not in config_profiles:
            raise KeyError(f"Unknown profile '{name}'. Available: {sorted(config_profiles.keys())}")
        spec = _to_profile(name, _coerce_mapping(config_profiles[name], f"profiles.{name}"))
        selected.append(spec)

    selected = sorted(selected, key=lambda p: (p.target_size, p.target_pos, p.target_neg, p.name))
    prev = None
    for spec in selected:
        if prev is not None:
            if spec.target_size < prev.target_size or spec.target_pos < prev.target_pos or spec.target_neg < prev.target_neg:
                raise ValueError(
                    "Profiles must be monotonic for nesting (size/pos/neg must not decrease): "
                    f"{prev.name} -> {spec.name}"
                )
        prev = spec
    return selected


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark-aware wrapper around run_simple_labeling.py")
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
    benchmark_profile_overrides = _coerce_mapping(benchmark_cfg.get("profiles"), "benchmarks.*.profiles")
    effective_profiles_cfg = dict(profiles_cfg)
    effective_profiles_cfg.update(benchmark_profile_overrides)
    requested_profiles = [x.strip() for x in args.profiles.split(",") if x.strip()]
    profiles = _select_profiles(effective_profiles_cfg, requested_profiles)
    largest = profiles[-1]

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
    left_schema_map = _coerce_mapping(benchmark_cfg.get("left_schema_map"), "left_schema_map")
    right_schema_map = _coerce_mapping(benchmark_cfg.get("right_schema_map"), "right_schema_map")
    _canonicalize_source_csv(left_src, left_schema_map, left_canonical)
    _canonicalize_source_csv(right_src, right_schema_map, right_canonical)

    labeling_defaults = _coerce_mapping(defaults.get("labeling_args"), "defaults.labeling_args")
    labeling_override = _coerce_mapping(benchmark_cfg.get("labeling_args"), "benchmarks.*.labeling_args")
    labeling_args = dict(labeling_defaults)
    labeling_args.update(labeling_override)
    for reserved in {"embeddings_dir", "left_csv", "right_csv", "target_size", "target_positives", "output_root", "run_name"}:
        labeling_args.pop(reserved, None)

    run_simple_cmd = [
        sys.executable,
        "scripts/run_simple_labeling.py",
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
    if args.resume:
        run_simple_cmd.append("--resume")
    run_simple_cmd.extend(_flatten_cli_args(labeling_args))

    print(f"Benchmark: {args.benchmark}")
    print(f"Run directory: {run_dir}")
    print(f"Largest target: total={largest.target_size} pos={largest.target_pos} neg={largest.target_neg}")
    if args.dry_run:
        print("Dry run: run_simple_labeling command:")
        print(" ".join(run_simple_cmd))
        for spec in profiles:
            print(
                f"Profile plan {spec.name}: total={spec.target_size} pos={spec.target_pos} neg={spec.target_neg}"
            )
        return
    subprocess.run(run_simple_cmd, check=True)

    master_path = run_dir / "labels_final.csv"
    if not master_path.exists():
        raise FileNotFoundError(f"Expected master labels at {master_path}")
    master = pd.read_csv(master_path).reset_index(drop=True)
    if "label" not in master.columns:
        raise ValueError(f"Missing label column in {master_path}")

    profiles_root = run_dir / "profiles"
    profiles_root.mkdir(parents=True, exist_ok=True)
    export_ditto = not args.no_export_ditto
    manifest: Dict[str, Any] = {
        "benchmark": args.benchmark,
        "run_dir": str(run_dir),
        "source_train_file": str(master_path),
        "canonical_left_csv": str(left_canonical),
        "canonical_right_csv": str(right_canonical),
        "profiles": {},
    }

    for spec in profiles:
        subset = _subset_from_master(master, target_pos=spec.target_pos, target_neg=spec.target_neg)
        profile_dir = profiles_root / spec.name
        profile_dir.mkdir(parents=True, exist_ok=True)
        labels_csv = profile_dir / "active_labels_latest.csv"
        subset.to_csv(labels_csv, index=False)
        # Keep naming consistent with the existing pipeline outputs.
        subset.to_csv(profile_dir / "labels_final.csv", index=False)

        pos = int((subset["label"].astype(str).str.upper() == "TRUE").sum())
        neg = int((subset["label"].astype(str).str.upper() == "FALSE").sum())
        profile_meta: Dict[str, Any] = {
            "target_total": int(spec.target_size),
            "target_pos": int(spec.target_pos),
            "target_neg": int(spec.target_neg),
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
            subprocess.run(convert_cmd, check=True)
            profile_meta["ditto_train_json_gz"] = str(ditto_json_gz)
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
