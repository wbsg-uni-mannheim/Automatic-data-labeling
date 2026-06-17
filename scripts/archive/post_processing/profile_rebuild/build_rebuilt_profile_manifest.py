#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "labeling" / "benchmarks_active.yaml"
DEFAULT_SOURCE_DIR = ROOT / "generated_labels" / "abt_ditto_active_labelling"
DEFAULT_REBUILT_DIR = ROOT / "generated_labels" / "abt_ditto_active_labelling_rebuilt_gpt-5-mini_agent_precision"
DEFAULT_BENCHMARK = "abt-buy"
PRICING_SOURCE_URL = "https://platform.openai.com/docs/models/gpt-5-mini/"
GPT5_MINI_PRICING = {"input": 0.25, "output": 2.00}


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return payload


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _count_csv_labels(path: Path) -> Tuple[int, int, int]:
    df = pd.read_csv(path)
    labels = df["label"].astype(bool)
    total = int(len(df))
    pos = int(labels.sum())
    neg = int(total - pos)
    return total, pos, neg


def _materialize_canonical_csv(src_csv: Path, out_csv: Path, fields: Dict[str, str], id_col: str) -> None:
    src = pd.read_csv(src_csv).reset_index(drop=True)
    if id_col not in src.columns:
        raise ValueError(f"Missing id column '{id_col}' in {src_csv}")
    out = pd.DataFrame()
    out["id"] = src[id_col].astype(str)
    for output_name, source_name in fields.items():
        if source_name in src.columns:
            out[output_name] = src[source_name]
        else:
            out[output_name] = ""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> Dict[str, Any]:
    input_cost = (prompt_tokens / 1_000_000.0) * GPT5_MINI_PRICING["input"]
    output_cost = (completion_tokens / 1_000_000.0) * GPT5_MINI_PRICING["output"]
    return {
        "model": model,
        "pricing_source_url": PRICING_SOURCE_URL,
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "cached_input_tokens_assumed": 0,
        "pricing_mode": "standard",
        "available": True,
        "input_usd_per_million": float(GPT5_MINI_PRICING["input"]),
        "output_usd_per_million": float(GPT5_MINI_PRICING["output"]),
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(input_cost + output_cost, 6),
    }


def _profile_target_meta(profile_name: str, original_profile_dir: Path, rebuilt_profile_dir: Path) -> Dict[str, int]:
    original_total, original_pos, original_neg = _count_csv_labels(original_profile_dir / "active_labels_latest.csv")
    rebuilt_total, rebuilt_pos, rebuilt_neg = _count_csv_labels(rebuilt_profile_dir / "active_labels_latest.csv")
    return {
        "target_total": int(original_total),
        "target_pos": int(original_pos),
        "target_neg": int(original_neg),
        "actual_total": int(rebuilt_total),
        "actual_pos": int(rebuilt_pos),
        "actual_neg": int(rebuilt_neg),
    }


def _profile_train_file(profile_dir: Path) -> Path:
    matches = sorted(profile_dir.glob("*train.json.gz"))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one train json.gz in {profile_dir}, found {len(matches)}")
    return matches[0]


def _to_repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a profile manifest for rebuilt generated labels.")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--rebuilt-dir", default=str(DEFAULT_REBUILT_DIR))
    args = parser.parse_args()

    benchmark = str(args.benchmark)
    config_path = Path(args.config).resolve()
    source_dir = Path(args.source_dir).resolve()
    rebuilt_dir = Path(args.rebuilt_dir).resolve()

    config = _load_yaml(config_path)
    defaults = dict(config.get("defaults") or {})
    benchmarks = dict(config.get("benchmarks") or {})
    benchmark_cfg = dict(benchmarks.get(benchmark) or {})
    if not benchmark_cfg:
        raise ValueError(f"Benchmark not found in config: {benchmark}")

    left_fields = dict(benchmark_cfg.get("fields") or {})
    right_fields = dict(benchmark_cfg.get("fields") or {})
    ditto_fields = list(left_fields.keys())
    id_col = str(benchmark_cfg.get("id_col", "id"))

    canonical_dir = rebuilt_dir / "canonical"
    left_canonical = canonical_dir / "left.csv"
    right_canonical = canonical_dir / "right.csv"
    _materialize_canonical_csv(ROOT / benchmark_cfg["left_csv"], left_canonical, left_fields, id_col)
    _materialize_canonical_csv(ROOT / benchmark_cfg["right_csv"], right_canonical, right_fields, id_col)

    master_profile_dir = rebuilt_dir / "all_plus20random"
    root_active = rebuilt_dir / "active_labels_latest.csv"
    root_final = rebuilt_dir / "labels_final.csv"
    shutil.copy2(master_profile_dir / "active_labels_latest.csv", root_active)
    shutil.copy2(master_profile_dir / "labels_final.csv", root_final)

    relabel_summary_path = master_profile_dir / "relabel_summary__gpt-5-mini__agent-precision-system-prompt.json"
    relabel_summary = _read_json(relabel_summary_path)
    labeling_cost = _estimate_cost(
        str(relabel_summary.get("model", "gpt-5-mini")),
        int(relabel_summary.get("prompt_tokens", 0) or 0),
        int(relabel_summary.get("completion_tokens", 0) or 0),
    )

    random_fraction = float(defaults.get("random_profile_fraction", 0.20))
    random_fraction_pct = max(1, int(round(random_fraction * 100)))
    random_profile_settings = {
        "enabled": bool(defaults.get("random_profile_enabled", True)),
        "fraction": random_fraction,
        "seed": int(defaults.get("random_profile_seed", 42)),
        "model": str(defaults.get("random_profile_model", "gpt-5.2")),
    }

    manifest: Dict[str, Any] = {
        "benchmark": benchmark,
        "run_dir": _to_repo_relative(rebuilt_dir),
        "runner_script": "scripts/labeling/relabel_generated_labels_realtime.py",
        "run_summary_json": _to_repo_relative(rebuilt_dir / "rebuild_summary.json"),
        "relabel_summary_json": _to_repo_relative(relabel_summary_path),
        "labeling_cost": labeling_cost,
        "source_train_file": _to_repo_relative(root_final),
        "source_all_examples_file": _to_repo_relative(root_active),
        "canonical_left_csv": _to_repo_relative(left_canonical),
        "canonical_right_csv": _to_repo_relative(right_canonical),
        "left_fields": left_fields,
        "right_fields": right_fields,
        "ditto_fields": ditto_fields,
        "random_profile_settings": random_profile_settings,
        "profiles": {},
        "random_profile_usage": {
            "available": False,
            "reason": "Original random profile token usage was not preserved in generated_labels/abt_ditto_active_labelling.",
        },
        "random_profile_cost": {
            "model": str(defaults.get("random_profile_model", "gpt-5.2")),
            "pricing_source_url": "https://platform.openai.com/docs/models/gpt-5.2/",
            "available": False,
            "reason": "Original random profile token usage was not preserved in generated_labels/abt_ditto_active_labelling.",
        },
        "random_profile_meta": {
            "candidate_count": 600,
            "labeled_count": 600,
            "inferred": True,
        },
    }

    profile_cfg = dict(benchmark_cfg.get("profiles") or {})
    order = ["small", "small_plus20random", "medium", "medium_plus20random", "all_plus20random"]
    for profile_name in order:
        rebuilt_profile_dir = rebuilt_dir / profile_name
        original_profile_dir = source_dir / profile_name
        train_file = _profile_train_file(rebuilt_profile_dir)
        labels_csv = rebuilt_profile_dir / "active_labels_latest.csv"
        base_meta = _profile_target_meta(profile_name, original_profile_dir, rebuilt_profile_dir)

        if profile_name in {"small", "medium"}:
            cfg = dict(profile_cfg.get(profile_name) or {})
            profile_meta: Dict[str, Any] = {
                "all_examples": False,
                "target_total": int(cfg.get("target_total", base_meta["target_total"])),
                "target_pos": int(cfg.get("target_pos", base_meta["target_pos"])),
                "target_neg": int(cfg.get("target_neg", base_meta["target_neg"])),
            }
        elif profile_name == "all_plus20random":
            profile_meta = {
                "all_examples": True,
                "base_profile": "all",
                "random_fraction": random_fraction,
                "random_additions": 600,
                "target_total": base_meta["target_total"],
                "target_pos": base_meta["target_pos"],
                "target_neg": base_meta["target_neg"],
                "shared_random_pool": True,
                "shared_random_model": str(defaults.get("random_profile_model", "gpt-5.2")),
            }
        else:
            base_name = profile_name.replace(f"_plus{random_fraction_pct}random", "")
            profile_meta = {
                "all_examples": False,
                "base_profile": base_name,
                "random_fraction": random_fraction,
                "random_additions": base_meta["target_total"] - _count_csv_labels(source_dir / base_name / "active_labels_latest.csv")[0],
                "target_total": base_meta["target_total"],
                "target_pos": base_meta["target_pos"],
                "target_neg": base_meta["target_neg"],
                "shared_random_pool": True,
                "shared_random_model": str(defaults.get("random_profile_model", "gpt-5.2")),
            }

        profile_meta.update(
            {
                "actual_total": base_meta["actual_total"],
                "actual_pos": base_meta["actual_pos"],
                "actual_neg": base_meta["actual_neg"],
                "labels_csv": _to_repo_relative(labels_csv),
                "ditto_train_json_gz": _to_repo_relative(train_file),
                "ditto_fields": ditto_fields,
            }
        )
        manifest["profiles"][profile_name] = profile_meta

    manifest_path = rebuilt_dir / "profile_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest_path": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
