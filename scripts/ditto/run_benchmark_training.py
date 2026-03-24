#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from third_party.ditto_modern.data import write_wdc_json_gz

TRAIN_CONFIG_KEYS: Sequence[str] = (
    "model_name",
    "batch_size",
    "max_len",
    "epochs",
    "lr",
    "weight_decay",
    "warmup_ratio",
    "grad_accum_steps",
    "early_stopping_patience",
    "seed",
    "num_workers",
    "fp16",
    "da",
    "alpha_aug",
    "summarize",
    "dk",
    "spacy_model",
    "max_field_len",
)

POS_LABELS = {"1", "TRUE", "T", "YES", "Y"}
NEG_LABELS = {"0", "FALSE", "F", "NO", "N"}


def _coerce_mapping(value: Any, name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _coerce_str_list(value: Any, name: str) -> List[str]:
    if isinstance(value, str):
        parts = [x.strip() for x in value.split(",")]
    elif isinstance(value, (list, tuple)):
        parts = [str(x).strip() for x in value]
    else:
        raise ValueError(f"{name} must be a list or comma-separated string")
    out = [x for x in parts if x]
    if not out:
        raise ValueError(f"{name} must not be empty")
    return out


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt")
    return path.open("r")


def _read_json_table(path: Path) -> pd.DataFrame:
    first: str | None = None
    second: str | None = None
    with _open_text(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if first is None:
                first = s
                continue
            second = s
            break
    if first is None:
        return pd.DataFrame()

    if second is not None:
        try:
            o1 = json.loads(first)
            o2 = json.loads(second)
            if isinstance(o1, dict) and isinstance(o2, dict):
                rows: List[Dict[str, Any]] = []
                with _open_text(path) as f:
                    for line in f:
                        s = line.strip()
                        if not s:
                            continue
                        rows.append(json.loads(s))
                return pd.DataFrame(rows)
        except Exception:
            pass

    with _open_text(path) as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        if payload and all(isinstance(v, dict) for v in payload.values()):
            return pd.DataFrame(payload)
        return pd.DataFrame([payload])
    raise ValueError(f"Unsupported JSON payload in {path}")


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    lower = path.name.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    if lower.endswith(".json") or lower.endswith(".json.gz"):
        return _read_json_table(path)
    raise ValueError(f"Unsupported input extension for {path}")


def _normalize_model_name(name: str) -> str:
    raw = str(name).strip()
    if raw.lower() == "robert-base":
        return "roberta-base"
    return raw


def _normalize_label(v: object) -> int:
    if pd.isna(v):
        raise ValueError("Label value is missing")
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        iv = int(v)
        if iv in {0, 1}:
            return iv
        raise ValueError(f"Unsupported numeric label value: {v}")
    s = str(v).strip().upper()
    if s in POS_LABELS:
        return 1
    if s in NEG_LABELS:
        return 0
    raise ValueError(f"Unsupported label value: {v}")


def _normalize_pair_id(df: pd.DataFrame) -> pd.Series:
    if "pair_id" in df.columns:
        out = df["pair_id"].astype(str).fillna("").str.strip()
        if (out != "").all():
            return out
    if "id_left" in df.columns and "id_right" in df.columns:
        left = df["id_left"].astype(str).fillna("").str.strip()
        right = df["id_right"].astype(str).fillna("").str.strip()
        return left + "#" + right
    return pd.Series([f"idx-{i}" for i in range(len(df))], index=df.index, dtype="object")


def _resolve_field_source(df: pd.DataFrame, field: str, aliases: Dict[str, List[str]]) -> str:
    candidates = [field] + aliases.get(field, [])
    seen = set()
    for cand in candidates:
        c = str(cand).strip()
        if not c or c in seen:
            continue
        seen.add(c)
        if f"{c}_left" in df.columns and f"{c}_right" in df.columns:
            return c
    raise ValueError(
        f"Could not resolve field '{field}' to source columns. "
        f"Tried: {[x for x in candidates if str(x).strip()]}"
    )


def _normalize_split_df(
    df: pd.DataFrame,
    *,
    fields: Sequence[str],
    field_aliases: Dict[str, List[str]],
    split_name: str,
    benchmark: str,
) -> pd.DataFrame:
    if "label" not in df.columns:
        raise ValueError(f"{benchmark}:{split_name} is missing label column")

    out = pd.DataFrame(index=df.index)
    out["pair_id"] = _normalize_pair_id(df)
    out["label"] = df["label"].map(_normalize_label).astype(int)
    for field in fields:
        src = _resolve_field_source(df, field=field, aliases=field_aliases)
        out[f"{field}_left"] = df[f"{src}_left"]
        out[f"{field}_right"] = df[f"{src}_right"]
    out = out.fillna("")
    return out.reset_index(drop=True)


def _build_valid_from_pair_ids(valid_df: pd.DataFrame, train_df: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    if "pair_id" not in valid_df.columns:
        raise ValueError(f"{benchmark}: valid_pair_id_only=true requires pair_id column in validation file")
    valid_ids = pd.DataFrame({"pair_id": valid_df["pair_id"].astype(str).str.strip()})
    valid_ids = valid_ids[valid_ids["pair_id"] != ""].copy()
    valid_ids = valid_ids.drop_duplicates(subset=["pair_id"], keep="first").reset_index(drop=True)
    merged = valid_ids.merge(train_df, on="pair_id", how="left")
    missing = int(merged["label"].isna().sum())
    if missing > 0:
        raise ValueError(f"{benchmark}: {missing} validation pair_ids are not present in train split")
    return merged.reset_index(drop=True)


def _resolve_valid_lookup_df(
    *,
    benchmark: str,
    bcfg: Dict[str, Any],
    fields: Sequence[str],
    field_aliases: Dict[str, List[str]],
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    lookup_path_raw = str(bcfg.get("valid_lookup_train", "")).strip()
    if not lookup_path_raw:
        return train_df
    lookup_raw = _read_table(Path(lookup_path_raw))
    return _normalize_split_df(
        lookup_raw,
        fields=fields,
        field_aliases=field_aliases,
        split_name="valid_lookup_train",
        benchmark=benchmark,
    )


def _label_stats(df: pd.DataFrame) -> Dict[str, int]:
    pos = int((df["label"] == 1).sum())
    neg = int((df["label"] == 0).sum())
    return {"rows": int(len(df)), "pos": pos, "neg": neg}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _prepare_train_config(defaults: Dict[str, Any], benchmark_cfg: Dict[str, Any], fields: Sequence[str]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    overrides = _coerce_mapping(benchmark_cfg.get("training"), "benchmarks.*.training")
    for key in TRAIN_CONFIG_KEYS:
        if key in overrides:
            cfg[key] = overrides[key]
        elif key in defaults:
            cfg[key] = defaults[key]
    cfg["model_name"] = _normalize_model_name(str(cfg.get("model_name", "roberta-base")))
    cfg["fields"] = ",".join(fields)
    return cfg


def _flatten_summary_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    flat: List[Dict[str, Any]] = []
    for row in rows:
        metrics = _coerce_mapping(row.get("metrics"), "row.metrics")
        test = _coerce_mapping(metrics.get("test"), "row.metrics.test")
        flat.append(
            {
                "benchmark": row.get("benchmark"),
                "status": row.get("status"),
                "train_rows": row.get("train_rows"),
                "valid_rows": row.get("valid_rows"),
                "test_rows": row.get("test_rows"),
                "best_epoch": metrics.get("best_epoch"),
                "best_val_f1": metrics.get("best_val_f1"),
                "test_f1": test.get("f1"),
                "test_precision": test.get("precision"),
                "test_recall": test.get("recall"),
                "test_accuracy": test.get("accuracy"),
                "run_dir": row.get("run_dir"),
                "error": row.get("error"),
            }
        )
    return pd.DataFrame(flat)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Ditto baseline training across configured benchmarks")
    parser.add_argument("--config", default="configs/ditto/benchmarks_training.yaml")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--benchmarks",
        default="",
        help="Comma-separated benchmark names. Default: all from config.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    payload = yaml.safe_load(cfg_path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError("Training benchmark config must be a mapping")

    defaults = _coerce_mapping(payload.get("defaults"), "defaults")
    benchmarks_cfg = _coerce_mapping(payload.get("benchmarks"), "benchmarks")
    if not benchmarks_cfg:
        raise ValueError("No benchmarks defined in config")

    selected = [x.strip() for x in args.benchmarks.split(",") if x.strip()]
    if not selected:
        selected = list(benchmarks_cfg.keys())
    missing = [x for x in selected if x not in benchmarks_cfg]
    if missing:
        raise KeyError(f"Unknown benchmarks: {missing}. Available: {sorted(benchmarks_cfg.keys())}")

    output_root = Path(args.output_root or defaults.get("output_root", "output/ditto_benchmark_runs"))
    run_name = args.run_name or f"run_{_timestamp()}"
    run_root = output_root / run_name
    run_root.mkdir(parents=True, exist_ok=False)

    _write_json(
        run_root / "run_manifest.json",
        {
            "created_at": datetime.now().isoformat(),
            "config": str(cfg_path),
            "benchmarks": selected,
            "run_root": str(run_root),
        },
    )

    results: List[Dict[str, Any]] = []
    failures: List[str] = []

    for benchmark in selected:
        bcfg = _coerce_mapping(benchmarks_cfg.get(benchmark), f"benchmarks.{benchmark}")
        bench_dir = run_root / benchmark
        bench_dir.mkdir(parents=True, exist_ok=True)
        try:
            fields = _coerce_str_list(bcfg.get("fields"), f"benchmarks.{benchmark}.fields")
            aliases_raw = _coerce_mapping(bcfg.get("field_aliases"), f"benchmarks.{benchmark}.field_aliases")
            field_aliases: Dict[str, List[str]] = {}
            for f in fields:
                aliases = aliases_raw.get(f, [])
                if isinstance(aliases, str):
                    alias_list = [x.strip() for x in aliases.split(",") if x.strip()]
                elif isinstance(aliases, (list, tuple)):
                    alias_list = [str(x).strip() for x in aliases if str(x).strip()]
                else:
                    alias_list = []
                field_aliases[f] = alias_list

            train_path = Path(str(bcfg["train"]))
            valid_path = Path(str(bcfg["valid"]))
            test_path = Path(str(bcfg["test"]))

            train_raw = _read_table(train_path)
            train_df = _normalize_split_df(
                train_raw,
                fields=fields,
                field_aliases=field_aliases,
                split_name="train",
                benchmark=benchmark,
            )

            valid_pair_id_only = bool(bcfg.get("valid_pair_id_only", False))
            if valid_pair_id_only:
                valid_raw = _read_table(valid_path)
                valid_lookup_df = _resolve_valid_lookup_df(
                    benchmark=benchmark,
                    bcfg=bcfg,
                    fields=fields,
                    field_aliases=field_aliases,
                    train_df=train_df,
                )
                valid_df = _build_valid_from_pair_ids(valid_raw, train_df=valid_lookup_df, benchmark=benchmark)
            else:
                valid_raw = _read_table(valid_path)
                valid_df = _normalize_split_df(
                    valid_raw,
                    fields=fields,
                    field_aliases=field_aliases,
                    split_name="valid",
                    benchmark=benchmark,
                )

            test_raw = _read_table(test_path)
            test_df = _normalize_split_df(
                test_raw,
                fields=fields,
                field_aliases=field_aliases,
                split_name="test",
                benchmark=benchmark,
            )

            exclude_valid = bool(bcfg.get("exclude_valid_from_train", defaults.get("exclude_valid_from_train", True)))
            dropped_for_valid = 0
            if exclude_valid:
                before = len(train_df)
                valid_ids = set(valid_df["pair_id"].astype(str).tolist())
                train_df = train_df[~train_df["pair_id"].astype(str).isin(valid_ids)].reset_index(drop=True)
                dropped_for_valid = int(before - len(train_df))

            splits_dir = bench_dir / "splits"
            splits_dir.mkdir(parents=True, exist_ok=True)
            train_json = splits_dir / "train.json.gz"
            valid_json = splits_dir / "valid.json.gz"
            test_json = splits_dir / "test.json.gz"
            write_wdc_json_gz(train_df, train_json)
            write_wdc_json_gz(valid_df, valid_json)
            write_wdc_json_gz(test_df, test_json)

            train_cfg = _prepare_train_config(defaults, bcfg, fields=fields)
            train_cfg_path = bench_dir / "train_config.yaml"
            train_cfg_path.write_text(yaml.safe_dump(train_cfg, sort_keys=False))

            training_out_dir = bench_dir / "training_output"
            cmd = [
                sys.executable,
                "scripts/ditto/train.py",
                "--train-json-gz",
                str(train_json),
                "--val-json-gz",
                str(valid_json),
                "--test-json-gz",
                str(test_json),
                "--output-dir",
                str(training_out_dir),
                "--config",
                str(train_cfg_path),
            ]

            if args.dry_run:
                run_dir_path = ""
                metrics: Dict[str, Any] = {}
            else:
                subprocess.run(cmd, check=True)
                latest_ptr = training_out_dir / "LATEST_RUN"
                if not latest_ptr.exists():
                    raise FileNotFoundError(f"Training output missing LATEST_RUN for {benchmark}")
                run_dir_path = latest_ptr.read_text().strip()
                run_dir = Path(run_dir_path)
                metrics_path = run_dir / "metrics.json"
                metrics = _load_json(metrics_path) if metrics_path.exists() else {}
                history_path = run_dir / "history.json"
                if history_path.exists():
                    (bench_dir / "history.json").write_text(history_path.read_text())
                if metrics_path.exists():
                    (bench_dir / "metrics.json").write_text(metrics_path.read_text())

            train_stats = _label_stats(train_df)
            valid_stats = _label_stats(valid_df)
            test_stats = _label_stats(test_df)
            report = {
                "benchmark": benchmark,
                "status": "ok",
                "paths": {
                    "train_input": str(train_path),
                    "valid_input": str(valid_path),
                    "test_input": str(test_path),
                    "train_json_gz": str(train_json),
                    "valid_json_gz": str(valid_json),
                    "test_json_gz": str(test_json),
                    "train_config": str(train_cfg_path),
                },
                "fields": fields,
                "field_aliases": field_aliases,
                "exclude_valid_from_train": exclude_valid,
                "dropped_train_rows_due_to_valid": dropped_for_valid,
                "train_stats": train_stats,
                "valid_stats": valid_stats,
                "test_stats": test_stats,
                "run_dir": run_dir_path,
                "metrics": metrics,
                "command": cmd,
                "dry_run": bool(args.dry_run),
            }
            _write_json(bench_dir / "benchmark_report.json", report)
            results.append(
                {
                    "benchmark": benchmark,
                    "status": "ok",
                    "train_rows": train_stats["rows"],
                    "valid_rows": valid_stats["rows"],
                    "test_rows": test_stats["rows"],
                    "run_dir": run_dir_path,
                    "metrics": metrics,
                    "error": "",
                }
            )
            print(
                f"[{benchmark}] ok: train={train_stats['rows']} valid={valid_stats['rows']} "
                f"test={test_stats['rows']} dropped_from_train={dropped_for_valid}"
            )

        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            failures.append(f"{benchmark}: {msg}")
            traceback_txt = traceback.format_exc()
            _write_json(
                bench_dir / "benchmark_report.json",
                {
                    "benchmark": benchmark,
                    "status": "failed",
                    "error": msg,
                    "traceback": traceback_txt,
                },
            )
            results.append(
                {
                    "benchmark": benchmark,
                    "status": "failed",
                    "train_rows": None,
                    "valid_rows": None,
                    "test_rows": None,
                    "run_dir": "",
                    "metrics": {},
                    "error": msg,
                }
            )
            print(f"[{benchmark}] failed: {msg}", file=sys.stderr)

    summary = {
        "run_root": str(run_root),
        "config": str(cfg_path),
        "benchmarks": selected,
        "results": results,
        "failures": failures,
        "dry_run": bool(args.dry_run),
        "created_at": datetime.now().isoformat(),
    }
    _write_json(run_root / "summary.json", summary)
    summary_df = _flatten_summary_rows(results)
    summary_df.to_csv(run_root / "summary.csv", index=False)

    print(f"Saved summary: {run_root / 'summary.json'}")
    print(f"Saved summary: {run_root / 'summary.csv'}")
    if failures:
        raise RuntimeError(f"{len(failures)} benchmark runs failed. See summary.json for details.")


if __name__ == "__main__":
    main()
