#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, SequentialSampler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from third_party.ditto_modern.data import PairDataset, load_wdc_json_gz, wdc_to_pair_examples
from third_party.ditto_modern.model import load_model, load_tokenizer
from third_party.ditto_modern.runtime import resolve_run_dir


CONFIG_PATH = ROOT / "configs" / "labeling" / "benchmarks_active.yaml"
AUTO_LABEL_V1_ROOT = ROOT / "data" / "auto_label_v1"
BASELINE_ROOT = ROOT / "output" / "baseline"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "noisy_autolabels_ditto_temperature"
DEFAULT_TOP_RATES = "0.01,0.02,0.05,0.10"

POS_LABELS = {"1", "TRUE", "T", "YES", "Y"}
NEG_LABELS = {"0", "FALSE", "F", "NO", "N"}


@dataclass(frozen=True)
class GeneratedSet:
    benchmark: str
    profile: str
    labels_csv: Path


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


def _prepare_generated_df(labels_csv: Path, benchmark_cfg: Dict[str, Any]) -> pd.DataFrame:
    left_source = pd.read_csv(ROOT / benchmark_cfg["left_csv"])
    right_source = pd.read_csv(ROOT / benchmark_cfg["right_csv"])
    left_source["id"] = left_source["id"].astype(str).str.strip()
    right_source["id"] = right_source["id"].astype(str).str.strip()
    left_clusters = left_source[["id", "cluster_id"]].rename(columns={"id": "id_left", "cluster_id": "cluster_id_left"})
    right_clusters = right_source[["id", "cluster_id"]].rename(columns={"id": "id_right", "cluster_id": "cluster_id_right"})
    out = _normalize_pair_frame(pd.read_csv(labels_csv))
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


def _parse_rates(text: str) -> List[float]:
    values: List[float] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = float(chunk)
        if value <= 0 or value >= 1:
            raise ValueError(f"Top rate must be between 0 and 1, got {value}")
        values.append(value)
    if not values:
        raise ValueError("Need at least one top-rate")
    return values


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _predict_logits(model, tokenizer, examples, max_len: int, batch_size: int, device: torch.device) -> np.ndarray:
    dataset = PairDataset(examples, tokenizer=tokenizer, max_len=max_len, da=None)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        num_workers=0,
        pin_memory=False,
        collate_fn=PairDataset.pad,
    )
    model.eval()
    rows: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x, att, _y, _idxs, _weights = batch
            x = x.to(device)
            att = att.to(device)
            logits = model(input_ids=x, attention_mask=att)
            rows.append(logits.detach().cpu().numpy())
    return np.concatenate(rows, axis=0) if rows else np.zeros((0, 2), dtype=np.float32)


def _fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    if logits.shape[0] == 0:
        return 1.0
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    temperature = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        t = torch.clamp(temperature, min=1e-3)
        loss = torch.nn.functional.cross_entropy(logits_t / t, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.clamp(temperature.detach(), min=1e-3).item())


def _softmax_pos(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / max(temperature, 1e-6)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    probs = np.exp(scaled)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    return probs[:, 1]


def _find_checkpoint_dir(benchmark: str) -> Optional[Path]:
    root = BASELINE_ROOT / benchmark / "training_output"
    run_dirs = sorted(root.glob("run_*"))
    if not run_dirs:
        return None
    latest = run_dirs[-1]
    best_dir = latest / "checkpoints" / "best"
    return best_dir if best_dir.exists() else None


def _train_checkpoint_if_needed(
    benchmark: str,
    benchmark_cfg: Dict[str, Any],
    output_root: Path,
    *,
    epochs: Optional[int],
    batch_size: Optional[int],
) -> Path:
    split_dir = BASELINE_ROOT / benchmark / "splits"
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing baseline splits for {benchmark}: {split_dir}")
    cmd = [
        sys.executable,
        "scripts/ditto/train.py",
        "--train-json-gz",
        str(split_dir / "train.json.gz"),
        "--val-json-gz",
        str(split_dir / "valid.json.gz"),
        "--test-json-gz",
        str(split_dir / "test.json.gz"),
        "--output-dir",
        str(output_root),
        "--fields",
        ",".join(str(k) for k in (benchmark_cfg.get("fields") or {}).keys()),
        "--no-fp16",
    ]
    if epochs is not None:
        cmd.extend(["--epochs", str(epochs)])
    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])
    subprocess.run(cmd, cwd=ROOT, check=True)
    run_dir = resolve_run_dir(output_root)
    best_dir = run_dir / "checkpoints" / "best"
    if not best_dir.exists():
        raise FileNotFoundError(f"Training finished but checkpoint missing: {best_dir}")
    return best_dir


def _budget_rows(scored: pd.DataFrame, top_rates: Sequence[float]) -> List[Dict[str, Any]]:
    proxy_total = int(scored["cluster_proxy_error"].sum())
    proxy_fp_total = int(scored["cluster_proxy_fp"].sum())
    proxy_fn_total = int(scored["cluster_proxy_fn"].sum())
    ranked = scored.sort_values(["suspicion_score", "temperature_scaled_prob"], ascending=[False, True]).reset_index(drop=True)
    rows: List[Dict[str, Any]] = []
    for rate in top_rates:
        top_n = max(1, int(np.ceil(rate * len(ranked))))
        flagged = ranked.head(top_n)
        found_total = int(flagged["cluster_proxy_error"].sum())
        found_fp = int(flagged["cluster_proxy_fp"].sum())
        found_fn = int(flagged["cluster_proxy_fn"].sum())
        rows.append(
            {
                "budget_value": float(rate),
                "flagged_pairs": top_n,
                "proxy_errors_found": found_total,
                "proxy_false_positive_found": found_fp,
                "proxy_false_negative_found": found_fn,
                "proxy_precision": float(found_total / top_n) if top_n else None,
                "proxy_recall": float(found_total / proxy_total) if proxy_total else None,
                "proxy_fp_recall": float(found_fp / proxy_fp_total) if proxy_fp_total else None,
                "proxy_fn_recall": float(found_fn / proxy_fn_total) if proxy_fn_total else None,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect noisy auto-labels with Ditto confidence + temperature scaling.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-rates", default=DEFAULT_TOP_RATES)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-if-missing", action="store_true")
    parser.add_argument("--train-epochs", type=int, default=None)
    parser.add_argument("--benchmarks", default=None, help="Optional comma-separated subset")
    args = parser.parse_args()

    config = _load_yaml(Path(args.config))
    top_rates = _parse_rates(args.top_rates)
    generated_sets = _discover_generated_sets(config)
    if args.benchmarks:
        wanted = {x.strip() for x in args.benchmarks.split(",") if x.strip()}
        generated_sets = [gs for gs in generated_sets if gs.benchmark in wanted]
    if not generated_sets:
        raise RuntimeError("No generated sets selected")

    output_dir = Path(args.output_dir)
    candidates_dir = output_dir / "candidates"
    models_dir = output_dir / "trained_models"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    device = _device()
    summary_rows: List[Dict[str, Any]] = []
    budget_rows: List[Dict[str, Any]] = []
    top_candidates: List[pd.DataFrame] = []
    skipped: List[Dict[str, Any]] = []

    checkpoint_cache: Dict[str, Path] = {}
    temperature_cache: Dict[str, float] = {}
    tokenizer_cache: Dict[str, Any] = {}
    model_cache: Dict[str, Any] = {}
    max_len_cache: Dict[str, int] = {}

    for generated in generated_sets:
        benchmark_cfg = (config.get("benchmarks") or {}).get(generated.benchmark) or {}
        checkpoint_dir = checkpoint_cache.get(generated.benchmark) or _find_checkpoint_dir(generated.benchmark)
        if checkpoint_dir is None and args.train_if_missing:
            checkpoint_dir = _train_checkpoint_if_needed(
                generated.benchmark,
                benchmark_cfg,
                models_dir / generated.benchmark / "training_output",
                epochs=args.train_epochs,
                batch_size=args.batch_size,
            )
        if checkpoint_dir is None:
            skipped.append(
                {
                    "benchmark": generated.benchmark,
                    "profile": generated.profile,
                    "reason": "missing_checkpoint",
                }
            )
            continue
        checkpoint_cache[generated.benchmark] = checkpoint_dir

        if generated.benchmark not in model_cache:
            model = load_model(str(checkpoint_dir))
            model.to(device)
            tokenizer = load_tokenizer(str(checkpoint_dir))
            cfg_path = checkpoint_dir.parents[1] / "config.json"
            train_cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
            model_cache[generated.benchmark] = model
            tokenizer_cache[generated.benchmark] = tokenizer
            max_len_cache[generated.benchmark] = int(train_cfg.get("max_len", 256))
        model = model_cache[generated.benchmark]
        tokenizer = tokenizer_cache[generated.benchmark]
        max_len = max_len_cache[generated.benchmark]

        if generated.benchmark not in temperature_cache:
            split_dir = BASELINE_ROOT / generated.benchmark / "splits"
            valid_df = load_wdc_json_gz(split_dir / "valid.json.gz")
            fields = list((benchmark_cfg.get("fields") or {}).keys())
            valid_examples = wdc_to_pair_examples(valid_df, fields=fields, max_field_len=350)
            valid_logits = _predict_logits(model, tokenizer, valid_examples, max_len=max_len, batch_size=args.batch_size, device=device)
            temperature_cache[generated.benchmark] = _fit_temperature(valid_logits, valid_df["label"].astype(int).to_numpy())
        temperature = temperature_cache[generated.benchmark]

        generated_df = _prepare_generated_df(generated.labels_csv, benchmark_cfg)
        fields = list((benchmark_cfg.get("fields") or {}).keys())
        generated_examples = wdc_to_pair_examples(generated_df.fillna(""), fields=fields, max_field_len=350)
        logits = _predict_logits(model, tokenizer, generated_examples, max_len=max_len, batch_size=args.batch_size, device=device)
        probs = _softmax_pos(logits, temperature)

        scored = generated_df.copy()
        scored["temperature"] = temperature
        scored["temperature_scaled_prob"] = probs
        scored["pred_label"] = (scored["temperature_scaled_prob"] >= 0.5).astype(int)
        scored["self_confidence"] = np.where(scored["label"] == 1, scored["temperature_scaled_prob"], 1.0 - scored["temperature_scaled_prob"])
        scored["suspicion_score"] = 1.0 - scored["self_confidence"]
        scored["margin"] = np.abs(scored["temperature_scaled_prob"] - 0.5)

        budget = _budget_rows(scored, top_rates)
        for row in budget:
            row["benchmark"] = generated.benchmark
            row["profile"] = generated.profile
        budget_rows.extend(budget)

        proxy_total = int(scored["cluster_proxy_error"].sum())
        proxy_fp_total = int(scored["cluster_proxy_fp"].sum())
        proxy_fn_total = int(scored["cluster_proxy_fn"].sum())
        summary_rows.append(
            {
                "benchmark": generated.benchmark,
                "profile": generated.profile,
                "labels_csv": str(generated.labels_csv.relative_to(ROOT)),
                "checkpoint_dir": str(checkpoint_dir.relative_to(ROOT)),
                "rows_total": int(len(scored)),
                "temperature": temperature,
                "cluster_proxy_errors": proxy_total,
                "cluster_proxy_false_positive": proxy_fp_total,
                "cluster_proxy_false_negative": proxy_fn_total,
                "mean_self_confidence": float(scored["self_confidence"].mean()),
                "mean_suspicion_score": float(scored["suspicion_score"].mean()),
            }
        )

        scored = scored.sort_values(["suspicion_score", "temperature_scaled_prob"], ascending=[False, True]).reset_index(drop=True)
        out_path = candidates_dir / f"{generated.benchmark}_{generated.profile}_ditto_temperature_candidates.csv.gz"
        scored.to_csv(out_path, index=False, compression="gzip")
        top_candidates.append(scored.head(200).assign(source_file=out_path.name))

    summary_json = output_dir / "summary.json"
    summary_xlsx = output_dir / "summary.xlsx"
    summary_json.write_text(
        json.dumps(
            {
                "method": "ditto_temperature_scaling",
                "device": str(device),
                "top_rates": top_rates,
                "summary": summary_rows,
                "budgets": budget_rows,
                "skipped": skipped,
            },
            indent=2,
        )
    )
    with pd.ExcelWriter(summary_xlsx) as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(budget_rows).to_excel(writer, sheet_name="budget_metrics", index=False)
        pd.DataFrame(top_candidates).to_excel(writer, sheet_name="top_candidates", index=False)
        pd.DataFrame(skipped).to_excel(writer, sheet_name="skipped", index=False)

    print(summary_xlsx)
    print(summary_json)
    print(candidates_dir)


if __name__ == "__main__":
    main()
