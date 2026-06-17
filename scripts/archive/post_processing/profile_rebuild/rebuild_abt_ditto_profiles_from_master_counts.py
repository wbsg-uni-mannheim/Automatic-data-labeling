#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = ROOT / "generated_labels" / "abt_ditto_active_labelling_rebuilt_gpt-5-mini_agent_precision"
DEFAULT_MASTER_PROFILE = "all_plus20random"
DEFAULT_MASTER_ACTIVE_CSV = (
    DEFAULT_SOURCE_DIR / DEFAULT_MASTER_PROFILE / "active_labels_latest__recall-then-skeptic__gpt-5-mini.csv"
)
DEFAULT_MASTER_FINAL_CSV = (
    DEFAULT_SOURCE_DIR / DEFAULT_MASTER_PROFILE / "labels_final__recall-then-skeptic__gpt-5-mini.csv"
)
DEFAULT_MASTER_TRAIN_GZ = (
    DEFAULT_SOURCE_DIR
    / DEFAULT_MASTER_PROFILE
    / "active_labels_latest_abt-buy_all_plus20random_train__recall-then-skeptic__gpt-5-mini.json.gz"
)
DEFAULT_OUTPUT_DIR = ROOT / "generated_labels" / "abt_ditto_active_labelling_rebuilt_gpt-5-mini_agent_precision_recall_then_skeptic"


def _load_train_records(path: Path) -> List[dict]:
    rows: List[dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_train_records(path: Path, rows: List[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _count_bool_labels(df: pd.DataFrame) -> Tuple[int, int]:
    labels = df["label"].astype(bool)
    pos = int(labels.sum())
    neg = int(len(df) - pos)
    return pos, neg


def _subset_from_master(
    master_df: pd.DataFrame,
    target_total: int,
    preferred_pos: int,
    preferred_neg: int,
) -> Tuple[pd.DataFrame, List[int]]:
    labels = master_df["label"].astype(bool)
    pos_idx = labels[labels].index.to_list()
    neg_idx = labels[~labels].index.to_list()
    take_pos = min(int(preferred_pos), len(pos_idx))
    take_neg = min(int(preferred_neg), len(neg_idx))
    keep = list(pos_idx[:take_pos]) + list(neg_idx[:take_neg])

    remaining_needed = int(target_total) - len(keep)
    if remaining_needed > 0:
        extra_pos = pos_idx[take_pos:]
        extra_neg = neg_idx[take_neg:]
        extra_pool = sorted(extra_pos + extra_neg)
        keep.extend(extra_pool[:remaining_needed])

    keep = sorted(set(keep))
    if len(keep) < int(target_total):
        raise RuntimeError(
            f"Master does not contain enough rows for target_total={target_total}. "
            f"Available total={len(master_df)} selected={len(keep)}."
        )
    return master_df.iloc[keep].copy().reset_index(drop=True), keep


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild ABT Ditto profiles from a master label set by matching per-profile pos/neg counts.")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--master-active-csv", default=str(DEFAULT_MASTER_ACTIVE_CSV))
    parser.add_argument("--master-final-csv", default=str(DEFAULT_MASTER_FINAL_CSV))
    parser.add_argument("--master-train-gz", default=str(DEFAULT_MASTER_TRAIN_GZ))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    master_active_csv = Path(args.master_active_csv)
    master_final_csv = Path(args.master_final_csv)
    master_train_gz = Path(args.master_train_gz)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    master_active_df = pd.read_csv(master_active_csv).reset_index(drop=True)
    master_final_df = pd.read_csv(master_final_csv).reset_index(drop=True)
    master_train_rows = _load_train_records(master_train_gz)
    if len(master_active_df) != len(master_final_df) or len(master_active_df) != len(master_train_rows):
        raise ValueError("Master active/final/train row counts do not match")

    summary: Dict[str, object] = {
        "source_dir": str(source_dir),
        "master_active_csv": str(master_active_csv),
        "master_final_csv": str(master_final_csv),
        "master_train_gz": str(master_train_gz),
        "output_dir": str(output_dir),
        "profiles": {},
    }

    for profile_dir in sorted(
        p
        for p in source_dir.iterdir()
        if p.is_dir() and (p / "active_labels_latest.csv").exists() and (p / "labels_final.csv").exists()
    ):
        profile_name = profile_dir.name
        out_profile_dir = output_dir / profile_name
        out_profile_dir.mkdir(parents=True, exist_ok=True)

        if profile_name == "all_plus20random":
            active_subset = master_active_df.copy().reset_index(drop=True)
            final_subset = master_final_df.copy().reset_index(drop=True)
            train_subset = [dict(row) for row in master_train_rows]
        else:
            source_active_df = pd.read_csv(profile_dir / "active_labels_latest.csv").reset_index(drop=True)
            target_pos, target_neg = _count_bool_labels(source_active_df)
            target_total = int(len(source_active_df))
            active_subset, keep_indices = _subset_from_master(
                master_active_df,
                target_total=target_total,
                preferred_pos=target_pos,
                preferred_neg=target_neg,
            )
            final_subset, final_keep_indices = _subset_from_master(
                master_final_df,
                target_total=target_total,
                preferred_pos=target_pos,
                preferred_neg=target_neg,
            )
            if keep_indices != final_keep_indices:
                raise RuntimeError(f"Master active/final subset indices diverged for profile {profile_name}")
            train_subset = []
            for row_idx in keep_indices:
                train_subset.append(dict(master_train_rows[int(row_idx)]))

        active_out = out_profile_dir / "active_labels_latest.csv"
        final_out = out_profile_dir / "labels_final.csv"
        train_out = out_profile_dir / next(profile_dir.glob("*train.json.gz")).name
        active_subset.to_csv(active_out, index=False)
        final_subset.to_csv(final_out, index=False)
        _write_train_records(train_out, train_subset)

        if profile_name == "all_plus20random":
            for extra_name in [
                "active_labels_latest__recall-then-skeptic__gpt-5-mini.csv",
                "labels_final__recall-then-skeptic__gpt-5-mini.csv",
                "active_labels_latest_abt-buy_all_plus20random_train__recall-then-skeptic__gpt-5-mini.json.gz",
                "recall-then-skeptic__results__gpt-5-mini.csv",
                "recall-then-skeptic__results__gpt-5-mini.jsonl",
                "recall-then-skeptic__summary__gpt-5-mini.json",
                "relabel_summary__gpt-5-mini__agent-precision-system-prompt.json",
            ]:
                src = profile_dir / extra_name
                if src.exists():
                    shutil.copy2(src, out_profile_dir / extra_name)

        pos, neg = _count_bool_labels(active_subset)
        summary["profiles"][profile_name] = {
            "row_count": int(len(active_subset)),
            "positive_labels": int(pos),
            "negative_labels": int(neg),
            "active_labels_latest": str(active_out),
            "labels_final": str(final_out),
            "train_json_gz": str(train_out),
        }

    (output_dir / "rebuild_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
