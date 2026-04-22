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
DEFAULT_SOURCE_DIR = ROOT / "generated_labels" / "abt_ditto_active_labelling"
DEFAULT_MASTER_PROFILE = "all_plus20random"
DEFAULT_MASTER_RELABEL_CSV = (
    DEFAULT_SOURCE_DIR
    / DEFAULT_MASTER_PROFILE
    / "active_labels_latest__relabeled__gpt-5-mini__agent-precision-system-prompt.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "generated_labels" / "abt_ditto_active_labelling_rebuilt_gpt-5-mini_agent_precision"


def _pair_key(id1: str, id2: str, rid1: str, rid2: str) -> Tuple[str, str, str, str]:
    return (str(id1), str(id2), str(rid1), str(rid2))


def _csv_pair_key(row: pd.Series | object) -> Tuple[str, str, str, str]:
    return _pair_key(row.id1, row.id2, row.rid1, row.rid2)


def _pair_key_from_pair_id(pair_id: str) -> Tuple[str, str, str, str]:
    parts = str(pair_id).split("__")
    if len(parts) < 3:
        raise ValueError(f"Unexpected pair_id format: {pair_id!r}")
    rid_parts = parts[2].split("_")
    if len(rid_parts) < 2:
        raise ValueError(f"Unexpected pair_id rid section: {pair_id!r}")
    return _pair_key(parts[0], parts[1], f"L:{rid_parts[0]}", f"R:{rid_parts[1]}")


def _load_label_lookup(path: Path) -> Dict[Tuple[str, str, str, str], bool]:
    df = pd.read_csv(path)
    return {_csv_pair_key(row): bool(row.label) for row in df.itertuples(index=False)}


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


def _update_csv_labels(
    input_path: Path,
    output_path: Path,
    label_lookup: Dict[Tuple[str, str, str, str], bool],
) -> Tuple[Dict[str, int], List[bool]]:
    df = pd.read_csv(input_path)
    old_labels = df["label"].astype(bool).tolist()
    new_labels = [label_lookup[_csv_pair_key(r)] for r in df.itertuples(index=False)]
    df["label"] = new_labels
    df.to_csv(output_path, index=False)

    changed = sum(int(old != new) for old, new in zip(old_labels, new_labels))
    changed_to_match = sum(int((not old) and new) for old, new in zip(old_labels, new_labels))
    changed_to_non_match = sum(int(old and (not new)) for old, new in zip(old_labels, new_labels))
    return (
        {
            "row_count": int(len(df)),
            "changed_labels": int(changed),
            "changed_to_match": int(changed_to_match),
            "changed_to_non_match": int(changed_to_non_match),
            "positive_labels": int(sum(int(v) for v in new_labels)),
        },
        new_labels,
    )


def _update_train_labels(input_path: Path, output_path: Path, csv_labels: List[bool]) -> Dict[str, int]:
    rows = _load_train_records(input_path)
    if len(rows) != len(csv_labels):
        raise ValueError(f"Train/CSV row mismatch for {input_path}: {len(rows)} vs {len(csv_labels)}")
    changed = 0
    positives = 0
    for row, csv_label in zip(rows, csv_labels):
        if _pair_key_from_pair_id(str(row.get("pair_id", "")))[:2] != (
            str(row.get("pair_id", "")).split("__")[0],
            str(row.get("pair_id", "")).split("__")[1],
        ):
            raise ValueError(f"Unexpected pair_id format while updating train labels: {row.get('pair_id')!r}")
        new_label = 1 if csv_label else 0
        if int(row.get("label", 0)) != new_label:
            changed += 1
        row["label"] = new_label
        positives += new_label
    _write_train_records(output_path, rows)
    return {
        "row_count": int(len(rows)),
        "changed_labels": int(changed),
        "positive_labels": int(positives),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild all ABT Ditto active-label profiles from a relabeled master profile.")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--master-profile", default=DEFAULT_MASTER_PROFILE)
    parser.add_argument("--master-relabeled-csv", default=str(DEFAULT_MASTER_RELABEL_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    master_profile = str(args.master_profile)
    master_relabeled_csv = Path(args.master_relabeled_csv)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    label_lookup = _load_label_lookup(master_relabeled_csv)

    summary = {
        "source_dir": str(source_dir),
        "master_profile": master_profile,
        "master_relabeled_csv": str(master_relabeled_csv),
        "output_dir": str(output_dir),
        "profiles": {},
    }

    for profile_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
        profile_name = profile_dir.name
        out_profile_dir = output_dir / profile_name
        out_profile_dir.mkdir(parents=True, exist_ok=True)

        active_in = profile_dir / "active_labels_latest.csv"
        final_in = profile_dir / "labels_final.csv"
        train_in = next(profile_dir.glob("*train.json.gz"))

        active_stats, active_labels = _update_csv_labels(
            active_in,
            out_profile_dir / "active_labels_latest.csv",
            label_lookup,
        )
        final_stats, _ = _update_csv_labels(
            final_in,
            out_profile_dir / "labels_final.csv",
            label_lookup,
        )
        train_stats = _update_train_labels(
            train_in,
            out_profile_dir / train_in.name,
            active_labels,
        )

        if profile_name == master_profile:
            for extra_name in [
                "active_labels_latest__relabeled__gpt-5-mini__agent-precision-system-prompt.csv",
                "labels_final__relabeled__gpt-5-mini__agent-precision-system-prompt.csv",
                "active_labels_latest_abt-buy_all_plus20random_train__relabeled__gpt-5-mini__agent-precision-system-prompt.json.gz",
                "relabel_results__gpt-5-mini__agent-precision-system-prompt.csv",
                "relabel_results__gpt-5-mini__agent-precision-system-prompt.jsonl",
                "relabel_summary__gpt-5-mini__agent-precision-system-prompt.json",
            ]:
                extra_path = profile_dir / extra_name
                if extra_path.exists():
                    shutil.copy2(extra_path, out_profile_dir / extra_name)

        summary["profiles"][profile_name] = {
            "active_labels_latest": active_stats,
            "labels_final": final_stats,
            "train_json_gz": train_stats,
        }

    summary_path = output_dir / "rebuild_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
