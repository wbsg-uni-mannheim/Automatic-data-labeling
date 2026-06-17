#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

BASE_OUTPUT_COLUMNS = [
    "pair_id",
    "label",
]
RESERVED_FEATURE_FIELDS = {"id", "__rid", "pair_id", "label", "is_hard_negative", "rid1", "rid2", "similarity"}


def _safe(v: object) -> str:
    if pd.isna(v):
        return ""
    return str(v)


def _load_source(csv_path: Path, side_prefix: str) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False).reset_index(drop=True).copy()
    required = {"id"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    rid_to_idx = {f"{side_prefix}:{i}": i for i in range(len(df))}
    # Fallback for files without rid columns; first occurrence only.
    id_to_idx: Dict[str, int] = {}
    for i, v in enumerate(df["id"].astype(str).tolist()):
        if v not in id_to_idx:
            id_to_idx[v] = i
    return df, rid_to_idx, id_to_idx


def _label_to_int(v: object) -> int:
    s = str(v).strip().upper()
    if s in {"1", "TRUE", "T", "YES"}:
        return 1
    if s in {"0", "FALSE", "F", "NO"}:
        return 0
    raise ValueError(f"Unsupported label value: {v}")


def _parse_extra_fields(raw: str | None) -> List[str]:
    if raw is None:
        return []
    vals = [x.strip() for x in str(raw).split(",")]
    vals = [v for v in vals if v]
    # Keep order and dedupe.
    return list(dict.fromkeys(vals))


def _resolve_fields(
    fields: Sequence[str] | None,
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
) -> List[str]:
    specified = [str(f).strip() for f in (fields or []) if str(f).strip()]
    if specified:
        return list(dict.fromkeys([f for f in specified if f not in RESERVED_FEATURE_FIELDS]))

    skip = set(RESERVED_FEATURE_FIELDS)
    inferred: List[str] = []
    right_cols = set(right_df.columns)
    for col in left_df.columns:
        c = str(col)
        if c in skip:
            continue
        if c not in right_cols:
            continue
        inferred.append(c)
    return inferred


def _pick_index(
    row: pd.Series,
    rid_col: str,
    id_col: str,
    rid_to_idx: Dict[str, int],
    id_to_idx: Dict[str, int],
) -> int:
    rid = str(row[rid_col]) if rid_col in row and pd.notna(row[rid_col]) else ""
    if rid and rid in rid_to_idx:
        return rid_to_idx[rid]
    sid = str(row[id_col]) if id_col in row and pd.notna(row[id_col]) else ""
    if sid and sid in id_to_idx:
        return id_to_idx[sid]
    raise KeyError(f"Cannot resolve pair row ({rid_col}={rid}, {id_col}={sid})")


def convert(
    labels_csv: Path,
    left_csv: Path,
    right_csv: Path,
    output_json_gz: Path,
    fields: Sequence[str] | None = None,
) -> None:
    labels = pd.read_csv(labels_csv, dtype=str, keep_default_na=False)
    needed = {"id1", "id2", "label"}
    missing = sorted(list(needed - set(labels.columns)))
    if missing:
        raise ValueError(f"Missing columns in {labels_csv}: {missing}")

    left_df, left_rid_to_idx, left_id_to_idx = _load_source(left_csv, side_prefix="L")
    right_df, right_rid_to_idx, right_id_to_idx = _load_source(right_csv, side_prefix="R")
    resolved_fields = _resolve_fields(fields=fields, left_df=left_df, right_df=right_df)

    out_records = []
    unresolved = 0
    for i, row in labels.iterrows():
        try:
            li = _pick_index(row, "rid1", "id1", left_rid_to_idx, left_id_to_idx)
            ri = _pick_index(row, "rid2", "id2", right_rid_to_idx, right_id_to_idx)
        except KeyError:
            unresolved += 1
            continue

        lrow = left_df.iloc[li]
        rrow = right_df.iloc[ri]
        label_int = _label_to_int(row["label"])
        pair_id = f"{_safe(lrow.get('id'))}__{_safe(rrow.get('id'))}__{li}_{ri}_{i}"

        rec = {
            "pair_id": pair_id,
            "label": int(label_int),
        }
        for field in resolved_fields:
            rec[f"{field}_left"] = _safe(lrow.get(field))
            rec[f"{field}_right"] = _safe(rrow.get(field))
        out_records.append(rec)

    out_df = pd.DataFrame(out_records)
    if out_df.empty:
        raise RuntimeError("No rows converted; check pair ids and source CSV inputs.")

    output_cols = list(BASE_OUTPUT_COLUMNS)
    for field in resolved_fields:
        output_cols.append(f"{field}_left")
        output_cols.append(f"{field}_right")
    for col in output_cols:
        if col not in out_df.columns:
            out_df[col] = ""
    out_df = out_df[output_cols].copy()
    output_json_gz.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_json_gz, "wt") as f:
        for _, r in out_df.iterrows():
            f.write(json.dumps({c: r[c] for c in output_cols}, ensure_ascii=False) + "\n")

    n_pos = int((out_df["label"] == 1).sum())
    n_neg = int((out_df["label"] == 0).sum())
    print(
        f"Wrote {len(out_df)} rows -> {output_json_gz} "
        f"({n_pos} pos, {n_neg} neg, unresolved={unresolved}, fields={len(resolved_fields)})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert active labels CSV (id/rid pairs) to Ditto WDC json.gz"
    )
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--left-csv", default="benchmarks/wdc/wdc_train_large_left.csv")
    parser.add_argument("--right-csv", default="benchmarks/wdc/wdc_train_large_right.csv")
    parser.add_argument("--output-json-gz", required=True)
    parser.add_argument(
        "--fields",
        default="",
        help="Comma-separated output fields to emit as <field>_left/<field>_right.",
    )
    parser.add_argument(
        "--extra-fields",
        default="",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    fields_arg = args.fields or args.extra_fields

    convert(
        labels_csv=Path(args.labels_csv),
        left_csv=Path(args.left_csv),
        right_csv=Path(args.right_csv),
        output_json_gz=Path(args.output_json_gz),
        fields=_parse_extra_fields(fields_arg),
    )


if __name__ == "__main__":
    main()
