#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

WDC_COLUMNS = [
    "id_left",
    "brand_left",
    "title_left",
    "description_left",
    "price_left",
    "priceCurrency_left",
    "cluster_id_left",
    "id_right",
    "brand_right",
    "title_right",
    "description_right",
    "price_right",
    "priceCurrency_right",
    "cluster_id_right",
    "pair_id",
    "label",
    "is_hard_negative",
]


def _safe(v: object) -> str:
    if pd.isna(v):
        return ""
    return str(v)


def _load_source(csv_path: Path, side_prefix: str) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
    df = pd.read_csv(csv_path).reset_index(drop=True).copy()
    required = {"id"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    for optional_col in ["title", "brand", "description", "price", "priceCurrency"]:
        if optional_col not in df.columns:
            df[optional_col] = ""
    if "cluster_id" not in df.columns:
        df["cluster_id"] = ""

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
) -> None:
    labels = pd.read_csv(labels_csv)
    needed = {"id1", "id2", "label"}
    missing = sorted(list(needed - set(labels.columns)))
    if missing:
        raise ValueError(f"Missing columns in {labels_csv}: {missing}")

    left_df, left_rid_to_idx, left_id_to_idx = _load_source(left_csv, side_prefix="L")
    right_df, right_rid_to_idx, right_id_to_idx = _load_source(right_csv, side_prefix="R")

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
            "id_left": _safe(lrow.get("id")),
            "brand_left": _safe(lrow.get("brand")),
            "title_left": _safe(lrow.get("title")),
            "description_left": _safe(lrow.get("description")),
            "price_left": _safe(lrow.get("price")),
            "priceCurrency_left": _safe(lrow.get("priceCurrency")),
            "cluster_id_left": _safe(lrow.get("cluster_id")),
            "id_right": _safe(rrow.get("id")),
            "brand_right": _safe(rrow.get("brand")),
            "title_right": _safe(rrow.get("title")),
            "description_right": _safe(rrow.get("description")),
            "price_right": _safe(rrow.get("price")),
            "priceCurrency_right": _safe(rrow.get("priceCurrency")),
            "cluster_id_right": _safe(rrow.get("cluster_id")),
            "pair_id": pair_id,
            "label": int(label_int),
            "is_hard_negative": int(0),
        }
        out_records.append(rec)

    out_df = pd.DataFrame(out_records)
    if out_df.empty:
        raise RuntimeError("No rows converted; check pair ids and source CSV inputs.")

    out_df = out_df[WDC_COLUMNS].copy()
    output_json_gz.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_json_gz, "wt") as f:
        for _, r in out_df.iterrows():
            f.write(json.dumps({c: r[c] for c in WDC_COLUMNS}, ensure_ascii=False) + "\n")

    n_pos = int((out_df["label"] == 1).sum())
    n_neg = int((out_df["label"] == 0).sum())
    print(
        f"Wrote {len(out_df)} rows -> {output_json_gz} "
        f"({n_pos} pos, {n_neg} neg, unresolved={unresolved})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert active labels CSV (id/rid pairs) to Ditto WDC json.gz"
    )
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--left-csv", default="data/wdc/wdc_train_large_left.csv")
    parser.add_argument("--right-csv", default="data/wdc/wdc_train_large_right.csv")
    parser.add_argument("--output-json-gz", required=True)
    args = parser.parse_args()

    convert(
        labels_csv=Path(args.labels_csv),
        left_csv=Path(args.left_csv),
        right_csv=Path(args.right_csv),
        output_json_gz=Path(args.output_json_gz),
    )


if __name__ == "__main__":
    main()
