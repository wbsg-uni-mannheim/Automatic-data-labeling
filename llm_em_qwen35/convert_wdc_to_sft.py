#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


SYSTEM_PROMPT = (
    "You are an entity matching classifier. Decide whether two entity descriptions "
    "refer to the same real-world entity. Answer only Yes or No."
)

USER_TEMPLATE = """Do these two entity descriptions refer to the same real-world entity?

Entity 1:
{left}

Entity 2:
{right}

Answer only Yes or No."""


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\n", " ").replace("\t", " ")).strip()


def parse_fields(raw: str) -> List[str]:
    fields = [x.strip() for x in raw.split(",") if x.strip()]
    if not fields:
        raise ValueError("--fields must contain at least one field")
    return fields


def load_jsonl_gz(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with gzip.open(path, "rt") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def normalize_label(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        iv = int(value)
        if iv in {0, 1}:
            return iv
    s = str(value).strip().upper()
    if s in {"1", "TRUE", "T", "YES", "Y", "MATCH"}:
        return 1
    if s in {"0", "FALSE", "F", "NO", "N", "NOT A MATCH", "NON-MATCH"}:
        return 0
    raise ValueError(f"Unsupported label value: {value!r}")


def serialize_entity(row: Dict[str, Any], side: str, fields: Sequence[str], max_field_len: int) -> str:
    parts: List[str] = []
    for field in fields:
        value = normalize_text(row.get(f"{field}_{side}", ""))
        if not value:
            continue
        if len(value) > max_field_len:
            value = value[:max_field_len].rstrip() + "..."
        parts.append(f"{field}: {value}")
    return "\n".join(parts) if parts else "(empty)"


def to_sft_record(row: Dict[str, Any], fields: Sequence[str], max_field_len: int) -> Dict[str, Any]:
    label = normalize_label(row.get("label"))
    answer = "Yes" if label == 1 else "No"
    left = serialize_entity(row, "left", fields, max_field_len)
    right = serialize_entity(row, "right", fields, max_field_len)
    pair_id = normalize_text(row.get("pair_id", ""))
    if not pair_id:
        left_id = normalize_text(row.get("id_left", ""))
        right_id = normalize_text(row.get("id_right", ""))
        pair_id = f"{left_id}#{right_id}" if left_id or right_id else ""
    return {
        "pair_id": pair_id,
        "label": label,
        "answer": answer,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(left=left, right=right)},
            {"role": "assistant", "content": answer},
        ],
    }


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def convert_split(name: str, input_path: Path, output_dir: Path, fields: Sequence[str], max_field_len: int) -> Dict[str, int]:
    rows = load_jsonl_gz(input_path)
    records = [to_sft_record(row, fields, max_field_len) for row in rows]
    out_path = output_dir / f"{name}.jsonl"
    count = write_jsonl(out_path, records)
    pos = sum(int(r["label"] == 1) for r in records)
    return {"rows": count, "pos": pos, "neg": count - pos}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert WDC/Ditto JSONL gzip splits to chat SFT JSONL.")
    parser.add_argument("--train-json-gz", required=True)
    parser.add_argument("--valid-json-gz", required=True)
    parser.add_argument("--test-json-gz", required=True)
    parser.add_argument("--fields", required=True, help="Comma-separated entity fields, e.g. title,description,price")
    parser.add_argument("--max-field-len", type=int, default=350)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    fields = parse_fields(args.fields)
    output_dir = Path(args.output_dir)
    split_paths = {
        "train": Path(args.train_json_gz),
        "valid": Path(args.valid_json_gz),
        "test": Path(args.test_json_gz),
    }
    summary = {
        name: convert_split(name, path, output_dir, fields, args.max_field_len)
        for name, path in split_paths.items()
    }
    manifest = {
        "fields": fields,
        "max_field_len": int(args.max_field_len),
        "inputs": {k: str(v) for k, v in split_paths.items()},
        "splits": summary,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

