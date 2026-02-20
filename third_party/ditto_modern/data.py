from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

from .augment import Augmenter

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


@dataclass(frozen=True)
class PairExample:
    idx: int
    pair_id: str
    left: str
    right: str
    label: int


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text.replace("\n", " ").replace("\t", " ")).strip()
    return text


def serialize_entity(record: Dict[str, object], side: str, fields: Sequence[str], max_field_len: int) -> str:
    parts: List[str] = []
    for field in fields:
        key = f"{field}_{side}"
        value = normalize_text(record.get(key, ""))
        if not value:
            continue
        if len(value) > max_field_len:
            value = value[:max_field_len] + "..."
        parts.append(f"COL {field} VAL {value}")
    return " ".join(parts)


def load_wdc_json_gz(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    with gzip.open(p, "rt") as f:
        records = [json.loads(line) for line in f]
    df = pd.DataFrame(records)

    missing = [c for c in WDC_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required WDC columns: {missing}")

    df["label"] = df["label"].astype(int)
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("label column must contain 0/1 values")
    return df


def write_wdc_json_gz(df: pd.DataFrame, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as f:
        for _, row in df.iterrows():
            record = {c: row[c] for c in df.columns}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def wdc_to_pair_examples(
    df: pd.DataFrame,
    fields: Sequence[str],
    max_field_len: int,
) -> List[PairExample]:
    rows: List[PairExample] = []
    for idx, row in enumerate(df.to_dict(orient="records")):
        pair_id = normalize_text(row.get("pair_id", f"idx-{idx}")) or f"idx-{idx}"
        left = serialize_entity(row, "left", fields, max_field_len)
        right = serialize_entity(row, "right", fields, max_field_len)
        rows.append(PairExample(idx=idx, pair_id=pair_id, left=left, right=right, label=int(row["label"])))
    return rows


def examples_to_ditto_lines(examples: Iterable[PairExample]) -> List[str]:
    return [f"{ex.left}\t{ex.right}\t{ex.label}" for ex in examples]


class PairDataset(Dataset):
    """
    Ditto-style pair dataset with optional MixDA augmentation.
    Returns variable-length token-id sequences and uses `pad` as collate_fn.
    """

    def __init__(
        self,
        examples: Sequence[PairExample],
        tokenizer,
        max_len: int,
        weights: Dict[int, float] | None = None,
        da: str | None = None,
    ):
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.weights = weights or {}
        self.da = da
        self.augmenter = Augmenter() if da else None

    def __len__(self) -> int:
        return len(self.examples)

    def _encode_pair(self, left: str, right: str) -> List[int]:
        return self.tokenizer.encode(
            text=left,
            text_pair=right,
            max_length=self.max_len,
            truncation=True,
        )

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        x = self._encode_pair(ex.left, ex.right)
        sample_weight = float(self.weights.get(ex.idx, 1.0))

        if self.augmenter is not None:
            combined = f"{ex.left} [SEP] {ex.right}"
            aug = self.augmenter.augment_sent(combined, op=self.da)
            if " [SEP] " in aug:
                left_aug, right_aug = aug.split(" [SEP] ", 1)
            else:
                # Fallback to original pair if op produced invalid split
                left_aug, right_aug = ex.left, ex.right
            x_aug = self._encode_pair(left_aug, right_aug)
            return x, x_aug, ex.label, ex.idx, sample_weight

        return x, ex.label, ex.idx, sample_weight

    @staticmethod
    def pad(batch):
        if len(batch[0]) == 5:
            x1, x2, y, idxs, weights = zip(*batch)
            maxlen = max(max(len(x) for x in x1), max(len(x) for x in x2))
            x1 = [xi + [0] * (maxlen - len(xi)) for xi in x1]
            x2 = [xi + [0] * (maxlen - len(xi)) for xi in x2]
            att1 = [[1 if tok != 0 else 0 for tok in xi] for xi in x1]
            att2 = [[1 if tok != 0 else 0 for tok in xi] for xi in x2]
            return (
                torch.LongTensor(x1),
                torch.LongTensor(att1),
                torch.LongTensor(x2),
                torch.LongTensor(att2),
                torch.LongTensor(y),
                torch.LongTensor(idxs),
                torch.FloatTensor(weights),
            )

        x, y, idxs, weights = zip(*batch)
        maxlen = max(len(xi) for xi in x)
        x = [xi + [0] * (maxlen - len(xi)) for xi in x]
        att = [[1 if tok != 0 else 0 for tok in xi] for xi in x]
        return (
            torch.LongTensor(x),
            torch.LongTensor(att),
            torch.LongTensor(y),
            torch.LongTensor(idxs),
            torch.FloatTensor(weights),
        )
