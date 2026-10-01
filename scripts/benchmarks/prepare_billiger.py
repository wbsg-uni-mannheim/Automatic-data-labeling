"""Prepare the billiger.de Products benchmark (WDC-Products layout) for the pipeline.

  python scripts/benchmarks/prepare_billiger.py --src /path/solute_de --name billiger-de --out benchmarks/billiger-de
  python scripts/benchmarks/prepare_billiger.py --src /path/solute_en --name billiger-en --out benchmarks/billiger-en

Mirrors the WDC Products setup used in the paper: 80% corner cases, large training
set, 100%-unseen gold standard. Files are gzipped JSONL pairs with name/desc/brand/
price and product_id (the entity cluster). Override with --train-variant / --gs-variant.
Source: https://wbsg-uni-mannheim.github.io/billiger-de-products/
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import emit_all, yaml_blocks  # noqa: E402

FIELDS = ["name", "desc", "brand", "price"]


def read_jsonl(p: Path) -> List[Dict[str, str]]:
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def split_records(pairs: List[Dict[str, str]], left: Dict[str, Dict[str, str]], right: Dict[str, Dict[str, str]],
                  clusters: Dict[str, int]) -> List[Dict[str, str]]:
    out = []
    for p in pairs:
        for side, tbl in (("left", left), ("right", right)):
            rid = str(p[f"id_{side}"])
            if rid not in tbl:
                tbl[rid] = {"id": rid, **{c: str(p.get(f"{c}_{side}", "") or "") for c in FIELDS}}
            pid = p.get(f"product_id_{side}")
            if pid not in (None, ""):
                clusters[rid] = int(pid)
        out.append({"id_left": str(p["id_left"]), "id_right": str(p["id_right"]), "label": str(int(p["label"]))})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="solute_de or solute_en directory")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-variant", default="products80cc20rnd000un_train_large")
    ap.add_argument("--valid-variant", default="products80cc20rnd000un_valid_large")
    ap.add_argument("--gs-variant", default="products80cc20rnd100un_gs")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)

    left: Dict[str, Dict[str, str]] = {}
    right: Dict[str, Dict[str, str]] = {}
    clusters: Dict[str, int] = {}
    train = split_records(read_jsonl(src / "training-sets" / f"{a.train_variant}.json.gz"), left, right, clusters)
    valid = split_records(read_jsonl(src / "validation-sets" / f"{a.valid_variant}.json.gz"), left, right, clusters)
    test = split_records(read_jsonl(src / "gold-standards_adjusted" / f"{a.gs_variant}.json.gz"), left, right, clusters)

    # emit_all derives clusters from positives; billiger ships product_id, so patch afterwards
    import _common
    _orig = _common.clusters_from_pairs
    _common.clusters_from_pairs = lambda _pairs: dict(clusters)  # type: ignore[assignment]
    try:
        emit_all(out, a.name, FIELDS, left, right, train, valid, test)
    finally:
        _common.clusters_from_pairs = _orig
    n = len(train)
    print(yaml_blocks(a.name, out, FIELDS, profiles={
        "small": {"target_total": 2500, "target_pos": 500, "target_neg": 2000, "label_budget": 4000},
        "medium": {"target_total": 10000, "target_pos": 2500, "target_neg": 7500, "label_budget": 15000},
        "large": {"target_total": n, "target_pos": int(0.25 * n), "target_neg": n - int(0.25 * n),
                  "label_budget": int(1.5 * n)}}))
    print("# fields map: title->name, description->desc (set field_aliases in the ditto config if you keep canonical names)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
