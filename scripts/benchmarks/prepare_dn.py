"""Convert a Papadakis et al. (ICDE 2024) Dn* dataset into the benchmark layout.

  python scripts/benchmarks/prepare_dn.py --src /path/Dn7/Dn7 --name dn7-walmart-amazon --out benchmarks/dn7-walmart-amazon

Source layout: tableA.csv, tableB.csv, train_set.csv, valid_set.csv, test_set.csv (Zenodo
10.5281/zenodo.8164151). Values are stored with literal doubled quotes inside the field and
are unquoted here.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import emit_all, yaml_blocks  # noqa: E402


def unquote(v: str) -> str:
    v = (v or "").strip()
    while len(v) >= 2 and v[0] == '"' and v[-1] == '"':
        v = v[1:-1].strip()
    return v


def read_csv(p: Path) -> List[Dict[str, str]]:
    with p.open(encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)

    tA, tB = read_csv(src / "tableA.csv"), read_csv(src / "tableB.csv")
    fields = [c for c in tA[0].keys() if c != "id"]
    lp, rp = f"{a.name}l_", f"{a.name}r_"
    left = {lp + r["id"]: {"id": lp + r["id"], **{c: unquote(r.get(c, "")) for c in fields}} for r in tA}
    right = {rp + r["id"]: {"id": rp + r["id"], **{c: unquote(r.get(c, "")) for c in fields}} for r in tB}

    def pairs(fn: str) -> List[Dict[str, str]]:
        return [{"id_left": lp + r["left_id"], "id_right": rp + r["right_id"], "label": str(int(r["label"]))}
                for r in read_csv(src / fn)]

    train, valid, test = pairs("train_set.csv"), pairs("valid_set.csv"), pairs("test_set.csv")
    emit_all(out, a.name, fields, left, right, train, valid, test)
    n = len(train)
    print(yaml_blocks(a.name, out, fields, profiles={
        "large": {"target_total": n, "target_pos": max(1, int(0.2 * n)), "target_neg": n - max(1, int(0.2 * n)),
                  "label_budget": int(1.5 * n)}}))


if __name__ == "__main__":
    main()
