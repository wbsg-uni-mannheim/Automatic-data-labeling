"""Convert Machamp Semi-HETER (or Semi-Rel) into the benchmark layout.

  python scripts/benchmarks/prepare_machamp.py --src /path/machamp/semi-heter --name semi-heter --out benchmarks/semi-heter

Machamp (Wang, Li, Hirota, CIKM 2021; github.com/megagonlabs/machamp) ships
left.{json,csv}, right.{json,csv} and header-less, comma-separated
train/valid/test.csv with 0-based *positions* into the record arrays.

Semi-HETER's left table is five concatenated Magellan Book sources with 127
distinct key sets, the right table has 56, and both nest dicts/lists. We map the
known key families onto one canonical schema so the pipeline gets a fixed field
list; the heterogeneity survives as attribute sparsity and format variance. All
keys not covered by the mapping are preserved in ``extra`` as "key: value"
pairs (not in the default field list, available to Ditto/LLM students if wanted).

Canonical fields: title, authors, publisher, year, isbn, pages, price, extra.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import emit_all, yaml_blocks  # noqa: E402

FIELDS = ["title", "authors", "publisher", "year", "isbn", "pages", "price", "extra"]

# canonical -> candidate keys (matched case-insensitively, after removing non-alphanumerics)
KEYMAP: Dict[str, List[str]] = {
    "title": ["title"],
    "authors": ["authors", "author", "firstauthor", "secondauthor", "thirdauthor"],
    "publisher": ["publisher", "soldby"],
    "year": ["pubyear", "publishdate", "publicationdate", "publication_date", "year"],
    "isbn": ["isbn13", "isbn", "isbn_13", "isbn_10", "isbn10", "asin"],
    "pages": ["pages", "pagecount"],
    "price": ["price", "newprice", "usedprice", "listprice"],
}
_AMBIGUOUS = {"paperback", "hardcover", "nookbook"}  # page count on one source, price on another
_MONEY = re.compile(r"^\s*[$€£]\s*\d")
_PAGES = re.compile(r"^\s*\d{1,5}(\s*pages?)?\s*$", re.I)


def _year_from(s: str) -> str:
    m = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", s)
    if m:
        return m.group(1)
    m = re.search(r"(?<!\d)(\d{1,2})\s*$", s)  # trailing 2-digit year: 12/31/13, or bare 8 / 99
    if m and (s.strip().isdigit() or "/" in s or "-" in s):
        n = int(m.group(1))
        return str(2000 + n if n < 30 else 1900 + n)
    return ""


_SKIP = {"id", "filename", "publisherdummy", "isbn13dummy", "salesrank", "rating", "ratingscount",
         "ratingvalue", "numberofratings", "numberofreviews", "cover", "format", "producttype"}


def _norm_key(k: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(k).lower())


def _flat(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if v != v:  # nan
            return ""
        return str(int(v)) if v.is_integer() and abs(v) < 1e15 else str(v)
    if isinstance(v, (list, tuple)):
        return ", ".join(x for x in (_flat(i) for i in v) if x)
    if isinstance(v, dict):
        return "; ".join(f"{k}: {_flat(x)}" for k, x in v.items() if _flat(x))
    return re.sub(r"\s+", " ", str(v)).strip()


def _walk(rec: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """Flatten one level of nesting into (leaf_key, value) pairs."""
    out: List[Tuple[str, Any]] = []
    for k, v in rec.items():
        if isinstance(v, dict):
            out.extend((kk, vv) for kk, vv in v.items())
        else:
            out.append((k, v))
    return out


def canonicalize(rec: Dict[str, Any]) -> Dict[str, str]:
    leaves = _walk(rec)
    got: Dict[str, List[str]] = {f: [] for f in FIELDS}
    extra: List[str] = []
    for k, v in leaves:
        nk = _norm_key(k)
        if nk in _SKIP or nk.startswith("unnamed"):
            continue
        s = _flat(v)
        if not s or s.lower() in {"nan", "none", "null", "0"} and nk in {"pubyear", "pages", "pagecount"}:
            continue
        if not s:
            continue
        if nk in _AMBIGUOUS:
            hit = "price" if _MONEY.match(s) else ("pages" if _PAGES.match(s) else None)
            if hit is None:
                extra.append(f"{k}: {s[:80]}")
                continue
        else:
            hit = next((f for f, keys in KEYMAP.items() if nk in {_norm_key(x) for x in keys}), None)
        if hit:
            got[hit].append(s)
        else:
            extra.append(f"{k}: {s[:80]}")
    out: Dict[str, str] = {}
    for f in FIELDS[:-1]:
        vals = [x for x in got[f] if x]
        if f == "year":
            years = [y for y in (_year_from(x) for x in vals) if y]
            out[f] = years[0] if years else ""
        elif f == "pages":
            digits = [re.sub(r"\D", "", x) for x in vals]
            out[f] = next((d for d in digits if d and d != "0"), "")
        elif f in {"isbn", "price", "publisher"}:
            out[f] = vals[0] if vals else ""
        else:
            out[f] = ", ".join(dict.fromkeys(vals)) if vals else ""
    out["extra"] = "; ".join(extra)[:400]
    return out


def load_table(path_stem: Path) -> List[Dict[str, Any]]:
    for ext in (".json", ".csv"):
        p = path_stem.with_suffix(ext)
        if p.exists():
            if ext == ".json":
                return json.loads(p.read_text(encoding="utf-8"))
            with p.open(encoding="utf-8", errors="replace") as f:
                return list(csv.DictReader(f))
    raise FileNotFoundError(path_stem)


def read_pairs(p: Path) -> List[Tuple[int, int, int]]:
    rows: List[Tuple[int, int, int]] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            parts = [x.strip() for x in line.strip().split(",")]
            if len(parts) < 3 or not parts[0].isdigit():
                continue
            rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Machamp task dir, e.g. machamp/semi-heter")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    L, R = load_table(src / "left"), load_table(src / "right")
    lp, rp = f"{a.name}l_", f"{a.name}r_"
    left = {f"{lp}{i}": {"id": f"{lp}{i}", **canonicalize(r)} for i, r in enumerate(L)}
    right = {f"{rp}{i}": {"id": f"{rp}{i}", **canonicalize(r)} for i, r in enumerate(R)}

    def pairs(fn: str) -> List[Dict[str, str]]:
        return [{"id_left": f"{lp}{l}", "id_right": f"{rp}{r}", "label": str(y)} for l, r, y in read_pairs(src / fn)]

    train, valid, test = pairs("train.csv"), pairs("valid.csv"), pairs("test.csv")
    emit_all(out, a.name, FIELDS, left, right, train, valid, test)
    n = len(train)
    print(yaml_blocks(a.name, out, FIELDS[:-1], profiles={
        "large": {"target_total": n, "target_pos": int(0.38 * n), "target_neg": n - int(0.38 * n),
                  "label_budget": int(1.5 * n)}},
        extra_labeling={"phase2_target_size": min(1000, n)}))
    print("# NOTE: 'extra' is written to the tables but left out of the default field list above.", file=sys.stderr)


if __name__ == "__main__":
    main()
