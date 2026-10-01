#!/usr/bin/env python3
"""Recompute the training-set composition of Table 10 from the released files.

Uses only the Python standard library. For every benchmark it compares the benchmark training set
with the GPT-5.2 sets of similarity search (SS), AL-ML, and AL-Ditto:

- positive and negative pairs and the positive rate,
- entity clusters: distinct cluster ids of the left and right records that occur in the set,
- hard pairs: surface similarity s is the token Jaccard over lower-cased [a-z0-9]+ tokens from all
  record attributes except id and cluster_id. tau_p is the 25% quantile of s over the benchmark
  positives and tau_n the 75% quantile over the benchmark negatives (linear interpolation).
  Hard positives are positives with s <= tau_p, hard negatives are negatives with s >= tau_n.

Writes results/table10_composition.csv.
"""
import csv
import gzip
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TD = ROOT / "artifacts/training_data"
BENCH = {
    "abt-buy": ("abt-buy-train.json", "abt-buy-valid.csv"),
    "walmart-amazon": ("walmart-amazon-train.json.gz", "walmart-amazon-valid.csv"),
    "dn7-walmart-amazon": ("dn7-walmart-amazon-train.json.gz", None),
    "wdc": ("wdcproducts80cc20rnd000un_train_large.json.gz", None),
    "billiger-de": ("billiger-de-train-medium.json.gz", None),
    "dblp-acm": ("dblp-acm-train.json.gz", "dblp-acm-valid.csv"),
    "dblp-scholar": ("dblp-scholar-train.json.gz", "dblp-scholar-valid.csv"),
    "semi-heter": ("semi-heter-train.json.gz", None),
}
METHODS = {"SS": "similarity_search", "AL-ML": "active_learning_ml", "AL-Ditto": "active_learning_ditto"}
# billiger.de: Table 10 uses the 5,897-pair sets that match the benchmark training size.
SUFFIX = {"billiger-de": "_5897"}
# Semi-HETER: s for the generated sets uses the attribute values stored in the training files.
FROM_ROWS = {"semi-heter"}


def read_pairs(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    text = opener(path, "rt").read().strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict):  # pandas column-oriented export
        keys = list(data["label"])
        data = [{column: data[column].get(k) for column in data} for k in keys]
    return data


def pair_ids(row):
    if row.get("id_left") is not None and row.get("id_right") is not None:
        return str(row["id_left"]), str(row["id_right"])
    left, right = re.split(r"#|__", str(row["pair_id"]))[:2]
    return left, right


def read_records(path):
    with open(path, newline="") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def record_tables(benchmark):
    folder = ROOT / "benchmarks" / benchmark
    if benchmark == "wdc":
        return read_records(folder / "wdc_train_large_left.csv"), read_records(folder / "wdc_train_large_right.csv")
    return read_records(folder / f"{benchmark}-train-left.csv"), read_records(folder / f"{benchmark}-train-right.csv")


def tokens(record):
    text = " ".join(value for key, value in record.items() if key not in ("id", "cluster_id", "extra") and value)
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def quantile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def row_side(row, side):
    return {key[: -len(side) - 1]: str(value) for key, value in row.items()
            if key.endswith("_" + side) and value is not None}


def scored(rows, records, from_rows=False):
    out = []
    for row in rows:
        a, b = (records[i] for i in pair_ids(row))
        ta, tb = (tokens(row_side(row, "left")), tokens(row_side(row, "right"))) if from_rows else (tokens(a), tokens(b))
        sim = len(ta & tb) / len(ta | tb) if ta | tb else 0.0
        out.append((int(row["label"]), sim, a.get("cluster_id", ""), b.get("cluster_id", "")))
    return out


def benchmark_rows(benchmark):
    train, valid = BENCH[benchmark]
    rows = read_pairs(ROOT / "benchmarks" / benchmark / train)
    if valid:  # the DeepMatcher train files also hold the validation pairs
        with open(ROOT / "benchmarks" / benchmark / valid, newline="") as handle:
            held_out = {pair_ids(r) for r in csv.DictReader(handle)}
        rows = [r for r in rows if pair_ids(r) not in held_out]
    return rows


def main():
    table = []
    for benchmark in BENCH:

        left, right = record_tables(benchmark)
        records = {**left, **right}  # some benchmark splits list a pair's records in swapped order
        sources = {"Bench.": scored(benchmark_rows(benchmark), records)}
        for name, folder in METHODS.items():
            path = TD / benchmark / "gpt-5.2" / (folder + SUFFIX.get(benchmark, "")) / f"{benchmark}_train.json.gz"
            sources[name] = scored(read_pairs(path), records, from_rows=benchmark in FROM_ROWS)
        bench = sources["Bench."]
        tau_p = quantile([s for label, s, _, _ in bench if label == 1], 0.25)
        tau_n = quantile([s for label, s, _, _ in bench if label == 0], 0.75)
        for name, pairs in sources.items():
            clusters = {c for _, _, cl, cr in pairs for c in (cl, cr) if c and c.lower() != "nan"}
            values = {
                "positives": sum(label == 1 for label, *_ in pairs),
                "negatives": sum(label == 0 for label, *_ in pairs),
                "clusters": len(clusters),
                "hard_positives": sum(label == 1 and s <= tau_p for label, s, *_ in pairs),
                "hard_negatives": sum(label == 0 and s >= tau_n for label, s, *_ in pairs),
            }
            table.append({"benchmark": benchmark, "source": name, **values,
                          "positive_rate": f"{100 * values['positives'] / len(pairs):.2f}",
                          "tau_p": f"{tau_p:.6f}", "tau_n": f"{tau_n:.6f}"})
    with open(ROOT / "results/table10_composition.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"wrote results/table10_composition.csv ({len(table)} rows)")


if __name__ == "__main__":
    main()
