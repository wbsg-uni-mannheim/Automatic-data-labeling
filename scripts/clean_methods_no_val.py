#!/usr/bin/env python3
"""No-validation cleaning for seed_round + simple_active (test + val pairs + dedup).

Drops, in order: test-set pairs, validation-set pairs, then duplicates.
Profile per benchmark matches the published comparison table.
Output: output/labeling_cleaned_no_val/<method>/<bm>/profiles/<profile>/
"""
from __future__ import annotations
import gzip, json
from collections import defaultdict
from pathlib import Path
import pandas as pd

ROOT = Path("/work/aasteine/Automatic-data-labeling")

TEST_FILES = {
    "abt-buy":        ROOT / "data/abt-buy/abt-buy-gs.json.gz",
    "dblp-acm":       ROOT / "data/dblp-acm/dblp-acm-gs.json.gz",
    "dblp-scholar":   ROOT / "data/dblp-scholar/dblp-scholar-gs.json.gz",
    "walmart-amazon": ROOT / "data/walmart-amazon/walmart-amazon-gs.json.gz",
    "wdc":            ROOT / "data/wdc/wdcproducts80cc20rnd100un_gs.json.gz",
}
VALID_FILES = {
    "abt-buy":        ROOT / "data/abt-buy/abt-buy-valid.csv",
    "dblp-acm":       ROOT / "data/dblp-acm/dblp-acm-valid.csv",
    "dblp-scholar":   ROOT / "data/dblp-scholar/dblp-scholar-valid.csv",
    "walmart-amazon": ROOT / "data/walmart-amazon/walmart-amazon-valid.csv",
    "wdc":            ROOT / "data/wdc/wdcproducts80cc20rnd000un_valid_medium.json.gz",
}

# (method, benchmark) -> (run_subdir, profile)  -- matches the published table
SPECS = {
    ("seed_round", "abt-buy"):        ("benchmark_abt-buy_20260415_190530",        "all_plus20random"),
    ("seed_round", "dblp-acm"):       ("benchmark_dblp-acm_20260415_190530",       "all_plus20random"),
    ("seed_round", "dblp-scholar"):   ("benchmark_dblp-scholar_20260415_190530",   "all_plus20random"),
    ("seed_round", "walmart-amazon"): ("benchmark_walmart-amazon_20260415_190530", "all"),
    ("seed_round", "wdc"):            ("benchmark_wdc_20260415_190530",            "medium"),
    ("simple_active", "abt-buy"):        ("benchmark_abt-buy_1",                      "large"),
    ("simple_active", "dblp-acm"):       ("benchmark_dblp-acm_20260302_113733",       "all"),
    ("simple_active", "dblp-scholar"):   ("benchmark_dblp-scholar_20260302_113733",   "all"),
    ("simple_active", "walmart-amazon"): ("benchmark_walmart-amazon_20260302_113733", "all"),
    ("simple_active", "wdc"):            ("benchmark_wdc_20260413_152105",            "large"),
}
METHOD_ROOT = {
    "seed_round":    ROOT / "output/seed_round_only_profiles",
    "simple_active": ROOT / "output/simple_active_learning_labeling",
}


def load_gs_pairs(path):
    pairs = set()
    with gzip.open(path, "rt") as f:
        for line in f:
            s = line.strip()
            if s:
                r = json.loads(s)
                pairs.add(frozenset([str(r["id_left"]), str(r["id_right"])]))
    return pairs


def load_valid_pairs(path):
    pairs = set()
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            for line in f:
                s = line.strip()
                if s:
                    r = json.loads(s)
                    pairs.add(frozenset([str(r["id_left"]), str(r["id_right"])]))
        return pairs
    df = pd.read_csv(path)
    if "id_left" in df.columns and "id_right" in df.columns:
        for l, r in zip(df["id_left"], df["id_right"]):
            pairs.add(frozenset([str(l), str(r)]))
    elif "pair_id" in df.columns:
        for pid in df["pair_id"]:
            l, r = str(pid).split("#", 1)
            pairs.add(frozenset([l, r]))
    else:
        raise ValueError(f"cannot parse {path}")
    return pairs


def extract_ids(pid):
    parts = pid.split("__")
    return frozenset([parts[0], parts[1]])


def load_train(p):
    with gzip.open(p, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_train(rows, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def clean(rows, test_pairs, valid_pairs):
    survivors = []
    t_p = t_n = v_p = v_n = 0
    for r in rows:
        ids = extract_ids(r["pair_id"])
        if ids in test_pairs:
            if r["label"] == 1: t_p += 1
            else: t_n += 1
            continue
        if ids in valid_pairs:
            if r["label"] == 1: v_p += 1
            else: v_n += 1
            continue
        survivors.append(r)
    groups = defaultdict(list)
    for r in survivors:
        groups[extract_ids(r["pair_id"])].append(r)
    kept = []
    dc_p = dc_n = dd_p = dd_n = 0
    for ids, g in groups.items():
        if len(g) == 1:
            kept.append(g[0]); continue
        if len({x["label"] for x in g}) == 1:
            kept.append(g[0])
            for r in g[1:]:
                if r["label"] == 1: dc_p += 1
                else: dc_n += 1
        else:
            for r in g:
                if r["label"] == 1: dd_p += 1
                else: dd_n += 1
    return kept, {
        "source_rows": len(rows),
        "dropped_test_pos": t_p, "dropped_test_neg": t_n, "dropped_test_total": t_p + t_n,
        "dropped_valid_pos": v_p, "dropped_valid_neg": v_n, "dropped_valid_total": v_p + v_n,
        "dup_collapsed_pos": dc_p, "dup_collapsed_neg": dc_n,
        "dup_disagreement_dropped_pos": dd_p, "dup_disagreement_dropped_neg": dd_n,
        "kept_rows": len(kept),
        "kept_pos": sum(1 for r in kept if r["label"] == 1),
        "kept_neg": sum(1 for r in kept if r["label"] == 0),
    }


def main():
    test_pairs = {bm: load_gs_pairs(p) for bm, p in TEST_FILES.items()}
    valid_pairs = {bm: load_valid_pairs(p) for bm, p in VALID_FILES.items()}
    report = {}
    for (method, bm), (run_subdir, profile) in SPECS.items():
        src = METHOD_ROOT[method] / run_subdir / "profiles" / profile / f"active_labels_latest_{bm}_{profile}_train.json.gz"
        if not src.exists():
            print(f"[skip] {method}/{bm}: missing {src}"); continue
        rows = load_train(src)
        kept, stats = clean(rows, test_pairs[bm], valid_pairs[bm])
        dst = ROOT / "output/labeling_cleaned_no_val" / method / bm / "profiles" / profile / f"active_labels_latest_{bm}_{profile}_train.json.gz"
        write_train(kept, dst)
        report.setdefault(method, {})[bm] = {"profile": profile, "source": str(src), "cleaned": str(dst), **stats}
        print(f"{method:>14} | {bm:<15} {profile:<18} src={stats['source_rows']:>6} "
              f"test={stats['dropped_test_total']:>4} valid={stats['dropped_valid_total']:>4} "
              f"kept={stats['kept_rows']} ({stats['kept_pos']}p/{stats['kept_neg']}n)")
    (ROOT / "output/labeling_cleaned_no_val/methods_cleanup_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
