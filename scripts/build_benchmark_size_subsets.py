#!/usr/bin/env python3
"""Build chronological training subsets that land within ±5% of each benchmark's
official train-only size (raw train rows minus official validation pairs), for
each of the 3 selection methods (sim search, AL ML, AL Ditto).

Strategy per (benchmark, method):
  - If an existing profile json.gz has a size in the ±5% band, reuse it.
  - Else build a head-N chronological subset from active_labels_latest.csv +
    random_profile_labels.csv (concat, dedupe, head N=benchmark_train_only).

Outputs:
  output/benchmark_size_runs/<method>/<benchmark>/<benchmark>_train.json.gz
  output/benchmark_size_runs/_manifest.json with picked source + size.
"""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

import pandas as pd
import gzip

ROOT = Path("/work/aasteine/Automatic-data-labeling")
OUT_ROOT = ROOT / "output/benchmark_size_runs"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

TARGETS = {
    "abt-buy":        5743,
    "walmart-amazon": 6144,
    "dblp-acm":       7417,
    "dblp-scholar":   17223,
    "wdc":            19835,
}
METHODS = {
    "sim":      "seed_round_only_profiles",
    "al_ml":    "simple_active_learning_labeling",
    "al_ditto": "three_phase_labeling_ditto_only_v2",
}
FIELDS_PER_BM = {
    "abt-buy":        "title,description,price",
    "walmart-amazon": "title,category,brand,modelno,price",
    "dblp-acm":       "title,authors,venue,year",
    "dblp-scholar":   "title,authors,venue,year",
    "wdc":            "title,brand,description,price,priceCurrency",
}


def count_gz(p):
    n = 0
    with gzip.open(p, "rt") as f:
        for line in f:
            if line.strip(): n += 1
    return n


def find_run_dir(method, benchmark):
    cand = list((ROOT / "output" / METHODS[method]).glob(f"benchmark_{benchmark}_*"))
    if not cand:
        return None
    # newest by name (timestamp suffix)
    return sorted(cand)[-1]


def list_profiles(run_dir, benchmark):
    out = []
    pdir = run_dir / "profiles"
    if not pdir.exists():
        return out
    for sub in sorted(pdir.iterdir()):
        f = sub / f"active_labels_latest_{benchmark}_{sub.name}_train.json.gz"
        if f.exists():
            out.append((sub.name, count_gz(f), f))
    return out


def pick_in_band(profiles, target):
    lo, hi = target * 0.95, target * 1.05
    matches = [p for p in profiles if lo <= p[1] <= hi]
    if matches:
        return min(matches, key=lambda x: abs(x[1] - target))
    return None


def build_trim(run_dir, benchmark, target, out_dir):
    """Build head-N chronological subset."""
    active = pd.read_csv(run_dir / "active_labels_latest.csv")
    rand_path = run_dir / "random_profile_labels.csv"
    if rand_path.exists():
        rand = pd.read_csv(rand_path)
        df = pd.concat([active, rand], ignore_index=True, sort=False)
    else:
        df = active
    df["label"] = df["label"].astype(str).str.upper()
    df = df.drop_duplicates(subset=["id1", "id2"], keep="first").reset_index(drop=True)
    if len(df) < target:
        return None, len(df)
    sub = df.head(target).copy()
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_csv = out_dir / "labels_subset.csv"
    sub.to_csv(labels_csv, index=False)
    out_jsongz = out_dir / f"{benchmark}_train.json.gz"
    cmd = [
        sys.executable,
        str(ROOT / "scripts/ditto/convert_active_labels_to_wdc.py"),
        "--labels-csv", str(labels_csv),
        "--left-csv", str(run_dir / "canonical/left.csv"),
        "--right-csv", str(run_dir / "canonical/right.csv"),
        "--output-json-gz", str(out_jsongz),
        "--fields", FIELDS_PER_BM[benchmark],
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    pos = int((sub["label"] == "TRUE").sum())
    neg = int((sub["label"] == "FALSE").sum())
    return (out_jsongz, target, pos, neg), len(df)


def main():
    manifest = {"targets": TARGETS, "results": {}}
    for bm, target in TARGETS.items():
        manifest["results"][bm] = {}
        for method in ("sim", "al_ml", "al_ditto"):
            entry = manifest["results"][bm][method] = {"target": target}
            run_dir = find_run_dir(method, bm)
            if not run_dir:
                entry["status"] = "no_run_dir"
                continue
            entry["run_dir"] = str(run_dir.relative_to(ROOT))
            profiles = list_profiles(run_dir, bm)
            picked = pick_in_band(profiles, target)
            if picked:
                entry["status"] = "in_band"
                entry["profile"] = picked[0]
                entry["size"] = picked[1]
                entry["train_json_gz"] = str(picked[2].relative_to(ROOT))
                continue
            # Need to trim from raw labels
            out_dir = OUT_ROOT / method / bm
            result, total_available = build_trim(run_dir, bm, target, out_dir)
            if result is None:
                entry["status"] = "insufficient_labels"
                entry["total_available"] = total_available
                continue
            entry["status"] = "trimmed_to_target"
            entry["size"] = target
            entry["total_available"] = total_available
            entry["pos"] = result[2]
            entry["neg"] = result[3]
            entry["train_json_gz"] = str(result[0].relative_to(ROOT))

    manifest_path = OUT_ROOT / "_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    # Pretty print summary
    print(f"{'benchmark':15s} {'method':10s} {'target':>7s} {'size':>7s} {'status':25s} {'pos':>5s} {'neg':>5s}  source")
    for bm in TARGETS:
        for method in ("sim", "al_ml", "al_ditto"):
            e = manifest["results"][bm][method]
            size = e.get("size", "-")
            pos = e.get("pos", "-")
            neg = e.get("neg", "-")
            src = e.get("profile") or e.get("train_json_gz", "")
            print(f"{bm:15s} {method:10s} {TARGETS[bm]:>7d} {str(size):>7s} {e['status']:25s} {str(pos):>5s} {str(neg):>5s}  {src}")
    print(f"\nmanifest: {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
