#!/usr/bin/env python3
"""Build all five paper cleaning variants from one frozen profile and review.

No labeling or training is performed. Review labels are joined by pair key;
original record text and row order are preserved in every training export.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.post_processing.build_closure_bridge_profiles import (
    PAIR_KEY_COLUMNS,
    _changed_label_keys,
    _detect_positive_bridge_edges,
    _filter_profile,
    _load_profile,
    _normalize_label,
    _pair_key,
    _write_json,
    _write_jsonl_gz,
)


def build_variants(original_profile: Path, reviewed_profile: Path, output_root: Path,
                   min_component_nodes: int = 3) -> dict:
    original_profile, reviewed_profile, output_root = (
        Path(p).resolve() for p in (original_profile, reviewed_profile, output_root)
    )
    active, final, train, _ = _load_profile(original_profile)
    reviewed = pd.read_csv(reviewed_profile / "active_labels_latest.csv")
    if not train or len(train) != len(active):
        raise ValueError("Original profile requires an aligned, nonempty Ditto training export")
    for frame in (active, reviewed):
        if not set(PAIR_KEY_COLUMNS).issubset(frame.columns):
            raise ValueError("Profile is missing pair key columns")
        if frame.duplicated(list(PAIR_KEY_COLUMNS)).any():
            raise ValueError("Profile contains duplicate pair keys")
    keys = [_pair_key(row) for row in active.to_dict("records")]
    review_labels = {
        _pair_key(row): _normalize_label(row["label"])
        for row in reviewed.to_dict("records")
    }
    if set(keys) != set(review_labels):
        raise ValueError("Review must cover exactly the frozen original pairs")
    if [_pair_key(row) for row in final.to_dict("records")] != keys:
        raise ValueError("Original CSV exports have different pair order")
    for row, source in zip(train, active.to_dict("records")):
        if any(str(row.get(f"id_{side}")) != str(source[f"id{number}"])
               for side, number in (("left", 1), ("right", 2))):
            raise ValueError("Ditto export and original CSV have different pair order")
        if _normalize_label(row["label"]) != _normalize_label(source["label"]):
            raise ValueError("Ditto export and original CSV have different labels")

    # These two calculations serve every variant. Never relabel per variant.
    changed = _changed_label_keys(active, reviewed)
    _, bridges = _detect_positive_bridge_edges(active, min_component_nodes=min_component_nodes)
    drop_sets = {
        "v_relabel": set(),
        "v_relabel_drop": changed,
        "v_closure_drop": bridges,
        "v_closure_and_relabel": bridges & changed,
        "v_closure_or_relabel": bridges | changed,
    }
    # Refuse to overwrite an existing experiment or either input profile.
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "original_profile": str(original_profile), "reviewed_profile": str(reviewed_profile),
        "min_component_nodes": min_component_nodes,
        "reviewed_pairs": len(keys), "changed_pairs": len(changed), "bridge_pairs": len(bridges),
        "variants": {},
    }
    identical = {}
    for name, dropped in drop_sets.items():
        out_active, out_final, rows, counts = _filter_profile(active, final, train, dropped)
        if name == "v_relabel":
            labels = [review_labels[key] for key in keys]
            out_active["label"] = labels
            out_final["label"] = labels
            rows = [{**row, "label": label} for row, label in zip(rows, labels)]
        directory = output_root / name
        directory.mkdir()
        out_active.to_csv(directory / "active_labels_latest.csv", index=False)
        out_final.to_csv(directory / "labels_final.csv", index=False)
        _write_jsonl_gz(directory / "train.json.gz", rows)
        # Hash ordered records, not gzip metadata. Equal hashes permit reuse
        # only when the seed, validation/test splits and training config match.
        digest = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        manifest["variants"][name] = {
            **counts, "train": str(directory / "train.json.gz"),
            "ordered_records_sha256": digest,
            "identical_training_variant": identical.get(digest),
        }
        identical.setdefault(digest, name)
    _write_json(output_root / "cleaning_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-profile", type=Path, required=True)
    parser.add_argument("--reviewed-profile", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-component-nodes", type=int, default=3)
    args = parser.parse_args()
    build_variants(args.original_profile, args.reviewed_profile, args.output_root, args.min_component_nodes)


if __name__ == "__main__":
    main()
