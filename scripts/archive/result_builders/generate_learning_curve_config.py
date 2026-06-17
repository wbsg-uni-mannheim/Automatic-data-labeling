#!/usr/bin/env python3
"""Generate configs/ditto/learning_curve_all.yaml with one benchmark entry per
(benchmark, method, N) point, by enumerating existing subset training files.
"""
from pathlib import Path
import yaml

ROOT = Path("/work/aasteine/Automatic-data-labeling")
OUT_CONFIG = ROOT / "configs/ditto/learning_curve_all.yaml"

VALID_TEST = {
    "abt-buy": {
        "valid": "benchmarks/abt-buy/abt-buy-valid.csv",
        "valid_pair_id_only": True,
        "valid_lookup_train": "benchmarks/abt-buy/abt-buy-train.json",
        "test": "benchmarks/abt-buy/abt-buy-gs.json.gz",
        "fields": ["title", "description", "price"],
        "field_aliases": {"title": ["name"]},
    },
    "walmart-amazon": {
        "valid": "benchmarks/walmart-amazon/walmart-amazon-valid.csv",
        "test":  "benchmarks/walmart-amazon/walmart-amazon-gs.json.gz",
        "fields": ["title", "category", "brand", "modelno", "price"],
    },
    "dblp-acm": {
        "valid": "benchmarks/dblp-acm/dblp-acm-valid.csv",
        "test":  "benchmarks/dblp-acm/dblp-acm-gs.json.gz",
        "fields": ["title", "authors", "venue", "year"],
    },
    "dblp-scholar": {
        "valid": "benchmarks/dblp-scholar/dblp-scholar-valid.csv",
        "test":  "benchmarks/dblp-scholar/dblp-scholar-gs.json.gz",
        "fields": ["title", "authors", "venue", "year"],
    },
    "wdc": {
        "valid": "benchmarks/wdc/wdcproducts80cc20rnd000un_valid_large.json.gz",
        "test":  "benchmarks/wdc/wdcproducts80cc20rnd100un_gs.json.gz",
        "fields": ["title", "brand", "description", "price", "priceCurrency"],
    },
}

METHOD_SLUG = {
    "similarity_selection": "sim",
    "simple_active_learning": "alml",
    "ditto_active_learning": "alditto",
}


def main():
    cfg = {
        "defaults": {
            "output_root": "output/ditto_learning_curve_all",
            "active_profiles": ["all"],
            "model_name": "roberta-base",
            "batch_size": 64,
            "max_len": 256,
            "epochs": 50,
            "lr": 5.0e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "grad_accum_steps": 1,
            "early_stopping_patience": 10,
            "seed": 42,
            "num_workers": 2,
            "fp16": True,
            "da": "del",
            "alpha_aug": 0.8,
            "summarize": False,
            "dk": None,
            "spacy_model": "en_core_web_sm",
            "max_field_len": 350,
            "exclude_valid_from_train": False,
        },
        "benchmarks": {},
    }

    for bm in ["abt-buy", "walmart-amazon", "dblp-acm", "dblp-scholar", "wdc"]:
        lc_root = ROOT / f"output/learning_curve_{bm}"
        if not lc_root.exists():
            continue
        for method, slug in METHOD_SLUG.items():
            mdir = lc_root / method
            if not mdir.exists():
                continue
            for ndir in sorted(mdir.iterdir(), key=lambda p: int(p.name[1:]) if p.name.startswith("N") else 0):
                if not ndir.name.startswith("N"):
                    continue
                train_file = next(ndir.glob(f"{bm}_N*_train.json.gz"), None)
                if train_file is None:
                    continue
                key = f"{bm}_{slug}_{ndir.name}"
                entry = {
                    "train": str(train_file.relative_to(ROOT)),
                    **{k: v for k, v in VALID_TEST[bm].items()},
                }
                cfg["benchmarks"][key] = entry

    OUT_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CONFIG.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    n = len(cfg["benchmarks"])
    print(f"Wrote {OUT_CONFIG.relative_to(ROOT)} with {n} variants")
    # Sample
    for k in list(cfg["benchmarks"].keys())[:5]:
        print(f"  {k}")
    print(f"  ... ({n} total)")


if __name__ == "__main__":
    main()
