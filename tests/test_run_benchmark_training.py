from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "ditto"
        / "run_benchmark_training.py"
    )
    spec = importlib.util.spec_from_file_location("run_benchmark_training", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_valid_lookup_df_uses_explicit_lookup_file(tmp_path: Path) -> None:
    module = _load_module()
    lookup_path = tmp_path / "lookup.json"
    lookup_rows = [
        {
            "pair_id": "a#b",
            "label": 1,
            "title_left": "left a",
            "title_right": "right b",
            "description_left": "dl",
            "description_right": "dr",
            "price_left": "10",
            "price_right": "11",
        }
    ]
    lookup_path.write_text(json.dumps(lookup_rows))

    train_df = pd.DataFrame(
        [
            {
                "pair_id": "x#y",
                "label": 0,
                "title_left": "other",
                "title_right": "other",
                "description_left": "",
                "description_right": "",
                "price_left": "",
                "price_right": "",
            }
        ]
    )

    lookup_df = module._resolve_valid_lookup_df(
        benchmark="abt-buy",
        bcfg={"valid_lookup_train": str(lookup_path)},
        fields=["title", "description", "price"],
        field_aliases={},
        train_df=train_df,
    )

    assert len(lookup_df) == 1
    assert lookup_df.iloc[0]["pair_id"] == "a#b"


def test_build_valid_from_pair_ids_can_use_lookup_df_outside_train_subset() -> None:
    module = _load_module()
    valid_ids = pd.DataFrame([{"pair_id": "a#b"}])
    lookup_df = pd.DataFrame(
        [
            {
                "pair_id": "a#b",
                "label": 1,
                "title_left": "left a",
                "title_right": "right b",
                "description_left": "dl",
                "description_right": "dr",
                "price_left": "10",
                "price_right": "11",
            }
        ]
    )
    train_subset_df = pd.DataFrame(
        [
            {
                "pair_id": "x#y",
                "label": 0,
                "title_left": "other",
                "title_right": "other",
                "description_left": "",
                "description_right": "",
                "price_left": "",
                "price_right": "",
            }
        ]
    )

    out = module._build_valid_from_pair_ids(valid_ids, train_df=lookup_df, benchmark="abt-buy")

    assert len(out) == 1
    assert out.iloc[0]["pair_id"] == "a#b"
    assert out.iloc[0]["label"] == 1
    assert "a#b" not in train_subset_df["pair_id"].tolist()
