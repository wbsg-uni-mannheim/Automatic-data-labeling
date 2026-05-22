from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "labeling"
        / "run_benchmark_labeling.py"
    )
    spec = importlib.util.spec_from_file_location("run_benchmark_labeling", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_random_profile_name_uses_percentage_suffix() -> None:
    module = _load_module()

    assert module._build_random_profile_name("small", 0.2) == "small_plus20random"


def test_select_random_augmented_profiles_only_large() -> None:
    module = _load_module()
    profiles = [
        module.ProfileSpec(name="small", all_examples=False, target_size=1000, target_pos=200, target_neg=800),
        module.ProfileSpec(name="medium", all_examples=False, target_size=3000, target_pos=600, target_neg=2400),
        module.ProfileSpec(name="large", all_examples=False, target_size=5000, target_pos=800, target_neg=4200),
        module.ProfileSpec(name="all", all_examples=True, target_size=0, target_pos=0, target_neg=0),
    ]

    selected = module._select_random_augmented_profiles(profiles)

    assert [spec.name for spec in selected] == ["large"]


def test_merge_base_with_random_labels_appends_requested_count() -> None:
    module = _load_module()
    base_subset = pd.DataFrame(
        [
            {"id1": "a", "id2": "b", "label": "TRUE"},
            {"id1": "c", "id2": "d", "label": "FALSE"},
        ]
    )
    random_labels = pd.DataFrame(
        [
            {"id1": "e", "id2": "f", "label": "FALSE"},
            {"id1": "g", "id2": "h", "label": "TRUE"},
        ]
    )

    out = module._merge_base_with_random_labels(base_subset, random_labels, extra_n=1)

    assert len(out) == 3
    assert set(out["label"].tolist()) == {"TRUE", "FALSE"}
    assert {("e", "f"), ("g", "h")} & set(zip(out["id1"], out["id2"]))
