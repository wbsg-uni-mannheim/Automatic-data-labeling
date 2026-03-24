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
