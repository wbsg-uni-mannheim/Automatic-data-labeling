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
        / "run_three_phase_labeling.py"
    )
    spec = importlib.util.spec_from_file_location("run_three_phase_labeling", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_phase_target_counts_uses_final_ratio() -> None:
    module = _load_module()

    pos, neg = module._resolve_phase_target_counts(
        final_target_size=2500,
        final_target_pos=500,
        stage_size=1000,
    )

    assert pos == 200
    assert neg == 800


def test_rank_probability_disagreements_prioritizes_score_variance() -> None:
    module = _load_module()

    correspondences = [
        {
            "matcher": "m1",
            "threshold": 0.5,
            "correspondences": pd.DataFrame(
                [
                    {"id1": "a", "id2": "b", "score": 0.90},
                    {"id1": "c", "id2": "d", "score": 0.90},
                    {"id1": "e", "id2": "f", "score": 0.60},
                ]
            ),
        },
        {
            "matcher": "m2",
            "threshold": 0.5,
            "correspondences": pd.DataFrame(
                [
                    {"id1": "a", "id2": "b", "score": 0.85},
                    {"id1": "c", "id2": "d", "score": 0.80},
                    {"id1": "e", "id2": "f", "score": 0.58},
                ]
            ),
        },
        {
            "matcher": "m3",
            "threshold": 0.5,
            "correspondences": pd.DataFrame(
                [
                    {"id1": "a", "id2": "b", "score": 0.20},
                    {"id1": "c", "id2": "d", "score": 0.40},
                    {"id1": "e", "id2": "f", "score": 0.62},
                ]
            ),
        },
        {
            "matcher": "m4",
            "threshold": 0.5,
            "correspondences": pd.DataFrame(
                [
                    {"id1": "a", "id2": "b", "score": 0.30},
                    {"id1": "c", "id2": "d", "score": 0.60},
                    {"id1": "e", "id2": "f", "score": 0.57},
                ]
            ),
        },
        {
            "matcher": "m5",
            "threshold": 0.5,
            "correspondences": pd.DataFrame(
                [
                    {"id1": "a", "id2": "b", "score": 0.75},
                    {"id1": "c", "id2": "d", "score": 0.70},
                    {"id1": "e", "id2": "f", "score": 0.61},
                ]
            ),
        },
    ]

    ranked = module._rank_probability_disagreements(correspondences)

    assert ranked[["id1", "id2"]].iloc[0].to_dict() == {"id1": "a", "id2": "b"}
    assert ranked.iloc[0]["score_variance"] > ranked.iloc[1]["score_variance"]
    assert ((ranked["id1"] == "e") & (ranked["id2"] == "f")).any()


def test_rank_probability_uncertainty_handles_no_disagreements() -> None:
    module = _load_module()

    correspondences = [
        {
            "matcher": "m1",
            "threshold": 0.5,
            "correspondences": pd.DataFrame(
                [
                    {"id1": "a", "id2": "b", "score": 0.51},
                    {"id1": "c", "id2": "d", "score": 0.80},
                    {"id1": "e", "id2": "f", "score": 0.62},
                ]
            ),
        },
        {
            "matcher": "m2",
            "threshold": 0.5,
            "correspondences": pd.DataFrame(
                [
                    {"id1": "a", "id2": "b", "score": 0.51},
                    {"id1": "c", "id2": "d", "score": 0.80},
                    {"id1": "e", "id2": "f", "score": 0.62},
                ]
            ),
        },
        {
            "matcher": "m3",
            "threshold": 0.5,
            "correspondences": pd.DataFrame(
                [
                    {"id1": "a", "id2": "b", "score": 0.51},
                    {"id1": "c", "id2": "d", "score": 0.80},
                    {"id1": "e", "id2": "f", "score": 0.62},
                ]
            ),
        },
    ]

    disagreements = module._rank_probability_disagreements(correspondences)
    ranked = module._rank_probability_uncertainty(correspondences)

    assert disagreements.empty
    assert ranked[["id1", "id2"]].iloc[0].to_dict() == {"id1": "a", "id2": "b"}
    assert ranked.iloc[0]["mean_margin"] < ranked.iloc[1]["mean_margin"]


def test_made_target_progress_ignores_already_satisfied_class_overshoot() -> None:
    module = _load_module()

    assert module._made_target_progress(
        prev_pos=10,
        prev_neg=20,
        cur_pos=10,
        cur_neg=30,
        target_pos=12,
        target_neg=20,
    ) is False
    assert module._made_target_progress(
        prev_pos=12,
        prev_neg=20,
        cur_pos=12,
        cur_neg=30,
        target_pos=12,
        target_neg=35,
    ) is True


def test_label_metadata_fills_seed_and_active_iterations() -> None:
    module = _load_module()
    df = pd.DataFrame(
        [
            {"id1": "a", "id2": "b", "label": "TRUE", "similarity": 0.9},
            {
                "id1": "c",
                "id2": "d",
                "label": "FALSE",
                "similarity": 0.2,
                "iteration": 3,
                "al_source": "uncertainty_mean_score",
            },
        ]
    )

    out = module._ensure_label_metadata(df, default_stage="seed", default_iteration=0)

    assert out["label_iteration"].tolist() == [0, 3]
    assert out["label_stage"].tolist() == ["seed", "active_learning"]


def test_remaining_class_budget_uses_class_deficits_not_total_size() -> None:
    module = _load_module()

    budget = module._remaining_class_budget(
        current_pos=1018,
        current_neg=1982,
        target_pos=600,
        target_neg=2400,
    )

    assert budget == 418


def test_make_bagged_split_preserves_both_classes() -> None:
    module = _load_module()
    labeled = pd.DataFrame(
        [
            {"pair_id": f"p{i}", "label": 1 if i % 3 == 0 else 0}
            for i in range(30)
        ]
    )

    train_df, valid_df = module._make_bagged_split(
        labeled,
        valid_fraction=0.2,
        bootstrap_fraction=1.0,
        seed=42,
    )

    assert not train_df.empty
    assert not valid_df.empty
    assert train_df["label"].nunique() == 2
    assert valid_df["label"].nunique() == 2


def test_make_bagged_split_targets_requested_ratio() -> None:
    module = _load_module()
    labeled = pd.DataFrame(
        [
            {"pair_id": f"p{i}", "label": 1 if i < 15 else 0}
            for i in range(100)
        ]
    )

    train_df, _valid_df = module._make_bagged_split(
        labeled,
        valid_fraction=0.2,
        bootstrap_fraction=1.0,
        seed=7,
        target_pos_ratio=0.4,
    )

    pos_ratio = float((train_df["label"] == 1).mean())
    assert len(train_df) == 80
    assert 0.35 <= pos_ratio <= 0.45


def test_estimate_usage_costs_for_gpt52() -> None:
    module = _load_module()

    summary = module._estimate_usage_costs(
        "gpt-5.2",
        {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
    )

    assert summary["available"] is True
    assert summary["input_cost_usd"] == 1.75
    assert summary["output_cost_usd"] == 14.0
    assert summary["total_cost_usd"] == 15.75
