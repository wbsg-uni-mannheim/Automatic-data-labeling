from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from third_party.ditto_modern.data import WDC_COLUMNS, load_wdc_json_gz, write_wdc_json_gz
from third_party.ditto_modern.pseudolabels import build_pseudolabels


def _source_row(pair_id: str, label: int):
    return {
        "id_left": "1",
        "brand_left": "B",
        "title_left": "T1",
        "description_left": "D1",
        "price_left": 1.0,
        "priceCurrency_left": "USD",
        "cluster_id_left": "c1",
        "id_right": "2",
        "brand_right": "B",
        "title_right": "T2",
        "description_right": "D2",
        "price_right": 2.0,
        "priceCurrency_right": "USD",
        "cluster_id_right": "c2",
        "pair_id": pair_id,
        "label": label,
        "is_hard_negative": False,
    }


def test_consensus_policy_filters_majority(tmp_path: Path):
    src_df = pd.DataFrame(
        [
            _source_row("p1", 0),
            _source_row("p2", 1),
            _source_row("p3", 0),
        ],
        columns=WDC_COLUMNS,
    )
    src_path = tmp_path / "source.json.gz"
    write_wdc_json_gz(src_df, src_path)

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame(
        {
            "pair_id": ["p1", "p2", "p3"],
            "pred_precision": [1, 0, 1],
            "pred_balanced": [1, 0, 1],
            "pred_recall": [1, 0, 0],
            "pred_vote": [1, 0, 1],
        }
    )
    pred_df.to_csv(run_dir / "predictions.csv", index=False)

    out_path = tmp_path / "pseudo.json.gz"
    report = build_pseudolabels(
        source_json_gz=str(src_path),
        multi_agent_run_dir=str(run_dir),
        output_json_gz=str(out_path),
        policy="consensus",
    )

    out_df = load_wdc_json_gz(out_path)
    assert len(out_df) == 2
    assert set(out_df["pair_id"]) == {"p1", "p2"}
    assert report["rows_kept"] == 2
    assert report["agreement"]["majority_2of3"] == 1


def test_consensus_with_majority_weights(tmp_path: Path):
    src_df = pd.DataFrame([_source_row("p1", 0), _source_row("p2", 1), _source_row("p3", 0)], columns=WDC_COLUMNS)
    src_path = tmp_path / "source.json.gz"
    write_wdc_json_gz(src_df, src_path)

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "pair_id": ["p1", "p2", "p3"],
            "pred_precision": [1, 0, 1],
            "pred_balanced": [1, 0, 1],
            "pred_recall": [1, 0, 0],
            "pred_vote": [1, 0, 1],
        }
    ).to_csv(run_dir / "predictions.csv", index=False)

    out_path = tmp_path / "pseudo.json.gz"
    report = build_pseudolabels(
        source_json_gz=str(src_path),
        multi_agent_run_dir=str(run_dir),
        output_json_gz=str(out_path),
        policy="consensus",
        include_majority_with_weight=True,
        majority_weight=0.4,
    )

    out_df = load_wdc_json_gz(out_path)
    assert len(out_df) == 3

    weights_csv = Path(report["sample_weights_csv"])
    weights = pd.read_csv(weights_csv)
    assert (weights[weights["agreement_count"] == 2]["sample_weight"] == 0.4).all()


def test_pair_id_mismatch_raises(tmp_path: Path):
    src_df = pd.DataFrame([_source_row("p1", 0)], columns=WDC_COLUMNS)
    src_path = tmp_path / "source.json.gz"
    write_wdc_json_gz(src_df, src_path)

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "pair_id": ["missing"],
            "pred_precision": [1],
            "pred_balanced": [1],
            "pred_recall": [1],
            "pred_vote": [1],
        }
    ).to_csv(run_dir / "predictions.csv", index=False)

    out_path = tmp_path / "pseudo.json.gz"
    try:
        build_pseudolabels(
            source_json_gz=str(src_path),
            multi_agent_run_dir=str(run_dir),
            output_json_gz=str(out_path),
            policy="consensus",
        )
        assert False, "Expected ValueError due to pair_id mismatch"
    except ValueError as exc:
        assert "Missing predictions" in str(exc)
