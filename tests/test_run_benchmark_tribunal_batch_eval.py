from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_benchmark_tribunal_batch_eval.py"
    spec = importlib.util.spec_from_file_location("run_benchmark_tribunal_batch_eval", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_parse_agent_response_supports_confidence_percent() -> None:
    module = _load_module()
    parsed = module.parse_agent_response('{"match": false, "confidence": 82}')

    assert parsed["match"] is False
    assert abs(parsed["confidence"] - 0.82) < 1e-9
    assert abs(parsed["p_match"] - 0.18) < 1e-9


def test_prepare_run_writes_five_requests_per_pair(tmp_path: Path) -> None:
    module = _load_module()
    data_root = tmp_path / "data"
    config_path = tmp_path / "benchmarks.yaml"
    (data_root / "alpha").mkdir(parents=True)
    _write_jsonl_gz(
        data_root / "alpha" / "alpha-gs.json.gz",
        [
            {"id_left": "l1", "id_right": "r1", "title_left": "A", "title_right": "B", "label": 1},
            {"id_left": "l2", "id_right": "r2", "title_left": "C", "title_right": "D", "label": 0},
        ],
    )
    config_path.write_text(
        "benchmarks:\n"
        "  alpha:\n"
        "    fields:\n"
        "      title: title\n",
        encoding="utf-8",
    )

    module.DEFAULT_CONFIG_PATH = config_path
    module.DEFAULT_DATA_ROOT = data_root
    out_dir = tmp_path / "out"

    module.prepare_run(output_dir=out_dir, model="gpt-5.2")

    batch_input = out_dir / "alpha__alpha-gs" / "batch_input.jsonl"
    lines = batch_input.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2 * len(module.AGENTS)

    payloads = [json.loads(line) for line in lines]
    assert payloads[0]["custom_id"] == "pair-0__agent-balanced"
    assert payloads[-1]["custom_id"] == "pair-1__agent-contextualist"


def test_confidence_weighted_vote_prefers_high_confidence_minority() -> None:
    module = _load_module()
    row = pd.Series(
        {
            "balanced_match": True,
            "balanced_confidence": 0.97,
            "balanced_p_match": 0.97,
            "precision_match": False,
            "precision_confidence": 0.55,
            "precision_p_match": 0.45,
            "recall_match": False,
            "recall_confidence": 0.55,
            "recall_p_match": 0.45,
            "variant_skeptic_match": False,
            "variant_skeptic_confidence": 0.55,
            "variant_skeptic_p_match": 0.45,
            "contextualist_match": False,
            "contextualist_confidence": 0.55,
            "contextualist_p_match": 0.45,
        }
    )

    match, score = module._confidence_weighted_vote(row)

    assert match is True
    assert score > 0.5


def test_evaluate_run_outputs_tribunal_and_baselines(tmp_path: Path) -> None:
    module = _load_module()
    run_dir = tmp_path / "run"
    dataset_dir = run_dir / "alpha__alpha-gs"
    dataset_dir.mkdir(parents=True)
    module._write_json(
        run_dir / "run_manifest.json",
        {
            "datasets": [
                {
                    "benchmark": "alpha",
                    "dataset_name": "alpha-gs",
                    "output_slug": "alpha__alpha-gs",
                    "model": "gpt-5.2",
                }
            ]
        },
    )
    pd.DataFrame(
        [
            {"pair_index": 0, "pair_id": "p0", "id_left": "l0", "id_right": "r0", "gold_label": "TRUE", "gold_match": True},
            {"pair_index": 1, "pair_id": "p1", "id_left": "l1", "id_right": "r1", "gold_label": "FALSE", "gold_match": False},
        ]
    ).to_csv(dataset_dir / "metadata.csv", index=False)

    lines = []
    for pair_index, label, confidence in [
        (0, True, 90),
        (1, False, 90),
    ]:
        for agent_key in module.AGENTS:
            lines.append(
                json.dumps(
                    {
                        "custom_id": f"pair-{pair_index}__agent-{agent_key}",
                        "response": {
                            "body": {
                                "choices": [
                                    {
                                        "message": {
                                            "content": json.dumps({"match": label, "confidence": confidence})
                                        }
                                    }
                                ]
                            }
                        },
                    }
                )
            )
    (dataset_dir / "batch_output.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_df = module.evaluate_run(run_dir)

    methods = set(summary_df["method"].tolist())
    assert {"tribunal", "weighted_vote", "majority_vote", "balanced_single"}.issubset(methods)
    tribunal_row = summary_df[summary_df["method"] == "tribunal"].iloc[0].to_dict()
    assert tribunal_row["pairs_scored"] == 2
    assert tribunal_row["accuracy"] == 1.0
