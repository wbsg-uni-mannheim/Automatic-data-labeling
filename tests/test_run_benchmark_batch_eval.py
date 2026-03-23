from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_benchmark_batch_eval.py"
    spec = importlib.util.spec_from_file_location("run_benchmark_batch_eval", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_discover_test_sets_uses_config_benchmarks(tmp_path: Path) -> None:
    module = _load_module()
    config_path = tmp_path / "benchmarks.yaml"
    data_root = tmp_path / "data"
    (data_root / "alpha").mkdir(parents=True)
    (data_root / "beta").mkdir(parents=True)

    config_path.write_text(
        "benchmarks:\n"
        "  alpha:\n"
        "    fields:\n"
        "      title: title\n"
        "  beta:\n"
        "    fields:\n"
        "      title: title\n",
        encoding="utf-8",
    )
    _write_jsonl_gz(data_root / "alpha" / "alpha-gs.json.gz", [{"label": 1}])
    _write_jsonl_gz(data_root / "beta" / "beta-a-gs.json.gz", [{"label": 0}])
    _write_jsonl_gz(data_root / "beta" / "beta-b-gs.json.gz", [{"label": 1}])

    specs = module.discover_test_sets(config_path=config_path, data_root=data_root)

    assert [spec.benchmark for spec in specs] == ["alpha", "beta", "beta"]
    assert [spec.dataset_name for spec in specs] == ["alpha-gs", "beta-a-gs", "beta-b-gs"]
    assert specs[0].prompt_fields == ("title",)


def test_build_messages_reuses_active_learning_prompt_style() -> None:
    module = _load_module()
    record = {
        "id_left": "l1",
        "id_right": "r1",
        "title_left": "Widget 1000",
        "title_right": "Widget 1000",
        "brand_left": "Acme",
        "brand_right": "Acme",
        "cluster_id_left": 1,
        "cluster_id_right": 2,
        "label": 1,
        "pair_id": "p1",
    }

    messages = module.build_messages(record, prompt_fields=("title", "brand"), max_field_length=200)

    assert messages[0]["content"] == module.ACTIVE_LEARNING_SYSTEM_PROMPT
    assert "Do the two entity descriptions refer to the same real-world entity?" in messages[1]["content"]
    assert '"title": "Widget 1000"' in messages[1]["content"]
    assert '"brand": "Acme"' in messages[1]["content"]
    assert "cluster_id" not in messages[1]["content"]
    assert "pair_id" not in messages[1]["content"]


def test_build_messages_uses_configured_field_aliases() -> None:
    module = _load_module()
    record = {
        "name_left": "Camera A",
        "name_right": "Camera B",
        "description_left": "left desc",
        "description_right": "right desc",
    }

    messages = module.build_messages(record, prompt_fields=("title", "description"), max_field_length=200)

    assert '"title": "Camera A"' in messages[1]["content"]
    assert '"description": "left desc"' in messages[1]["content"]


def test_evaluate_run_computes_metrics_from_batch_output(tmp_path: Path) -> None:
    module = _load_module()
    run_dir = tmp_path / "run_001"
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
            {
                "custom_id": "alpha__alpha-gs__req_0",
                "pair_index": 0,
                "pair_id": "p0",
                "id_left": "l0",
                "id_right": "r0",
                "gold_label": "TRUE",
                "gold_match": True,
            },
            {
                "custom_id": "alpha__alpha-gs__req_1",
                "pair_index": 1,
                "pair_id": "p1",
                "id_left": "l1",
                "id_right": "r1",
                "gold_label": "FALSE",
                "gold_match": False,
            },
        ]
    ).to_csv(dataset_dir / "metadata.csv", index=False)

    (run_dir / "batch_output.jsonl").write_text(
        json.dumps(
            {
                "custom_id": "alpha__alpha-gs__req_0",
                "response": {"body": {"choices": [{"message": {"content": '{"match": true}'}}]}},
            }
        )
        + "\n"
        + json.dumps(
            {
                "custom_id": "alpha__alpha-gs__req_1",
                "response": {"body": {"choices": [{"message": {"content": '{"match": false}'}}]}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary_df = module.evaluate_run(run_dir)

    assert len(summary_df) == 1
    row = summary_df.iloc[0].to_dict()
    assert row["pairs_scored"] == 2
    assert row["parse_failures"] == 0
    assert row["accuracy"] == 1.0
    assert row["precision"] == 1.0
    assert row["recall"] == 1.0
    assert row["f1"] == 1.0


def test_prepare_run_writes_single_combined_batch_file(tmp_path: Path) -> None:
    module = _load_module()

    data_root = tmp_path / "data"
    config_path = tmp_path / "benchmarks.yaml"
    (data_root / "alpha").mkdir(parents=True)
    (data_root / "beta").mkdir(parents=True)
    _write_jsonl_gz(
        data_root / "alpha" / "alpha-gs.json.gz",
        [{"id_left": "l1", "id_right": "r1", "title_left": "A", "title_right": "B", "label": 1}],
    )
    _write_jsonl_gz(
        data_root / "beta" / "beta-gs.json.gz",
        [{"id_left": "l2", "id_right": "r2", "title_left": "C", "title_right": "D", "label": 0}],
    )
    config_path.write_text(
        "benchmarks:\n"
        "  alpha:\n"
        "    fields:\n"
        "      title: title\n"
        "  beta:\n"
        "    fields:\n"
        "      title: title\n",
        encoding="utf-8",
    )

    module.DEFAULT_CONFIG_PATH = config_path
    module.DEFAULT_DATA_ROOT = data_root
    out_dir = tmp_path / "out"

    module.prepare_run(output_dir=out_dir, model="gpt-5.2")

    batch_input = out_dir / "batch_input.jsonl"
    assert batch_input.exists()
    assert not (out_dir / "alpha__alpha-gs" / "batch_input.jsonl").exists()
    lines = batch_input.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    payloads = [json.loads(line) for line in lines]
    assert payloads[0]["custom_id"] == "alpha__alpha-gs__req_0"
    assert payloads[1]["custom_id"] == "beta__beta-gs__req_0"


def test_evaluate_run_supports_legacy_per_dataset_outputs(tmp_path: Path) -> None:
    module = _load_module()
    run_dir = tmp_path / "run_legacy"
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
            {
                "custom_id": "req_0",
                "pair_index": 0,
                "pair_id": "p0",
                "id_left": "l0",
                "id_right": "r0",
                "gold_label": "TRUE",
                "gold_match": True,
            }
        ]
    ).to_csv(dataset_dir / "metadata.csv", index=False)

    (dataset_dir / "batch_output.jsonl").write_text(
        json.dumps(
            {
                "custom_id": "req_0",
                "response": {"body": {"choices": [{"message": {"content": '{"match": true}'}}]}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary_df = module.evaluate_run(run_dir)

    assert len(summary_df) == 1
    row = summary_df.iloc[0].to_dict()
    assert row["pairs_scored"] == 1
    assert row["accuracy"] == 1.0
