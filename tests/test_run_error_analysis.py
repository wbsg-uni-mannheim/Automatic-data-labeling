from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_error_analysis.py"
    spec = importlib.util.spec_from_file_location("run_error_analysis", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_predictions_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_collect_cases_from_run_dir_enriches_records(tmp_path: Path) -> None:
    module = _load_module()
    data_root = tmp_path / "data"
    run_dir = tmp_path / "output" / "batch_eval"
    dataset_dir = run_dir / "alpha__alpha-gs"
    dataset_dir.mkdir(parents=True)

    _write_jsonl_gz(
        data_root / "alpha" / "alpha-gs.json.gz",
        [
            {
                "id_left": "l0",
                "id_right": "r0",
                "title_left": "Alpha Camera",
                "brand_left": "Acme",
                "title_right": "Beta Camera",
                "brand_right": "Other",
                "pair_id": "l0#r0",
                "label": 0,
            }
        ],
    )

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
                "pair_id": "l0#r0",
                "id_left": "l0",
                "id_right": "r0",
                "gold_match": False,
                "predicted_match": True,
                "response_text": '{"match": true}',
            }
        ]
    ).to_csv(dataset_dir / "predictions.csv", index=False)

    out_dir = tmp_path / "analysis"
    cases = module.collect_cases(
        artifacts=[run_dir],
        output_dir=out_dir,
        data_root=data_root,
        benchmark=None,
        dataset_name=None,
        model=None,
        sample_per_error_type=None,
    )

    assert len(cases) == 1
    row = cases.iloc[0].to_dict()
    assert row["error_type"] == "false_positive"
    assert row["benchmark"] == "alpha"
    assert row["dataset_name"] == "alpha-gs"
    assert '"title": "Alpha Camera"' in row["left_record_json"]
    assert '"brand": "Other"' in row["right_record_json"]
    assert (out_dir / "cases.csv").exists()
    assert (out_dir / "cases.jsonl").exists()


def test_collect_cases_from_exported_ditto_predictions_uses_benchmark_report(tmp_path: Path) -> None:
    module = _load_module()
    data_root = tmp_path / "data"
    export_dir = tmp_path / "export" / "ditto_baseline" / "alpha"
    run_dir = export_dir / "training_output" / "run_001"
    run_dir.mkdir(parents=True)

    _write_jsonl_gz(
        data_root / "alpha" / "alpha-gs.json.gz",
        [
            {
                "id_left": "l0",
                "id_right": "r0",
                "title_left": "Alpha Camera",
                "title_right": "Beta Camera",
                "pair_id": "l0#r0",
                "label": 0,
            }
        ],
    )
    module._write_json(
        export_dir / "benchmark_report.json",
        {
            "benchmark": "alpha",
            "paths": {"test_input": "data/alpha/alpha-gs.json.gz"},
        },
    )
    pd.DataFrame(
        [
            {
                "idx": 0,
                "pair_id": "l0#r0",
                "gold": 0,
                "pred": 1,
                "prob": 0.91,
            }
        ]
    ).to_csv(run_dir / "predictions.csv", index=False)

    out_dir = tmp_path / "analysis"
    cases = module.collect_cases(
        artifacts=[run_dir / "predictions.csv"],
        output_dir=out_dir,
        data_root=data_root,
        benchmark=None,
        dataset_name=None,
        model=None,
        sample_per_error_type=None,
    )

    assert len(cases) == 1
    row = cases.iloc[0].to_dict()
    assert row["benchmark"] == "alpha"
    assert row["dataset_name"] == "alpha-gs"
    assert row["model"] == "ditto"
    assert row["id_left"] == "l0"
    assert row["id_right"] == "r0"
    assert row["error_type"] == "false_positive"


def test_default_collect_output_dir_uses_export_run_folder(tmp_path: Path) -> None:
    module = _load_module()
    module.DEFAULT_EXPORT_ROOT = tmp_path / "export"

    artifact = module.DEFAULT_EXPORT_ROOT / "ditto_baseline" / "alpha" / "training_output" / "run_001" / "predictions.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("pair_id,gold,pred\n", encoding="utf-8")

    output_dir = module._default_collect_output_dir([artifact])

    assert output_dir == artifact.parent / "error_analysis"


def test_default_child_output_dir_uses_sibling_folder_for_explanations_csv(tmp_path: Path) -> None:
    module = _load_module()
    cases_csv = tmp_path / "export" / "foo" / "training_output" / "run_001" / "error_analysis" / "explanations" / "cases_with_explanations.csv"
    cases_csv.parent.mkdir(parents=True)
    cases_csv.write_text("", encoding="utf-8")

    output_dir = module._default_child_output_dir(cases_csv, "summarization")

    assert output_dir == cases_csv.parent.parent / "summarization"


def test_discover_export_prediction_artifacts_ignores_error_analysis(tmp_path: Path) -> None:
    module = _load_module()
    export_dir = tmp_path / "export_001"
    keep = export_dir / "training_from_generated_labels" / "method_a" / "run_a" / "alpha" / "training_output" / "run_001" / "predictions.csv"
    ignore = export_dir / "error_analysis" / "explanations" / "predictions.csv"
    _write_predictions_csv(keep, [{"pair_id": "l0#r0", "gold": 0, "pred": 1}])
    _write_predictions_csv(ignore, [{"pair_id": "l0#r0", "gold": 0, "pred": 1}])

    discovered = module.discover_export_prediction_artifacts(export_dir)

    assert discovered == [keep.resolve()]


def test_prepare_export_analysis_collects_all_export_cases_and_writes_batch_input(tmp_path: Path) -> None:
    module = _load_module()
    export_dir = tmp_path / "export_001"
    data_root = export_dir / "data"
    _write_jsonl_gz(
        data_root / "alpha" / "alpha-gs.json.gz",
        [
            {
                "id_left": "l0",
                "id_right": "r0",
                "title_left": "Alpha Camera",
                "title_right": "Beta Camera",
                "pair_id": "l0#r0",
                "label": 0,
            },
            {
                "id_left": "l1",
                "id_right": "r1",
                "title_left": "Gamma Camera",
                "title_right": "Gamma Camera",
                "pair_id": "l1#r1",
                "label": 0,
            },
        ],
    )

    generated_predictions = (
        export_dir
        / "training_from_generated_labels"
        / "method_a"
        / "generated_ditto_benchmark_alpha_20260414_120000_small_20260414_123000"
        / "alpha"
        / "training_output"
        / "run_001"
        / "predictions.csv"
    )
    _write_predictions_csv(
        generated_predictions,
        [{"pair_id": "l0#r0", "gold": 0, "pred": 1, "prob": 0.91}],
    )
    module._write_json(
        generated_predictions.parents[2] / "benchmark_report.json",
        {
            "benchmark": "alpha",
            "paths": {"test_input": "data/alpha/alpha-gs.json.gz"},
        },
    )

    baseline_predictions = (
        export_dir
        / "ditto_baseline"
        / "alpha"
        / "training_output"
        / "run_002"
        / "predictions.csv"
    )
    _write_predictions_csv(
        baseline_predictions,
        [{"pair_id": "l1#r1", "gold": 0, "pred": 1, "prob": 0.88}],
    )
    module._write_json(
        export_dir / "ditto_baseline" / "alpha" / "benchmark_report.json",
        {
            "benchmark": "alpha",
            "paths": {"test_input": "data/alpha/alpha-gs.json.gz"},
        },
    )

    output_dir = export_dir / "error_analysis"
    payload = module.prepare_export_analysis(
        export_dir=export_dir,
        output_dir=output_dir,
        data_root=data_root,
        explanation_model="gpt-5-mini",
        schema_mode="none",
        sample_per_error_type=None,
        best_profile_only=False,
        best_run_only=False,
        submit_batch=False,
    )

    assert payload["case_count"] == 2
    cases = pd.read_csv(output_dir / "cases.csv")
    assert sorted(cases["method_name"].unique().tolist()) == ["ditto_baseline", "method_a"]
    generated_row = cases[cases["method_name"] == "method_a"].iloc[0].to_dict()
    assert generated_row["source_family"] == "training_from_generated_labels"
    assert generated_row["profile_name"] == "small"
    baseline_row = cases[cases["method_name"] == "ditto_baseline"].iloc[0].to_dict()
    assert baseline_row["profile_name"] == "official"
    assert (output_dir / "explanations" / "batch_input.jsonl").exists()
    assert (output_dir / "export_analysis_manifest.json").exists()


def test_extract_export_context_supports_training_root_subdir() -> None:
    module = _load_module()
    export_dir = Path("/tmp/export/training_from_generated_labels_3runs")
    artifact = export_dir / "method_a" / "generated_ditto_benchmark_alpha_20260323_202820_small_r2_20260416_211130" / "alpha" / "training_output" / "run_001" / "predictions.csv"
    context = module._extract_export_context(artifact, export_dir)
    assert context["source_family"] == "training_from_generated_labels_3runs"
    assert context["method_name"] == "method_a"
    assert context["run_name"] == "generated_ditto_benchmark_alpha_20260323_202820_small_r2_20260416_211130"
    assert context["profile_name"] == "small"
    assert context["benchmark_dir_name"] == "alpha"
    assert context["training_run_name"] == "run_001"


def test_select_best_profiles_and_filter_export_cases(tmp_path: Path) -> None:
    module = _load_module()
    export_dir = tmp_path / "export" / "training_from_generated_labels_3runs"
    data_root = tmp_path / "export" / "data"
    _write_jsonl_gz(
        data_root / "alpha" / "alpha-gs.json.gz",
        [
            {
                "id_left": "l0",
                "id_right": "r0",
                "title_left": "Alpha Camera",
                "title_right": "Beta Camera",
                "pair_id": "l0#r0",
                "label": 0,
            }
        ],
    )

    for repeat, f1 in [(1, 0.80), (2, 0.82), (3, 0.84)]:
        run_dir = export_dir / "method_a" / f"generated_ditto_benchmark_alpha_20260323_202820_small_r{repeat}_20260416_211130"
        _write_predictions_csv(
            run_dir / "alpha" / "training_output" / f"run_{repeat:03d}" / "predictions.csv",
            [{"pair_id": "l0#r0", "gold": 0, "pred": 1, "prob": 0.91}],
        )
        module._write_json(
            run_dir / "alpha" / "benchmark_report.json",
            {"benchmark": "alpha", "paths": {"test_input": "data/alpha/alpha-gs.json.gz"}},
        )
        _write_predictions_csv(run_dir / "summary.csv", [{"benchmark": "alpha", "status": "ok", "test_f1": f1}])

    for repeat, f1 in [(1, 0.70), (2, 0.72), (3, 0.74)]:
        run_dir = export_dir / "method_a" / f"generated_ditto_benchmark_alpha_20260323_202820_medium_r{repeat}_20260416_211130"
        _write_predictions_csv(
            run_dir / "alpha" / "training_output" / f"run_x{repeat}" / "predictions.csv",
            [{"pair_id": "l0#r0", "gold": 0, "pred": 1, "prob": 0.88}],
        )
        module._write_json(
            run_dir / "alpha" / "benchmark_report.json",
            {"benchmark": "alpha", "paths": {"test_input": "data/alpha/alpha-gs.json.gz"}},
        )
        _write_predictions_csv(run_dir / "summary.csv", [{"benchmark": "alpha", "status": "ok", "test_f1": f1}])

    selected = module.select_best_profiles_by_method(export_dir)
    assert len(selected) == 1
    selected_row = selected.iloc[0].to_dict()
    assert selected_row["method_name"] == "method_a"
    assert selected_row["benchmark"] == "alpha"
    assert selected_row["profile_name"] == "small"

    out_dir = tmp_path / "analysis"
    cases = module.collect_export_cases(
        export_dir=export_dir,
        output_dir=out_dir,
        data_root=data_root,
        sample_per_error_type=None,
        best_profile_only=True,
        best_run_only=False,
    )
    assert len(cases) == 3
    assert set(cases["profile_name"].unique().tolist()) == {"small"}
    selected_profiles = pd.read_csv(out_dir / "selected_profiles.csv")
    assert selected_profiles.iloc[0]["benchmark"] == "alpha"
    assert selected_profiles.iloc[0]["profile_name"] == "small"


def test_select_best_runs_and_filter_export_cases(tmp_path: Path) -> None:
    module = _load_module()
    export_dir = tmp_path / "export" / "training_from_generated_labels_3runs"
    data_root = tmp_path / "export" / "data"
    _write_jsonl_gz(
        data_root / "alpha" / "alpha-gs.json.gz",
        [
            {
                "id_left": "l0",
                "id_right": "r0",
                "title_left": "Alpha Camera",
                "title_right": "Beta Camera",
                "pair_id": "l0#r0",
                "label": 0,
            }
        ],
    )

    for repeat, f1 in [(1, 0.80), (2, 0.85), (3, 0.82)]:
        run_dir = export_dir / "method_a" / f"generated_ditto_benchmark_alpha_20260323_202820_small_r{repeat}_20260416_211130"
        _write_predictions_csv(
            run_dir / "alpha" / "training_output" / f"run_{repeat:03d}" / "predictions.csv",
            [{"pair_id": "l0#r0", "gold": 0, "pred": 1, "prob": 0.91}],
        )
        module._write_json(
            run_dir / "alpha" / "benchmark_report.json",
            {"benchmark": "alpha", "paths": {"test_input": "data/alpha/alpha-gs.json.gz"}},
        )
        _write_predictions_csv(run_dir / "summary.csv", [{"benchmark": "alpha", "status": "ok", "test_f1": f1}])

    selected = module.select_best_runs_by_method(export_dir)
    assert len(selected) == 1
    selected_row = selected.iloc[0].to_dict()
    assert selected_row["method_name"] == "method_a"
    assert selected_row["run_name"].endswith("small_r2_20260416_211130")

    out_dir = tmp_path / "analysis"
    cases = module.collect_export_cases(
        export_dir=export_dir,
        output_dir=out_dir,
        data_root=data_root,
        sample_per_error_type=None,
        best_profile_only=False,
        best_run_only=True,
    )
    assert len(cases) == 1
    assert set(cases["run_name"].unique().tolist()) == {selected_row["run_name"]}
    selected_runs = pd.read_csv(out_dir / "selected_runs.csv")
    assert selected_runs.iloc[0]["run_name"] == selected_row["run_name"]


def test_select_best_profiles_per_method_and_benchmark(tmp_path: Path) -> None:
    module = _load_module()
    export_dir = tmp_path / "export" / "training_from_generated_labels_3runs"

    small_run = export_dir / "method_a" / "generated_ditto_benchmark_alpha_20260323_202820_small_r1_20260416_211130"
    medium_run = export_dir / "method_a" / "generated_ditto_benchmark_alpha_20260323_202820_medium_r1_20260416_211130"
    for run_dir, rows in [
        (
            small_run,
            [
                {"benchmark": "alpha", "status": "ok", "test_f1": 0.92},
                {"benchmark": "beta", "status": "ok", "test_f1": 0.78},
            ],
        ),
        (
            medium_run,
            [
                {"benchmark": "alpha", "status": "ok", "test_f1": 0.90},
                {"benchmark": "beta", "status": "ok", "test_f1": 0.84},
            ],
        ),
    ]:
        _write_predictions_csv(run_dir / "summary.csv", rows)

    selected = module.select_best_profiles_by_method(export_dir)
    assert len(selected) == 2
    selected_rows = {
        (row["method_name"], row["benchmark"]): row["profile_name"]
        for row in selected.to_dict("records")
    }
    assert selected_rows[("method_a", "alpha")] == "small"
    assert selected_rows[("method_a", "beta")] == "medium"


def test_prepare_explanations_writes_batch_input(tmp_path: Path) -> None:
    module = _load_module()
    cases_csv = tmp_path / "cases.csv"
    pd.DataFrame(
        [
            {
                "case_id": "alpha__alpha-gs__gpt-5-2__false-positive__l0-r0",
                "benchmark": "alpha",
                "dataset_name": "alpha-gs",
                "model": "gpt-5.2",
                "error_type": "false_positive",
                "gold_label": "non_match",
                "predicted_label": "match",
                "pair_id": "l0#r0",
                "left_record_json": '{"title":"Alpha Camera"}',
                "right_record_json": '{"title":"Beta Camera"}',
            }
        ]
    ).to_csv(cases_csv, index=False)

    out_dir = tmp_path / "explanations"
    metadata = module.prepare_explanations(
        cases_csv=cases_csv,
        output_dir=out_dir,
        model="gpt-5-mini",
        schema_mode="schema",
    )

    assert len(metadata) == 1
    batch_lines = (out_dir / "batch_input.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(batch_lines) == 1
    payload = json.loads(batch_lines[0])
    assert payload["custom_id"] == "alpha__alpha-gs__gpt-5-2__false-positive__l0-r0"
    assert payload["body"]["model"] == "gpt-5-mini"
    assert payload["body"]["response_format"]["type"] == "json_object"
    prompt_text = payload["body"]["messages"][0]["content"]
    assert "Do the two entity descriptions refer to the same real-world entity?" in prompt_text
    assert "Decision: Yes" in prompt_text
    assert "attribute=brand|||importance=0.05" in prompt_text
    assert (out_dir / "metadata.csv").exists()
    assert (out_dir / "batch_manifest.json").exists()


def test_prepare_category_classification_batch_writes_batch_input(tmp_path: Path) -> None:
    module = _load_module()
    cases_csv = tmp_path / "cases_with_explanations.csv"
    pd.DataFrame(
        [
            {
                "case_id": "case_1",
                "benchmark": "alpha",
                "dataset_name": "alpha-gs",
                "model": "gpt-5.2",
                "method_name": "method_a",
                "profile_name": "small",
                "error_type": "false_positive",
                "gold_match": False,
                "predicted_match": True,
                "left_record_json": '{"title":"Alpha Camera"}',
                "right_record_json": '{"title":"Beta Camera"}',
                "explanation_raw": "Decision: Yes",
            }
        ]
    ).to_csv(cases_csv, index=False)
    registry_path = tmp_path / "category_registry.json"
    module._write_json(
        registry_path,
        {
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
            "next_category_index": 3,
            "categories": [
                {"id": "C001", "name": "Title mismatch", "description": "Title differs", "applies_to": ["false_positive"]},
                {"id": "C002", "name": "Price mismatch", "description": "Price differs", "applies_to": ["false_negative"]},
            ],
        },
    )

    out_dir = tmp_path / "category_batch"
    metadata = module.prepare_category_classification_batch(
        cases_csv=cases_csv,
        registry_path=registry_path,
        output_dir=out_dir,
        model="gpt-5-mini",
    )

    assert len(metadata) == 1
    payload = json.loads((out_dir / "batch_input.jsonl").read_text(encoding="utf-8").strip())
    assert payload["custom_id"] == "case_1"
    assert payload["body"]["response_format"]["type"] == "json_object"
    assert (out_dir / "metadata.csv").exists()


def test_merge_category_classification_batch_builds_pivot_csv(tmp_path: Path) -> None:
    module = _load_module()
    cases_csv = tmp_path / "cases_with_explanations.csv"
    pd.DataFrame(
        [
            {
                "case_id": "case_1",
                "benchmark": "alpha",
                "dataset_name": "alpha-gs",
                "model": "gpt-5.2",
                "method_name": "method_a",
                "profile_name": "small",
                "run_name": "run_001",
                "pair_id": "l0#r0",
                "error_type": "false_positive",
                "left_record_json": '{"title":"Alpha Camera"}',
                "right_record_json": '{"title":"Beta Camera"}',
                "what_went_wrong": "title dominated price",
                "strongest_positive_attribute": "title",
                "strongest_negative_attribute": "price",
            }
        ]
    ).to_csv(cases_csv, index=False)
    registry_path = tmp_path / "category_registry.json"
    module._write_json(
        registry_path,
        {
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
            "next_category_index": 2,
            "categories": [
                {"id": "C001", "name": "Price mismatch", "description": "Price differs", "applies_to": ["false_positive"]},
            ],
        },
    )
    out_dir = tmp_path / "category_batch"
    out_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "custom_id": "case_1",
                "case_id": "case_1",
                "benchmark": "alpha",
                "method_name": "method_a",
                "profile_name": "small",
                "error_type": "false_positive",
                "registry_number_to_id": json.dumps({"1": "C001"}),
            }
        ]
    ).to_csv(out_dir / "metadata.csv", index=False)
    (out_dir / "batch_output.jsonl").write_text(
        json.dumps(
            {
                "custom_id": "case_1",
                "response": {
                    "body": {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps({"categories": [{"number": "1", "confidence": "91"}]})
                                }
                            }
                        ]
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    merged = module.merge_category_classification_batch(
        cases_csv=cases_csv,
        registry_path=registry_path,
        output_dir=out_dir,
    )

    row = merged.iloc[0].to_dict()
    assert row["assigned_category_id"] == "C001"
    assert row["assigned_category_name"] == "Price mismatch"
    assert (out_dir / "pivot_error_cases.csv").exists()


def test_collect_export_cases_builds_unique_case_ids_per_method_benchmark_profile(tmp_path: Path) -> None:
    module = _load_module()
    export_dir = tmp_path / "export" / "training_from_generated_labels_3runs"
    data_root = tmp_path / "export" / "data"
    _write_jsonl_gz(
        data_root / "alpha" / "alpha-gs.json.gz",
        [
            {
                "id_left": "l0",
                "id_right": "r0",
                "title_left": "Alpha Camera",
                "title_right": "Beta Camera",
                "pair_id": "l0#r0",
                "label": 0,
            }
        ],
    )

    for run_name in [
        "generated_ditto_benchmark_alpha_20260323_202820_small_r1_20260416_211130",
        "generated_ditto_benchmark_alpha_20260323_202820_small_r2_20260416_211130",
    ]:
        predictions_path = export_dir / "method_a" / run_name / "alpha" / "training_output" / "run_001" / "predictions.csv"
        _write_predictions_csv(
            predictions_path,
            [{"pair_id": "l0#r0", "gold": 0, "pred": 1, "prob": 0.91}],
        )
        module._write_json(
            predictions_path.parents[2] / "benchmark_report.json",
            {"benchmark": "alpha", "paths": {"test_input": "data/alpha/alpha-gs.json.gz"}},
        )
        _write_predictions_csv(predictions_path.parents[3] / "summary.csv", [{"benchmark": "alpha", "status": "ok", "test_f1": 0.8}])

    out_dir = tmp_path / "analysis"
    cases = module.collect_export_cases(
        export_dir=export_dir,
        output_dir=out_dir,
        data_root=data_root,
        sample_per_error_type=None,
        best_profile_only=False,
        best_run_only=False,
    )

    assert len(cases) == 2
    assert cases["case_id"].duplicated().sum() == 0
    for case_id in cases["case_id"].tolist():
        assert case_id.startswith("method-a__alpha__small__false-positive__l0-r0__")


def test_merge_explanations_parses_paper_format_and_builds_edge_summary(tmp_path: Path) -> None:
    module = _load_module()
    out_dir = tmp_path / "explanations"
    out_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "case_id": "case_1",
                "benchmark": "alpha",
                "dataset_name": "alpha-gs",
                "model": "gpt-5.2",
                "error_type": "false_positive",
                "gold_match": False,
                "predicted_match": True,
                "left_record_json": '{"title":"Alpha Camera"}',
                "right_record_json": '{"title":"Beta Camera"}',
            }
        ]
    ).to_csv(out_dir / "cases.csv", index=False)

    (out_dir / "batch_output.jsonl").write_text(
        json.dumps(
            {
                "custom_id": "case_1",
                "response": {
                    "body": {
                        "choices": [
                            {
                                "message": {
                                    "content": (
                                        "Yes\n\n"
                                        "Decision: Yes\n"
                                        "Similarity: 82%\n"
                                        "Confidence: 77%\n\n"
                                        "attribute=brand|||importance=0.40|||values=Acme###Acme|||similarity=1.00\n"
                                        "attribute=model|||importance=-0.80|||values=A100###B200|||similarity=0.10"
                                    )
                                }
                            }
                        ]
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    merged = module.merge_explanations(out_dir)
    row = merged.iloc[0].to_dict()
    assert row["paper_decision_label"] == "match"
    assert row["paper_similarity_pct"] == 82
    assert row["paper_confidence_pct"] == 77
    assert row["strongest_positive_attribute"] == "brand"
    assert row["strongest_negative_attribute"] == "model"
    assert "match evidence" in row["what_went_wrong"]


def test_add_registry_category_updates_counter() -> None:
    module = _load_module()
    registry = module._default_registry()

    first = module._add_registry_category(
        registry,
        name="Variant mismatch",
        description="Records describe different product variants.",
        error_type="false_positive",
        example_case_id="case_1",
    )
    second = module._add_registry_category(
        registry,
        name="Missing evidence",
        description="Relevant fields are too sparse to confirm the match.",
        error_type="false_negative",
        example_case_id="case_2",
    )

    assert first["id"] == "C001"
    assert second["id"] == "C002"
    assert registry["next_category_index"] == 3
    assert registry["categories"][0]["applies_to"] == ["false_positive"]
    assert registry["categories"][1]["example_case_ids"] == ["case_2"]
