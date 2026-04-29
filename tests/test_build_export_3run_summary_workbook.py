from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "build_export_3run_summary_workbook.py"
    spec = importlib.util.spec_from_file_location("build_export_3run_summary_workbook", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_parse_profile_supports_repeat_suffix() -> None:
    module = _load_module()
    assert (
        module._parse_profile(
            "generated_ditto_benchmark_abt-buy_20260323_202820_small_r1_20260416_211130",
            "three_phase_labeling_ditto_only_v2",
        )
        == "small"
    )
    assert module._parse_profile("baseline_r2_seed52", "ditto_baseline") == "official"
    assert module._parse_repeat("generated_ditto_benchmark_abt-buy_small_r3_20260416_211130") == 3
    assert module._parse_seed("baseline_r2_seed52") == 52


def test_collect_runs_and_aggregate_three_run_export(tmp_path: Path) -> None:
    module = _load_module()
    export_dir = tmp_path / "export"

    _write_summary(
        export_dir
        / "training_from_generated_labels_3runs"
        / "ditto_baseline"
        / "baseline_r1_seed42"
        / "summary.csv",
        [{"benchmark": "abt-buy", "status": "ok", "test_f1": 0.90, "test_precision": 0.91, "test_recall": 0.89, "test_accuracy": 0.98, "best_val_f1": 0.91, "train_rows": 1000}],
    )
    _write_summary(
        export_dir
        / "training_from_generated_labels_3runs"
        / "ditto_baseline"
        / "baseline_r2_seed52"
        / "summary.csv",
        [{"benchmark": "abt-buy", "status": "ok", "test_f1": 0.87, "test_precision": 0.88, "test_recall": 0.86, "test_accuracy": 0.97, "best_val_f1": 0.88, "train_rows": 1000}],
    )
    _write_summary(
        export_dir
        / "training_from_generated_labels_3runs"
        / "ditto_baseline"
        / "baseline_r3_seed62"
        / "summary.csv",
        [{"benchmark": "abt-buy", "status": "ok", "test_f1": 0.93, "test_precision": 0.94, "test_recall": 0.92, "test_accuracy": 0.99, "best_val_f1": 0.94, "train_rows": 1000}],
    )

    for repeat, value in [(1, 0.80), (2, 0.82), (3, 0.84)]:
        _write_summary(
            export_dir
            / "training_from_generated_labels_3runs"
            / "three_phase_labeling_ditto_only_v2_drop_changed"
            / f"generated_ditto_benchmark_abt-buy_20260323_202820_small_r{repeat}_20260416_211130"
            / "summary.csv",
            [{"benchmark": "abt-buy", "status": "ok", "test_f1": value, "test_precision": value, "test_recall": value, "test_accuracy": value, "best_val_f1": value, "train_rows": 500}],
        )

    all_runs = module.collect_runs(export_dir)
    assert len(all_runs) == 6

    benchmark_profile = module.build_benchmark_profile_summary(all_runs)
    baseline_row = benchmark_profile[benchmark_profile["family"] == "ditto_baseline"].iloc[0].to_dict()
    assert baseline_row["n_runs"] == 3
    assert round(baseline_row["mean_test_f1"], 4) == 0.9
    assert round(baseline_row["std_test_f1"], 4) == 0.03

    method_row = benchmark_profile[
        benchmark_profile["family"] == "three_phase_labeling_ditto_only_v2_drop_changed"
    ].iloc[0].to_dict()
    assert method_row["profile"] == "small"
    assert method_row["n_runs"] == 3
    assert round(method_row["mean_test_f1"], 4) == 0.82
    assert round(method_row["std_test_f1"], 4) == 0.02
    assert round(method_row["delta_mean_f1_vs_baseline"], 4) == -0.08

    method_summary = module.build_method_summary(benchmark_profile)
    assert len(method_summary) == 2

    selected_runs = module.build_selected_runs_summary(all_runs, benchmark_profile)
    assert len(selected_runs) == 2
    baseline_selected = selected_runs[selected_runs["method"] == "ditto_baseline"].iloc[0].to_dict()
    assert baseline_selected["benchmark"] == "abt-buy"
    assert baseline_selected["selected_profile"] == "official"
    assert round(baseline_selected["avg_f1_over_3_runs"], 4) == 0.9
    assert round(baseline_selected["std_f1_over_3_runs"], 4) == 0.03
    assert round(baseline_selected["baseline_avg_f1_over_3_runs"], 4) == 0.9
    assert round(baseline_selected["delta_vs_baseline_avg_f1"], 4) == 0.0
    assert baseline_selected["selected_run"] == "baseline_r3_seed62"

    method_selected = selected_runs[
        selected_runs["method"] == "three_phase_labeling_ditto_only_v2_drop_changed"
    ].iloc[0].to_dict()
    assert method_selected["selected_profile"] == "small"
    assert round(method_selected["avg_f1_over_3_runs"], 4) == 0.82
    assert round(method_selected["std_f1_over_3_runs"], 4) == 0.02
    assert round(method_selected["baseline_avg_f1_over_3_runs"], 4) == 0.9
    assert round(method_selected["delta_vs_baseline_avg_f1"], 4) == -0.08
    assert method_selected["selected_run"].endswith("small_r3_20260416_211130")
