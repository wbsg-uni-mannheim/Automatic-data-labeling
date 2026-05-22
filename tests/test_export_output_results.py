from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "export_output_results.py"
    spec = importlib.util.spec_from_file_location("export_output_results", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_should_copy_error_analysis_files_when_enabled() -> None:
    module = _load_module()

    should_copy_pred, category_pred = module._should_copy(
        Path("batch_eval_gpt52/alpha/predictions.csv"),
        include_predictions=False,
        include_error_analysis=True,
    )
    should_copy_jsonl, category_jsonl = module._should_copy(
        Path("openrouter/test/model/results.jsonl"),
        include_predictions=False,
        include_error_analysis=True,
    )

    assert should_copy_pred is True
    assert category_pred == "predictions"
    assert should_copy_jsonl is True
    assert category_jsonl == "results_jsonl"


def test_collect_error_analysis_support_files_includes_data_and_script(tmp_path: Path) -> None:
    module = _load_module()
    original_root = module.ROOT
    original_data_dir = module.DEFAULT_DATA_DIR
    original_support = list(module.ERROR_ANALYSIS_SUPPORT_FILES)

    try:
        module.ROOT = tmp_path
        module.DEFAULT_DATA_DIR = tmp_path / "data"
        data_file = module.DEFAULT_DATA_DIR / "alpha" / "alpha-gs.json.gz"
        data_file.parent.mkdir(parents=True)
        data_file.write_text("{}", encoding="utf-8")

        script_file = tmp_path / "scripts" / "run_error_analysis.py"
        script_file.parent.mkdir(parents=True)
        script_file.write_text("# stub", encoding="utf-8")
        module.ERROR_ANALYSIS_SUPPORT_FILES = [script_file]

        selected = module._collect_error_analysis_support_files()
        rel_paths = {rel_path.as_posix(): category for _, rel_path, category in selected}

        assert rel_paths["data/alpha/alpha-gs.json.gz"] == "benchmark_data"
        assert rel_paths["scripts/run_error_analysis.py"] == "script"
    finally:
        module.ROOT = original_root
        module.DEFAULT_DATA_DIR = original_data_dir
        module.ERROR_ANALYSIS_SUPPORT_FILES = original_support
