import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.labeling.relabel_benchmark_split_openrouter import (
    _build_messages,
    _fields_from_spec,
    _load_benchmark_specs,
    _model_slug,
    _benchmark_output_dir,
    _parse_benchmark_arg,
    _path_for_split,
)


def test_wdc_spec_has_valid_and_test_paths():
    specs = _load_benchmark_specs()
    assert _path_for_split("wdc", specs["wdc"], "valid") == ROOT / "data/wdc/wdcproducts80cc20rnd000un_valid_large.json.gz"
    assert _path_for_split("wdc", specs["wdc"], "test") == ROOT / "data/wdc/wdcproducts80cc20rnd100un_gs.json.gz"


def test_parse_benchmark_arg_supports_all_and_lists():
    specs = _load_benchmark_specs()
    selected_all = _parse_benchmark_arg("all", specs)
    assert "wdc" in selected_all
    assert "abt-buy" in selected_all
    assert _parse_benchmark_arg("wdc,abt-buy", specs) == ["wdc", "abt-buy"]


def test_output_dir_uses_passed_model_slug():
    model = "meta-llama/llama-3.3-70b-instruct"
    output_dir = _benchmark_output_dir("wdc", "test", model)
    assert output_dir == ROOT / "output/benchmark_split_openrouter/wdc/test/meta-llama-llama-3-3-70b-instruct"
    assert _model_slug(model) == "meta-llama-llama-3-3-70b-instruct"


def test_build_messages_uses_config_fields_and_ditto_serialization():
    specs = _load_benchmark_specs()
    fields = _fields_from_spec("wdc", specs["wdc"])
    record = {
        "title_left": "Sony WH-1000XM5",
        "brand_left": "Sony",
        "description_left": "Noise cancelling headphones",
        "price_left": "299.99",
        "priceCurrency_left": "USD",
        "title_right": "Sony WH1000XM5 Wireless Headphones",
        "brand_right": "Sony",
        "description_right": "Over-ear ANC headphones",
        "price_right": "299.99",
        "priceCurrency_right": "USD",
    }

    messages = _build_messages(record, fields)
    assert len(messages) == 2
    assert "COL title VAL Sony WH-1000XM5" in messages[1]["content"]
    assert "COL brand VAL Sony" in messages[1]["content"]
    assert "COL priceCurrency VAL USD" in messages[1]["content"]
