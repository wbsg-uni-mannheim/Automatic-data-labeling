from __future__ import annotations

import pandas as pd

from third_party.ditto_modern.data import WDC_COLUMNS, examples_to_ditto_lines, load_wdc_json_gz, wdc_to_pair_examples, write_wdc_json_gz


def _row(pair_id: str, label: int):
    return {
        "id_left": "1",
        "brand_left": "BrandA",
        "title_left": "Product A 128GB",
        "description_left": "A long description",
        "price_left": 10.0,
        "priceCurrency_left": "USD",
        "cluster_id_left": "c1",
        "id_right": "2",
        "brand_right": "BrandA",
        "title_right": "Product A 128 GB",
        "description_right": "Another description",
        "price_right": 12.0,
        "priceCurrency_right": "USD",
        "cluster_id_right": "c2",
        "pair_id": pair_id,
        "label": label,
        "is_hard_negative": False,
    }


def test_wdc_to_ditto_lines_format():
    df = pd.DataFrame([_row("p1", 1)], columns=WDC_COLUMNS)
    examples = wdc_to_pair_examples(df, fields=["title", "brand", "price"], max_field_len=100)
    lines = examples_to_ditto_lines(examples)

    assert len(lines) == 1
    line = lines[0]
    parts = line.split("\t")
    assert len(parts) == 3
    assert parts[2] == "1"
    assert "COL title VAL Product A 128GB" in parts[0]
    assert "COL brand VAL BrandA" in parts[0]
    assert "COL title VAL Product A 128 GB" in parts[1]


def test_generic_pair_schema_supported(tmp_path):
    df = pd.DataFrame(
        [
            {
                "id_left": "l1",
                "id_right": "r1",
                "pair_id": "p1",
                "label": "TRUE",
                "is_hard_negative": 0,
                "title_left": "Paper A",
                "title_right": "Paper A Extended",
                "authors_left": "Alice;Bob",
                "authors_right": "Alice;Bob",
            }
        ]
    )
    path = tmp_path / "generic.json.gz"
    write_wdc_json_gz(df, path)

    loaded = load_wdc_json_gz(path)
    examples = wdc_to_pair_examples(loaded, fields=["title", "authors"], max_field_len=100)
    lines = examples_to_ditto_lines(examples)

    assert len(lines) == 1
    left, right, label = lines[0].split("\t")
    assert label == "1"
    assert "COL title VAL Paper A" in left
    assert "COL authors VAL Alice;Bob" in left
    assert "COL title VAL Paper A Extended" in right
