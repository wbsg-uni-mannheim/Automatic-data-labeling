#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd


AUTO_ROOT = Path("output/autolabel_v1")
BASELINE_ROOT = Path("output/baseline")
OUTPUT_XLSX = Path("output/results_comparison_autolabel_vs_baseline.xlsx")

SPLIT_ORDER = {"official": 0, "small": 1, "medium": 2, "large": 3, "all": 4}
SOURCE_ORDER = {"baseline": 0, "autolabel_v1": 1}


def normalize_name(raw: str) -> str:
    return raw.replace("_", "-")


def dataset_from_profile_dir(profile_dir: str) -> str:
    prefix = profile_dir.split("_profiles", 1)[0]
    return normalize_name(prefix)


def find_run_part(path: Path) -> str:
    for part in path.parts:
        if part.startswith("run_"):
            return part
    return ""


def parse_autolabel_metrics_path(path: Path) -> Optional[Tuple[str, str, Path, str]]:
    rel = path.relative_to(AUTO_ROOT)
    parts = rel.parts
    if len(parts) < 3:
        return None

    # Shape C: <dataset>/run_x/metrics.json
    if len(parts) >= 3 and parts[1].startswith("run_"):
        dataset = normalize_name(parts[0])
        split = "unspecified"
        run_id = parts[1]
        return dataset, split, path.parent, run_id

    # Shape A: <profile>/<split>/run_x/metrics.json
    if parts[2].startswith("run_"):
        dataset = dataset_from_profile_dir(parts[0])
        split = parts[1]
        run_id = parts[2]
        return dataset, split, path.parent, run_id

    # Shape B: <profile>/<dataset>/<split>/run_x/metrics.json
    if len(parts) >= 5 and parts[3].startswith("run_"):
        dataset = normalize_name(parts[1])
        split = parts[2]
        run_id = parts[3]
        return dataset, split, path.parent, run_id

    return None


def parse_baseline_metrics_path(path: Path) -> Optional[Tuple[str, str, Path, str]]:
    rel = path.relative_to(BASELINE_ROOT)
    parts = rel.parts
    if len(parts) < 4:
        return None
    if "training_output" not in parts:
        return None
    run_id = find_run_part(path)
    if not run_id:
        return None
    dataset = normalize_name(parts[0])
    return dataset, "official", path.parent, run_id


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_labeled_rows(path: Path) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    if not path.exists():
        return None, None, None

    rows = 0
    pos = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows += 1
            label = line.rsplit("\t", 1)[-1].strip()
            if label == "1":
                pos += 1

    neg = rows - pos
    return rows, pos, neg


def collect_metrics_rows() -> Iterable[Dict]:
    latest: Dict[Tuple[str, str, str], Dict] = {}

    for metrics_path in AUTO_ROOT.rglob("metrics.json"):
        if "checkpoints" in metrics_path.parts:
            continue
        parsed = parse_autolabel_metrics_path(metrics_path)
        if not parsed:
            continue
        dataset, split, run_dir, run_id = parsed
        key = ("autolabel_v1", dataset, split)
        current = latest.get(key)
        if current is None or run_id > current["run_id"]:
            metrics = load_json(metrics_path)
            test = metrics.get("test", {})
            latest[key] = {
                "source": "autolabel_v1",
                "dataset": dataset,
                "training_split": split,
                "best_val_f1": metrics.get("best_val_f1"),
                "test_f1": test.get("f1"),
                "test_precision": test.get("precision"),
                "test_recall": test.get("recall"),
                "test_accuracy": test.get("accuracy"),
                "best_epoch": metrics.get("best_epoch"),
                "best_threshold": metrics.get("best_threshold"),
                "test_threshold": test.get("threshold"),
                "run_dir": str(run_dir),
                "run_id": run_id,
            }

    for metrics_path in BASELINE_ROOT.rglob("metrics.json"):
        if "checkpoints" in metrics_path.parts:
            continue
        parsed = parse_baseline_metrics_path(metrics_path)
        if not parsed:
            continue
        dataset, split, run_dir, run_id = parsed
        key = ("baseline", dataset, split)
        current = latest.get(key)
        if current is None or run_id > current["run_id"]:
            metrics = load_json(metrics_path)
            test = metrics.get("test", {})
            latest[key] = {
                "source": "baseline",
                "dataset": dataset,
                "training_split": split,
                "best_val_f1": metrics.get("best_val_f1"),
                "test_f1": test.get("f1"),
                "test_precision": test.get("precision"),
                "test_recall": test.get("recall"),
                "test_accuracy": test.get("accuracy"),
                "best_epoch": metrics.get("best_epoch"),
                "best_threshold": metrics.get("best_threshold"),
                "test_threshold": test.get("threshold"),
                "run_dir": str(run_dir),
                "run_id": run_id,
            }

    return latest.values()


def build_dataframes() -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = list(collect_metrics_rows())
    if not rows:
        raise RuntimeError("No metrics rows found in output/autolabel_v1 or output/baseline.")

    df = pd.DataFrame(rows)

    baseline_f1 = (
        df[(df["source"] == "baseline") & (df["training_split"] == "official")]
        .set_index("dataset")["test_f1"]
        .to_dict()
    )

    def baseline_value(dataset: str) -> Optional[float]:
        return baseline_f1.get(dataset)

    df["baseline_test_f1"] = df["dataset"].map(baseline_value)
    df["delta_test_f1_vs_baseline"] = df["test_f1"] - df["baseline_test_f1"]
    df.loc[df["source"] == "baseline", "delta_test_f1_vs_baseline"] = 0.0

    df["source_order"] = df["source"].map(lambda s: SOURCE_ORDER.get(s, 99))
    df["split_order"] = df["training_split"].map(lambda s: SPLIT_ORDER.get(s, 99))
    df = df.sort_values(
        by=["dataset", "source_order", "split_order"],
        ascending=[True, True, True],
        kind="stable",
    ).drop(columns=["run_id", "source_order", "split_order"])

    results_cols = [
        "source",
        "dataset",
        "training_split",
        "best_val_f1",
        "test_f1",
        "baseline_test_f1",
        "delta_test_f1_vs_baseline",
        "test_precision",
        "test_recall",
        "test_accuracy",
        "best_epoch",
        "best_threshold",
        "test_threshold",
        "run_dir",
    ]
    results_df = df[results_cols].copy()

    size_rows = []
    for row in rows:
        run_dir = Path(row["run_dir"])
        train_rows, train_pos, train_neg = count_labeled_rows(run_dir / "data" / "train.txt")
        valid_rows, valid_pos, valid_neg = count_labeled_rows(run_dir / "data" / "valid.txt")
        test_rows, test_pos, test_neg = count_labeled_rows(run_dir / "data" / "test.txt")
        size_rows.append(
            {
                "source": row["source"],
                "dataset": row["dataset"],
                "training_split": row["training_split"],
                "train_rows": train_rows,
                "train_pos": train_pos,
                "train_neg": train_neg,
                "valid_rows": valid_rows,
                "valid_pos": valid_pos,
                "valid_neg": valid_neg,
                "test_rows": test_rows,
                "test_pos": test_pos,
                "test_neg": test_neg,
                "run_dir": row["run_dir"],
            }
        )

    sizes_df = pd.DataFrame(size_rows)

    baseline_train = (
        sizes_df[(sizes_df["source"] == "baseline") & (sizes_df["training_split"] == "official")]
        .set_index("dataset")["train_rows"]
        .to_dict()
    )

    sizes_df["baseline_train_rows"] = sizes_df["dataset"].map(lambda d: baseline_train.get(d))
    sizes_df["train_rows_vs_baseline_pct"] = (
        sizes_df["train_rows"] / sizes_df["baseline_train_rows"] * 100.0
    )
    sizes_df.loc[sizes_df["source"] == "baseline", "train_rows_vs_baseline_pct"] = 100.0

    sizes_df["source_order"] = sizes_df["source"].map(lambda s: SOURCE_ORDER.get(s, 99))
    sizes_df["split_order"] = sizes_df["training_split"].map(lambda s: SPLIT_ORDER.get(s, 99))
    sizes_df = sizes_df.sort_values(
        by=["dataset", "source_order", "split_order"],
        ascending=[True, True, True],
        kind="stable",
    ).drop(columns=["source_order", "split_order"])

    size_cols = [
        "source",
        "dataset",
        "training_split",
        "train_rows",
        "baseline_train_rows",
        "train_rows_vs_baseline_pct",
        "train_pos",
        "train_neg",
        "valid_rows",
        "valid_pos",
        "valid_neg",
        "test_rows",
        "test_pos",
        "test_neg",
        "run_dir",
    ]
    sizes_df = sizes_df[size_cols]

    return results_df, sizes_df


def build_baseline_vs_splits_df(results_df: pd.DataFrame) -> pd.DataFrame:
    baseline_map = (
        results_df[
            (results_df["source"] == "baseline") & (results_df["training_split"] == "official")
        ]
        .set_index("dataset")["test_f1"]
        .to_dict()
    )

    split_maps = {}
    for split in ["small", "medium", "large", "all", "unspecified"]:
        split_maps[split] = (
            results_df[
                (results_df["source"] == "autolabel_v1") & (results_df["training_split"] == split)
            ]
            .set_index("dataset")["test_f1"]
            .to_dict()
        )

    datasets = sorted(
        set(baseline_map.keys())
        | set(split_maps["small"].keys())
        | set(split_maps["medium"].keys())
        | set(split_maps["large"].keys())
        | set(split_maps["all"].keys())
        | set(split_maps["unspecified"].keys())
    )
    rows = []
    for dataset in datasets:
        baseline_f1 = baseline_map.get(dataset)
        small_f1 = split_maps["small"].get(dataset)
        medium_f1 = split_maps["medium"].get(dataset)
        large_f1 = split_maps["large"].get(dataset)
        all_f1 = split_maps["all"].get(dataset)
        unspecified_f1 = split_maps["unspecified"].get(dataset)

        # If a dataset has no explicit split folder (e.g., direct run), use it as small fallback.
        if small_f1 is None:
            small_f1 = unspecified_f1

        def delta(value: Optional[float]) -> Optional[float]:
            if baseline_f1 is None or value is None:
                return None
            return value - baseline_f1

        rows.append(
            {
                "dataset": dataset,
                "baseline_f1": baseline_f1,
                "small_f1": small_f1,
                "small_delta_to_baseline": delta(small_f1),
                "medium_f1": medium_f1,
                "medium_delta_to_baseline": delta(medium_f1),
                "large_f1": large_f1,
                "large_delta_to_baseline": delta(large_f1),
                "all_f1": all_f1,
                "all_delta_to_baseline": delta(all_f1),
            }
        )

    return pd.DataFrame(rows)


def autosize_columns(sheet, dataframe: pd.DataFrame) -> None:
    for idx, col in enumerate(dataframe.columns, start=1):
        max_val_len = dataframe[col].astype(str).map(len).max() if not dataframe.empty else 0
        width = min(max(len(col), max_val_len) + 2, 60)
        sheet.column_dimensions[sheet.cell(row=1, column=idx).column_letter].width = width


def main() -> None:
    results_df, sizes_df = build_dataframes()
    baseline_vs_splits_df = build_baseline_vs_splits_df(results_df)
    OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        results_df.to_excel(writer, index=False, sheet_name="results")
        sizes_df.to_excel(writer, index=False, sheet_name="training_sizes")
        baseline_vs_splits_df.to_excel(writer, index=False, sheet_name="baseline_vs_splits")

        results_sheet = writer.sheets["results"]
        sizes_sheet = writer.sheets["training_sizes"]
        baseline_vs_splits_sheet = writer.sheets["baseline_vs_splits"]

        results_sheet.freeze_panes = "A2"
        sizes_sheet.freeze_panes = "A2"
        baseline_vs_splits_sheet.freeze_panes = "A2"
        autosize_columns(results_sheet, results_df)
        autosize_columns(sizes_sheet, sizes_df)
        autosize_columns(baseline_vs_splits_sheet, baseline_vs_splits_df)

    print(f"Wrote workbook: {OUTPUT_XLSX}")
    print(f"Rows in results sheet: {len(results_df)}")
    print(f"Rows in training_sizes sheet: {len(sizes_df)}")
    print(f"Rows in baseline_vs_splits sheet: {len(baseline_vs_splits_df)}")


if __name__ == "__main__":
    main()
