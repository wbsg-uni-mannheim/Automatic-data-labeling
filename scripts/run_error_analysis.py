#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data"
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "error_analysis"
DEFAULT_EXPORT_ROOT = ROOT / "export"
DEFAULT_EXPLANATION_MODEL = "gpt-5.4-mini"
DEFAULT_CATEGORY_MODEL = "gpt-5.4-mini"
DEFAULT_BATCH_ENDPOINT = "/v1/chat/completions"
DEFAULT_BATCH_WINDOW = "24h"
PROFILE_ORDER = {
    "official": 0,
    "small": 1,
    "small_plus20random": 2,
    "medium": 3,
    "medium_plus20random": 4,
    "large": 5,
    "all": 6,
    "all_plus20random": 7,
}
PROFILE_PARSE_ORDER = sorted(PROFILE_ORDER, key=len, reverse=True)

PAIR_META_KEYS = {
    "label",
    "pair_id",
    "id_left",
    "id_right",
    "cluster_id_left",
    "cluster_id_right",
    "is_hard_negative",
}
SIDE_SUFFIXES = {"_left": "left", "_right": "right"}

PAPER_EXPLANATION_SUFFIX = """Now output your decision as a simple 'Yes' or 'No'. Also provide a similarity score for the items and a confidence score for your decision as percentage values, 100% referring to perfect similarity or full confidence. Please use the following format which includes an example output:

Decision: Yes
Similarity: 75%
Confidence: 60%

Now explain your decision in a structured format, listing the attributes that you compared for reaching your decision. Each attribute should be accompanied by the attribute values and a score between -1 and 1 that shows the importance of the attribute for the decision. If the attribute influenced the decision towards non-match the importance score should be negative. If the attribute pointed towards a match, the importance score should be positive. Also provide a similarity score for the attribute values. If an attribute only occurs in one item, specify the value of that attribute for the other item as "missing". An example output is the following:

attribute=brand|||importance=0.05|||values=Logitech###Logitech|||similarity=1.00
attribute=model|||importance=-0.95|||values=MX G500###MX Master 3S|||similarity=0.20
attribute=color|||importance=0.00|||values=missing###Graphite|||similarity=0.00"""

PAPER_SUMMARIZATION_PREFIX_PRODUCT = """The following list contains false positive and false negative product pairs from the output of a product matching classification system. Given the product pairs and the associated explanations, come up with a set of error classes, separately for both false positives and false negatives, that explain why the classification systems fails on these examples. The importance score of the attribute comparisons in the explanations is based on a scale from -1 to 1. If the score is negative the attribute comparison points to a non-match, if it is positive it points to a match.

"""

PAPER_SUMMARIZATION_PREFIX_PUBLICATION = """The following list contains false positive and false negative publication pairs from the output of a publication matching classification system. Given the publication pairs and the associated explanations, come up with a set of error classes, separately for both false positives and false negatives, that explain why the classification systems fails on these examples. The importance score of the attribute comparisons in the explanations is based on a scale from -1 to 1. If the score is negative the attribute comparison points to a non-match, if it is positive it points to a match.

"""

PAPER_CLASSIFICATION_PREFIX_PRODUCT = """Given the following error classes for a product matching classification system, please classify the following product pair into all error classes by their number if they are relevant for this pair and its explanation. The importance score of the attributes in the explanation is on a scale of -1 to 1. It is negative if the attribute comparison pointed toward a non-match and positive if the comparison pointed toward a match. Please give a short explanation of every decision as a list first. Finally also provide a confidence score for each classification adhering to the JSON format of the following example: ```{"2":"90","4":"30","5":"75"}```

After your analysis, if you think the pair was actually correctly labeled by the matching system and the original label was wrong, please additionally state this in combination with your overall confidence in that statement.

"""

PAPER_CLASSIFICATION_PREFIX_PUBLICATION = """Given the following error classes for a publication matching classification system, please classify the following publication pair into all error classes by their number if they are relevant for this pair and its explanation. The importance score of the attributes in the explanation is on a scale from -1 to 1. It is negative if the attribute comparison pointed toward a non-match and positive if the comparison pointed toward a match. Please give a short explanation of every decision as a list first. Finally also provide a confidence score for each classification adhering to the JSON format of the following example: ```{"2":"90","4":"30","5":"75"}```

After your analysis, if you think the pair was actually correctly labeled by the matching system and the original label was wrong, please additionally state this in combination with your overall confidence in that statement.

"""

REGISTRY_GENERATION_SUFFIX = """
Return JSON with exactly two fields:
{
  "false_negative_classes": [{"name": "...", "description": "..."}],
  "false_positive_classes": [{"name": "...", "description": "..."}]
}

Use short stable class names and concise descriptions. Do not include markdown.
"""

REGISTRY_CONSOLIDATION_PROMPT = """You are consolidating error classes for an entity matching error analysis across multiple benchmarks.

Input: benchmark-specific proposed error classes for false positives and false negatives.

Task:
1. Merge duplicates and near-duplicates.
2. Keep category names short and stable.
3. Keep descriptions concise and specific enough for annotation.
4. Preserve whether a category applies to false positives, false negatives, or both.
5. Produce one global registry that can be reused consistently across methods and runs.

Return JSON with exactly one field:
{
  "categories": [
    {
      "name": "...",
      "description": "...",
      "applies_to": ["false_positive"] | ["false_negative"] | ["false_positive","false_negative"]
    }
  ]
}
"""


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(int(value))
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"true", "1", "yes", "y", "t", "match"}:
        return True
    if text in {"false", "0", "no", "n", "f", "non_match", "non-match"}:
        return False
    return None


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _slugify(text: str) -> str:
    chars: List[str] = []
    prev_dash = False
    for ch in str(text):
        if ch.isalnum():
            chars.append(ch.lower())
            prev_dash = False
            continue
        if not prev_dash:
            chars.append("-")
        prev_dash = True
    return "".join(chars).strip("-") or "unknown"


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _default_collect_output_dir(artifacts: List[Path]) -> Path:
    if len(artifacts) != 1:
        return DEFAULT_OUTPUT_ROOT / "latest"

    artifact = artifacts[0].resolve()
    if not _is_within(artifact, DEFAULT_EXPORT_ROOT):
        return DEFAULT_OUTPUT_ROOT / "latest"

    if artifact.is_dir():
        return artifact / "error_analysis"
    if artifact.name in {"predictions.csv", "results.jsonl", "summary.csv"}:
        return artifact.parent / "error_analysis"
    return DEFAULT_OUTPUT_ROOT / "latest"


def _default_child_output_dir(cases_csv: Path, child_name: str) -> Path:
    parent = cases_csv.resolve().parent
    if parent.name == "explanations":
        return parent.parent / child_name
    return parent / child_name


def _default_export_output_dir(export_dir: Path) -> Path:
    return export_dir.resolve() / "error_analysis"


def _default_export_data_root(export_dir: Path) -> Path:
    for candidate_root in [export_dir.resolve(), *export_dir.resolve().parents]:
        candidate = candidate_root / "data"
        if candidate.exists():
            return candidate
    return DEFAULT_DATA_ROOT


def _parse_profile_from_run_name(run_name: str) -> str:
    for profile in PROFILE_PARSE_ORDER:
        if re.search(rf"_{re.escape(profile)}(?:_r\d+)?_\d{{8}}_\d{{6}}$", run_name):
            return profile
    return "official" if "baseline" in run_name.lower() else "unknown"


def _find_profile_in_parts(parts: Tuple[str, ...]) -> str:
    for part in parts:
        text = _normalize_text(part)
        if text in PROFILE_ORDER:
            return text
    return "unknown"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_jsonl_gz(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def discover_export_prediction_artifacts(export_dir: Path) -> List[Path]:
    artifacts: List[Path] = []
    for pattern in ("predictions.csv", "results.jsonl"):
        for path in sorted(export_dir.rglob(pattern)):
            if "error_analysis" in path.parts:
                continue
            artifacts.append(path.resolve())
    deduped = sorted({path for path in artifacts})
    return deduped


def discover_export_summary_files(export_dir: Path) -> List[Path]:
    summaries: List[Path] = []
    for path in sorted(export_dir.rglob("summary.csv")):
        if "error_analysis" in path.parts:
            continue
        summaries.append(path.resolve())
    return summaries


def _extract_export_context(artifact: Path, export_dir: Path) -> Dict[str, str]:
    rel = artifact.resolve().relative_to(export_dir.resolve())
    parts = rel.parts
    context = {
        "export_name": export_dir.name,
        "export_rel_path": str(rel),
        "source_family": "",
        "method_name": "",
        "run_name": "",
        "profile_name": "unknown",
        "benchmark_dir_name": "",
        "training_run_name": "",
    }
    if not parts:
        return context

    if export_dir.name.startswith("training_from_generated_labels") and len(parts) >= 2:
        context["source_family"] = export_dir.name
        context["method_name"] = parts[0]
        context["run_name"] = parts[1]
        context["profile_name"] = _parse_profile_from_run_name(parts[1])
        if len(parts) >= 3 and parts[2] != "summary.csv":
            context["benchmark_dir_name"] = parts[2]
        if "training_output" in parts:
            training_idx = parts.index("training_output")
            if training_idx + 1 < len(parts):
                context["training_run_name"] = parts[training_idx + 1]
        return context

    top = parts[0]
    context["method_name"] = top

    if top.startswith("training_from_generated_labels") and len(parts) >= 3:
        context["source_family"] = top
        context["method_name"] = parts[1]
        context["run_name"] = parts[2]
        context["profile_name"] = _parse_profile_from_run_name(parts[2])
        if len(parts) >= 4 and parts[3] != "summary.csv":
            context["benchmark_dir_name"] = parts[3]
        if "training_output" in parts:
            training_idx = parts.index("training_output")
            if training_idx + 1 < len(parts):
                context["training_run_name"] = parts[training_idx + 1]
    elif top in {"ditto_baseline", "ditto_benchmark_runs"} and len(parts) >= 4:
        context["source_family"] = "baseline"
        context["method_name"] = top
        context["run_name"] = parts[1]
        context["profile_name"] = "official"
        if parts[2] == "training_output":
            context["training_run_name"] = parts[3]
        else:
            context["benchmark_dir_name"] = parts[2]
            if len(parts) >= 5 and parts[3] == "training_output":
                context["training_run_name"] = parts[4]
    else:
        context["source_family"] = "export_artifact"
        if len(parts) >= 2:
            context["run_name"] = parts[1]
        context["profile_name"] = _find_profile_in_parts(parts)
        training_idx = next((idx for idx, value in enumerate(parts) if value == "training_output"), None)
        if training_idx is not None and training_idx + 1 < len(parts):
            context["training_run_name"] = parts[training_idx + 1]
        if training_idx is not None and training_idx >= 1:
            context["benchmark_dir_name"] = parts[training_idx - 1]
    return context


def _annotate_frame_with_context(frame: pd.DataFrame, context: Dict[str, str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    annotated = frame.copy()
    for key, value in context.items():
        annotated[key] = value
    return annotated


def select_best_profiles_by_method(export_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for summary_path in discover_export_summary_files(export_dir):
        context = _extract_export_context(summary_path, export_dir)
        if not _normalize_text(context.get("method_name")):
            continue
        frame = pd.read_csv(summary_path)
        if frame.empty:
            continue
        for record in frame.to_dict("records"):
            if _normalize_text(record.get("status")).lower() != "ok":
                continue
            test_f1 = _safe_float(record.get("test_f1"))
            if test_f1 is None:
                continue
            rows.append(
                {
                    "method_name": context["method_name"],
                    "profile_name": context["profile_name"],
                    "run_name": context["run_name"],
                    "benchmark": _normalize_text(record.get("benchmark")),
                    "test_f1": test_f1,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["method_name", "benchmark", "profile_name", "avg_test_f1", "run_count"])

    frame = pd.DataFrame(rows)
    grouped = (
        frame.groupby(["method_name", "benchmark", "profile_name"], dropna=False)
        .agg(
            avg_test_f1=("test_f1", "mean"),
            run_count=("run_name", "nunique"),
        )
        .reset_index()
    )
    grouped["profile_order"] = grouped["profile_name"].map(lambda value: PROFILE_ORDER.get(str(value), 99))
    grouped = grouped.sort_values(
        ["method_name", "benchmark", "avg_test_f1", "run_count", "profile_order"],
        ascending=[True, True, False, False, True],
        kind="stable",
    )
    return grouped.drop_duplicates(subset=["method_name", "benchmark"], keep="first").drop(
        columns=["profile_order"]
    ).reset_index(drop=True)


def select_best_runs_by_method(
    export_dir: Path,
    selected_profile_map: Optional[Dict[Any, str]] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for summary_path in discover_export_summary_files(export_dir):
        context = _extract_export_context(summary_path, export_dir)
        if not _normalize_text(context.get("method_name")) or not _normalize_text(context.get("run_name")):
            continue
        if selected_profile_map:
            selected_profile = selected_profile_map.get(
                (context["method_name"], context["benchmark_dir_name"]),
                selected_profile_map.get(context["method_name"]),
            )
            if selected_profile is None or context["profile_name"] != selected_profile:
                continue
        frame = pd.read_csv(summary_path)
        if frame.empty:
            continue
        for record in frame.to_dict("records"):
            if _normalize_text(record.get("status")).lower() != "ok":
                continue
            test_f1 = _safe_float(record.get("test_f1"))
            if test_f1 is None:
                continue
            rows.append(
                {
                    "method_name": context["method_name"],
                    "profile_name": context["profile_name"],
                    "run_name": context["run_name"],
                    "benchmark": _normalize_text(record.get("benchmark")),
                    "test_f1": test_f1,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["method_name", "profile_name", "run_name", "avg_test_f1", "benchmark_count"])

    frame = pd.DataFrame(rows)
    grouped = (
        frame.groupby(["method_name", "profile_name", "benchmark", "run_name"], dropna=False)
        .agg(
            avg_test_f1=("test_f1", "mean"),
        )
        .reset_index()
    )
    grouped["profile_order"] = grouped["profile_name"].map(lambda value: PROFILE_ORDER.get(str(value), 99))
    grouped = grouped.sort_values(
        ["method_name", "benchmark", "avg_test_f1", "profile_order", "run_name"],
        ascending=[True, True, False, True, True],
        kind="stable",
    )
    return grouped.drop_duplicates(subset=["method_name", "benchmark"], keep="first").drop(
        columns=["profile_order"]
    ).reset_index(drop=True)


def _infer_dataset_path(benchmark: str, dataset_name: str, data_root: Path) -> Path:
    candidate = data_root / benchmark / f"{dataset_name}.json.gz"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not resolve dataset file for {benchmark}/{dataset_name}: {candidate}")


def _load_dataset_records(dataset_path: Path, pair_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    wanted = {str(pair_id) for pair_id in pair_ids if str(pair_id)}
    records: Dict[str, Dict[str, Any]] = {}
    if not wanted:
        return records
    with gzip.open(dataset_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            pair_id = _normalize_text(row.get("pair_id"))
            if pair_id in wanted:
                records[pair_id] = row
            if len(records) == len(wanted):
                break
    missing = sorted(wanted.difference(records.keys()))
    if missing:
        raise KeyError(
            f"Missing {len(missing)} pair_ids in {dataset_path}. First missing ids: {', '.join(missing[:5])}"
        )
    return records


def _split_record_sides(record: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    left: Dict[str, str] = {}
    right: Dict[str, str] = {}
    for key, value in record.items():
        if key in PAIR_META_KEYS:
            continue
        normalized = _normalize_text(value)
        if not normalized:
            continue
        for suffix, side in SIDE_SUFFIXES.items():
            if not key.endswith(suffix):
                continue
            field = key[: -len(suffix)]
            if field in {"cluster_id"}:
                continue
            if side == "left":
                left[field] = normalized
            else:
                right[field] = normalized
            break
    return left, right


def _format_bool_label(value: bool) -> str:
    return "match" if value else "non_match"


def _parse_prediction_content(content: str) -> Optional[bool]:
    text = _normalize_text(content)
    if not text:
        return None
    if text.startswith("```"):
        parts = text.split("\n", 1)
        text = parts[1] if len(parts) == 2 else text
        text = text.rsplit("```", 1)[0].strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            if "match" in payload:
                return _coerce_bool(payload.get("match"))
            if "predicted_match" in payload:
                return _coerce_bool(payload.get("predicted_match"))
    except json.JSONDecodeError:
        pass

    lowered = text.lower()
    if '"match":true' in lowered or '"match": true' in lowered:
        return True
    if '"match":false' in lowered or '"match": false' in lowered:
        return False
    return None


def _safe_model_slug(model: str) -> str:
    return _slugify(str(model).replace("/", "-"))


def _error_type(gold_match: bool, predicted_match: bool) -> str:
    if gold_match and not predicted_match:
        return "false_negative"
    if not gold_match and predicted_match:
        return "false_positive"
    raise ValueError("Only mispredictions have an error type")


def _build_case_id(
    benchmark: str,
    dataset_name: str,
    model: str,
    pair_id: str,
    error_type: str,
) -> str:
    return "__".join(
        [
            _slugify(benchmark),
            _slugify(dataset_name),
            _safe_model_slug(model),
            error_type,
            _slugify(pair_id),
        ]
    )


def _short_hash(*parts: str, length: int = 12) -> str:
    payload = "||".join(_normalize_text(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def _build_export_case_id(row: Dict[str, Any]) -> str:
    return "__".join(
        [
            _slugify(row.get("method_name")),
            _slugify(row.get("benchmark")),
            _slugify(row.get("profile_name")),
            _slugify(row.get("error_type")),
            _slugify(row.get("pair_id")),
            _short_hash(
                str(row.get("method_name", "")),
                str(row.get("benchmark", "")),
                str(row.get("profile_name", "")),
                str(row.get("run_name", "")),
                str(row.get("training_run_name", "")),
                str(row.get("dataset_name", "")),
                str(row.get("model", "")),
                str(row.get("error_type", "")),
                str(row.get("pair_id", "")),
            ),
        ]
    )


def _parse_predictions_csv(
    path: Path,
    *,
    benchmark: str,
    dataset_name: str,
    model: str,
    source_artifact: Path,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rows: List[Dict[str, Any]] = []
    for row in frame.to_dict("records"):
        gold_match = _coerce_bool(row.get("gold_match"))
        if gold_match is None:
            gold_match = _coerce_bool(row.get("gold"))
        predicted_match = _coerce_bool(row.get("predicted_match"))
        if predicted_match is None:
            predicted_match = _coerce_bool(row.get("pred"))
        if gold_match is None or predicted_match is None or gold_match == predicted_match:
            continue
        pair_id = _normalize_text(row.get("pair_id"))
        id_left = _normalize_text(row.get("id_left"))
        id_right = _normalize_text(row.get("id_right"))
        if pair_id and (not id_left or not id_right) and "#" in pair_id:
            left, right = pair_id.split("#", 1)
            id_left = id_left or left
            id_right = id_right or right
        rows.append(
            {
                "benchmark": benchmark,
                "dataset_name": dataset_name,
                "model": model,
                "source_kind": "predictions_csv",
                "source_artifact": str(source_artifact),
                "pair_id": pair_id,
                "id_left": id_left,
                "id_right": id_right,
                "gold_match": bool(gold_match),
                "predicted_match": bool(predicted_match),
                "gold_label": _format_bool_label(bool(gold_match)),
                "predicted_label": _format_bool_label(bool(predicted_match)),
                "error_type": _error_type(bool(gold_match), bool(predicted_match)),
                "raw_response": _normalize_text(row.get("response_text") or row.get("prob")),
            }
        )
    return pd.DataFrame(rows)


def _parse_results_jsonl(
    path: Path,
    *,
    benchmark: str,
    dataset_name: str,
    model: str,
    source_artifact: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for row in _read_jsonl(path):
        gold_match = (
            _coerce_bool(row.get("gold_match"))
            if "gold_match" in row
            else _coerce_bool(row.get("gold_label"))
        )
        predicted_match = (
            _coerce_bool(row.get("predicted_match"))
            if "predicted_match" in row
            else _coerce_bool(row.get("pred_label"))
        )
        if predicted_match is None and "predicted_label" in row:
            predicted_match = _coerce_bool(row.get("predicted_label"))
        if predicted_match is None and "response_text" in row:
            predicted_match = _parse_prediction_content(_normalize_text(row.get("response_text")))
        if predicted_match is None and "raw_response" in row:
            predicted_match = _parse_prediction_content(_normalize_text(row.get("raw_response")))
        if gold_match is None or predicted_match is None or gold_match == predicted_match:
            continue
        pair_id = _normalize_text(row.get("pair_id"))
        rows.append(
            {
                "benchmark": benchmark,
                "dataset_name": dataset_name,
                "model": model,
                "source_kind": "results_jsonl",
                "source_artifact": str(source_artifact),
                "pair_id": pair_id,
                "id_left": _normalize_text(row.get("id_left")),
                "id_right": _normalize_text(row.get("id_right")),
                "gold_match": bool(gold_match),
                "predicted_match": bool(predicted_match),
                "gold_label": _format_bool_label(bool(gold_match)),
                "predicted_label": _format_bool_label(bool(predicted_match)),
                "error_type": _error_type(bool(gold_match), bool(predicted_match)),
                "raw_response": _normalize_text(row.get("response_text") or row.get("raw_response")),
            }
        )
    return pd.DataFrame(rows)


def _collect_dataset_from_run_dir(run_dir: Path) -> List[Dict[str, Any]]:
    run_manifest = _load_json(run_dir / "run_manifest.json")
    datasets = run_manifest.get("datasets") or []
    if not isinstance(datasets, list):
        raise ValueError(f"Invalid datasets list in {run_dir / 'run_manifest.json'}")
    return [dict(item) for item in datasets if isinstance(item, dict)]


def _resolve_cases_from_artifact(
    artifact: Path,
    *,
    data_root: Path,
    benchmark: Optional[str],
    dataset_name: Optional[str],
    model: Optional[str],
) -> pd.DataFrame:
    artifact = artifact.resolve()
    if artifact.is_dir() and (artifact / "run_manifest.json").exists():
        frames: List[pd.DataFrame] = []
        for dataset in _collect_dataset_from_run_dir(artifact):
            slug = str(dataset.get("output_slug", "")).strip()
            if not slug:
                continue
            prediction_path = artifact / slug / "predictions.csv"
            if not prediction_path.exists():
                continue
            dataset_frame = _parse_predictions_csv(
                prediction_path,
                benchmark=str(dataset.get("benchmark", "")),
                dataset_name=str(dataset.get("dataset_name", "")),
                model=str(dataset.get("model", model or "")),
                source_artifact=artifact,
            )
            if not dataset_frame.empty:
                frames.append(dataset_frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if artifact.is_file() and artifact.name == "summary.csv" and (artifact.parent / "run_manifest.json").exists():
        return _resolve_cases_from_artifact(
            artifact.parent,
            data_root=data_root,
            benchmark=benchmark,
            dataset_name=dataset_name,
            model=model,
        )

    if artifact.is_file() and artifact.name == "predictions.csv":
        ds_manifest_path = artifact.parent / "dataset_manifest.json"
        benchmark_report_path = artifact.parents[2] / "benchmark_report.json" if len(artifact.parents) >= 3 else None
        ds_manifest = _load_json(ds_manifest_path) if ds_manifest_path.exists() else {}
        benchmark_report = (
            _load_json(benchmark_report_path)
            if benchmark_report_path is not None and benchmark_report_path.exists()
            else {}
        )
        resolved_benchmark = benchmark or str(ds_manifest.get("benchmark", ""))
        if not resolved_benchmark:
            resolved_benchmark = str(benchmark_report.get("benchmark", ""))
        resolved_dataset_name = dataset_name or str(ds_manifest.get("dataset_name", ""))
        if not resolved_dataset_name:
            test_input = (
                ((benchmark_report.get("paths") or {}).get("test_input"))
                if isinstance(benchmark_report, dict)
                else None
            )
            if test_input:
                test_name = Path(str(test_input)).name
                if test_name.endswith(".json.gz"):
                    resolved_dataset_name = test_name[: -len(".json.gz")]
                else:
                    resolved_dataset_name = Path(str(test_input)).stem
        resolved_model = model or str(ds_manifest.get("model", ""))
        if not resolved_model and benchmark_report:
            resolved_model = "ditto"
        if not resolved_benchmark or not resolved_dataset_name:
            raise ValueError(
                f"Need benchmark and dataset_name for {artifact}; pass flags or provide dataset_manifest.json"
            )
        return _parse_predictions_csv(
            artifact,
            benchmark=resolved_benchmark,
            dataset_name=resolved_dataset_name,
            model=resolved_model,
            source_artifact=artifact,
        )

    if artifact.is_file() and artifact.name == "results.jsonl":
        resolved_benchmark = benchmark
        resolved_dataset_name = dataset_name
        resolved_model = model or ""
        summary_path = artifact.parent / "summary.json"
        if summary_path.exists():
            summary_payload = _load_json(summary_path)
            resolved_benchmark = resolved_benchmark or str(summary_payload.get("benchmark", ""))
            resolved_dataset_name = resolved_dataset_name or str(summary_payload.get("dataset_name", ""))
            resolved_model = resolved_model or str(summary_payload.get("model", ""))
        if not resolved_benchmark or not resolved_dataset_name:
            raise ValueError(
                f"Need benchmark and dataset_name for {artifact}; pass --benchmark and --dataset-name"
            )
        return _parse_results_jsonl(
            artifact,
            benchmark=resolved_benchmark,
            dataset_name=resolved_dataset_name,
            model=resolved_model,
            source_artifact=artifact,
        )

    raise ValueError(f"Unsupported artifact path: {artifact}")


def _enrich_cases_with_records(frame: pd.DataFrame, *, data_root: Path) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    enriched_frames: List[pd.DataFrame] = []
    for (benchmark, dataset_name), group in frame.groupby(["benchmark", "dataset_name"], sort=False):
        dataset_path = _infer_dataset_path(str(benchmark), str(dataset_name), data_root)
        records = _load_dataset_records(dataset_path, group["pair_id"].astype(str).tolist())
        rows: List[Dict[str, Any]] = []
        for row in group.to_dict("records"):
            pair_record = records[str(row["pair_id"])]
            left_payload, right_payload = _split_record_sides(pair_record)
            row["dataset_path"] = str(dataset_path)
            row["id_left"] = row["id_left"] or _normalize_text(pair_record.get("id_left"))
            row["id_right"] = row["id_right"] or _normalize_text(pair_record.get("id_right"))
            row["left_record_json"] = json.dumps(left_payload, ensure_ascii=False, sort_keys=True)
            row["right_record_json"] = json.dumps(right_payload, ensure_ascii=False, sort_keys=True)
            row["case_id"] = _build_case_id(
                str(row["benchmark"]),
                str(row["dataset_name"]),
                str(row["model"]),
                str(row["pair_id"]),
                str(row["error_type"]),
            )
            rows.append(row)
        enriched_frames.append(pd.DataFrame(rows))
    return pd.concat(enriched_frames, ignore_index=True)


def collect_cases(
    *,
    artifacts: List[Path],
    output_dir: Path,
    data_root: Path,
    benchmark: Optional[str],
    dataset_name: Optional[str],
    model: Optional[str],
    sample_per_error_type: Optional[int],
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: List[pd.DataFrame] = []
    for artifact in artifacts:
        frame = _resolve_cases_from_artifact(
            artifact,
            data_root=data_root,
            benchmark=benchmark,
            dataset_name=dataset_name,
            model=model,
        )
        if not frame.empty:
            frames.append(frame)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if combined.empty:
        combined.to_csv(output_dir / "cases.csv", index=False)
        _write_json(
            output_dir / "manifest.json",
            {
                "created_at": _timestamp(),
                "artifacts": [str(path.resolve()) for path in artifacts],
                "case_count": 0,
            },
        )
        return combined

    combined = _enrich_cases_with_records(combined, data_root=data_root)
    combined = combined.sort_values(
        ["benchmark", "dataset_name", "model", "error_type", "pair_id"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)

    if sample_per_error_type is not None:
        combined = (
            combined.groupby(["benchmark", "dataset_name", "model", "error_type"], group_keys=False)
            .head(int(sample_per_error_type))
            .reset_index(drop=True)
        )

    combined.to_csv(output_dir / "cases.csv", index=False)
    with (output_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for row in combined.to_dict("records"):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_rows = (
        combined.groupby(["benchmark", "dataset_name", "model", "error_type"])
        .size()
        .reset_index(name="count")
        .to_dict("records")
    )
    _write_json(
        output_dir / "manifest.json",
        {
            "created_at": _timestamp(),
            "artifacts": [str(path.resolve()) for path in artifacts],
            "case_count": int(len(combined)),
            "datasets": summary_rows,
        },
    )
    return combined


def collect_export_cases(
    *,
    export_dir: Path,
    output_dir: Path,
    data_root: Path,
    sample_per_error_type: Optional[int],
    best_profile_only: bool,
    best_run_only: bool,
) -> pd.DataFrame:
    artifacts = discover_export_prediction_artifacts(export_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_profiles_df = select_best_profiles_by_method(export_dir) if best_profile_only else pd.DataFrame()
    selected_profile_map = (
        {
            (row["method_name"], row["benchmark"]): row["profile_name"]
            for row in selected_profiles_df.to_dict("records")
        }
        if not selected_profiles_df.empty
        else {}
    )
    selected_runs_df = (
        select_best_runs_by_method(export_dir, selected_profile_map=selected_profile_map or None)
        if best_run_only
        else pd.DataFrame()
    )
    selected_run_map = (
        {
            (row["method_name"], row["benchmark"]): row["run_name"]
            for row in selected_runs_df.to_dict("records")
        }
        if not selected_runs_df.empty
        else {}
    )

    frames: List[pd.DataFrame] = []
    for artifact in artifacts:
        context = _extract_export_context(artifact, export_dir)
        if selected_run_map:
            selected_run = selected_run_map.get((context["method_name"], context["benchmark_dir_name"]))
            if selected_run is None or context["run_name"] != selected_run:
                continue
        if selected_profile_map:
            selected_profile = selected_profile_map.get((context["method_name"], context["benchmark_dir_name"]))
            if selected_profile is None or context["profile_name"] != selected_profile:
                continue
        frame = _resolve_cases_from_artifact(
            artifact,
            data_root=data_root,
            benchmark=None,
            dataset_name=None,
            model=None,
        )
        if frame.empty:
            continue
        frames.append(_annotate_frame_with_context(frame, context))

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if combined.empty:
        combined.to_csv(output_dir / "cases.csv", index=False)
        _write_json(
            output_dir / "manifest.json",
            {
                "created_at": _timestamp(),
                "export_dir": str(export_dir.resolve()),
                "artifact_count": int(len(artifacts)),
                "selected_profile_count": int(len(selected_profile_map)),
                "selected_run_count": int(len(selected_run_map)),
                "case_count": 0,
            },
        )
        return combined

    combined = _enrich_cases_with_records(combined, data_root=data_root)
    combined["case_id"] = [
        _build_export_case_id(row)
        for row in combined.to_dict("records")
    ]
    combined = combined.sort_values(
        [
            "method_name",
            "benchmark",
            "profile_name",
            "run_name",
            "training_run_name",
            "error_type",
            "pair_id",
        ],
        ascending=[True, True, True, True, True, True, True],
    ).reset_index(drop=True)

    if sample_per_error_type is not None:
        combined = (
            combined.groupby(
                ["method_name", "benchmark", "profile_name", "model", "error_type"],
                group_keys=False,
            )
            .head(int(sample_per_error_type))
            .reset_index(drop=True)
        )

    combined.to_csv(output_dir / "cases.csv", index=False)
    with (output_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for row in combined.to_dict("records"):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    datasets = (
        combined.groupby(["method_name", "benchmark", "profile_name", "model", "error_type"], dropna=False)
        .size()
        .reset_index(name="count")
        .to_dict("records")
    )
    methods = sorted({_normalize_text(value) for value in combined["method_name"].tolist() if _normalize_text(value)})
    _write_json(
        output_dir / "manifest.json",
        {
            "created_at": _timestamp(),
            "export_dir": str(export_dir.resolve()),
            "data_root": str(data_root.resolve()),
            "artifact_count": int(len(artifacts)),
            "case_count": int(len(combined)),
            "method_count": int(len(methods)),
            "methods": methods,
            "best_profile_only": bool(best_profile_only),
            "best_run_only": bool(best_run_only),
            "selected_profiles": selected_profiles_df.to_dict("records") if not selected_profiles_df.empty else [],
            "selected_runs": selected_runs_df.to_dict("records") if not selected_runs_df.empty else [],
            "datasets": datasets,
        },
    )
    if not selected_profiles_df.empty:
        selected_profiles_df.to_csv(output_dir / "selected_profiles.csv", index=False)
    if not selected_runs_df.empty:
        selected_runs_df.to_csv(output_dir / "selected_runs.csv", index=False)
    return combined


def _explanation_response_format(schema_mode: str) -> Optional[Dict[str, Any]]:
    if schema_mode == "none":
        return None
    if schema_mode == "json_object":
        return {"type": "json_object"}
    return {"type": "json_object"}


def _is_publication_benchmark(benchmark: str) -> bool:
    return str(benchmark).strip().lower() in {"dblp-acm", "dblp-scholar"}


def _paper_pair_prompt(row: Dict[str, Any]) -> str:
    left = _normalize_text(row.get("left_record_json"))
    right = _normalize_text(row.get("right_record_json"))
    if _is_publication_benchmark(_normalize_text(row.get("benchmark"))):
        return (
            "Do the two publications refer to the same real-world publication?\n"
            f"Publication 1: '{left}'\n"
            f"Publication 2: '{right}'"
        )
    return (
        "Do the two entity descriptions refer to the same real-world entity? "
        "Answer with 'Yes' if they do and 'No' if they do not.\n"
        f"Entity 1: '{left}'\n"
        f"Entity 2: '{right}'"
    )


def build_explanation_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    return [{"role": "user", "content": f"{_paper_pair_prompt(row)}\n\n{PAPER_EXPLANATION_SUFFIX}"}]


def prepare_explanations(
    *,
    cases_csv: Path,
    output_dir: Path,
    model: str,
    schema_mode: str,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(cases_csv)
    if cases.empty:
        raise ValueError(f"No cases found in {cases_csv}")
    cases.to_csv(output_dir / "cases.csv", index=False)

    metadata_rows: List[Dict[str, Any]] = []
    response_format = _explanation_response_format(schema_mode)
    batch_input_path = output_dir / "batch_input.jsonl"
    with batch_input_path.open("w", encoding="utf-8") as handle:
        for row in cases.to_dict("records"):
            custom_id = _normalize_text(row.get("custom_id")) or str(row["case_id"])
            body: Dict[str, Any] = {
                "model": model,
                "messages": build_explanation_messages(row),
            }
            if response_format is not None:
                body["response_format"] = response_format
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": DEFAULT_BATCH_ENDPOINT,
                "body": body,
            }
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")
            metadata_rows.append(
                {
                    "custom_id": custom_id,
                    "case_id": str(row["case_id"]),
                    "benchmark": row["benchmark"],
                    "dataset_name": row["dataset_name"],
                    "model": row["model"],
                    "error_type": row["error_type"],
                    "pair_id": row["pair_id"],
                }
            )

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(output_dir / "metadata.csv", index=False)
    _write_json(
        output_dir / "batch_manifest.json",
        {
            "created_at": _timestamp(),
            "cases_csv": str(cases_csv.resolve()),
            "request_count": int(len(metadata_df)),
            "model": model,
            "endpoint": DEFAULT_BATCH_ENDPOINT,
            "completion_window": DEFAULT_BATCH_WINDOW,
            "schema_mode": schema_mode,
            "batch_input_path": str(batch_input_path.resolve()),
        },
    )
    return metadata_df


def submit_explanations_batch(output_dir: Path, *, kind: str = "error_analysis_explanations") -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()
    manifest = _load_json(output_dir / "batch_manifest.json")
    with (output_dir / "batch_input.jsonl").open("rb") as handle:
        file_obj = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint=str(manifest.get("endpoint", DEFAULT_BATCH_ENDPOINT)),
        completion_window=str(manifest.get("completion_window", DEFAULT_BATCH_WINDOW)),
        metadata={
            "kind": kind,
            "request_count": str(manifest.get("request_count", "")),
        },
    )
    payload = {
        "submitted_at": _timestamp(),
        "status": batch.status,
        "file_id": file_obj.id,
        "batch_id": batch.id,
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
    }
    _write_json(output_dir / "batch_info.json", payload)
    return payload


def _download_file_if_missing(client: Any, file_id: Optional[str], target_path: Path) -> None:
    if not file_id or target_path.exists():
        return
    content = client.files.content(file_id)
    with target_path.open("wb") as handle:
        handle.write(content.read())


def refresh_explanations_batch(output_dir: Path) -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()
    batch_info = _load_json(output_dir / "batch_info.json")
    batch = client.batches.retrieve(str(batch_info["batch_id"]))
    request_counts = getattr(batch, "request_counts", None)
    payload = {
        **batch_info,
        "checked_at": _timestamp(),
        "status": batch.status,
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
        "request_counts": {
            "completed": int(getattr(request_counts, "completed", 0) or 0),
            "failed": int(getattr(request_counts, "failed", 0) or 0),
            "total": int(getattr(request_counts, "total", 0) or 0),
        },
    }
    _write_json(output_dir / "batch_info.json", payload)
    _download_file_if_missing(client, payload.get("output_file_id"), output_dir / "batch_output.jsonl")
    _download_file_if_missing(client, payload.get("error_file_id"), output_dir / "batch_error.jsonl")
    return payload


def _extract_content_from_batch_row(row: Dict[str, Any]) -> Tuple[str, str]:
    if row.get("error"):
        return "", json.dumps(row["error"], ensure_ascii=False)
    try:
        content = row["response"]["body"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return "", "missing_response_content"
    return str(content), ""


def _parse_json_payload(text: str) -> Optional[Dict[str, Any]]:
    cleaned = _normalize_text(text)
    if not cleaned:
        return None
    if cleaned.startswith("```"):
        parts = cleaned.split("\n", 1)
        cleaned = parts[1] if len(parts) == 2 else cleaned
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _parse_structured_attribute_line(line: str) -> Optional[Dict[str, Any]]:
    text = _normalize_text(line)
    if not text or "attribute=" not in text.lower():
        return None
    parts = [part.strip() for part in text.split("|||") if part.strip()]
    values: Dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        values[key.strip().lower()] = value.strip()
    if "attribute" not in values:
        return None
    pair_values = values.get("values", "").split("###", 1)
    left_value = pair_values[0] if pair_values else ""
    right_value = pair_values[1] if len(pair_values) == 2 else ""
    try:
        importance = float(values.get("importance", "0"))
    except ValueError:
        importance = 0.0
    try:
        similarity = float(values.get("similarity", "0"))
    except ValueError:
        similarity = 0.0
    return {
        "attribute": values.get("attribute", ""),
        "importance": importance,
        "left_value": left_value,
        "right_value": right_value,
        "similarity": similarity,
        "raw": text,
    }


def _parse_paper_explanation(content: str) -> Dict[str, Any]:
    text = _normalize_text(content)
    decision_match = None
    similarity_pct = None
    confidence_pct = None
    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("decision:"):
            decision_match = _coerce_bool(stripped.split(":", 1)[1].strip())
        elif lower.startswith("similarity:"):
            raw = stripped.split(":", 1)[1].strip().rstrip("%")
            try:
                similarity_pct = int(raw)
            except ValueError:
                similarity_pct = None
        elif lower.startswith("confidence:"):
            raw = stripped.split(":", 1)[1].strip().rstrip("%")
            try:
                confidence_pct = int(raw)
            except ValueError:
                confidence_pct = None
    attributes = []
    for line in text.splitlines():
        parsed = _parse_structured_attribute_line(line)
        if parsed is not None:
            attributes.append(parsed)
    return {
        "paper_decision_match": decision_match,
        "paper_decision_label": (
            _format_bool_label(bool(decision_match)) if decision_match is not None else ""
        ),
        "paper_similarity_pct": similarity_pct,
        "paper_confidence_pct": confidence_pct,
        "paper_attribute_rows": attributes,
    }


def _edge_failure_summary(error_type: str, attributes: List[Dict[str, Any]]) -> str:
    if not attributes:
        return ""
    strongest_positive = max(attributes, key=lambda item: item["importance"])
    strongest_negative = min(attributes, key=lambda item: item["importance"])
    pos_text = (
        f"{strongest_positive['attribute']} ({strongest_positive['left_value']} vs "
        f"{strongest_positive['right_value']}, importance={strongest_positive['importance']:.2f})"
    )
    neg_text = (
        f"{strongest_negative['attribute']} ({strongest_negative['left_value']} vs "
        f"{strongest_negative['right_value']}, importance={strongest_negative['importance']:.2f})"
    )
    if error_type == "false_positive":
        return (
            f"The edge likely failed because match evidence from {pos_text} dominated a conflicting "
            f"signal on {neg_text}."
        )
    if error_type == "false_negative":
        return (
            f"The edge likely failed because the model over-weighted a conflicting signal on {neg_text} "
            f"despite match evidence from {pos_text}."
        )
    return ""


def merge_explanations(output_dir: Path) -> pd.DataFrame:
    cases = pd.read_csv(output_dir / "cases.csv")
    metadata_path = output_dir / "metadata.csv"
    metadata_by_custom_id: Dict[str, str] = {}
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        metadata_by_custom_id = {
            _normalize_text(row.get("custom_id")): str(row.get("case_id"))
            for row in metadata.to_dict("records")
            if _normalize_text(row.get("custom_id"))
        }
    output_path = output_dir / "batch_output.jsonl"
    if not output_path.exists():
        raise FileNotFoundError(f"Batch output missing: {output_path}")

    explanation_rows: Dict[str, Dict[str, Any]] = {}
    for row in _read_jsonl(output_path):
        custom_id = _normalize_text(row.get("custom_id"))
        if not custom_id:
            continue
        case_id = metadata_by_custom_id.get(custom_id, custom_id)
        content, error = _extract_content_from_batch_row(row)
        parsed = _parse_paper_explanation(content) if content else {}
        attrs = parsed.get("paper_attribute_rows", []) if isinstance(parsed, dict) else []
        explanation_rows[case_id] = {
            "explanation_raw": content,
            "explanation_parse_error": error,
            "paper_decision_match": parsed.get("paper_decision_match"),
            "paper_decision_label": parsed.get("paper_decision_label", ""),
            "paper_similarity_pct": parsed.get("paper_similarity_pct"),
            "paper_confidence_pct": parsed.get("paper_confidence_pct"),
            "paper_attribute_rows": json.dumps(attrs, ensure_ascii=False),
        }

    merged = cases.copy()
    for column in [
        "explanation_raw",
        "explanation_parse_error",
        "paper_decision_match",
        "paper_decision_label",
        "paper_similarity_pct",
        "paper_confidence_pct",
        "paper_attribute_rows",
    ]:
        merged[column] = merged["case_id"].map(
            lambda case_id, c=column: explanation_rows.get(str(case_id), {}).get(c, "")
        )
    strongest_positive_attributes: List[str] = []
    strongest_negative_attributes: List[str] = []
    strongest_positive_values: List[str] = []
    strongest_negative_values: List[str] = []
    failure_summaries: List[str] = []
    for row in merged.to_dict("records"):
        try:
            attrs = json.loads(_normalize_text(row.get("paper_attribute_rows")) or "[]")
        except json.JSONDecodeError:
            attrs = []
        parsed_attrs = [item for item in attrs if isinstance(item, dict)]
        if parsed_attrs:
            strongest_positive = max(parsed_attrs, key=lambda item: float(item.get("importance", 0.0)))
            strongest_negative = min(parsed_attrs, key=lambda item: float(item.get("importance", 0.0)))
            strongest_positive_attributes.append(_normalize_text(strongest_positive.get("attribute")))
            strongest_negative_attributes.append(_normalize_text(strongest_negative.get("attribute")))
            strongest_positive_values.append(
                f"{_normalize_text(strongest_positive.get('left_value'))} ### "
                f"{_normalize_text(strongest_positive.get('right_value'))}"
            )
            strongest_negative_values.append(
                f"{_normalize_text(strongest_negative.get('left_value'))} ### "
                f"{_normalize_text(strongest_negative.get('right_value'))}"
            )
            failure_summaries.append(_edge_failure_summary(_normalize_text(row.get("error_type")), parsed_attrs))
        else:
            strongest_positive_attributes.append("")
            strongest_negative_attributes.append("")
            strongest_positive_values.append("")
            strongest_negative_values.append("")
            failure_summaries.append("")
    merged["strongest_positive_attribute"] = strongest_positive_attributes
    merged["strongest_negative_attribute"] = strongest_negative_attributes
    merged["strongest_positive_values"] = strongest_positive_values
    merged["strongest_negative_values"] = strongest_negative_values
    merged["what_went_wrong"] = failure_summaries
    merged.to_csv(output_dir / "cases_with_explanations.csv", index=False)
    return merged


def prepare_summarization_prompts(cases_csv: Path, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(cases_csv)
    if cases.empty:
        raise ValueError(f"No cases found in {cases_csv}")

    rows: List[Dict[str, Any]] = []
    for (benchmark, dataset_name, model), group in cases.groupby(
        ["benchmark", "dataset_name", "model"],
        dropna=False,
        sort=False,
    ):
        prefix = (
            PAPER_SUMMARIZATION_PREFIX_PUBLICATION
            if _is_publication_benchmark(str(benchmark))
            else PAPER_SUMMARIZATION_PREFIX_PRODUCT
        )
        parts: List[str] = [prefix, "False Negatives:\n"]
        fn_group = group[group["error_type"].astype(str) == "false_negative"]
        fp_group = group[group["error_type"].astype(str) == "false_positive"]

        for idx, row in enumerate(fn_group.to_dict("records")):
            label = "Publication" if _is_publication_benchmark(str(benchmark)) else "Product"
            parts.append(
                f"FN{idx}\n"
                f"{label} 1: '{_normalize_text(row.get('left_record_json'))}'\n"
                f"{label} 2: '{_normalize_text(row.get('right_record_json'))}'\n"
                f"Explanation: {_normalize_text(row.get('explanation_raw'))}\n"
            )

        parts.append("\nFalse Positives:\n")
        for idx, row in enumerate(fp_group.to_dict("records")):
            label = "Publication" if _is_publication_benchmark(str(benchmark)) else "Product"
            parts.append(
                f"FP{idx}\n"
                f"{label} 1: '{_normalize_text(row.get('left_record_json'))}'\n"
                f"{label} 2: '{_normalize_text(row.get('right_record_json'))}'\n"
                f"Explanation: {_normalize_text(row.get('explanation_raw'))}\n"
            )

        prompt_text = "\n".join(parts).strip() + "\n"
        slug = f"{_slugify(str(benchmark))}__{_slugify(str(dataset_name))}__{_safe_model_slug(str(model))}"
        prompt_path = output_dir / f"{slug}.txt"
        prompt_path.write_text(prompt_text, encoding="utf-8")
        rows.append(
            {
                "benchmark": benchmark,
                "dataset_name": dataset_name,
                "model": model,
                "prompt_path": str(prompt_path),
                "false_negative_count": int(len(fn_group)),
                "false_positive_count": int(len(fp_group)),
            }
        )

    manifest_df = pd.DataFrame(rows)
    manifest_df.to_csv(output_dir / "summarization_prompt_manifest.csv", index=False)
    return manifest_df


def _trim_summarization_prompt(prompt_text: str, max_per_error_type: int) -> str:
    text = _normalize_text(prompt_text)
    if not text:
        return text
    split_token = "\nFalse Positives:\n"
    if split_token not in text:
        return text
    left, right = text.split(split_token, 1)
    fn_match = re.match(r"(?s)(.*?False Negatives:\n)(.*)$", left)
    fn_prefix = "False Negatives:\n"
    fn_body = left
    if fn_match:
        fn_prefix = fn_match.group(1)
        fn_body = fn_match.group(2)

    fn_entries = re.findall(r"(?s)(FN\d+\n.*?)(?=(?:\nFN\d+\n)|\Z)", fn_body)
    fp_entries = re.findall(r"(?s)(FP\d+\n.*?)(?=(?:\nFP\d+\n)|\Z)", right)
    fn_selected = "".join(fn_entries[:max_per_error_type]).strip()
    fp_selected = "".join(fp_entries[:max_per_error_type]).strip()
    out = f"{fn_prefix}{fn_selected}\n\nFalse Positives:\n{fp_selected}\n"
    return out


def _default_registry() -> Dict[str, Any]:
    return {
        "created_at": _timestamp(),
        "updated_at": _timestamp(),
        "next_category_index": 1,
        "categories": [],
    }


def _load_registry(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return _default_registry()
    payload = _load_json(path)
    payload.setdefault("created_at", _timestamp())
    payload.setdefault("updated_at", _timestamp())
    payload.setdefault("next_category_index", 1)
    payload.setdefault("categories", [])
    return payload


def _save_registry(path: Path, payload: Dict[str, Any]) -> None:
    payload["updated_at"] = _timestamp()
    _write_json(path, payload)


def _registry_categories_for_prompt(registry: Dict[str, Any]) -> List[Dict[str, Any]]:
    categories = registry.get("categories") or []
    out: List[Dict[str, Any]] = []
    for item in categories:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "description": item.get("description"),
                "applies_to": item.get("applies_to", []),
            }
        )
    return out


def _parse_registry_class_list(payload: Dict[str, Any], key: str, error_type: str) -> List[Dict[str, str]]:
    values = payload.get(key) or []
    out: List[Dict[str, str]] = []
    if not isinstance(values, list):
        return out
    for item in values:
        if not isinstance(item, dict):
            continue
        name = _normalize_text(item.get("name"))
        description = _normalize_text(item.get("description"))
        if not name or not description:
            continue
        out.append(
            {
                "name": name,
                "description": description,
                "applies_to": [error_type],
            }
        )
    return out


def _dedupe_registry_categories(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in items:
        name = _normalize_text(item.get("name"))
        description = _normalize_text(item.get("description"))
        applies_to = list(item.get("applies_to") or [])
        if not name or not description:
            continue
        key = (name.lower(), description.lower())
        existing = by_key.get(key)
        if existing is None:
            record = {
                "name": name,
                "description": description,
                "applies_to": sorted(set(str(value) for value in applies_to if value)),
            }
            by_key[key] = record
            deduped.append(record)
            continue
        existing["applies_to"] = sorted(
            set(existing.get("applies_to") or []).union(str(value) for value in applies_to if value)
        )
    return deduped


def _next_category_id(registry: Dict[str, Any]) -> str:
    index = int(registry.get("next_category_index", 1) or 1)
    registry["next_category_index"] = index + 1
    return f"C{index:03d}"


def _add_registry_category(
    registry: Dict[str, Any],
    *,
    name: str,
    description: str,
    error_type: str,
    example_case_id: str,
) -> Dict[str, Any]:
    category = {
        "id": _next_category_id(registry),
        "name": name.strip(),
        "description": description.strip(),
        "applies_to": [error_type],
        "example_case_ids": [example_case_id],
        "created_at": _timestamp(),
    }
    registry.setdefault("categories", []).append(category)
    return category


def _find_registry_category(registry: Dict[str, Any], category_id: str) -> Optional[Dict[str, Any]]:
    for category in registry.get("categories") or []:
        if str(category.get("id")) == str(category_id):
            return category
    return None


def _registry_numbering(registry: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
    lines: List[str] = []
    number_to_registry_id: Dict[str, str] = {}
    for index, category in enumerate(registry.get("categories") or [], start=1):
        if not isinstance(category, dict):
            continue
        number = str(index)
        registry_id = _normalize_text(category.get("id"))
        name = _normalize_text(category.get("name"))
        description = _normalize_text(category.get("description"))
        lines.append(f"{number}. **{name}**: {description}")
        number_to_registry_id[number] = registry_id
    return "\n\n".join(lines), number_to_registry_id


def _classification_prefix_for_row(row: Dict[str, Any]) -> str:
    if _is_publication_benchmark(_normalize_text(row.get("benchmark"))):
        return PAPER_CLASSIFICATION_PREFIX_PUBLICATION
    return PAPER_CLASSIFICATION_PREFIX_PRODUCT


def build_category_messages(row: Dict[str, Any], registry: Dict[str, Any]) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    numbered_categories, number_to_registry_id = _registry_numbering(registry)
    pair_label = "Publication" if _is_publication_benchmark(_normalize_text(row.get("benchmark"))) else "Product"
    structured_explanation = _normalize_text(row.get("explanation_raw"))
    user_prompt = (
        f"{_classification_prefix_for_row(row)}"
        "Error classes:\n\n"
        f"{numbered_categories}\n\n"
        "Now classify this pair:\n\n"
        f"Original Label: {'Match' if _coerce_bool(row.get('gold_match')) else 'Non-Match'}\n"
        f"Predicted Label: {'Match' if _coerce_bool(row.get('predicted_match')) else 'Non-Match'}\n\n"
        f"{pair_label} 1: '{_normalize_text(row.get('left_record_json'))}'\n"
        f"{pair_label} 2: '{_normalize_text(row.get('right_record_json'))}'\n"
        f"Explanation: {structured_explanation}"
    )
    return [{"role": "user", "content": user_prompt}], number_to_registry_id


def _parse_classification_json(text: str) -> Dict[str, str]:
    payload = _parse_json_payload(text) or {}
    out: Dict[str, str] = {}
    for key, value in payload.items():
        key_text = _normalize_text(key)
        value_text = _normalize_text(value)
        if key_text:
            out[key_text] = value_text
    return out


def _parse_category_assignment_payload(text: str) -> List[Dict[str, str]]:
    payload = _parse_json_payload(text) or {}
    categories = payload.get("categories")
    rows: List[Dict[str, str]] = []
    if isinstance(categories, list):
        for item in categories:
            if not isinstance(item, dict):
                continue
            number = _normalize_text(item.get("number"))
            confidence = _normalize_text(item.get("confidence"))
            if number:
                rows.append({"number": number, "confidence": confidence})
        if rows:
            return rows
    for key, value in payload.items():
        number = _normalize_text(key)
        confidence = _normalize_text(value)
        if number.isdigit():
            rows.append({"number": number, "confidence": confidence})
    return rows


def _propose_new_category(
    client: Any,
    *,
    row: Dict[str, Any],
    registry: Dict[str, Any],
    model: str,
) -> Optional[Dict[str, str]]:
    categories_json = json.dumps(_registry_categories_for_prompt(registry), ensure_ascii=False, indent=2)
    prompt = (
        "The following entity-matching error does not fit the existing registry well. "
        "Propose exactly one new stable error category with a short name and description.\n\n"
        f"Existing registry: {categories_json}\n\n"
        f"Error type: {_normalize_text(row.get('error_type'))}\n"
        f"Gold label: {_normalize_text(row.get('gold_label'))}\n"
        f"Predicted label: {_normalize_text(row.get('predicted_label'))}\n"
        f"Left record: {_normalize_text(row.get('left_record_json'))}\n"
        f"Right record: {_normalize_text(row.get('right_record_json'))}\n"
        f"Explanation: {_normalize_text(row.get('explanation_raw'))}\n\n"
        'Return JSON with exactly two fields: {"name":"...","description":"..."}'
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    payload = _parse_json_payload(response.choices[0].message.content or "")
    if payload is None:
        return None
    name = _normalize_text(payload.get("name"))
    description = _normalize_text(payload.get("description"))
    if not name or not description:
        return None
    return {"name": name, "description": description}


def generate_category_registry(
    *,
    summarization_dir: Path,
    registry_path: Path,
    model: str,
    max_per_error_type: int = 30,
) -> Dict[str, Any]:
    from openai import OpenAI

    manifest_path = summarization_dir / "summarization_prompt_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing summarization manifest: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    if manifest.empty:
        raise ValueError(f"No summarization prompts found in {manifest_path}")

    client = OpenAI()
    proposals: List[Dict[str, Any]] = []
    benchmark_rows: List[Dict[str, Any]] = []
    for row in manifest.to_dict("records"):
        prompt_path = Path(str(row["prompt_path"]))
        prompt_text = _trim_summarization_prompt(
            prompt_path.read_text(encoding="utf-8"),
            max_per_error_type=max_per_error_type,
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"{prompt_text}\n\n{REGISTRY_GENERATION_SUFFIX}"}],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        payload = _parse_json_payload(content) or {}
        fn_classes = _parse_registry_class_list(payload, "false_negative_classes", "false_negative")
        fp_classes = _parse_registry_class_list(payload, "false_positive_classes", "false_positive")
        proposals.extend(fn_classes)
        proposals.extend(fp_classes)
        benchmark_rows.append(
            {
                "benchmark": row["benchmark"],
                "dataset_name": row["dataset_name"],
                "model": row["model"],
                "prompt_path": str(prompt_path),
                "raw_response": content,
                "false_negative_count": len(fn_classes),
                "false_positive_count": len(fp_classes),
            }
        )

    deduped = _dedupe_registry_categories(proposals)
    consolidation_prompt = (
        f"{REGISTRY_CONSOLIDATION_PROMPT}\n\n"
        f"Proposed categories:\n{json.dumps(deduped, ensure_ascii=False, indent=2)}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": consolidation_prompt}],
        response_format={"type": "json_object"},
    )
    payload = _parse_json_payload(response.choices[0].message.content or "") or {}
    categories = _dedupe_registry_categories(list(payload.get("categories") or []))

    registry = _default_registry()
    registry["categories"] = []
    registry["source"] = {
        "kind": "summarization_prompts",
        "summarization_dir": str(summarization_dir.resolve()),
        "model": model,
    }
    for item in categories:
        category = {
            "id": _next_category_id(registry),
            "name": _normalize_text(item.get("name")),
            "description": _normalize_text(item.get("description")),
            "applies_to": sorted(set(str(value) for value in item.get("applies_to") or [] if value)),
            "example_case_ids": [],
            "created_at": _timestamp(),
        }
        registry["categories"].append(category)

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    _save_registry(registry_path, registry)
    pd.DataFrame(benchmark_rows).to_csv(registry_path.parent / "registry_generation_benchmarks.csv", index=False)
    _write_json(
        registry_path.parent / "registry_generation_manifest.json",
        {
            "created_at": _timestamp(),
            "registry_path": str(registry_path.resolve()),
            "category_count": int(len(registry["categories"])),
            "model": model,
            "max_per_error_type": int(max_per_error_type),
            "summarization_dir": str(summarization_dir.resolve()),
        },
    )
    return registry


def build_category_batch_messages(row: Dict[str, Any], registry: Dict[str, Any]) -> List[Dict[str, str]]:
    numbered_categories, _ = _registry_numbering(registry)
    pair_label = "Publication" if _is_publication_benchmark(_normalize_text(row.get("benchmark"))) else "Product"
    system_prompt = (
        "You assign stable entity-matching error classes. "
        'Return JSON only with the shape {"categories":[{"number":"<class number>","confidence":"<0-100>"}]}. '
        "Use an empty categories list if none apply."
    )
    user_prompt = (
        f"{_classification_prefix_for_row(row)}"
        "Error classes:\n\n"
        f"{numbered_categories}\n\n"
        "Now classify this pair. Return JSON only.\n\n"
        f"Original Label: {'Match' if _coerce_bool(row.get('gold_match')) else 'Non-Match'}\n"
        f"Predicted Label: {'Match' if _coerce_bool(row.get('predicted_match')) else 'Non-Match'}\n"
        f"Error Type: {_normalize_text(row.get('error_type'))}\n\n"
        f"{pair_label} 1: '{_normalize_text(row.get('left_record_json'))}'\n"
        f"{pair_label} 2: '{_normalize_text(row.get('right_record_json'))}'\n"
        f"Explanation: {_normalize_text(row.get('explanation_raw'))}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def prepare_category_classification_batch(
    *,
    cases_csv: Path,
    registry_path: Path,
    output_dir: Path,
    model: str,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(cases_csv)
    if cases.empty:
        raise ValueError(f"No cases found in {cases_csv}")
    registry = _load_registry(registry_path)
    _, number_to_registry_id = _registry_numbering(registry)
    mapping_json = json.dumps(number_to_registry_id, ensure_ascii=False, sort_keys=True)

    metadata_rows: List[Dict[str, Any]] = []
    batch_input_path = output_dir / "batch_input.jsonl"
    with batch_input_path.open("w", encoding="utf-8") as handle:
        for row in cases.to_dict("records"):
            custom_id = str(row["case_id"])
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": DEFAULT_BATCH_ENDPOINT,
                "body": {
                    "model": model,
                    "messages": build_category_batch_messages(row, registry),
                    "response_format": {"type": "json_object"},
                },
            }
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")
            metadata_rows.append(
                {
                    "custom_id": custom_id,
                    "case_id": custom_id,
                    "benchmark": row["benchmark"],
                    "method_name": row.get("method_name", ""),
                    "profile_name": row.get("profile_name", ""),
                    "error_type": row["error_type"],
                    "registry_number_to_id": mapping_json,
                }
            )

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(output_dir / "metadata.csv", index=False)
    _write_json(
        output_dir / "batch_manifest.json",
        {
            "created_at": _timestamp(),
            "cases_csv": str(cases_csv.resolve()),
            "registry_path": str(registry_path.resolve()),
            "request_count": int(len(metadata_df)),
            "model": model,
            "endpoint": DEFAULT_BATCH_ENDPOINT,
            "completion_window": DEFAULT_BATCH_WINDOW,
            "batch_input_path": str(batch_input_path.resolve()),
        },
    )
    return metadata_df


def submit_category_classification_batch(output_dir: Path) -> Dict[str, Any]:
    return submit_explanations_batch(output_dir, kind="error_analysis_categories")


def refresh_category_classification_batch(output_dir: Path) -> Dict[str, Any]:
    return refresh_explanations_batch(output_dir)


def merge_category_classification_batch(
    *,
    cases_csv: Path,
    registry_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    cases = pd.read_csv(cases_csv)
    metadata = pd.read_csv(output_dir / "metadata.csv")
    metadata_map = {
        _normalize_text(row.get("custom_id")): {
            "case_id": str(row.get("case_id")),
            "registry_number_to_id": json.loads(_normalize_text(row.get("registry_number_to_id")) or "{}"),
        }
        for row in metadata.to_dict("records")
        if _normalize_text(row.get("custom_id"))
    }
    registry = _load_registry(registry_path)
    output_path = output_dir / "batch_output.jsonl"
    if not output_path.exists():
        raise FileNotFoundError(f"Batch output missing: {output_path}")

    parsed_rows: Dict[str, List[Dict[str, Any]]] = {}
    for row in _read_jsonl(output_path):
        custom_id = _normalize_text(row.get("custom_id"))
        if not custom_id:
            continue
        content, error = _extract_content_from_batch_row(row)
        meta = metadata_map.get(custom_id, {"case_id": custom_id, "registry_number_to_id": {}})
        assignments: List[Dict[str, Any]] = []
        for item in _parse_category_assignment_payload(content):
            registry_id = meta["registry_number_to_id"].get(str(item["number"]))
            category = _find_registry_category(registry, str(registry_id)) if registry_id else None
            if category is None:
                continue
            assignments.append(
                {
                    "assigned_category_id": category["id"],
                    "assigned_category_name": category["name"],
                    "assigned_category_description": category["description"],
                    "assigned_category_number": item["number"],
                    "assigned_confidence": item["confidence"],
                    "category_raw_response": content,
                    "category_parse_error": error,
                }
            )
        if not assignments:
            assignments.append(
                {
                    "assigned_category_id": "",
                    "assigned_category_name": "",
                    "assigned_category_description": "",
                    "assigned_category_number": "",
                    "assigned_confidence": "",
                    "category_raw_response": content,
                    "category_parse_error": error,
                }
            )
        parsed_rows[str(meta["case_id"])] = assignments

    assignments: List[Dict[str, Any]] = []
    for row in cases.to_dict("records"):
        case_rows = parsed_rows.get(str(row["case_id"])) or [
            {
                "assigned_category_id": "",
                "assigned_category_name": "",
                "assigned_category_description": "",
                "assigned_category_number": "",
                "assigned_confidence": "",
                "category_raw_response": "",
                "category_parse_error": "missing_batch_response",
            }
        ]
        for item in case_rows:
            assignments.append({**row, **item})

    assigned_df = pd.DataFrame(assignments)
    assigned_df.to_csv(output_dir / "categorized_cases.csv", index=False)
    summary_df = (
        assigned_df.groupby(
            ["benchmark", "dataset_name", "model", "error_type", "assigned_category_id", "assigned_category_name"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )
    summary_df.to_csv(output_dir / "category_summary.csv", index=False)
    pivot_columns = [
        "benchmark",
        "method_name",
        "profile_name",
        "run_name",
        "case_id",
        "pair_id",
        "error_type",
        "assigned_category_id",
        "assigned_category_name",
        "assigned_category_description",
        "assigned_confidence",
        "what_went_wrong",
        "strongest_positive_attribute",
        "strongest_negative_attribute",
        "left_record_json",
        "right_record_json",
    ]
    assigned_df[pivot_columns].to_csv(output_dir / "pivot_error_cases.csv", index=False)
    return assigned_df


def categorize_cases(
    *,
    cases_csv: Path,
    registry_path: Path,
    output_dir: Path,
    model: str,
    max_cases: Optional[int],
) -> pd.DataFrame:
    from openai import OpenAI

    output_dir.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(cases_csv)
    if cases.empty:
        raise ValueError(f"No cases found in {cases_csv}")
    if max_cases is not None:
        cases = cases.head(int(max_cases)).copy()

    client = OpenAI()
    registry = _load_registry(registry_path)
    assignments: List[Dict[str, Any]] = []

    for row in cases.to_dict("records"):
        messages, number_to_registry_id = build_category_messages(row, registry)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        content = response.choices[0].message.content or ""
        selected = _parse_classification_json(content)
        selected_registry_ids = [
            number_to_registry_id[number]
            for number in selected.keys()
            if number in number_to_registry_id
        ]

        assignment_rows: List[Dict[str, Any]] = []
        for number, confidence in selected.items():
            registry_id = number_to_registry_id.get(number)
            if not registry_id:
                continue
            category = _find_registry_category(registry, registry_id)
            if category is None:
                continue
            applies_to = list(category.get("applies_to") or [])
            if str(row["error_type"]) not in applies_to:
                applies_to.append(str(row["error_type"]))
                category["applies_to"] = applies_to
            examples = list(category.get("example_case_ids") or [])
            if str(row["case_id"]) not in examples and len(examples) < 10:
                examples.append(str(row["case_id"]))
                category["example_case_ids"] = examples
            assignment_rows.append(
                {
                    **row,
                    "assigned_category_id": category["id"],
                    "assigned_category_name": category["name"],
                    "assigned_category_description": category["description"],
                    "assigned_category_number": number,
                    "assigned_confidence": confidence,
                    "is_new_category": False,
                    "category_raw_response": content,
                }
            )

        if not assignment_rows:
            proposed = _propose_new_category(client, row=row, registry=registry, model=model)
            if proposed:
                category = _add_registry_category(
                    registry,
                    name=proposed["name"],
                    description=proposed["description"],
                    error_type=str(row["error_type"]),
                    example_case_id=str(row["case_id"]),
                )
                assignment_rows.append(
                    {
                        **row,
                        "assigned_category_id": category["id"],
                        "assigned_category_name": category["name"],
                        "assigned_category_description": category["description"],
                        "assigned_category_number": "",
                        "assigned_confidence": "",
                        "is_new_category": True,
                        "category_raw_response": content,
                    }
                )
            else:
                assignment_rows.append(
                    {
                        **row,
                        "assigned_category_id": "",
                        "assigned_category_name": "",
                        "assigned_category_description": "",
                        "assigned_category_number": "",
                        "assigned_confidence": "",
                        "is_new_category": False,
                        "category_raw_response": content,
                    }
                )

        assignments.extend(assignment_rows)

    _save_registry(registry_path, registry)
    assigned_df = pd.DataFrame(assignments)
    assigned_df.to_csv(output_dir / "categorized_cases.csv", index=False)
    summary_df = (
        assigned_df.groupby(
            ["benchmark", "dataset_name", "model", "error_type", "assigned_category_id", "assigned_category_name"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )
    summary_df.to_csv(output_dir / "category_summary.csv", index=False)
    return assigned_df


def write_export_category_summaries(assigned_df: pd.DataFrame, output_dir: Path) -> None:
    if assigned_df.empty:
        return

    by_method = (
        assigned_df.groupby(
            ["method_name", "error_type", "assigned_category_id", "assigned_category_name"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )
    by_method.to_csv(output_dir / "category_summary_by_method.csv", index=False)

    by_method_profile = (
        assigned_df.groupby(
            [
                "method_name",
                "profile_name",
                "error_type",
                "assigned_category_id",
                "assigned_category_name",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )
    by_method_profile.to_csv(output_dir / "category_summary_by_method_profile.csv", index=False)

    by_method_benchmark_profile = (
        assigned_df.groupby(
            [
                "method_name",
                "benchmark",
                "profile_name",
                "error_type",
                "assigned_category_id",
                "assigned_category_name",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )
    by_method_benchmark_profile.to_csv(
        output_dir / "category_summary_by_method_benchmark_profile.csv",
        index=False,
    )


def prepare_export_analysis(
    *,
    export_dir: Path,
    output_dir: Path,
    data_root: Path,
    explanation_model: str,
    schema_mode: str,
    sample_per_error_type: Optional[int],
    best_profile_only: bool,
    best_run_only: bool,
    submit_batch: bool,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = collect_export_cases(
        export_dir=export_dir,
        output_dir=output_dir,
        data_root=data_root,
        sample_per_error_type=sample_per_error_type,
        best_profile_only=best_profile_only,
        best_run_only=best_run_only,
    )
    explanations_dir = output_dir / "explanations"
    prepare_explanations(
        cases_csv=output_dir / "cases.csv",
        output_dir=explanations_dir,
        model=explanation_model,
        schema_mode=schema_mode,
    )
    batch_info: Optional[Dict[str, Any]] = None
    if submit_batch:
        batch_info = submit_explanations_batch(explanations_dir)

    payload = {
        "created_at": _timestamp(),
        "export_dir": str(export_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "data_root": str(data_root.resolve()),
        "case_count": int(len(cases)),
        "best_profile_only": bool(best_profile_only),
        "best_run_only": bool(best_run_only),
        "explanations_dir": str(explanations_dir.resolve()),
        "batch_submitted": bool(submit_batch),
        "batch_info": batch_info,
    }
    _write_json(output_dir / "export_analysis_manifest.json", payload)
    return payload


def finalize_export_analysis(
    *,
    export_dir: Path,
    output_dir: Path,
    registry_path: Path,
    category_model: str,
    max_cases: Optional[int],
) -> Dict[str, Any]:
    explanations_dir = output_dir / "explanations"
    if (explanations_dir / "batch_info.json").exists():
        batch_info = refresh_explanations_batch(explanations_dir)
        if batch_info.get("status") != "completed" and not (explanations_dir / "batch_output.jsonl").exists():
            raise RuntimeError(
                f"Explanation batch is not completed yet (status={batch_info.get('status')})."
            )

    merged = merge_explanations(explanations_dir)
    summarization_dir = output_dir / "summarization"
    prepare_summarization_prompts(explanations_dir / "cases_with_explanations.csv", summarization_dir)

    categories_dir = output_dir / "categories"
    assigned = categorize_cases(
        cases_csv=explanations_dir / "cases_with_explanations.csv",
        registry_path=registry_path,
        output_dir=categories_dir,
        model=category_model,
        max_cases=max_cases,
    )
    write_export_category_summaries(assigned, categories_dir)

    payload = {
        "created_at": _timestamp(),
        "export_dir": str(export_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "registry_path": str(registry_path.resolve()),
        "merged_case_count": int(len(merged)),
        "categorized_case_count": int(len(assigned)),
        "categories_dir": str(categories_dir.resolve()),
    }
    _write_json(output_dir / "export_analysis_finalize_manifest.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build error-analysis artifacts from model outputs, generate explanation batches, and maintain a category registry."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect-cases", help="Extract model errors from prediction artifacts.")
    collect_parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        help="Artifact path: run dir, summary.csv, predictions.csv, or results.jsonl. Can be passed multiple times.",
    )
    collect_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to the export run's error_analysis folder when the artifact is inside export/.",
    )
    collect_parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Benchmark data root.")
    collect_parser.add_argument("--benchmark", default=None, help="Benchmark override for results.jsonl inputs.")
    collect_parser.add_argument("--dataset-name", default=None, help="Dataset-name override for results.jsonl inputs.")
    collect_parser.add_argument("--model", default=None, help="Model override when the artifact does not store it.")
    collect_parser.add_argument(
        "--sample-per-error-type",
        type=int,
        default=None,
        help="Optional cap per benchmark/dataset/model/error_type group.",
    )

    explain_parser = subparsers.add_parser(
        "prepare-explanations",
        help="Prepare a Batch API input file for structured explanations of the collected errors.",
    )
    explain_parser.add_argument("--cases-csv", required=True, help="Path to cases.csv from collect-cases.")
    explain_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for batch artifacts. Defaults to <error_analysis>/explanations.",
    )
    explain_parser.add_argument("--model", default=DEFAULT_EXPLANATION_MODEL, help="Explanation model.")
    explain_parser.add_argument(
        "--schema-mode",
        choices=["schema", "json_object", "none"],
        default="none",
        help="Chat Completions response_format mode.",
    )

    submit_parser = subparsers.add_parser("submit-explanations", help="Submit the prepared explanation batch.")
    submit_parser.add_argument("--output-dir", required=True, help="Batch output directory.")

    refresh_parser = subparsers.add_parser("refresh-explanations", help="Refresh batch status and download outputs.")
    refresh_parser.add_argument("--output-dir", required=True, help="Batch output directory.")

    merge_parser = subparsers.add_parser(
        "merge-explanations",
        help="Merge batch explanation outputs back into the collected cases.",
    )
    merge_parser.add_argument("--output-dir", required=True, help="Batch output directory containing cases.csv.")

    summarize_parser = subparsers.add_parser(
        "prepare-summarization",
        help="Prepare paper-style FP/FN summarization prompts from structured explanations.",
    )
    summarize_parser.add_argument("--cases-csv", required=True, help="Path to cases_with_explanations.csv.")
    summarize_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for summarization prompts. Defaults to a sibling summarization/ folder.",
    )

    registry_parser = subparsers.add_parser(
        "generate-category-registry",
        help="Generate a global error-category registry from summarization prompts.",
    )
    registry_parser.add_argument("--summarization-dir", required=True, help="Directory with summarization prompts.")
    registry_parser.add_argument("--registry-path", required=True, help="Target registry JSON path.")
    registry_parser.add_argument("--model", default=DEFAULT_CATEGORY_MODEL, help="Model for registry generation.")
    registry_parser.add_argument(
        "--max-per-error-type",
        type=int,
        default=30,
        help="Maximum number of FN and FP examples per benchmark prompt used during registry generation.",
    )

    prep_category_batch_parser = subparsers.add_parser(
        "prepare-category-classification",
        help="Prepare a Batch API input file for case classification against a fixed category registry.",
    )
    prep_category_batch_parser.add_argument("--cases-csv", required=True, help="Path to cases_with_explanations.csv.")
    prep_category_batch_parser.add_argument("--registry-path", required=True, help="Path to the category registry JSON.")
    prep_category_batch_parser.add_argument("--output-dir", required=True, help="Output directory for category batch artifacts.")
    prep_category_batch_parser.add_argument("--model", default=DEFAULT_CATEGORY_MODEL, help="Classification model.")

    submit_category_batch_parser = subparsers.add_parser(
        "submit-category-classification",
        help="Submit the prepared category-classification batch.",
    )
    submit_category_batch_parser.add_argument("--output-dir", required=True, help="Batch output directory.")

    refresh_category_batch_parser = subparsers.add_parser(
        "refresh-category-classification",
        help="Refresh category-classification batch status and download outputs.",
    )
    refresh_category_batch_parser.add_argument("--output-dir", required=True, help="Batch output directory.")

    merge_category_batch_parser = subparsers.add_parser(
        "merge-category-classification",
        help="Merge category-classification batch outputs back into the cases and export pivot CSVs.",
    )
    merge_category_batch_parser.add_argument("--cases-csv", required=True, help="Path to cases_with_explanations.csv.")
    merge_category_batch_parser.add_argument("--registry-path", required=True, help="Path to the category registry JSON.")
    merge_category_batch_parser.add_argument("--output-dir", required=True, help="Batch output directory.")

    categorize_parser = subparsers.add_parser(
        "categorize",
        help="Assign stable error categories using a persistent registry.",
    )
    categorize_parser.add_argument("--cases-csv", required=True, help="Path to cases_with_explanations.csv.")
    categorize_parser.add_argument("--registry-path", required=True, help="Path to the category registry JSON.")
    categorize_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for categorized tables. Defaults to a sibling categories/ folder.",
    )
    categorize_parser.add_argument("--model", default=DEFAULT_CATEGORY_MODEL, help="Categorization model.")
    categorize_parser.add_argument("--max-cases", type=int, default=None, help="Optional case cap for dry runs.")

    prepare_export_parser = subparsers.add_parser(
        "prepare-export-analysis",
        help="Collect all prediction artifacts in an export directory and prepare one shared explanation batch.",
    )
    prepare_export_parser.add_argument("--export-dir", required=True, help="Path to the exported result directory.")
    prepare_export_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <export-dir>/error_analysis.",
    )
    prepare_export_parser.add_argument(
        "--data-root",
        default=None,
        help="Benchmark data root. Defaults to <export-dir>/data when present, else repo data/.",
    )
    prepare_export_parser.add_argument("--model", default=DEFAULT_EXPLANATION_MODEL, help="Explanation model.")
    prepare_export_parser.add_argument(
        "--schema-mode",
        choices=["schema", "json_object", "none"],
        default="none",
        help="Chat Completions response_format mode.",
    )
    prepare_export_parser.add_argument(
        "--sample-per-error-type",
        type=int,
        default=None,
        help="Optional cap per method/benchmark/profile/model/error_type group.",
    )
    prepare_export_parser.add_argument(
        "--best-profile-only",
        action="store_true",
        help="Restrict the analysis to the best average-F1 profile per method based on summary.csv files.",
    )
    prepare_export_parser.add_argument(
        "--best-run-only",
        action="store_true",
        help="Restrict the analysis to the single best average-F1 run per method based on summary.csv files.",
    )
    prepare_export_parser.add_argument(
        "--submit",
        action="store_true",
        help="Immediately submit the prepared explanation batch to the Batch API.",
    )

    finalize_export_parser = subparsers.add_parser(
        "finalize-export-analysis",
        help="Refresh the shared batch, merge explanations, and classify errors across the whole export.",
    )
    finalize_export_parser.add_argument("--export-dir", required=True, help="Path to the exported result directory.")
    finalize_export_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <export-dir>/error_analysis.",
    )
    finalize_export_parser.add_argument(
        "--registry-path",
        default=None,
        help="Path to the shared category registry. Defaults to <output-dir>/category_registry.json.",
    )
    finalize_export_parser.add_argument("--model", default=DEFAULT_CATEGORY_MODEL, help="Categorization model.")
    finalize_export_parser.add_argument("--max-cases", type=int, default=None, help="Optional case cap for dry runs.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "collect-cases":
        artifacts = [Path(value) for value in args.artifact]
        frame = collect_cases(
            artifacts=artifacts,
            output_dir=Path(args.output_dir) if args.output_dir else _default_collect_output_dir(artifacts),
            data_root=Path(args.data_root),
            benchmark=args.benchmark,
            dataset_name=args.dataset_name,
            model=args.model,
            sample_per_error_type=args.sample_per_error_type,
        )
        print(frame.to_string(index=False) if not frame.empty else "No error cases found.")
        return

    if args.command == "prepare-explanations":
        cases_csv = Path(args.cases_csv)
        frame = prepare_explanations(
            cases_csv=cases_csv,
            output_dir=Path(args.output_dir) if args.output_dir else _default_child_output_dir(cases_csv, "explanations"),
            model=args.model,
            schema_mode=args.schema_mode,
        )
        print(frame.to_string(index=False))
        return

    if args.command == "submit-explanations":
        payload = submit_explanations_batch(Path(args.output_dir))
        print(json.dumps(payload, indent=2))
        return

    if args.command == "refresh-explanations":
        payload = refresh_explanations_batch(Path(args.output_dir))
        print(json.dumps(payload, indent=2))
        return

    if args.command == "merge-explanations":
        frame = merge_explanations(Path(args.output_dir))
        print(frame.to_string(index=False))
        return

    if args.command == "prepare-summarization":
        cases_csv = Path(args.cases_csv)
        frame = prepare_summarization_prompts(
            cases_csv,
            Path(args.output_dir) if args.output_dir else _default_child_output_dir(cases_csv, "summarization"),
        )
        print(frame.to_string(index=False))
        return

    if args.command == "generate-category-registry":
        registry = generate_category_registry(
            summarization_dir=Path(args.summarization_dir),
            registry_path=Path(args.registry_path),
            model=args.model,
            max_per_error_type=args.max_per_error_type,
        )
        print(json.dumps({"category_count": int(len(registry.get("categories") or []))}, indent=2))
        return

    if args.command == "prepare-category-classification":
        frame = prepare_category_classification_batch(
            cases_csv=Path(args.cases_csv),
            registry_path=Path(args.registry_path),
            output_dir=Path(args.output_dir),
            model=args.model,
        )
        print(frame.to_string(index=False))
        return

    if args.command == "submit-category-classification":
        payload = submit_category_classification_batch(Path(args.output_dir))
        print(json.dumps(payload, indent=2))
        return

    if args.command == "refresh-category-classification":
        payload = refresh_category_classification_batch(Path(args.output_dir))
        print(json.dumps(payload, indent=2))
        return

    if args.command == "merge-category-classification":
        frame = merge_category_classification_batch(
            cases_csv=Path(args.cases_csv),
            registry_path=Path(args.registry_path),
            output_dir=Path(args.output_dir),
        )
        print(frame.to_string(index=False))
        return

    if args.command == "categorize":
        cases_csv = Path(args.cases_csv)
        frame = categorize_cases(
            cases_csv=cases_csv,
            registry_path=Path(args.registry_path),
            output_dir=Path(args.output_dir) if args.output_dir else _default_child_output_dir(cases_csv, "categories"),
            model=args.model,
            max_cases=args.max_cases,
        )
        print(frame.to_string(index=False))
        return

    if args.command == "prepare-export-analysis":
        export_dir = Path(args.export_dir)
        output_dir = Path(args.output_dir) if args.output_dir else _default_export_output_dir(export_dir)
        data_root = Path(args.data_root) if args.data_root else _default_export_data_root(export_dir)
        payload = prepare_export_analysis(
            export_dir=export_dir,
            output_dir=output_dir,
            data_root=data_root,
            explanation_model=args.model,
            schema_mode=args.schema_mode,
            sample_per_error_type=args.sample_per_error_type,
            best_profile_only=bool(args.best_profile_only),
            best_run_only=bool(args.best_run_only),
            submit_batch=bool(args.submit),
        )
        print(json.dumps(payload, indent=2))
        return

    if args.command == "finalize-export-analysis":
        export_dir = Path(args.export_dir)
        output_dir = Path(args.output_dir) if args.output_dir else _default_export_output_dir(export_dir)
        registry_path = Path(args.registry_path) if args.registry_path else (output_dir / "category_registry.json")
        payload = finalize_export_analysis(
            export_dir=export_dir,
            output_dir=output_dir,
            registry_path=registry_path,
            category_model=args.model,
            max_cases=args.max_cases,
        )
        print(json.dumps(payload, indent=2))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
