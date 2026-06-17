#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "wdc" / "wdcproducts80cc20rnd100un_gs.json.gz"
OUTPUT_ROOT = ROOT / "output" / "wdc_test_minimax_openrouter"
DEFAULT_MODEL = "minimax/minimax-m2.5"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PROMPT_FIELDS = ("title", "brand", "description", "price", "priceCurrency")
MAX_FIELD_LENGTH = 200
SYSTEM_PROMPT = (
    "You are an expert entity matcher. Decide if two records refer to the same real-world entity. Return only valid JSON with exactly one field: {\"match\": true|false}."
)
PROVIDER_PREFS = {
    "order": ["sambanova"],
    "allow_fallbacks": False,
    "quantizations": ["fp8"],
    "require_parameters": True,
}

load_dotenv(ROOT / ".env")

_thread_local = threading.local()


def _model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def _output_dir_for_model(model: str) -> Path:
    return OUTPUT_ROOT / _model_slug(model)


def _get_client() -> OpenAI:
    client = getattr(_thread_local, "client", None)
    if client is None:
        api_key = os.getenv("OPEN_ROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPEN_ROUTER_API_KEY in environment or .env")
        client = OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            default_headers={"X-Title": "automatic-data-labeling-wdc-test"},
        )
        _thread_local.client = client
    return client


def _read_jsonl_gz(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def _extract_entity_payload(record: Dict[str, Any], suffix: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    suffix_token = f"_{suffix}"
    for field in PROMPT_FIELDS:
        value = _normalize_text(record.get(f"{field}{suffix_token}"))
        if not value:
            continue
        if len(value) > MAX_FIELD_LENGTH:
            value = value[:MAX_FIELD_LENGTH] + "..."
        out[field] = value
    return out


def _build_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    left_json = json.dumps(_extract_entity_payload(record, "left"), ensure_ascii=False)
    right_json = json.dumps(_extract_entity_payload(record, "right"), ensure_ascii=False)
    user_prompt = (
        "Do the two entity descriptions refer to the same real-world entity? "
        f"Entity 1: '{left_json}'. "
        f"Entity 2: '{right_json}'."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _json_from_text(text: str) -> Dict[str, Any]:
    raw = text.strip()
    if not raw:
        raise ValueError("Empty model response")
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Model response JSON is not an object")
    return payload


def _coerce_match(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Unsupported match value: {value!r}")


def _coerce_confidence(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        conf = float(value)
    except Exception:
        return None
    if conf < 0:
        return None
    return conf


def _extract_usage(response: Any) -> Dict[str, Optional[int]]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _call_model(record: Dict[str, Any], model: str, max_retries: int = 3) -> Dict[str, Any]:
    client = _get_client()
    messages = _build_messages(record)
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=messages,
                extra_body={"provider": PROVIDER_PREFS},
            )
            content = response.choices[0].message.content or ""
            payload = _json_from_text(content)
            return {
                "match": int(_coerce_match(payload.get("match"))),
                "confidence": _coerce_confidence(payload.get("confidence")),
                "raw_response": content,
                **_extract_usage(response),
            }
        except Exception as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Model call failed after retries: {last_error}")


def _compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    ok = df[df["status"] == "ok"].copy()
    prompt_tokens = int(pd.to_numeric(df.get("prompt_tokens"), errors="coerce").fillna(0).sum())
    completion_tokens = int(pd.to_numeric(df.get("completion_tokens"), errors="coerce").fillna(0).sum())
    total_tokens = int(pd.to_numeric(df.get("total_tokens"), errors="coerce").fillna(0).sum())
    if ok.empty:
        return {
            "pair_count": int(len(df)),
            "pairs_scored": 0,
            "parse_failures": int((df["status"] != "ok").sum()),
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
    y_true = ok["gold_label"].astype(int)
    y_pred = ok["pred_label"].astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    accuracy = float((tp + tn) / len(ok)) if len(ok) else None
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0
    return {
        "pair_count": int(len(df)),
        "pairs_scored": int(len(ok)),
        "parse_failures": int((df["status"] != "ok").sum()),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Relabel the WDC 100un GS test set with MiniMax via OpenRouter.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing WDC test set: {DATA_PATH}")

    output_dir = _output_dir_for_model(args.model)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_jsonl = output_dir / "results.jsonl"
    predictions_csv = output_dir / "predictions.csv.gz"
    predictions_xlsx = output_dir / "predictions.xlsx"
    summary_json = output_dir / "summary.json"
    manifest_json = output_dir / "manifest.json"

    records = _read_jsonl_gz(DATA_PATH)
    if args.limit is not None:
        records = records[: args.limit]

    manifest_json.write_text(
        json.dumps(
            {
                "model": args.model,
                "data_path": str(DATA_PATH),
                "output_dir": str(output_dir),
                "pair_count": len(records),
                "max_workers": args.max_workers,
                "prompt_fields": list(PROMPT_FIELDS),
                "max_field_length": MAX_FIELD_LENGTH,
                "prompt_source": "balanced_single from output/tribunal_v1/gpt5mini/summary.csv",
                "system_prompt": SYSTEM_PROMPT,
                "provider": PROVIDER_PREFS,
                "token_accounting": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    existing: Dict[str, Dict[str, Any]] = {}
    if results_jsonl.exists():
        for line in results_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            existing[str(payload["pair_id"])] = payload

    pending = [record for record in records if str(record.get("pair_id", "")) not in existing]

    with tqdm(total=len(records), initial=len(records) - len(pending), desc="Labeling WDC test pairs", unit="pair") as progress:
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
            futures = {executor.submit(_call_model, record, args.model): record for record in pending}
            for future in as_completed(futures):
                record = futures[future]
                pair_id = str(record.get("pair_id", ""))
                try:
                    response = future.result()
                    result_row = {
                        "pair_id": pair_id,
                        "gold_label": int(record.get("label", 0)),
                        "pred_label": int(response["match"]),
                        "confidence": response.get("confidence"),
                        "status": "ok",
                        "prompt_tokens": response.get("prompt_tokens"),
                        "completion_tokens": response.get("completion_tokens"),
                        "total_tokens": response.get("total_tokens"),
                        "raw_response": response.get("raw_response", ""),
                    }
                except Exception as exc:
                    result_row = {
                        "pair_id": pair_id,
                        "gold_label": int(record.get("label", 0)),
                        "pred_label": None,
                        "confidence": None,
                        "status": "error",
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "total_tokens": None,
                        "raw_response": str(exc),
                    }
                with results_jsonl.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result_row, ensure_ascii=False) + "\n")
                progress.update(1)

    result_rows: List[Dict[str, Any]] = []
    for line in results_jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            result_rows.append(json.loads(line))
    results_df = pd.DataFrame(result_rows).drop_duplicates(subset=["pair_id"], keep="last")

    base_rows: List[Dict[str, Any]] = []
    for record in records:
        base_rows.append(
            {
                "pair_id": str(record.get("pair_id", "")),
                "gold_label": int(record.get("label", 0)),
                "title_left": _normalize_text(record.get("title_left")),
                "brand_left": _normalize_text(record.get("brand_left")),
                "price_left": _normalize_text(record.get("price_left")),
                "priceCurrency_left": _normalize_text(record.get("priceCurrency_left")),
                "title_right": _normalize_text(record.get("title_right")),
                "brand_right": _normalize_text(record.get("brand_right")),
                "price_right": _normalize_text(record.get("price_right")),
                "priceCurrency_right": _normalize_text(record.get("priceCurrency_right")),
            }
        )
    base_df = pd.DataFrame(base_rows)
    merged = base_df.merge(results_df, on=["pair_id", "gold_label"], how="left")
    merged["is_correct"] = merged["status"].eq("ok") & (merged["gold_label"] == merged["pred_label"])

    merged.to_csv(predictions_csv, index=False, compression="gzip")
    with pd.ExcelWriter(predictions_xlsx) as writer:
        merged.to_excel(writer, sheet_name="predictions", index=False)

    summary = _compute_metrics(merged)
    summary.update(
        {
            "model": args.model,
            "data_path": str(DATA_PATH),
            "output_dir": str(output_dir),
            "prompt_source": "balanced_single from output/tribunal_v1/gpt5mini/summary.csv",
        }
    )
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
