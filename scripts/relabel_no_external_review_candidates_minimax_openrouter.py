#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "reports" / "no_external_review_candidates" / "review_candidates.csv.gz"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "no_external_review_candidates" / "relabel_run_minimax_openrouter"
DEFAULT_MODEL = "minimax/minimax-m2.5"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PROVIDER_PREFS = {
    "order": ["inceptron"],
    "allow_fallbacks": False,
    "quantizations": ["fp8"],
    "require_parameters": True,
}

load_dotenv(ROOT / ".env")

_thread_local = threading.local()


def _get_client() -> OpenAI:
    client = getattr(_thread_local, "client", None)
    if client is None:
        api_key = os.getenv("OPEN_ROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPEN_ROUTER_API_KEY in environment or .env")
        client = OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            default_headers={
                "X-Title": "automatic-data-labeling-relabel",
            },
        )
        _thread_local.client = client
    return client


def _json_from_text(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Model response JSON is not an object")
    return payload


def _entity_text(row: pd.Series, side: str) -> str:
    cols = [c for c in row.index if c.endswith(f"_{side}") and c not in {f"cluster_id_{side}", f"id_{side}"}]
    parts: List[str] = []
    for col in cols:
        value = row.get(col)
        if pd.isna(value) or str(value).strip() == "":
            continue
        suffix_len = 5 if side == "left" else 6
        parts.append(f"{col[:-suffix_len]}: {value}")
    return "\n".join(parts) if parts else "(no attributes)"


def _prompt_for_row(row: pd.Series) -> List[Dict[str, str]]:
    original_label = int(row["label"])
    current_label_text = "match (1)" if original_label == 1 else "non-match (0)"
    user_prompt = f"""You are reviewing an automatically labeled entity-matching training pair.

This pair was selected because the strongest label-free review signal in our pipeline fired:
- it was in the high-disagreement slice of a self-kNN embedding consistency check inside the generated training set
- and it also violated transitive-closure consistency in the positive match graph built from the generated labels

These heuristics are warning signals only. They are not proof that the label is wrong.

Benchmark: {row['benchmark']}
Profile: {row['profile']}
Pair ID: {row['pair_key']}
Current auto-label: {current_label_text}

Selection signals:
- self-kNN weighted positive rate: {row.get('self_weighted_pos_rate')}
- self-kNN mean similarity: {row.get('self_mean_similarity')}
- transitive candidate type: {row.get('candidate_type')}
- transitive contradiction count: {row.get('contradiction_count')}
- closure component size: {row.get('closure_component_size')}

Left record:
{_entity_text(row, 'left')}

Right record:
{_entity_text(row, 'right')}

Task:
1. Reason from the actual entity values only.
2. Decide whether the current label should stay as-is or be flipped.
3. Be conservative. Only flip if the evidence is strong.

Return JSON with exactly these keys:
- reasoning: short explanation
- decision: "keep" or "flip"
- suggested_label: 0 or 1
- confidence: number between 0 and 1
"""
    return [
        {
            "role": "system",
            "content": "You are an expert entity matcher. Be conservative and precise. Output valid JSON only.",
        },
        {"role": "user", "content": user_prompt},
    ]


def _call_model(row: pd.Series, model: str, max_retries: int = 3) -> Dict[str, Any]:
    messages = _prompt_for_row(row)
    client = _get_client()
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=messages,
                extra_body={
                    "provider": PROVIDER_PREFS,
                },
            )
            content = response.choices[0].message.content or ""
            payload = _json_from_text(content)
            payload["raw_response"] = content
            return payload
        except Exception as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Model call failed after retries: {last_error}")


def _evaluate_row(row: pd.Series, suggested_label: int) -> Dict[str, Any]:
    cluster_known = bool(row.get("cluster_known", False))
    cluster_match = bool(row.get("cluster_match", False))
    original_label = int(row["label"])
    original_wrong = bool(row.get("cluster_proxy_error", False))
    if not cluster_known:
        new_wrong = None
    else:
        new_wrong = (suggested_label == 1 and not cluster_match) or (suggested_label == 0 and cluster_match)
    fixed = bool(original_wrong and new_wrong is False)
    destroyed = bool((not original_wrong) and new_wrong is True)
    changed = suggested_label != original_label
    return {
        "original_label": original_label,
        "original_wrong_proxy": original_wrong,
        "new_wrong_proxy": new_wrong,
        "fixed_proxy_error": fixed,
        "destroyed_correct_label": destroyed,
        "changed_label": changed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Relabel high-precision no-external-label review candidates with OpenRouter MiniMax M2.5.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = output_dir / "results.jsonl"
    results_csv = output_dir / "results.csv.gz"
    results_xlsx = output_dir / "results.xlsx"
    summary_json = output_dir / "summary.json"

    candidates = pd.read_csv(input_path)
    if args.limit is not None:
        candidates = candidates.head(args.limit).copy()

    existing: Dict[str, Dict[str, Any]] = {}
    if results_jsonl.exists():
        for line in results_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            existing[str(payload["pair_key"])] = payload

    pending = [row for _, row in candidates.iterrows() if str(row["pair_key"]) not in existing]
    completed_initial = int(len(candidates) - len(pending))

    with tqdm(total=len(candidates), initial=completed_initial, desc="Relabeling pairs", unit="pair") as progress:
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
            futures = {
                executor.submit(_call_model, row, args.model): row
                for row in pending
            }
            for future in as_completed(futures):
                row = futures[future]
                try:
                    payload = future.result()
                    suggested_label = int(payload.get("suggested_label", row["label"]))
                    eval_payload = _evaluate_row(row, suggested_label)
                    result_row = {
                        "benchmark": row["benchmark"],
                        "profile": row["profile"],
                        "pair_key": row["pair_key"],
                        "decision": str(payload.get("decision", "keep")).strip().lower(),
                        "suggested_label": suggested_label,
                        "confidence": payload.get("confidence"),
                        "reasoning": payload.get("reasoning"),
                        "raw_response": payload.get("raw_response"),
                        **eval_payload,
                    }
                except Exception as exc:
                    result_row = {
                        "benchmark": row["benchmark"],
                        "profile": row["profile"],
                        "pair_key": row["pair_key"],
                        "decision": "error",
                        "suggested_label": row["label"],
                        "confidence": None,
                        "reasoning": str(exc),
                        "raw_response": "",
                        **_evaluate_row(row, int(row["label"])),
                    }
                with results_jsonl.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result_row, ensure_ascii=False) + "\n")
                progress.update(1)

    result_rows: List[Dict[str, Any]] = []
    for line in results_jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            result_rows.append(json.loads(line))
    results_df = pd.DataFrame(result_rows)
    merged = candidates.merge(results_df, on=["benchmark", "profile", "pair_key"], how="left")
    merged.to_csv(results_csv, index=False, compression="gzip")
    with pd.ExcelWriter(results_xlsx) as writer:
        merged.to_excel(writer, sheet_name="results", index=False)

    fixed = int(merged["fixed_proxy_error"].fillna(False).sum())
    destroyed = int(merged["destroyed_correct_label"].fillna(False).sum())
    changed = int(merged["changed_label"].fillna(False).sum())
    errors = int((merged["decision"] == "error").sum())
    summary = {
        "input_rows": int(len(candidates)),
        "completed_rows": int(len(merged)),
        "changed_labels": changed,
        "fixed_proxy_errors": fixed,
        "destroyed_correct_labels": destroyed,
        "net_proxy_gain": fixed - destroyed,
        "errors": errors,
        "model": args.model,
        "provider": PROVIDER_PREFS,
    }
    summary_json.write_text(json.dumps(summary, indent=2))

    print(results_csv)
    print(results_xlsx)
    print(summary_json)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
