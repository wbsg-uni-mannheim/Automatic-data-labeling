#!/usr/bin/env python3
"""Zero-shot EM baseline using plain transformers (no unsloth).

Use this for models unsloth 2026.4.6 doesn't support well (e.g.
openai/gpt-oss-20b). For unsloth-supported models, prefer baseline_zero_shot_eval.py
(2x faster).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


THINK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
# gpt-oss harmony channels — strip everything except the final channel
CHANNEL_FINAL_RE = re.compile(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)", re.DOTALL)
CHANNEL_ANY_RE = re.compile(r"<\|channel\|>\w+<\|message\|>", re.DOTALL)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def strip_gold_answer(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [m for m in messages if m.get("role") != "assistant"]


def apply_chat_template(tokenizer, messages: List[Dict[str, str]], *, add_generation_prompt: bool, reasoning_effort: str = "low") -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    # gpt-oss harmony templates support reasoning_effort=low|medium|high.
    # Pass it through; templates that don't use it will ignore the kwarg.
    try:
        return tokenizer.apply_chat_template(messages, reasoning_effort=reasoning_effort, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def clean_response(text: str) -> str:
    text = THINK_RE.sub("", text)
    # If the response has a final channel marker, extract just the final channel
    m = CHANNEL_FINAL_RE.search(text)
    if m:
        text = m.group(1)
    # Strip leftover channel markers
    text = CHANNEL_ANY_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_yes_no(text: str) -> int | None:
    cleaned = clean_response(text).lower()
    match = re.search(r"\b(yes|no)\b", cleaned)
    if not match:
        return None
    return 1 if match.group(1) == "yes" else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model-name", default="openai/gpt-oss-20b",
                        help="Base model repo OR a path to a saved LoRA adapter dir")
    parser.add_argument("--base-model-name", default=None,
                        help="Explicit base model name when --model-name points at an adapter")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--reasoning-effort", default="low", help="For gpt-oss harmony templates")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(Path(args.data_path))
    if args.limit > 0:
        rows = rows[: args.limit]

    # If --model-name points at an adapter dir, load base model then apply adapter.
    adapter_config = Path(args.model_name) / "adapter_config.json"
    is_adapter = adapter_config.exists()
    if is_adapter:
        base_name = args.base_model_name
        if base_name is None:
            base_name = json.loads(adapter_config.read_text()).get("base_model_name_or_path")
        if base_name is None:
            raise SystemExit("Could not determine base model for adapter; pass --base-model-name")
        print(f"loading base {base_name}, adapter {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(base_name)
        base_model = AutoModelForCausalLM.from_pretrained(base_name, dtype=torch.bfloat16, device_map="auto")
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, args.model_name)
    else:
        print(f"loading model: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    model.eval()

    records: List[Dict[str, Any]] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    parse_failures = 0

    for row in tqdm(rows, desc="zero-shot eval"):
        prompt_messages = strip_gold_answer(row["messages"])
        prompt = apply_chat_template(tokenizer, prompt_messages, add_generation_prompt=True, reasoning_effort=args.reasoning_effort)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if args.temperature > 0:
            gen_kwargs["temperature"] = args.temperature
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **gen_kwargs)
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(generated_ids, skip_special_tokens=False)
        pred = parse_yes_no(raw)
        if pred is None:
            parse_failures += 1
            pred = 0
        gold = int(row["label"])
        y_true.append(gold)
        y_pred.append(pred)
        records.append(
            {
                "pair_id": row.get("pair_id", ""),
                "label": gold,
                "prediction": pred,
                "raw_response": raw,
                "clean_response": clean_response(raw),
            }
        )

    metrics = {
        "model_name": args.model_name,
        "data_path": str(args.data_path),
        "rows": len(rows),
        "parse_failures": parse_failures,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    pd.DataFrame(records).to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
