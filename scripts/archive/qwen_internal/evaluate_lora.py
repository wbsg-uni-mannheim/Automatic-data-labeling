#!/usr/bin/env python3
from __future__ import annotations

# Import unsloth first to apply its transformers/trl patches before any other
# imports touch those modules (see train_unsloth_lora.py for the full story).
from unsloth import FastLanguageModel

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import AutoTokenizer


THINK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)


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


def apply_chat_template(tokenizer, messages: List[Dict[str, str]], *, add_generation_prompt: bool) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": False,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def clean_response(text: str) -> str:
    text = THINK_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_yes_no(text: str) -> int | None:
    cleaned = clean_response(text).lower()
    match = re.search(r"\b(yes|no)\b", cleaned)
    if not match:
        return None
    return 1 if match.group(1) == "yes" else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Qwen3.5 EM LoRA adapter on SFT JSONL.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model-path", required=True, help="Adapter dir produced by train_unsloth_lora.py")
    parser.add_argument("--base-model-name", default="Qwen/Qwen3.5-9B", help="Base model for loading a plain text tokenizer")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(Path(args.data_path))

    model, _processor = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
    )
    FastLanguageModel.for_inference(model)

    # Use the plain text tokenizer from the base model (the LoRA adapter dir
    # typically saves the text tokenizer, but the base has it for sure).
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)

    records: List[Dict[str, Any]] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    parse_failures = 0

    # Warm-up to exclude JIT/lazy-init cost from timing
    if rows:
        warm_prompt = apply_chat_template(tokenizer, strip_gold_answer(rows[0]["messages"]), add_generation_prompt=True)
        warm_in = tokenizer(warm_prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            _ = model.generate(**warm_in, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _eval_start = time.perf_counter()

    for row in tqdm(rows, desc="evaluate"):
        prompt_messages = strip_gold_answer(row["messages"])
        prompt = apply_chat_template(tokenizer, prompt_messages, add_generation_prompt=True)
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
        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
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

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _eval_elapsed = time.perf_counter() - _eval_start

    metrics = {
        "rows": len(rows),
        "parse_failures": parse_failures,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "inference_time_s": float(_eval_elapsed),
        "ms_per_pair": float(1000 * _eval_elapsed / max(len(rows), 1)),
        "pairs_per_s": float(len(rows) / max(_eval_elapsed, 1e-9)),
    }
    pd.DataFrame(records).to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
