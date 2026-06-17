#!/usr/bin/env python3
"""Plain transformers + peft LoRA SFT training.

Use this for models unsloth 2026.4.6 doesn't properly support, e.g. the MoE
openai/gpt-oss-20b (unsloth_zoo has only a `temporary_patches/gpt_oss.py`
WIP patch). For unsloth-supported models, prefer train_unsloth_lora.py (~2x
faster).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer


def build_text_formatter(tokenizer, *, reasoning_effort: str = "low"):
    def apply_chat_template(messages, *, add_generation_prompt: bool) -> str:
        kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
        # gpt-oss harmony templates accept reasoning_effort; other templates
        # will raise TypeError, in which case fall back to the default call.
        try:
            return tokenizer.apply_chat_template(messages, reasoning_effort=reasoning_effort, **kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)

    def formatter(example):
        messages = example["messages"]
        # trl calls this per-example and per-batch; detect the shape and
        # always return list[str].
        if messages and isinstance(messages[0], dict):
            return [apply_chat_template(messages, add_generation_prompt=False)]
        return [apply_chat_template(m, add_generation_prompt=False) for m in messages]

    return formatter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="openai/gpt-oss-20b")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="MoE models: stay attention-only (skip experts / grouped_mm)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument("--load-best-model-at-end", action="store_true")
    parser.add_argument("--dataset-num-proc", type=int, default=1)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--report-to", default="none")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"loading model {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False

    target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.lora_r
    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    # Gradient checkpointing for memory. use_reentrant=False plays nicer with
    # PEFT adapters.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # enable_input_require_grads is needed with gradient checkpointing + PEFT
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "valid.jsonl"),
        },
    )

    use_early_stopping = args.early_stopping_patience > 0
    effective_save_steps = (
        args.eval_steps if (use_early_stopping or args.load_best_model_at_end) else args.save_steps
    )
    load_best = args.load_best_model_at_end or use_early_stopping

    training_args = SFTConfig(
        output_dir=str(output_dir),
        max_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=effective_save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_torch",
        bf16=True,
        fp16=False,
        packing=False,
        seed=args.seed,
        dataset_num_proc=args.dataset_num_proc,
        report_to=[] if args.report_to == "none" else [args.report_to],
        gradient_checkpointing=True,
        max_grad_norm=1.0,
    )

    callbacks = []
    if use_early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        formatting_func=build_text_formatter(tokenizer, reasoning_effort=args.reasoning_effort),
        args=training_args,
        callbacks=callbacks,
    )
    trainer.train()

    final_dir = output_dir / "final_adapter"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"Saved final adapter: {final_dir}")


if __name__ == "__main__":
    main()
