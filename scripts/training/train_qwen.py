#!/usr/bin/env python3
from __future__ import annotations

# Unsloth MUST be imported before trl/transformers/peft. It patches trl.SFTConfig
# and trl.SFTTrainer at import time. Otherwise SFTTrainer hits an isinstance
# mismatch, calls TrainingArguments.to_dict() (which in transformers 5.x redacts
# any *_token field to the literal sentinel "<EOS_TOKEN>"), rebuilds SFTConfig
# with the sentinel baked in, then fails vocab lookup at validation time.
from unsloth import FastLanguageModel

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer, EarlyStoppingCallback


def build_text_formatter(tokenizer):
    def apply_chat_template(messages, *, add_generation_prompt: bool) -> str:
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

    def formatter(example):
        # Unsloth's patched SFTTrainer requires formatting_func to return a
        # list[str]. It invokes the func both per-example (test call in
        # _prepare_dataset) and per-batch. Handle both by inspecting the
        # shape of example["messages"].
        messages = example["messages"]
        if messages and isinstance(messages[0], dict):
            # Single example: messages is a list of chat turns.
            return [apply_chat_template(messages, add_generation_prompt=False)]
        # Batched: messages is a list of message-lists.
        return [apply_chat_template(m, add_generation_prompt=False) for m in messages]

    return formatter


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3.5-9B for entity matching with bf16 LoRA.")
    parser.add_argument("--data-dir", required=True, help="Directory containing train.jsonl and valid.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated list. Default covers attention + MLP. For MoE "
        "models like gpt-oss whose MLP is expert-routed, pass only attention "
        "modules (e.g. q_proj,k_proj,v_proj,o_proj).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataset-num-proc", type=int, default=1)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Number of eval_steps with no eval_loss improvement before stopping. 0 disables.",
    )
    parser.add_argument(
        "--early-stopping-threshold",
        type=float,
        default=0.0,
        help="Minimum eval_loss improvement to count as non-stagnant.",
    )
    parser.add_argument(
        "--load-best-model-at-end",
        action="store_true",
        help="Restore the best-eval checkpoint before saving the final adapter.",
    )
    parser.add_argument("--report-to", default="none", help="Set to wandb if desired")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, processor = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )

    # Qwen3.5-9B loads as a multimodal Qwen3VLProcessor; we want text-only SFT,
    # so pass a plain AutoTokenizer to TRL instead of the VL processor.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.lora_r
    target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        max_seq_length=args.max_seq_length,
    )

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "valid.jsonl"),
        },
    )

    # For early-stopping + best-model reload, save_steps must equal eval_steps
    # so that every eval has a matching checkpoint to possibly restore.
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
        optim="adamw_8bit",
        bf16=True,
        fp16=False,
        packing=False,
        seed=args.seed,
        dataset_num_proc=args.dataset_num_proc,
        report_to=[] if args.report_to == "none" else [args.report_to],
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
        formatting_func=build_text_formatter(tokenizer),
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
