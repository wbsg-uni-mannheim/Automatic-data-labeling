#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from third_party.ditto_modern.data import PairExample, load_wdc_json_gz, wdc_to_pair_examples
from third_party.ditto_modern.metrics import compute_binary_metrics, tune_threshold_for_f1


@dataclass
class CrossEncoderConfig:
    model_name: str = "BAAI/bge-reranker-v2-m3"
    batch_size: int = 16
    max_len: int = 512
    epochs: int = 5
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_accum_steps: int = 1
    early_stopping_patience: int = 2
    seed: int = 42
    num_workers: int = 2
    fp16: bool = True
    max_field_len: int = 350
    fields: str = "title,brand,description,price,priceCurrency"


class CrossEncoderDataset(Dataset):
    def __init__(self, examples: Sequence[PairExample], tokenizer, max_len: int):
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_len = int(max_len)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        ex = self.examples[idx]
        encoded = self.tokenizer(
            ex.left,
            ex.right,
            truncation=True,
            max_length=self.max_len,
            padding=False,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "label": int(ex.label),
            "idx": int(ex.idx),
            "pad_token_id": int(self.tokenizer.pad_token_id or 0),
        }

    @staticmethod
    def collate(batch: Sequence[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(row["input_ids"]) for row in batch)
        input_ids = []
        attention_mask = []
        labels = []
        idxs = []
        pad_token_id = int(batch[0].get("pad_token_id", 0))
        for row in batch:
            ids = list(row["input_ids"])
            mask = list(row["attention_mask"])
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_token_id] * pad_len)
            attention_mask.append(mask + [0] * pad_len)
            labels.append(int(row["label"]))
            idxs.append(int(row["idx"]))
        return {
            "input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.LongTensor(attention_mask),
            "labels": torch.LongTensor(labels),
            "idxs": torch.LongTensor(idxs),
        }


def _parse_fields(raw: str) -> List[str]:
    fields = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not fields:
        raise ValueError("At least one field must be provided")
    return fields


def _load_yaml_config(path: str | None) -> Dict[str, object]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    data = yaml.safe_load(p.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("Config YAML must be a mapping")
    return data


def _pick(name: str, cli_value, cfg: Dict[str, object], default):
    if cli_value is not None:
        return cli_value
    if name in cfg:
        return cfg[name]
    return default


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _create_run_dir(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (root / "LATEST_RUN").write_text(str(run_dir))
    return run_dir


def _save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _make_examples(path: str, fields: Sequence[str], max_field_len: int) -> List[PairExample]:
    df = load_wdc_json_gz(path)
    return wdc_to_pair_examples(df, fields=fields, max_field_len=max_field_len)


def _loss_and_probs(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim == 1 or logits.shape[-1] == 1:
        flat_logits = logits.view(-1)
        loss = F.binary_cross_entropy_with_logits(flat_logits, labels.float())
        probs = torch.sigmoid(flat_logits)
        return loss, probs
    loss = F.cross_entropy(logits, labels)
    probs = torch.softmax(logits, dim=-1)[:, 1]
    return loss, probs


def _eval(
    model,
    examples: Sequence[PairExample],
    tokenizer,
    cfg: CrossEncoderConfig,
    device: torch.device,
    threshold: float | None = None,
) -> Dict[str, object]:
    dataset = CrossEncoderDataset(examples, tokenizer=tokenizer, max_len=cfg.max_len)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=SequentialSampler(dataset),
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=CrossEncoderDataset.collate,
    )
    model.eval()
    idxs: List[int] = []
    labels: List[int] = []
    probs: List[float] = []
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss, p = _loss_and_probs(outputs.logits, y)
            total_loss += float(loss.detach().cpu().item()) * int(y.size(0))
            total_items += int(y.size(0))
            idxs.extend(batch["idxs"].cpu().tolist())
            labels.extend(y.detach().cpu().tolist())
            probs.extend(p.detach().cpu().tolist())

    used_threshold = 0.5 if threshold is None else float(threshold)
    preds = [1 if p > used_threshold else 0 for p in probs]
    metrics = compute_binary_metrics(labels, preds)
    metrics.update(
        {
            "loss": total_loss / max(1, total_items),
            "idxs": idxs,
            "labels": labels,
            "probs": probs,
            "preds": preds,
            "threshold": used_threshold,
        }
    )
    return metrics


def train(
    train_examples: Sequence[PairExample],
    val_examples: Sequence[PairExample],
    test_examples: Sequence[PairExample] | None,
    cfg: CrossEncoderConfig,
    run_dir: Path,
) -> Dict[str, object]:
    _set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    train_dataset = CrossEncoderDataset(train_examples, tokenizer=tokenizer, max_len=cfg.max_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=RandomSampler(train_dataset),
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=CrossEncoderDataset.collate,
    )
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(1, (len(train_loader) * cfg.epochs) // max(1, cfg.grad_accum_steps))
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.fp16 and device.type == "cuda"))

    checkpoints_dir = run_dir / "checkpoints"
    best_dir = checkpoints_dir / "best"
    best_val_f1 = -1.0
    best_epoch = -1
    best_threshold = 0.5
    no_improve = 0
    history: List[Dict[str, object]] = []

    _save_json(run_dir / "config.json", asdict(cfg))

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_train_loss = 0.0
        total_train_items = 0
        iterator = tqdm(train_loader, desc=f"train epoch {epoch}/{cfg.epochs}", leave=False)
        for step, batch in enumerate(iterator, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.cuda.amp.autocast(enabled=(cfg.fp16 and device.type == "cuda")):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss, _ = _loss_and_probs(outputs.logits, labels)
                loss = loss / max(1, cfg.grad_accum_steps)

            scaler.scale(loss).backward()
            if step % cfg.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            total_train_loss += float(loss.detach().cpu().item()) * int(labels.size(0)) * max(1, cfg.grad_accum_steps)
            total_train_items += int(labels.size(0))

        val_raw = _eval(model, val_examples, tokenizer, cfg, device, threshold=0.5)
        tuned_threshold, tuned_metrics = tune_threshold_for_f1(val_raw["labels"], val_raw["probs"])
        entry = {
            "epoch": epoch,
            "train_loss": total_train_loss / max(1, total_train_items),
            "val_loss": val_raw["loss"],
            "val_threshold": tuned_threshold,
            "val_precision": tuned_metrics["precision"],
            "val_recall": tuned_metrics["recall"],
            "val_f1": tuned_metrics["f1"],
            "val_accuracy": tuned_metrics["accuracy"],
        }
        history.append(entry)
        _save_json(run_dir / "history.json", {"epochs": history})

        if float(tuned_metrics["f1"]) > best_val_f1:
            best_val_f1 = float(tuned_metrics["f1"])
            best_epoch = epoch
            best_threshold = float(tuned_threshold)
            no_improve = 0
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            _save_json(checkpoints_dir / "best_metrics.json", entry)
        else:
            no_improve += 1
            if no_improve >= cfg.early_stopping_patience:
                break

    summary: Dict[str, object] = {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "best_threshold": best_threshold,
    }

    if test_examples is not None:
        best_model = AutoModelForSequenceClassification.from_pretrained(best_dir)
        best_model.to(device)
        test_metrics = _eval(best_model, test_examples, tokenizer, cfg, device, threshold=best_threshold)
        summary["test"] = {
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1": test_metrics["f1"],
            "accuracy": test_metrics["accuracy"],
            "tp": test_metrics["tp"],
            "fp": test_metrics["fp"],
            "fn": test_metrics["fn"],
            "tn": test_metrics["tn"],
            "loss": test_metrics["loss"],
            "threshold": best_threshold,
        }
        idx_to_pair = {ex.idx: ex.pair_id for ex in test_examples}
        pd.DataFrame(
            {
                "idx": test_metrics["idxs"],
                "pair_id": [idx_to_pair.get(i, f"idx-{i}") for i in test_metrics["idxs"]],
                "gold": test_metrics["labels"],
                "pred": test_metrics["preds"],
                "prob": test_metrics["probs"],
            }
        ).to_csv(run_dir / "predictions.csv", index=False)

    _save_json(run_dir / "metrics.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a plain HF cross-encoder matcher on WDC-style json.gz pairs")
    parser.add_argument("--train-json-gz", required=True)
    parser.add_argument("--val-json-gz", required=True)
    parser.add_argument("--test-json-gz", default=None)
    parser.add_argument("--config", default="configs/cross_encoder/default_train.yaml")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-dir", default="output/cross_encoder_runs")
    parser.add_argument("--fields", default=None)
    parser.add_argument("--max-field-len", type=int, default=None)
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()

    cfg_map = _load_yaml_config(args.config)
    cfg = CrossEncoderConfig(
        model_name=str(_pick("model_name", args.model_name, cfg_map, CrossEncoderConfig.model_name)),
        batch_size=int(_pick("batch_size", args.batch_size, cfg_map, CrossEncoderConfig.batch_size)),
        max_len=int(_pick("max_len", args.max_len, cfg_map, CrossEncoderConfig.max_len)),
        epochs=int(_pick("epochs", args.epochs, cfg_map, CrossEncoderConfig.epochs)),
        lr=float(_pick("lr", args.lr, cfg_map, CrossEncoderConfig.lr)),
        weight_decay=float(_pick("weight_decay", args.weight_decay, cfg_map, CrossEncoderConfig.weight_decay)),
        warmup_ratio=float(_pick("warmup_ratio", args.warmup_ratio, cfg_map, CrossEncoderConfig.warmup_ratio)),
        grad_accum_steps=int(_pick("grad_accum_steps", args.grad_accum_steps, cfg_map, CrossEncoderConfig.grad_accum_steps)),
        early_stopping_patience=int(
            _pick("early_stopping_patience", args.early_stopping_patience, cfg_map, CrossEncoderConfig.early_stopping_patience)
        ),
        seed=int(_pick("seed", args.seed, cfg_map, CrossEncoderConfig.seed)),
        num_workers=int(_pick("num_workers", args.num_workers, cfg_map, CrossEncoderConfig.num_workers)),
        fp16=(bool(_pick("fp16", None, cfg_map, CrossEncoderConfig.fp16)) and not args.no_fp16),
        max_field_len=int(_pick("max_field_len", args.max_field_len, cfg_map, CrossEncoderConfig.max_field_len)),
        fields=str(_pick("fields", args.fields, cfg_map, CrossEncoderConfig.fields)),
    )

    fields = _parse_fields(cfg.fields)
    train_examples = _make_examples(args.train_json_gz, fields, cfg.max_field_len)
    val_examples = _make_examples(args.val_json_gz, fields, cfg.max_field_len)
    test_examples = _make_examples(args.test_json_gz, fields, cfg.max_field_len) if args.test_json_gz else None
    run_dir = _create_run_dir(args.output_dir)
    summary = train(train_examples, val_examples, test_examples, cfg, run_dir)
    print(f"Run directory: {run_dir}")
    print(summary)


if __name__ == "__main__":
    main()
