#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SequentialSampler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_base_module():
    base_path = Path(__file__).resolve().with_name("run_simple_labeling.py")
    spec = importlib.util.spec_from_file_location("_run_simple_labeling_base", base_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base labeling module from {base_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base_module()

from third_party.ditto_modern.data import PairDataset, wdc_to_pair_examples
from third_party.ditto_modern.knowledge import inject_knowledge
from third_party.ditto_modern.model import load_model, load_tokenizer
from third_party.ditto_modern.runtime import DistContext, save_json as ditto_save_json
from third_party.ditto_modern.summarize import summarize_examples
from third_party.ditto_modern.trainer import TrainConfig, train_loop

RESERVED_FEATURE_FIELDS = base.RESERVED_FEATURE_FIELDS
_build_candidates = base._build_candidates
_build_preview = base._build_preview
_build_seed_queue = base._build_seed_queue
_count_labels = base._count_labels
_estimate_usage_costs = base._estimate_usage_costs
_fit_matcher_ensemble = base._fit_matcher_ensemble
_label_iteratively_per_query_seed = base._label_iteratively_per_query_seed
_label_pair = base._label_pair
_load_df = base._load_df
_load_json = base._load_json
_load_resume_labels = base._load_resume_labels
_load_seed_for_summary = base._load_seed_for_summary
_materialize_output_ids = base._materialize_output_ids
_parse_field_list_arg = base._parse_field_list_arg
_parse_schema_map_arg = base._parse_schema_map_arg
_plan_adaptive_round = base._plan_adaptive_round
_run_active_learning_same_prompt = base._run_active_learning_same_prompt
_run_matchers_on_pool = base._run_matchers_on_pool
_save_json = base._save_json
_trim_exact = base._trim_exact


@dataclass(frozen=True)
class DittoMember:
    name: str
    checkpoint_dir: Path
    threshold: float
    val_f1: float
    train_rows: int
    valid_rows: int
    cfg: TrainConfig
    max_field_len: int


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _context() -> DistContext:
    return DistContext(
        enabled=False,
        rank=0,
        world_size=1,
        local_rank=0,
        device=_device(),
    )


def _normalize_label_to_int(value: object) -> int:
    text = str(value).strip().upper()
    if text in {"1", "TRUE", "T", "YES", "Y"}:
        return 1
    if text in {"0", "FALSE", "F", "NO", "N"}:
        return 0
    raise ValueError(f"Unsupported label value: {value!r}")


def _label_stats(df: pd.DataFrame) -> Dict[str, int]:
    labels = df["label"].astype(int)
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    return {"rows": int(len(df)), "pos": pos, "neg": neg}


def _load_yaml_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    payload = yaml.safe_load(cfg_path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {cfg_path}")
    return payload


def _resolve_phase_target_counts(
    final_target_size: int,
    final_target_pos: int,
    stage_size: int,
) -> Tuple[int, int]:
    final_target_size = int(final_target_size)
    final_target_pos = int(final_target_pos)
    stage_size = max(0, min(int(stage_size), final_target_size))
    if stage_size <= 0:
        return 0, 0
    if final_target_size <= 0:
        return 0, 0
    pos_ratio = float(final_target_pos / final_target_size)
    stage_pos = int(round(stage_size * pos_ratio))
    stage_pos = max(0, min(stage_pos, stage_size))
    stage_neg = int(stage_size - stage_pos)
    return stage_pos, stage_neg


def _remaining_class_budget(
    current_pos: int,
    current_neg: int,
    target_pos: int,
    target_neg: int,
) -> int:
    pos_gap = max(0, int(target_pos) - int(current_pos))
    neg_gap = max(0, int(target_neg) - int(current_neg))
    return int(pos_gap + neg_gap)


def _pairs_to_ditto_df(
    pairs: pd.DataFrame,
    *,
    left_map: Dict[str, Dict[str, object]],
    right_map: Dict[str, Dict[str, object]],
    left_rid_to_id: Dict[str, str],
    right_rid_to_id: Dict[str, str],
    fields: Sequence[str],
    default_label: int = 0,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in pairs.iterrows():
        rid1 = str(row["id1"])
        rid2 = str(row["id2"])
        left_row = left_map[rid1]
        right_row = right_map[rid2]
        out: Dict[str, object] = {
            "pair_id": f"{left_rid_to_id.get(rid1, rid1)}__{right_rid_to_id.get(rid2, rid2)}__{rid1}__{rid2}",
            "label": int(default_label),
            "is_hard_negative": 0,
            "id_left": left_rid_to_id.get(rid1, rid1),
            "id_right": right_rid_to_id.get(rid2, rid2),
        }
        if "label" in row and pd.notna(row["label"]):
            out["label"] = _normalize_label_to_int(row["label"])
        for field in fields:
            out[f"{field}_left"] = left_row.get(field, "")
            out[f"{field}_right"] = right_row.get(field, "")
        rows.append(out)
    return pd.DataFrame(rows)


def _apply_ditto_transforms(
    *,
    train_examples,
    valid_examples,
    test_examples,
    cfg: TrainConfig,
):
    if cfg.summarize:
        train_examples, valid_examples, test_examples = summarize_examples(
            train_examples=train_examples,
            val_examples=valid_examples,
            test_examples=test_examples,
            lm=cfg.model_name,
            max_len=cfg.max_len,
        )

    if cfg.dk is not None:
        dk_norm = str(cfg.dk).strip().lower()
        if dk_norm not in {"product", "general"}:
            raise ValueError("--phase3-ditto-dk must be one of: product, general")
        train_examples = inject_knowledge(train_examples, dk=dk_norm, spacy_model=cfg.spacy_model)
        valid_examples = inject_knowledge(valid_examples, dk=dk_norm, spacy_model=cfg.spacy_model)
        if test_examples is not None:
            test_examples = inject_knowledge(test_examples, dk=dk_norm, spacy_model=cfg.spacy_model)

    return train_examples, valid_examples, test_examples


def _build_phase3_ditto_config(args: argparse.Namespace) -> Tuple[TrainConfig, int]:
    cfg_map = _load_yaml_config(args.phase3_ditto_config)
    train_cfg_fields = set(TrainConfig.__dataclass_fields__.keys())
    cfg_kwargs = {k: v for k, v in cfg_map.items() if k in train_cfg_fields}

    if args.phase3_ditto_model_name:
        cfg_kwargs["model_name"] = args.phase3_ditto_model_name
    if args.phase3_ditto_batch_size is not None:
        cfg_kwargs["batch_size"] = int(args.phase3_ditto_batch_size)
    if args.phase3_ditto_max_len is not None:
        cfg_kwargs["max_len"] = int(args.phase3_ditto_max_len)
    if args.phase3_ditto_epochs is not None:
        cfg_kwargs["epochs"] = int(args.phase3_ditto_epochs)
    if args.phase3_ditto_lr is not None:
        cfg_kwargs["lr"] = float(args.phase3_ditto_lr)
    if args.phase3_ditto_no_fp16:
        cfg_kwargs["fp16"] = False
    if args.phase3_ditto_seed is not None:
        cfg_kwargs["seed"] = int(args.phase3_ditto_seed)

    cfg = TrainConfig(**cfg_kwargs)
    max_field_len = int(args.phase3_ditto_max_field_len or cfg_map.get("max_field_len", 350))
    return cfg, max_field_len


def _make_bagged_split(
    labeled_wdc: pd.DataFrame,
    *,
    valid_fraction: float,
    bootstrap_fraction: float,
    seed: int,
    target_pos_ratio: float | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if labeled_wdc.empty:
        raise ValueError("Cannot bag an empty labeled set")
    if labeled_wdc["label"].nunique() < 2:
        raise ValueError("Ditto bagging requires both classes to be present")

    valid_fraction = float(min(max(valid_fraction, 0.05), 0.4))
    bootstrap_fraction = float(max(bootstrap_fraction, 0.25))
    stratify = labeled_wdc["label"]
    test_size = max(2, int(round(len(labeled_wdc) * valid_fraction)))
    test_size = min(test_size, max(len(labeled_wdc) - 2, 2))

    for offset in range(10):
        train_base, valid_df = train_test_split(
            labeled_wdc,
            test_size=test_size,
            random_state=seed + offset,
            stratify=stratify,
            shuffle=True,
        )
        train_base = train_base.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        if train_base["label"].nunique() < 2 or valid_df["label"].nunique() < 2:
            continue

        train_n = max(2, int(round(len(train_base) * bootstrap_fraction)))
        ratio = float(target_pos_ratio) if target_pos_ratio is not None else float(train_base["label"].mean())
        ratio = float(min(max(ratio, 1e-6), 1.0 - 1e-6))
        pos_base = train_base[train_base["label"] == 1].reset_index(drop=True)
        neg_base = train_base[train_base["label"] == 0].reset_index(drop=True)
        if pos_base.empty or neg_base.empty:
            continue

        pos_n = int(round(train_n * ratio))
        pos_n = max(1, min(pos_n, train_n - 1))
        neg_n = max(1, train_n - pos_n)
        rng = np.random.default_rng(seed + offset)

        pos_idx = rng.choice(len(pos_base), size=pos_n, replace=True)
        neg_idx = rng.choice(len(neg_base), size=neg_n, replace=True)
        train_df = pd.concat(
            [
                pos_base.iloc[pos_idx],
                neg_base.iloc[neg_idx],
            ],
            ignore_index=True,
        )
        train_df = train_df.sample(frac=1.0, random_state=seed + offset).reset_index(drop=True)
        return train_df, valid_df

    raise RuntimeError("Could not create a bagged train/validation split with both classes present")


def _train_ditto_bagged_ensemble(
    *,
    labeled: pd.DataFrame,
    left_map: Dict[str, Dict[str, object]],
    right_map: Dict[str, Dict[str, object]],
    left_rid_to_id: Dict[str, str],
    right_rid_to_id: Dict[str, str],
    fields: Sequence[str],
    run_dir: Path,
    round_idx: int,
    num_models: int,
    valid_fraction: float,
    bootstrap_fraction: float,
    target_pos_ratio: float,
    train_cfg: TrainConfig,
    max_field_len: int,
) -> Tuple[List[DittoMember], Dict[str, object]]:
    phase3_dir = run_dir / "phase3" / f"round_{int(round_idx):02d}"
    phase3_dir.mkdir(parents=True, exist_ok=True)

    labeled_wdc = _pairs_to_ditto_df(
        labeled[["id1", "id2", "label"]],
        left_map=left_map,
        right_map=right_map,
        left_rid_to_id=left_rid_to_id,
        right_rid_to_id=right_rid_to_id,
        fields=fields,
    )
    label_summary = _label_stats(labeled_wdc)
    if label_summary["pos"] <= 0 or label_summary["neg"] <= 0:
        raise RuntimeError("Phase 3 requires both positive and negative labeled examples")

    ctx = _context()
    members: List[DittoMember] = []
    summary_rows: List[Dict[str, object]] = []
    base_seed = int(train_cfg.seed)

    for model_idx in range(int(num_models)):
        member_name = f"ditto_bag_{model_idx + 1:02d}"
        member_dir = phase3_dir / member_name
        member_dir.mkdir(parents=True, exist_ok=True)

        train_df, valid_df = _make_bagged_split(
            labeled_wdc,
            valid_fraction=valid_fraction,
            bootstrap_fraction=bootstrap_fraction,
            seed=base_seed + model_idx,
            target_pos_ratio=target_pos_ratio,
        )
        member_cfg = TrainConfig(**{**asdict(train_cfg), "seed": base_seed + model_idx})

        train_examples = wdc_to_pair_examples(train_df, fields=fields, max_field_len=max_field_len)
        valid_examples = wdc_to_pair_examples(valid_df, fields=fields, max_field_len=max_field_len)
        train_examples, valid_examples, _ = _apply_ditto_transforms(
            train_examples=train_examples,
            valid_examples=valid_examples,
            test_examples=None,
            cfg=member_cfg,
        )

        t0 = time.perf_counter()
        summary = train_loop(
            train_examples=train_examples,
            val_examples=valid_examples,
            cfg=member_cfg,
            run_dir=member_dir,
            ctx=ctx,
            test_examples=None,
            sample_weights=None,
        )
        dt = time.perf_counter() - t0
        checkpoint_dir = member_dir / "checkpoints" / "best"
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Missing Ditto checkpoint for {member_name}: {checkpoint_dir}")

        member = DittoMember(
            name=member_name,
            checkpoint_dir=checkpoint_dir,
            threshold=float(summary.get("best_threshold", 0.5) or 0.5),
            val_f1=float(summary.get("best_val_f1", 0.0) or 0.0),
            train_rows=int(len(train_df)),
            valid_rows=int(len(valid_df)),
            cfg=member_cfg,
            max_field_len=max_field_len,
        )
        members.append(member)
        summary_rows.append(
            {
                "name": member_name,
                "checkpoint_dir": str(checkpoint_dir),
                "best_val_f1": member.val_f1,
                "threshold": member.threshold,
                "train_rows": member.train_rows,
                "valid_rows": member.valid_rows,
                "duration_seconds": dt,
            }
        )
        print(
            f"Phase 3 round {round_idx}: trained {member_name} "
            f"(train={member.train_rows}, valid={member.valid_rows}, val_f1={member.val_f1:.3f})",
            flush=True,
        )

    summary_path = phase3_dir / "ditto_ensemble_summary.json"
    ditto_save_json(
        summary_path,
        {
            "round": int(round_idx),
            "labeled_rows": label_summary["rows"],
            "labeled_pos": label_summary["pos"],
            "labeled_neg": label_summary["neg"],
            "num_models": int(num_models),
            "target_pos_ratio": float(target_pos_ratio),
            "members": summary_rows,
        },
    )
    return members, {"summary_path": str(summary_path), "members": summary_rows}


def _predict_ditto_scores(
    *,
    member: DittoMember,
    pool: pd.DataFrame,
    left_map: Dict[str, Dict[str, object]],
    right_map: Dict[str, Dict[str, object]],
    left_rid_to_id: Dict[str, str],
    right_rid_to_id: Dict[str, str],
    fields: Sequence[str],
    inference_batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    if pool.empty:
        return pd.DataFrame(columns=["id1", "id2", "score"])

    pool_wdc = _pairs_to_ditto_df(
        pool[["id1", "id2"]],
        left_map=left_map,
        right_map=right_map,
        left_rid_to_id=left_rid_to_id,
        right_rid_to_id=right_rid_to_id,
        fields=fields,
        default_label=0,
    )
    examples = wdc_to_pair_examples(pool_wdc, fields=fields, max_field_len=member.max_field_len)
    _, _, examples = _apply_ditto_transforms(
        train_examples=[],
        valid_examples=[],
        test_examples=examples,
        cfg=member.cfg,
    )
    if examples is None:
        return pd.DataFrame(columns=["id1", "id2", "score"])

    tokenizer = load_tokenizer(str(member.checkpoint_dir))
    model = load_model(str(member.checkpoint_dir), alpha_aug=member.cfg.alpha_aug)
    model.to(device)
    model.eval()

    dataset = PairDataset(examples, tokenizer=tokenizer, max_len=member.cfg.max_len, da=None)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(inference_batch_size)),
        sampler=SequentialSampler(dataset),
        num_workers=0,
        pin_memory=False,
        collate_fn=PairDataset.pad,
    )

    scores: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x, att, _y, _idxs, _weights = batch
            x = x.to(device)
            att = att.to(device)
            logits = model(input_ids=x, attention_mask=att)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            scores.append(probs.detach().cpu().numpy())

    probs_np = np.concatenate(scores, axis=0) if scores else np.zeros((0,), dtype=np.float32)
    out = pool[["id1", "id2"]].copy().reset_index(drop=True)
    out["score"] = probs_np.astype(float)
    return out


def _rank_probability_disagreements(
    correspondences_list: List[Dict[str, object]],
) -> pd.DataFrame:
    valid = [
        result
        for result in correspondences_list
        if isinstance(result.get("correspondences"), pd.DataFrame) and not result["correspondences"].empty
    ]
    if len(valid) < 2:
        return pd.DataFrame(
            columns=[
                "id1",
                "id2",
                "model_count",
                "votes_pos",
                "votes_neg",
                "conflict_count",
                "vote_entropy",
                "score_variance",
                "mean_score",
            ]
        )

    pair_payloads: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    for result in valid:
        corr = result["correspondences"].copy()
        corr["score"] = pd.to_numeric(corr["score"], errors="coerce")
        threshold = float(result.get("threshold", 0.5) or 0.5)
        for _, row in corr.iterrows():
            score = row["score"]
            if pd.isna(score):
                continue
            key = (str(row["id1"]), str(row["id2"]))
            payload = pair_payloads.setdefault(key, {"scores": [], "votes": []})
            s = float(score)
            payload["scores"].append(s)
            payload["votes"].append(1.0 if s >= threshold else 0.0)

    rows: List[Dict[str, object]] = []
    for (id1, id2), payload in pair_payloads.items():
        scores = payload["scores"]
        votes = payload["votes"]
        if len(votes) < 2:
            continue
        votes_pos = int(sum(votes))
        votes_neg = int(len(votes) - votes_pos)
        conflict_count = int(min(votes_pos, votes_neg))
        p = float(votes_pos / len(votes))
        vote_entropy = 0.0
        if 0.0 < p < 1.0:
            vote_entropy = float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))
        mean_score = float(np.mean(scores))
        score_variance = float(np.var(scores))
        if score_variance <= 0.0 and vote_entropy <= 0.0:
            continue
        rows.append(
            {
                "id1": id1,
                "id2": id2,
                "model_count": int(len(votes)),
                "votes_pos": votes_pos,
                "votes_neg": votes_neg,
                "conflict_count": conflict_count,
                "vote_entropy": vote_entropy,
                "score_variance": score_variance,
                "mean_score": mean_score,
                "mean_margin": float(abs(mean_score - 0.5)),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "id1",
                "id2",
                "model_count",
                "votes_pos",
                "votes_neg",
                "conflict_count",
                "vote_entropy",
                "score_variance",
                "mean_score",
                "mean_margin",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        ["score_variance", "vote_entropy", "mean_margin", "conflict_count"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)


def _select_phase3_pool(
    pool_all: pd.DataFrame,
    *,
    labeled: pd.DataFrame,
    pool_cap: int,
    round_idx: int,
) -> pd.DataFrame:
    labeled_keys = set(
        (
            labeled["id1"].astype(str)
            + "||"
            + labeled["id2"].astype(str)
        ).tolist()
    )
    pool_keys = pool_all["id1"].astype(str) + "||" + pool_all["id2"].astype(str)
    pool = pool_all.loc[~pool_keys.isin(labeled_keys)].reset_index(drop=True)
    if pool.empty:
        return pool

    if pool_cap > 0 and len(pool) > int(pool_cap):
        keep_top = max(int(pool_cap) // 2, 1)
        head = pool.sort_values("similarity", ascending=False).head(keep_top)
        tail_n = int(pool_cap) - len(head)
        if tail_n > 0:
            rem = pool.drop(index=head.index)
            if not rem.empty:
                tail = rem.sample(n=min(tail_n, len(rem)), random_state=84 + round_idx)
                pool = pd.concat([head, tail], ignore_index=True)
            else:
                pool = head.reset_index(drop=True)
        else:
            pool = head.reset_index(drop=True)
        pool = pool.drop_duplicates(subset=["id1", "id2"], keep="first").reset_index(drop=True)
    return pool


def _run_phase3_active_learning(
    *,
    client: OpenAI,
    model: str,
    labeled: pd.DataFrame,
    candidates: pd.DataFrame,
    left_map: Dict[str, Dict[str, object]],
    right_map: Dict[str, Dict[str, object]],
    left_idx: Dict[str, int],
    right_idx: Dict[str, int],
    left_emb: np.ndarray,
    right_emb: np.ndarray,
    left_rid_to_id: Dict[str, str],
    right_rid_to_id: Dict[str, str],
    feature_fields: Sequence[str],
    target_pos: int,
    target_neg: int,
    labels_per_iteration: int,
    active_candidates: int,
    max_iterations: int,
    usage_stats: Dict[str, int],
    run_dir: Path,
    ensemble_mode: str,
    ditto_model_count: int,
    ditto_valid_fraction: float,
    ditto_bootstrap_fraction: float,
    ditto_target_pos_ratio: float,
    ditto_train_cfg: TrainConfig,
    ditto_max_field_len: int,
    ditto_inference_batch_size: int,
    classic_top_matchers: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    training_set = labeled[["id1", "id2", "label", "similarity"]].copy()
    training_set["id1"] = training_set["id1"].astype(str)
    training_set["id2"] = training_set["id2"].astype(str)
    training_set["label"] = training_set["label"].astype(str).str.upper().str.strip()
    training_set["label"] = training_set["label"].replace({"1": "TRUE", "0": "FALSE"})
    training_set["similarity"] = pd.to_numeric(training_set.get("similarity", 0.0), errors="coerce").fillna(0.0)
    training_set = training_set.drop_duplicates(subset=["id1", "id2"], keep="last").reset_index(drop=True)

    pool_all = candidates[["id1", "id2", "similarity"]].copy()
    pool_all["id1"] = pool_all["id1"].astype(str)
    pool_all["id2"] = pool_all["id2"].astype(str)
    pool_all["similarity"] = pd.to_numeric(pool_all["similarity"], errors="coerce").fillna(0.0).astype(float)
    pool_all = pool_all.drop_duplicates(subset=["id1", "id2"], keep="first").reset_index(drop=True)

    rounds: List[Dict[str, object]] = []
    device = _device()

    for iteration in range(1, int(max_iterations) + 1):
        cur_pos, cur_neg = _count_labels(training_set)
        if cur_pos >= target_pos and cur_neg >= target_neg:
            break

        remaining_budget = _remaining_class_budget(
            current_pos=cur_pos,
            current_neg=cur_neg,
            target_pos=target_pos,
            target_neg=target_neg,
        )
        quota = min(int(labels_per_iteration), remaining_budget) if remaining_budget > 0 else 0
        if quota <= 0:
            break

        pool = _select_phase3_pool(
            pool_all,
            labeled=training_set,
            pool_cap=active_candidates,
            round_idx=iteration,
        )
        print(
            f"Phase 3 iter {iteration}: train={len(training_set)} "
            f"(pos={cur_pos}, neg={cur_neg}), unlabeled_pool={len(pool)}",
            flush=True,
        )
        if pool.empty:
            rounds.append(
                {
                    "iteration": int(iteration),
                    "status": "stopped",
                    "reason": "candidate_pool_exhausted",
                }
            )
            break

        ditto_members, ditto_summary = _train_ditto_bagged_ensemble(
            labeled=training_set,
            left_map=left_map,
            right_map=right_map,
            left_rid_to_id=left_rid_to_id,
            right_rid_to_id=right_rid_to_id,
            fields=feature_fields,
            run_dir=run_dir,
            round_idx=iteration,
            num_models=ditto_model_count,
            valid_fraction=ditto_valid_fraction,
            bootstrap_fraction=ditto_bootstrap_fraction,
            target_pos_ratio=ditto_target_pos_ratio,
            train_cfg=ditto_train_cfg,
            max_field_len=ditto_max_field_len,
        )

        correspondences_list: List[Dict[str, object]] = []
        for member in ditto_members:
            score_t0 = time.perf_counter()
            corr = _predict_ditto_scores(
                member=member,
                pool=pool,
                left_map=left_map,
                right_map=right_map,
                left_rid_to_id=left_rid_to_id,
                right_rid_to_id=right_rid_to_id,
                fields=feature_fields,
                inference_batch_size=ditto_inference_batch_size,
                device=device,
            )
            correspondences_list.append(
                {
                    "matcher": member.name,
                    "f1": member.val_f1,
                    "threshold": member.threshold,
                    "correspondences": corr,
                    "family": "ditto",
                    "duration_seconds": time.perf_counter() - score_t0,
                }
            )

        classic_matchers_used: List[str] = []
        if ensemble_mode == "hybrid":
            fitted = _fit_matcher_ensemble(
                training_set,
                left_map,
                right_map,
                left_idx,
                right_idx,
                left_emb,
                right_emb,
                feature_fields=feature_fields,
                verbose=True,
            )
            top_classic = fitted[: max(0, int(classic_top_matchers))]
            if top_classic:
                classic_corrs = _run_matchers_on_pool(
                    pool,
                    top_classic,
                    left_map,
                    right_map,
                    left_idx,
                    right_idx,
                    left_emb,
                    right_emb,
                    feature_fields=feature_fields,
                )
                for item in classic_corrs:
                    item["family"] = "classic"
                correspondences_list.extend(classic_corrs)
                classic_matchers_used = [str(x.get("matcher", "")) for x in top_classic]

        disagreements = _rank_probability_disagreements(correspondences_list)
        if disagreements.empty:
            rounds.append(
                {
                    "iteration": int(iteration),
                    "status": "stopped",
                    "reason": "no_probability_disagreements",
                    "ensemble_mode": ensemble_mode,
                }
            )
            break

        select = disagreements.merge(pool[["id1", "id2", "similarity"]], on=["id1", "id2"], how="left")
        select = select.head(quota).reset_index(drop=True)
        print(
            f"Phase 3 iter {iteration}: selected {len(select)} pairs "
            f"from {len(disagreements)} probability disagreements using mode={ensemble_mode}",
            flush=True,
        )
        if select.empty:
            rounds.append(
                {
                    "iteration": int(iteration),
                    "status": "stopped",
                    "reason": "no_pairs_selected",
                    "ensemble_mode": ensemble_mode,
                }
            )
            break

        labeled_pairs = set(zip(training_set["id1"].astype(str), training_set["id2"].astype(str)))
        new_rows: List[Dict[str, object]] = []
        label_t0 = time.perf_counter()
        for _, row in select.iterrows():
            id1 = str(row["id1"])
            id2 = str(row["id2"])
            if (id1, id2) in labeled_pairs:
                continue
            label, usage = _label_pair(client, model, left_map[id1], right_map[id2])
            usage_stats["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
            usage_stats["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
            usage_stats["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
            usage_stats["active_prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
            usage_stats["active_completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
            usage_stats["active_total_tokens"] += int(usage.get("total_tokens", 0) or 0)
            new_rows.append(
                {
                    "id1": id1,
                    "id2": id2,
                    "label": label,
                    "similarity": float(row.get("similarity", 0.0) or 0.0),
                    "al_source": f"phase3_{ensemble_mode}_probability_disagreement",
                    "iteration": int(iteration),
                    "phase3_conflict_count": int(row.get("conflict_count", 0) or 0),
                    "phase3_votes_pos": int(row.get("votes_pos", 0) or 0),
                    "phase3_votes_neg": int(row.get("votes_neg", 0) or 0),
                }
            )
            labeled_pairs.add((id1, id2))

        if not new_rows:
            rounds.append(
                {
                    "iteration": int(iteration),
                    "status": "stopped",
                    "reason": "no_new_labels",
                    "ensemble_mode": ensemble_mode,
                }
            )
            break

        add_df = pd.DataFrame(new_rows)
        training_set = pd.concat([training_set, add_df], ignore_index=True)
        training_set = training_set.drop_duplicates(subset=["id1", "id2"], keep="last").reset_index(drop=True)
        cur_pos, cur_neg = _count_labels(training_set)
        print(
            f"Phase 3 iter {iteration}: added {len(add_df)} labels "
            f"in {time.perf_counter() - label_t0:.1f}s; totals pos={cur_pos}, neg={cur_neg}",
            flush=True,
        )
        rounds.append(
            {
                "iteration": int(iteration),
                "status": "ok",
                "ensemble_mode": ensemble_mode,
                "pool_size": int(len(pool)),
                "disagreements_found": int(len(disagreements)),
                "selected_pairs": int(len(select)),
                "new_labels": int(len(add_df)),
                "total_size": int(len(training_set)),
                "total_pos": int(cur_pos),
                "total_neg": int(cur_neg),
                "classic_matchers": classic_matchers_used,
                "ditto_summary": ditto_summary,
            }
        )

        _materialize_output_ids(training_set, left_rid_to_id, right_rid_to_id).to_csv(
            run_dir / "active_labels_latest.csv", index=False
        )
        _save_json(
            run_dir / "run_state.json",
            {
                "stage": "phase3_active_learning",
                "active_learning_impl": "three_phase_ditto_ensemble",
                "total": int(len(training_set)),
                "pos": int(cur_pos),
                "neg": int(cur_neg),
                "target_pos": int(target_pos),
                "target_neg": int(target_neg),
                "phase3_rounds": rounds,
                "token_usage": usage_stats,
            },
        )

    final_pos, final_neg = _count_labels(training_set)
    return training_set, {
        "impl": "three_phase_ditto_ensemble",
        "ensemble_mode": ensemble_mode,
        "ditto_model_count": int(ditto_model_count),
        "labels_per_iteration": int(labels_per_iteration),
        "active_candidates": int(active_candidates),
        "max_iterations": int(max_iterations),
        "rounds": rounds,
        "final_total": int(len(training_set)),
        "final_pos": int(final_pos),
        "final_neg": int(final_neg),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Three-phase active labeling with a Ditto-based phase 3 ensemble.")
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--left-csv", default="data/wdc/wdc_train_large_left.csv")
    parser.add_argument("--right-csv", default="data/wdc/wdc_train_large_right.csv")
    parser.add_argument("--left-schema-map", default="")
    parser.add_argument("--right-schema-map", default="")
    parser.add_argument("--strict-schema", action="store_true")
    parser.add_argument("--left-emb", default="wdc_left_embeddings.npy")
    parser.add_argument("--right-emb", default="wdc_right_embeddings.npy")
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--faiss-k", type=int, default=20)
    parser.add_argument("--faiss-random-state", type=int, default=42)
    parser.add_argument("--candidate-cap", type=int, default=0)

    parser.add_argument("--seed-size", type=int, default=100)
    parser.add_argument("--seed-positives", type=int, default=30)
    parser.add_argument("--seed-max-calls", type=int, default=4000)
    parser.add_argument("--seed-pos-per-query", type=int, default=1)
    parser.add_argument("--seed-neg-per-query", type=int, default=4)
    parser.add_argument("--seed-batch-size", type=int, default=5)
    parser.add_argument("--seed-bottom-k", type=int, default=2)
    parser.add_argument("--seed-query-order", type=str, default="random", choices=["similarity", "random", "left"])

    parser.add_argument("--target-size", type=int, default=2500)
    parser.add_argument("--target-positives", type=int, default=500)
    parser.add_argument("--labels-per-iteration", type=int, default=100)
    parser.add_argument("--active-candidates", type=int, default=5000)
    parser.add_argument("--active-top-matchers", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--adaptive-round-size", type=int, default=0)
    parser.add_argument("--adaptive-neg-min-share", type=float, default=0.60)
    parser.add_argument("--adaptive-neg-max-share", type=float, default=0.90)
    parser.add_argument(
        "--feature-fields",
        default="title,brand,description,price,priceCurrency",
        help="Comma-separated fields to use for classic and Ditto phase-3 features.",
    )

    parser.add_argument("--phase2-target-size", type=int, default=1000)
    parser.add_argument(
        "--phase2-target-positives",
        type=int,
        default=None,
        help="Optional explicit positive target for phase 2. Default derives from the final class ratio.",
    )
    parser.add_argument(
        "--phase2-target-negatives",
        type=int,
        default=None,
        help="Optional explicit negative target for phase 2. Default derives from the final class ratio.",
    )

    parser.add_argument("--phase3-batch-size", type=int, default=500)
    parser.add_argument("--phase3-candidates", type=int, default=5000)
    parser.add_argument("--phase3-max-iterations", type=int, default=30)
    parser.add_argument(
        "--phase3-ensemble-mode",
        choices=["ditto_only", "hybrid"],
        default="hybrid",
        help="Use only the five bagged Ditto models, or combine them with the classic in-script ensemble.",
    )
    parser.add_argument("--phase3-ditto-models", type=int, default=5)
    parser.add_argument("--phase3-ditto-valid-fraction", type=float, default=0.20)
    parser.add_argument("--phase3-ditto-bootstrap-fraction", type=float, default=1.0)
    parser.add_argument("--phase3-classic-top-matchers", type=int, default=5)
    parser.add_argument("--phase3-ditto-config", default="configs/ditto/default_train.yaml")
    parser.add_argument("--phase3-ditto-model-name", default=None)
    parser.add_argument("--phase3-ditto-batch-size", type=int, default=None)
    parser.add_argument("--phase3-ditto-inference-batch-size", type=int, default=32)
    parser.add_argument("--phase3-ditto-max-len", type=int, default=None)
    parser.add_argument("--phase3-ditto-max-field-len", type=int, default=None)
    parser.add_argument("--phase3-ditto-epochs", type=int, default=None)
    parser.add_argument("--phase3-ditto-lr", type=float, default=None)
    parser.add_argument("--phase3-ditto-seed", type=int, default=42)
    parser.add_argument("--phase3-ditto-no-fp16", action="store_true")

    parser.add_argument("--preview-k", type=int, default=0)
    parser.add_argument("--output-root", default="output/simple_labeling")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-enforce-exact-final", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    load_dotenv()

    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "run_state.json"
    prev_state = _load_json(state_path) if args.resume else {}
    prev_usage = prev_state.get("token_usage", {}) if isinstance(prev_state, dict) else {}

    left_schema_map = _parse_schema_map_arg(args.left_schema_map, side="left")
    right_schema_map = _parse_schema_map_arg(args.right_schema_map, side="right")
    left_df = _load_df(Path(args.left_csv), side="L", schema_map=left_schema_map, strict_schema=args.strict_schema)
    right_df = _load_df(Path(args.right_csv), side="R", schema_map=right_schema_map, strict_schema=args.strict_schema)

    feature_fields = _parse_field_list_arg(args.feature_fields)
    feature_fields = [f for f in feature_fields if f not in RESERVED_FEATURE_FIELDS]
    feature_fields = [f for f in feature_fields if f in left_df.columns and f in right_df.columns]
    if not feature_fields:
        raise ValueError(
            "No valid --feature-fields found in both left/right dataframes. "
            f"Requested={_parse_field_list_arg(args.feature_fields)}"
        )

    emb_dir = Path(args.embeddings_dir)
    left_emb = np.load(emb_dir / args.left_emb).astype(np.float32)
    right_emb = np.load(emb_dir / args.right_emb).astype(np.float32)
    if len(left_df) != left_emb.shape[0]:
        raise ValueError(f"Left rows {len(left_df)} != left embedding rows {left_emb.shape[0]}")
    if len(right_df) != right_emb.shape[0]:
        raise ValueError(f"Right rows {len(right_df)} != right embedding rows {right_emb.shape[0]}")
    if left_emb.shape[1] != right_emb.shape[1]:
        raise ValueError("Left/right embedding dimensions differ")

    left_df = left_df.reset_index(drop=True)
    right_df = right_df.reset_index(drop=True)
    left_ids = left_df["__rid"].astype(str).to_numpy()
    right_ids = right_df["__rid"].astype(str).to_numpy()
    left_idx = {str(v): i for i, v in enumerate(left_ids.tolist())}
    right_idx = {str(v): i for i, v in enumerate(right_ids.tolist())}
    left_map = {str(r["__rid"]): r for r in left_df.to_dict("records")}
    right_map = {str(r["__rid"]): r for r in right_df.to_dict("records")}
    left_rid_to_id = {str(r["__rid"]): str(r["id"]) for r in left_df.to_dict("records")}
    right_rid_to_id = {str(r["__rid"]): str(r["id"]) for r in right_df.to_dict("records")}

    usage_stats: Dict[str, int] = {
        "prompt_tokens": int(prev_usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(prev_usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(prev_usage.get("total_tokens", 0) or 0),
        "seed_prompt_tokens": int(prev_usage.get("seed_prompt_tokens", 0) or 0),
        "seed_completion_tokens": int(prev_usage.get("seed_completion_tokens", 0) or 0),
        "seed_total_tokens": int(prev_usage.get("seed_total_tokens", 0) or 0),
        "active_prompt_tokens": int(prev_usage.get("active_prompt_tokens", 0) or 0),
        "active_completion_tokens": int(prev_usage.get("active_completion_tokens", 0) or 0),
        "active_total_tokens": int(prev_usage.get("active_total_tokens", 0) or 0),
    }

    _save_json(
        state_path,
        {
            "stage": "init",
            "left_rows": len(left_df),
            "right_rows": len(right_df),
            "embedding_dim": int(left_emb.shape[1]),
            "left_schema_map": left_schema_map,
            "right_schema_map": right_schema_map,
            "feature_fields": feature_fields,
            "token_usage": usage_stats,
        },
    )

    candidates, faiss_stats = _build_candidates(
        left_ids=left_ids,
        right_ids=right_ids,
        right_source_ids=right_df["id"].astype(str).to_numpy(),
        left_emb=left_emb,
        right_emb=right_emb,
        k=args.faiss_k,
        candidate_cap=args.candidate_cap,
        bottom_k=args.seed_bottom_k,
        random_state=args.faiss_random_state,
    )
    candidates_dedup = candidates.copy()
    candidates_dedup["src_id1"] = candidates_dedup["id1"].astype(str).map(left_rid_to_id)
    candidates_dedup["src_id2"] = candidates_dedup["id2"].astype(str).map(right_rid_to_id)
    before_source_pair_dedup = len(candidates_dedup)
    candidates_dedup = candidates_dedup.drop_duplicates(subset=["src_id1", "src_id2"], keep="first")
    after_source_pair_dedup = len(candidates_dedup)
    candidates = candidates_dedup.drop(columns=["src_id1", "src_id2"]).reset_index(drop=True)
    faiss_stats["source_pair_dedup_before"] = int(before_source_pair_dedup)
    faiss_stats["source_pair_dedup_after"] = int(after_source_pair_dedup)
    faiss_stats["source_pair_dedup_dropped"] = int(before_source_pair_dedup - after_source_pair_dedup)

    candidates_with_ids = _materialize_output_ids(candidates, left_rid_to_id, right_rid_to_id)
    candidates_with_ids.to_csv(run_dir / "faiss_candidates.csv", index=False)
    _save_json(
        state_path,
        {
            "stage": "candidates_done",
            "candidates": int(len(candidates)),
            "faiss": faiss_stats,
            "token_usage": usage_stats,
        },
    )

    all_neighbors = candidates.rename(columns={"id1": "query_id", "id2": "neighbor_id"}).copy()
    all_neighbors["rank"] = (
        all_neighbors.groupby("query_id")["similarity"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    seed_queue = _build_seed_queue(candidates)
    if args.preview_k and args.preview_k > 0:
        preview = _build_preview(
            queue=seed_queue,
            left_map=left_map,
            right_map=right_map,
            left_rid_to_id=left_rid_to_id,
            right_rid_to_id=right_rid_to_id,
            k=args.preview_k,
        )
        preview_path = run_dir / "preview_first_k.csv"
        preview.to_csv(preview_path, index=False)
        print(f"Preview mode: saved {preview_path}")
        return

    client = OpenAI()
    resumed_labeled: pd.DataFrame | None = None
    resumed_from: str | None = None
    if args.resume:
        resumed_labeled, resumed_from = _load_resume_labels(run_dir)
        if resumed_labeled is not None:
            rpos, rneg = _count_labels(resumed_labeled)
            print(f"Resume: loaded {len(resumed_labeled)} labels from {resumed_from} ({rpos} pos, {rneg} neg)")
        else:
            print("Resume: no checkpoint found, starting fresh.")

    if resumed_labeled is None:
        seed = _label_iteratively_per_query_seed(
            client=client,
            model=args.model,
            all_neighbors=all_neighbors,
            left_map=left_map,
            right_map=right_map,
            target_positives_per_query=args.seed_pos_per_query,
            target_negatives_per_query=args.seed_neg_per_query,
            total_target_positives=args.seed_positives,
            total_target_size=args.seed_size,
            max_calls=args.seed_max_calls,
            batch_size=args.seed_batch_size,
            query_order=args.seed_query_order,
            bottom_k=args.seed_bottom_k,
            state_path=state_path,
            out_csv=run_dir / "seed_labels_internal.csv",
            usage_stats=usage_stats,
        )
        if seed.empty:
            raise RuntimeError("Seed labeling produced zero rows")
        _materialize_output_ids(seed, left_rid_to_id, right_rid_to_id).to_csv(run_dir / "seed_labels.csv", index=False)
        seed_pos, seed_neg = _count_labels(seed)
        print(f"Seed: {len(seed)} ({seed_pos} pos, {seed_neg} neg)")
        labeled = seed.copy().reset_index(drop=True)
        labeled["label"] = labeled["label"].astype(str).str.upper()
    else:
        seed = _load_seed_for_summary(run_dir)
        seed_pos, seed_neg = _count_labels(seed) if not seed.empty else (0, 0)
        labeled = resumed_labeled.copy().reset_index(drop=True)
        labeled["label"] = labeled["label"].astype(str).str.upper()

    final_target_pos = int(args.target_positives)
    final_target_neg = int(args.target_size - args.target_positives)
    if final_target_neg < 0:
        raise ValueError("--target-size must be >= --target-positives")

    phase2_target_size = min(int(args.phase2_target_size), int(args.target_size))
    if phase2_target_size < int(args.seed_size):
        raise ValueError("--phase2-target-size must be >= --seed-size")

    if args.phase2_target_positives is not None or args.phase2_target_negatives is not None:
        phase2_target_pos = int(args.phase2_target_positives if args.phase2_target_positives is not None else 0)
        phase2_target_neg = int(
            args.phase2_target_negatives
            if args.phase2_target_negatives is not None
            else phase2_target_size - phase2_target_pos
        )
        if phase2_target_pos + phase2_target_neg != phase2_target_size:
            raise ValueError("Phase 2 targets must sum to --phase2-target-size")
    else:
        phase2_target_pos, phase2_target_neg = _resolve_phase_target_counts(
            final_target_size=args.target_size,
            final_target_pos=args.target_positives,
            stage_size=phase2_target_size,
        )

    adaptive_round_size = args.adaptive_round_size if args.adaptive_round_size > 0 else args.labels_per_iteration
    adaptive_round_size = max(1, int(adaptive_round_size))

    pos, neg = _count_labels(labeled)
    phase2_rounds: List[Dict[str, object]] = []
    if pos < phase2_target_pos or neg < phase2_target_neg:
        print(
            f"Phase 2: start total={len(labeled)} pos={pos} neg={neg} "
            f"target={phase2_target_pos}/{phase2_target_neg}",
            flush=True,
        )
        round_idx = 0
        while pos < phase2_target_pos or neg < phase2_target_neg:
            plan = _plan_adaptive_round(
                current_pos=pos,
                current_neg=neg,
                target_pos=phase2_target_pos,
                target_neg=phase2_target_neg,
                round_size=adaptive_round_size,
                min_neg_share=float(args.adaptive_neg_min_share),
                max_neg_share=float(args.adaptive_neg_max_share),
            )
            budget = int(plan["budget"])
            round_target_pos = int(plan["target_pos"])
            round_target_neg = int(plan["target_neg"])
            if budget <= 0 or (round_target_pos <= pos and round_target_neg <= neg):
                break

            round_idx += 1
            prev_pos, prev_neg, prev_total = pos, neg, len(labeled)
            labeled, active_summary = _run_active_learning_same_prompt(
                client=client,
                model=args.model,
                labeled=labeled,
                candidates=candidates,
                left_map=left_map,
                right_map=right_map,
                left_idx=left_idx,
                right_idx=right_idx,
                left_emb=left_emb,
                right_emb=right_emb,
                feature_fields=feature_fields,
                target_pos=round_target_pos,
                target_neg=round_target_neg,
                labels_per_iteration=min(args.labels_per_iteration, budget),
                active_candidates=args.active_candidates,
                active_top_matchers=args.active_top_matchers,
                max_iterations=args.max_iterations,
                max_total_labels_override=budget,
                usage_stats=usage_stats,
            )
            pos, neg = _count_labels(labeled)
            phase2_rounds.append(
                {
                    "round": int(round_idx),
                    "start_pos": int(prev_pos),
                    "start_neg": int(prev_neg),
                    "end_pos": int(pos),
                    "end_neg": int(neg),
                    "gain_total": int(len(labeled) - prev_total),
                    "budget": int(budget),
                    "plan": plan,
                    "active_summary": active_summary,
                }
            )
            _materialize_output_ids(labeled, left_rid_to_id, right_rid_to_id).to_csv(
                run_dir / "active_labels_latest.csv", index=False
            )
            if pos <= prev_pos and neg <= prev_neg:
                print("Phase 2: no class progress in this round, stopping early.", flush=True)
                break

    ditto_cfg, ditto_max_field_len = _build_phase3_ditto_config(args)
    pos, neg = _count_labels(labeled)
    phase3_summary: Dict[str, object] = {"skipped": True}
    if pos < final_target_pos or neg < final_target_neg:
        print(
            f"Phase 3: start total={len(labeled)} pos={pos} neg={neg} "
            f"target={final_target_pos}/{final_target_neg} mode={args.phase3_ensemble_mode}",
            flush=True,
        )
        final_target_ratio = float(final_target_pos / max(final_target_pos + final_target_neg, 1))
        labeled, phase3_summary = _run_phase3_active_learning(
            client=client,
            model=args.model,
            labeled=labeled,
            candidates=candidates,
            left_map=left_map,
            right_map=right_map,
            left_idx=left_idx,
            right_idx=right_idx,
            left_emb=left_emb,
            right_emb=right_emb,
            left_rid_to_id=left_rid_to_id,
            right_rid_to_id=right_rid_to_id,
            feature_fields=feature_fields,
            target_pos=final_target_pos,
            target_neg=final_target_neg,
            labels_per_iteration=int(args.phase3_batch_size),
            active_candidates=int(args.phase3_candidates),
            max_iterations=int(args.phase3_max_iterations),
            usage_stats=usage_stats,
            run_dir=run_dir,
            ensemble_mode=str(args.phase3_ensemble_mode),
            ditto_model_count=int(args.phase3_ditto_models),
            ditto_valid_fraction=float(args.phase3_ditto_valid_fraction),
            ditto_bootstrap_fraction=float(args.phase3_ditto_bootstrap_fraction),
            ditto_target_pos_ratio=final_target_ratio,
            ditto_train_cfg=ditto_cfg,
            ditto_max_field_len=ditto_max_field_len,
            ditto_inference_batch_size=int(args.phase3_ditto_inference_batch_size),
            classic_top_matchers=int(args.phase3_classic_top_matchers),
        )

    final_df = labeled.copy()
    if not args.no_enforce_exact_final:
        final_df = _trim_exact(final_df, final_target_pos, final_target_neg)
    final_df = final_df.reset_index(drop=True)
    final_out = _materialize_output_ids(final_df, left_rid_to_id, right_rid_to_id)
    final_out.to_csv(run_dir / "labels_final.csv", index=False)

    final_pos, final_neg = _count_labels(final_df)
    cost_summary = _estimate_usage_costs(args.model, usage_stats)
    _save_json(
        run_dir / "summary.json",
        {
            "seed_total": int(len(seed)),
            "seed_pos": int(seed_pos),
            "seed_neg": int(seed_neg),
            "phase2_target_total": int(phase2_target_size),
            "phase2_target_pos": int(phase2_target_pos),
            "phase2_target_neg": int(phase2_target_neg),
            "phase2_rounds": phase2_rounds,
            "phase3_summary": phase3_summary,
            "final_total": int(len(final_out)),
            "final_pos": int(final_pos),
            "final_neg": int(final_neg),
            "target_total": int(args.target_size),
            "target_pos": int(final_target_pos),
            "target_neg": int(final_target_neg),
            "faiss": faiss_stats,
            "token_usage": usage_stats,
            "labeling_cost": cost_summary,
        },
    )
    _save_json(
        state_path,
        {
            "stage": "done",
            "final_total": int(len(final_df)),
            "final_pos": int(final_pos),
            "final_neg": int(final_neg),
            "phase2_rounds": phase2_rounds,
            "phase3_summary": phase3_summary,
            "token_usage": usage_stats,
            "labeling_cost": cost_summary,
        },
    )
    print(f"Final: {len(final_out)} ({final_pos} pos, {final_neg} neg)")
    print(
        "Token usage: "
        f"prompt={usage_stats['prompt_tokens']}, "
        f"completion={usage_stats['completion_tokens']}, "
        f"total={usage_stats['total_tokens']}"
    )
    if bool(cost_summary.get("available")):
        print(f"Estimated labeling cost (USD): {float(cost_summary['total_cost_usd']):.6f}")
    print(f"Output: {run_dir / 'labels_final.csv'}")


if __name__ == "__main__":
    main()
