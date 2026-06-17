"""
Single-Batch Multi-Agent Labeling for WDC Entity Matching
=========================================================

Design:
- Keep multiple agent prompts (precision / balanced / recall)
- Submit exactly one batch per run (all agent requests in one JSONL)
- No sampling (use full dataset)
- Aggregate with majority vote (tie-break via balanced agent)

Commands
--------
Generate one batch input file:
  python scripts/multi_agent_crowd.py generate \
      --data benchmarks/wdc/wdcproducts80cc20rnd100un_gs.json.gz \
      --model gpt-5.2mini \
      --output-dir output/multi_agent_single_batch

Submit single batch:
  python scripts/multi_agent_crowd.py submit --output-dir output/multi_agent_single_batch

Check status and download outputs:
  python scripts/multi_agent_crowd.py status --output-dir output/multi_agent_single_batch

Evaluate per-agent + voted performance:
  python scripts/multi_agent_crowd.py evaluate --output-dir output/multi_agent_single_batch
"""

import argparse
import gzip
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file, if present

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_FIELDS = ["title", "brand", "description", "price", "priceCurrency"]
DEFAULT_DATA = "benchmarks/wdc/wdcproducts80cc20rnd100un_gs.json.gz"
DEFAULT_MODEL = "gpt-5-mini-2025-08-07"
DEFAULT_OUTPUT_DIR = "output/multi_agent_single_batch"

# Keep prompts close to prior style, but role-specialized
AGENTS: Dict[str, Dict[str, str]] = {
    "precision": {
        "name": "Precision Sentinel",
        "system_prompt": (
            "Do the two product descriptions refer to the same real-world product? "
            "Be conservative: only return match=true when core identifiers align and there is no hard conflict. "
            "Hard conflicts include different model/part numbers, capacities, versions, sizes, or variants. "
            "Answer with ONLY valid JSON: {\"match\": true|false}."
        ),
    },
    "balanced": {
        "name": "Balanced Resolver",
        "system_prompt": (
            "Do the two product descriptions refer to the same real-world product? "
            "Balance precision and recall. Treat wording/language/formatting differences as normal, "
            "but reject clear model/spec conflicts. "
            "Answer with ONLY valid JSON: {\"match\": true|false}."
        ),
    },
    "recall": {
        "name": "Recall Scout",
        "system_prompt": (
            "Do the two product descriptions refer to the same real-world product? "
            "Be permissive for incomplete/noisy listings: if evidence points to the same underlying model and there is "
            "no explicit contradiction, prefer match=true. "
            "Answer with ONLY valid JSON: {\"match\": true|false}."
        ),
    },
}


def _read_jsonl_gz(path: Path) -> List[Dict[str, Any]]:
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_run_dir(base_output_dir: str, create_new: bool) -> Path:
    """
    Resolve a concrete run directory.

    Behavior:
    - generate (create_new=True): create base/run_<timestamp>, update base/LATEST_RUN
    - submit/status/evaluate (create_new=False):
        - if base path already contains batch_input.jsonl, treat it as run dir
        - else use base/LATEST_RUN pointer
    """
    base = Path(base_output_dir)
    if create_new:
        base.mkdir(parents=True, exist_ok=True)
        run_dir = base / f"run_{_timestamp()}"
        run_dir.mkdir(parents=True, exist_ok=False)
        (base / "LATEST_RUN").write_text(str(run_dir))
        return run_dir

    # If caller passed a concrete run directory, use it directly.
    if (base / "batch_input.jsonl").exists() or (base / "metadata.csv").exists():
        return base

    latest_ptr = base / "LATEST_RUN"
    if not latest_ptr.exists():
        raise FileNotFoundError(
            f"No run directory found under {base}. "
            "Run 'generate' first or pass a concrete run directory."
        )
    run_dir = Path(latest_ptr.read_text().strip())
    if not run_dir.exists():
        raise FileNotFoundError(f"LATEST_RUN points to missing directory: {run_dir}")
    return run_dir


def _serialize_listing(record: Dict[str, Any], suffix: str, fields: List[str], max_len: int) -> str:
    payload: Dict[str, str] = {}
    for field in fields:
        key = f"{field}_{suffix}"
        value = record.get(key)
        if value is None:
            continue
        value_str = str(value)
        if value_str.strip() == "":
            continue
        if len(value_str) > max_len:
            value_str = value_str[:max_len] + "..."
        payload[field] = value_str
    return json.dumps(payload, ensure_ascii=False)


def _build_user_prompt(record: Dict[str, Any], fields: List[str], max_len: int) -> str:
    left = _serialize_listing(record, "left", fields, max_len)
    right = _serialize_listing(record, "right", fields, max_len)
    return f"Entity 1: {left}\nEntity 2: {right}"


def _custom_id(idx: int, agent_key: str) -> str:
    return f"pair-{idx}__agent-{agent_key}"


def _parse_custom_id(custom_id: str) -> Optional[tuple[int, str]]:
    # Expected: pair-123__agent-balanced
    try:
        left, right = custom_id.split("__", 1)
        idx = int(left.replace("pair-", ""))
        agent = right.replace("agent-", "")
        if agent not in AGENTS:
            return None
        return idx, agent
    except Exception:
        return None


def generate_batch_file(
    data_path: str = DEFAULT_DATA,
    model: str = DEFAULT_MODEL,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    fields: Optional[List[str]] = None,
    max_field_len: int = 350,
) -> Path:
    fields = fields or DEFAULT_FIELDS
    out_dir = _resolve_run_dir(output_dir, create_new=True)

    data_file = Path(data_path)
    logger.info("Loading data from %s", data_file)
    records = _read_jsonl_gz(data_file)
    logger.info("Loaded %d pairs", len(records))

    metadata = pd.DataFrame(
        {
            "idx": list(range(len(records))),
            "pair_id": [r.get("pair_id", "") for r in records],
            "gold": [int(r.get("label", 0)) for r in records],
        }
    )
    metadata_path = out_dir / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)
    logger.info("Saved metadata -> %s", metadata_path)

    batch_input_path = out_dir / "batch_input.jsonl"
    n_requests = 0
    with open(batch_input_path, "w") as f:
        for idx, record in enumerate(records):
            user_msg = _build_user_prompt(record, fields=fields, max_len=max_field_len)
            for agent_key, agent_cfg in AGENTS.items():
                req = {
                    "custom_id": _custom_id(idx, agent_key),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": agent_cfg["system_prompt"]},
                            {"role": "user", "content": user_msg},
                        ],
                    },
                }
                f.write(json.dumps(req, ensure_ascii=False) + "\n")
                n_requests += 1

    run_config = {
        "data_path": str(data_file),
        "model": model,
        "fields": fields,
        "max_field_len": max_field_len,
        "n_pairs": len(records),
        "agents": list(AGENTS.keys()),
        "n_requests": n_requests,
        "batch_input": str(batch_input_path),
    }
    run_config_path = out_dir / "run_config.json"
    with open(run_config_path, "w") as f:
        json.dump(run_config, f, indent=2)

    logger.info("Wrote %d requests (%d pairs x %d agents) -> %s", n_requests, len(records), len(AGENTS), batch_input_path)
    logger.info("Saved run config -> %s", run_config_path)
    logger.info("Run directory -> %s", out_dir)
    return batch_input_path


def submit_batch(output_dir: str = DEFAULT_OUTPUT_DIR) -> None:
    from openai import OpenAI

    out_dir = _resolve_run_dir(output_dir, create_new=False)
    batch_input_path = out_dir / "batch_input.jsonl"
    if not batch_input_path.exists():
        raise FileNotFoundError(f"Missing {batch_input_path}. Run 'generate' first.")

    client = OpenAI()
    logger.info("Uploading %s", batch_input_path)
    file_obj = client.files.create(file=open(batch_input_path, "rb"), purpose="batch")

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"experiment": "multi_agent_single_batch"},
    )

    batch_info = {
        "file_id": file_obj.id,
        "batch_id": batch.id,
        "status": batch.status,
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
    }
    info_path = out_dir / "batch_info.json"
    with open(info_path, "w") as f:
        json.dump(batch_info, f, indent=2)

    logger.info("Submitted batch_id=%s (status=%s)", batch.id, batch.status)
    logger.info("Saved batch info -> %s", info_path)


def check_status(output_dir: str = DEFAULT_OUTPUT_DIR, download: bool = True) -> None:
    from openai import OpenAI

    out_dir = _resolve_run_dir(output_dir, create_new=False)
    info_path = out_dir / "batch_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing {info_path}. Run 'submit' first.")

    with open(info_path) as f:
        info = json.load(f)

    client = OpenAI()
    batch = client.batches.retrieve(info["batch_id"])

    logger.info(
        "Batch %s status=%s completed=%s/%s failed=%s",
        batch.id,
        batch.status,
        batch.request_counts.completed,
        batch.request_counts.total,
        batch.request_counts.failed,
    )

    info["status"] = batch.status
    info["output_file_id"] = getattr(batch, "output_file_id", None)
    info["error_file_id"] = getattr(batch, "error_file_id", None)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    if not download:
        return

    if batch.output_file_id:
        output_path = out_dir / "batch_output.jsonl"
        if not output_path.exists():
            content = client.files.content(batch.output_file_id)
            with open(output_path, "wb") as f:
                f.write(content.read())
            logger.info("Downloaded output -> %s", output_path)
        else:
            logger.info("Output already exists -> %s", output_path)

    if batch.error_file_id:
        error_path = out_dir / "batch_error.jsonl"
        if not error_path.exists():
            content = client.files.content(batch.error_file_id)
            with open(error_path, "wb") as f:
                f.write(content.read())
            logger.info("Downloaded errors -> %s", error_path)
        else:
            logger.info("Error file already exists -> %s", error_path)


def _parse_match_from_content(content: str) -> Optional[bool]:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "match" in obj:
            return bool(obj["match"])
    except json.JSONDecodeError:
        pass

    lower = text.lower()
    if '"match": true' in lower or '"match":true' in lower:
        return True
    if '"match": false' in lower or '"match":false' in lower:
        return False
    return None


def parse_batch_output(output_file: Path) -> Dict[int, Dict[str, Optional[bool]]]:
    preds: Dict[int, Dict[str, Optional[bool]]] = {}
    parse_failures = 0

    with open(output_file, "r") as f:
        for line in f:
            row = json.loads(line)
            cid = row.get("custom_id", "")
            parsed = _parse_custom_id(cid)
            if parsed is None:
                continue
            idx, agent = parsed
            preds.setdefault(idx, {})

            if row.get("error"):
                preds[idx][agent] = None
                continue

            try:
                content = row["response"]["body"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                preds[idx][agent] = None
                parse_failures += 1
                continue

            pred = _parse_match_from_content(content)
            if pred is None:
                parse_failures += 1
            preds[idx][agent] = pred

    if parse_failures:
        logger.warning("Could not parse %d responses", parse_failures)
    return preds


def _compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, Any]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _majority_vote(row: pd.Series) -> int:
    votes = [int(row[f"pred_{k}"]) for k in AGENTS]
    s = sum(votes)
    if s >= 2:
        return 1
    if s <= 1:
        return 0
    # unreachable with 3 agents, kept for safety
    return int(row["pred_balanced"])


def evaluate(output_dir: str = DEFAULT_OUTPUT_DIR) -> None:
    out_dir = _resolve_run_dir(output_dir, create_new=False)
    metadata_path = out_dir / "metadata.csv"
    output_path = out_dir / "batch_output.jsonl"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}. Run 'generate' first.")
    if not output_path.exists():
        raise FileNotFoundError(f"Missing {output_path}. Run 'status' after completion.")

    meta = pd.read_csv(metadata_path)
    pred_map = parse_batch_output(output_path)

    for agent_key in AGENTS:
        meta[f"pred_{agent_key}"] = meta["idx"].map(lambda i: pred_map.get(int(i), {}).get(agent_key))
        # fall back to non-match on parse failures
        meta[f"pred_{agent_key}"] = meta[f"pred_{agent_key}"].fillna(False).astype(bool).astype(int)

    meta["pred_vote"] = meta.apply(_majority_vote, axis=1)

    logger.info("\nPer-agent metrics")
    metrics_out: Dict[str, Any] = {}
    for agent_key, cfg in AGENTS.items():
        m = _compute_metrics(meta["gold"], meta[f"pred_{agent_key}"])
        metrics_out[agent_key] = m
        logger.info(
            "  %-18s P=%.4f R=%.4f F1=%.4f Acc=%.4f (TP=%d FP=%d FN=%d TN=%d)",
            cfg["name"],
            m["precision"],
            m["recall"],
            m["f1"],
            m["accuracy"],
            m["tp"],
            m["fp"],
            m["fn"],
            m["tn"],
        )

    m_vote = _compute_metrics(meta["gold"], meta["pred_vote"])
    metrics_out["vote"] = m_vote
    logger.info("\nVoted metrics")
    logger.info(
        "  %-18s P=%.4f R=%.4f F1=%.4f Acc=%.4f (TP=%d FP=%d FN=%d TN=%d)",
        "Majority Vote",
        m_vote["precision"],
        m_vote["recall"],
        m_vote["f1"],
        m_vote["accuracy"],
        m_vote["tp"],
        m_vote["fp"],
        m_vote["fn"],
        m_vote["tn"],
    )

    preds_path = out_dir / "predictions.csv"
    keep_cols = ["idx", "pair_id", "gold"] + [f"pred_{k}" for k in AGENTS] + ["pred_vote"]
    meta[keep_cols].to_csv(preds_path, index=False)
    logger.info("Saved predictions -> %s", preds_path)

    summary = {
        "n_pairs": int(len(meta)),
        "agents": {k: AGENTS[k]["name"] for k in AGENTS},
        "metrics": {
            k: {mk: (round(mv, 6) if isinstance(mv, float) else mv) for mk, mv in v.items()}
            for k, v in metrics_out.items()
        },
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved metrics -> %s", metrics_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-batch multi-agent GPT entity matching workflow")
    subparsers = parser.add_subparsers(dest="command")

    gen = subparsers.add_parser("generate", help="Generate one batch_input.jsonl (all agents, all pairs)")
    gen.add_argument("--data", default=DEFAULT_DATA)
    gen.add_argument("--model", default=DEFAULT_MODEL)
    gen.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    gen.add_argument("--max-field-len", type=int, default=350)

    sub = subparsers.add_parser("submit", help="Submit single batch")
    sub.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)

    stat = subparsers.add_parser("status", help="Check status and download files")
    stat.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    stat.add_argument("--no-download", action="store_true")

    ev = subparsers.add_parser("evaluate", help="Evaluate per-agent and voted metrics")
    ev.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)

    args = parser.parse_args()

    if args.command == "generate":
        generate_batch_file(
            data_path=args.data,
            model=args.model,
            output_dir=args.output_dir,
            max_field_len=args.max_field_len,
        )
    elif args.command == "submit":
        submit_batch(output_dir=args.output_dir)
    elif args.command == "status":
        check_status(output_dir=args.output_dir, download=not args.no_download)
    elif args.command == "evaluate":
        evaluate(output_dir=args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
