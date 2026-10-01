"""Build text-embedding-3-small (256-d) record embeddings for a benchmark.

Reads the benchmark block from the labeling config, concatenates the configured
fields in config order (the same text the existing pools were built from), writes
the OpenAI batch request files + row maps + manifest in the layout used by the
released benchmarks, and produces ``embeddings/<name>_{left,right}_embeddings.npy``.

  python scripts/labeling/build_embeddings.py --benchmark dn7-walmart-amazon
  python scripts/labeling/build_embeddings.py --benchmark dn7-walmart-amazon --dry-run   # request files only

Default mode embeds synchronously through /v1/embeddings with a thread pool
(fast for <100k records). ``--submit-batch`` submits the jsonl files as OpenAI
batch jobs instead and records the batch ids in the manifest; pull them later
with ``--pull-batch``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "labeling"))
from pool_builders import record_text  # noqa: E402

MODEL = "text-embedding-3-small"
DIM = 256


def _load_cfg(config: Path, benchmark: str) -> Dict[str, Any]:
    cfg = yaml.safe_load(config.read_text()) or {}
    b = (cfg.get("benchmarks") or {}).get(benchmark)
    if not b:
        raise KeyError(f"benchmark {benchmark!r} not in {config}")
    return b


def _fields(bcfg: Dict[str, Any], side: str) -> List[str]:
    fm = bcfg.get(f"{side}_fields") or bcfg.get("fields") or {}
    return [str(v) for v in fm.values()]  # source column names, config order


def _write_requests(df: pd.DataFrame, fields: List[str], side: str, name: str, out_dir: Path) -> tuple[Path, Path, List[str]]:
    jsonl = out_dir / f"{name}-batch_embed_{side}.jsonl"
    mapcsv = out_dir / f"{name}-batch_embed_{side}_map.csv"
    texts: List[str] = []
    with jsonl.open("w", encoding="utf-8") as f, mapcsv.open("w", encoding="utf-8") as m:
        m.write("custom_id,id\n")
        for i, row in enumerate(df.to_dict("records")):
            t = record_text(row, fields) or str(row.get("id", ""))
            texts.append(t)
            req = {"custom_id": f"{side}-{i}", "method": "POST", "url": "/v1/embeddings",
                   "body": {"model": MODEL, "dimensions": DIM, "input": t}}
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
            m.write(f"{side}-{i},{row['id']}\n")
    return jsonl, mapcsv, texts


def _embed_sync(texts: List[str], concurrency: int, chunk: int = 64) -> np.ndarray:
    from openai import OpenAI

    client = OpenAI()
    out = np.zeros((len(texts), DIM), dtype=np.float32)

    def work(start: int) -> tuple[int, List[List[float]]]:
        batch = [t[:8000] for t in texts[start:start + chunk]]
        for attempt in range(6):
            try:
                r = client.embeddings.create(model=MODEL, input=batch, dimensions=DIM)
                return start, [d.embedding for d in r.data]
            except Exception as e:  # rate limits / transient
                time.sleep(2 ** attempt)
                last = e
        raise RuntimeError(f"embedding chunk at {start} failed: {last}")

    starts = list(range(0, len(texts), chunk))
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(work, s) for s in starts]
        done = 0
        for fut in as_completed(futs):
            s, vecs = fut.result()
            out[s:s + len(vecs)] = np.asarray(vecs, dtype=np.float32)
            done += 1
            if done % 20 == 0 or done == len(starts):
                print(f"  embedded {min(done * chunk, len(texts))}/{len(texts)}", file=sys.stderr)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/labeling/benchmarks_active.yaml")
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="write request files + manifest only")
    ap.add_argument("--submit-batch", action="store_true", help="submit OpenAI batch jobs instead of sync calls")
    ap.add_argument("--pull-batch", action="store_true", help="download finished batch jobs recorded in the manifest")
    a = ap.parse_args()
    os.chdir(ROOT)

    bcfg = _load_cfg(Path(a.config), a.benchmark)
    left_csv, right_csv = Path(bcfg["left_csv"]), Path(bcfg["right_csv"])
    emb = bcfg["embeddings"]
    emb_dir = Path(emb["dir"])
    out_dir = left_csv.parent
    emb_dir.mkdir(parents=True, exist_ok=True)
    name = a.benchmark
    manifest_path = out_dir / f"{name}-batch_embed_manifest.json"
    manifest: Dict[str, Any] = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    if a.pull_batch:
        from openai import OpenAI
        client = OpenAI()
        for side, npy in (("left", emb["left_file"]), ("right", emb["right_file"])):
            b = manifest["submitted_batches"][side]
            job = client.batches.retrieve(b["batch_id"])
            if job.status != "completed":
                print(f"{side}: batch {b['batch_id']} status={job.status}", file=sys.stderr)
                continue
            content = client.files.content(job.output_file_id).text
            n = sum(1 for _ in open(b["jsonl_path"], encoding="utf-8"))
            arr = np.zeros((n, DIM), dtype=np.float32)
            for line in content.splitlines():
                o = json.loads(line)
                i = int(o["custom_id"].split("-")[1])
                arr[i] = np.asarray(o["response"]["body"]["data"][0]["embedding"], dtype=np.float32)
            np.save(emb_dir / npy, arr)
            manifest.setdefault("pulled_embeddings", {})[side] = {
                "batch_id": b["batch_id"], "saved_file": str(emb_dir / npy), "rows": n, "dim": DIM,
                "pulled_at_utc": datetime.now(timezone.utc).isoformat()}
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return

    ldf, rdf = pd.read_csv(left_csv), pd.read_csv(right_csv)
    lf, rf = _fields(bcfg, "left"), _fields(bcfg, "right")
    l_jsonl, l_map, l_texts = _write_requests(ldf, lf, "left", name, out_dir)
    r_jsonl, r_map, r_texts = _write_requests(rdf, rf, "right", name, out_dir)
    manifest.update({
        "dataset": name, "model": MODEL, "dimensions": DIM,
        "left_csv": str(left_csv), "right_csv": str(right_csv),
        "left_unique_rows": int(len(ldf)), "right_unique_rows": int(len(rdf)),
        "embed_fields": {"left": lf, "right": rf},
        "left_batch_jsonl": str(l_jsonl), "right_batch_jsonl": str(r_jsonl),
        "left_map_csv": str(l_map), "right_map_csv": str(r_map),
    })
    print(f"wrote {l_jsonl} ({len(ldf)} rows), {r_jsonl} ({len(rdf)} rows)", file=sys.stderr)

    if a.dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print("dry run: request files written, no API calls", file=sys.stderr)
        return

    if a.submit_batch:
        from openai import OpenAI
        client = OpenAI()
        manifest["submitted_batches"] = {}
        for side, jp in (("left", l_jsonl), ("right", r_jsonl)):
            up = client.files.create(file=open(jp, "rb"), purpose="batch")
            job = client.batches.create(input_file_id=up.id, endpoint="/v1/embeddings", completion_window="24h")
            manifest["submitted_batches"][side] = {"jsonl_path": str(jp), "input_file_id": up.id,
                                                   "batch_id": job.id, "status": job.status}
            print(f"{side}: submitted batch {job.id}", file=sys.stderr)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return

    for side, texts, npy in (("left", l_texts, emb["left_file"]), ("right", r_texts, emb["right_file"])):
        print(f"embedding {side} ({len(texts)} rows) ...", file=sys.stderr)
        arr = _embed_sync(texts, a.concurrency)
        np.save(emb_dir / npy, arr)
        manifest.setdefault("pulled_embeddings", {})[side] = {
            "mode": "sync", "saved_file": str(emb_dir / npy), "rows": int(len(texts)), "dim": DIM,
            "pulled_at_utc": datetime.now(timezone.utc).isoformat()}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"saved embeddings to {emb_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
