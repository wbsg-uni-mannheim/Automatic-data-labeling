#!/bin/bash
set -euo pipefail
cd /work/aasteine/Automatic-data-labeling
set -a; source .env; set +a
source scripts/archive/qwen_internal/.venv/bin/activate
export HF_HOME=/work/shared_data/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
python - <<'PY'
import os, time
from huggingface_hub import snapshot_download
t0 = time.time()
path = snapshot_download(
    repo_id="Qwen/Qwen3-8B",
    cache_dir=os.environ["HF_HOME"],
    token=os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN"),
    allow_patterns=["*.json","*.txt","*.safetensors","*.model","tokenizer*"],
)
print(f"DONE path={path} elapsed={time.time()-t0:.1f}s")
PY
