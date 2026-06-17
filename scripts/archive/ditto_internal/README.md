# Ditto Modern Training Pipeline

This directory contains a modernized Ditto-style entity matching stack that trains on WDC-style `json.gz` files and supports pseudo-labeling from multi-agent outputs.

## Conda environment

```bash
conda env create -f configs/ditto/environment.yml
conda activate ditto-modern
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords')"
```

## Data Contract
Input files must use the same WDC schema as:
- `benchmarks/wdc/wdcproducts80cc20rnd000un_train_*.json.gz`
- `benchmarks/wdc/wdcproducts80cc20rnd000un_valid_*.json.gz`

## 1) Build pseudo-labeled datasets (train and val separately)

```bash
python scripts/archive/ditto_internal/build_wdc_pseudolabels.py \
  --source-json-gz benchmarks/wdc/wdcproducts80cc20rnd000un_train_medium.json.gz \
  --multi-agent-run-dir output/multi_agent_single_batch/run_YYYYMMDD_HHMMSS \
  --output-json-gz output/pseudolabels/train_pseudo.json.gz \
  --policy consensus
```

```bash
python scripts/archive/ditto_internal/build_wdc_pseudolabels.py \
  --source-json-gz benchmarks/wdc/wdcproducts80cc20rnd000un_valid_medium.json.gz \
  --multi-agent-run-dir output/multi_agent_single_batch/run_YYYYMMDD_HHMMSS \
  --output-json-gz output/pseudolabels/val_pseudo.json.gz \
  --policy consensus
```

Optional inclusion of 2/3 majority samples with lower weight:

```bash
python scripts/archive/ditto_internal/build_wdc_pseudolabels.py \
  --source-json-gz ... \
  --multi-agent-run-dir ... \
  --output-json-gz ... \
  --policy consensus \
  --include-majority-with-weight \
  --majority-weight 0.5
```

## 2) Convert WDC json.gz to Ditto text format

```bash
python scripts/archive/ditto_internal/convert_wdc_to_ditto.py \
  --input-json-gz output/pseudolabels/train_pseudo.json.gz \
  --output-txt output/pseudolabels/train_pseudo.txt
```

## 3) Train (single GPU)

```bash
python scripts/archive/ditto_internal/train.py \
  --train-json-gz output/pseudolabels/train_pseudo.json.gz \
  --val-json-gz output/pseudolabels/val_pseudo.json.gz \
  --test-json-gz benchmarks/wdc/wdcproducts80cc20rnd100un_gs.json.gz \
  --output-dir output/ditto_runs
```

Optional original-Ditto features (ported):

```bash
python scripts/archive/ditto_internal/train.py \
  --train-json-gz ... \
  --val-json-gz ... \
  --test-json-gz ... \
  --output-dir output/ditto_runs \
  --summarize \
  --dk product \
  --da all \
  --alpha-aug 0.8
```

Convenience script for existing WDC splits (`small|medium|large`):

```bash
SPLIT_SIZE=medium bash scripts/archive/ditto_internal/run_existing_wdc.sh
```

## 4) Train (DDP / multi-GPU)

```bash
torchrun --standalone --nproc_per_node 2 scripts/archive/ditto_internal/train.py \
  --ddp \
  --train-json-gz output/pseudolabels/train_pseudo.json.gz \
  --val-json-gz output/pseudolabels/val_pseudo.json.gz \
  --test-json-gz benchmarks/wdc/wdcproducts80cc20rnd100un_gs.json.gz \
  --output-dir output/ditto_runs
```

## DWS SLURM quick submit

```bash
sbatch cluster/slurm/ditto_test_existing_wdc.sbatch
```

Optional before submit:

```bash
export SPLIT_SIZE=small   # or medium/large
export CONDA_ENV_NAME=ditto-modern
sbatch cluster/slurm/ditto_test_existing_wdc.sbatch
```

## 5) Evaluate saved checkpoint

```bash
python scripts/archive/ditto_internal/evaluate.py \
  --checkpoint output/ditto_runs/run_YYYYMMDD_HHMMSS/checkpoints/best \
  --test-json-gz benchmarks/wdc/wdcproducts80cc20rnd100un_gs.json.gz \
  --output-dir output/ditto_runs
```

## Output structure

Training run outputs:
- `output/ditto_runs/run_<timestamp>/config.json`
- `output/ditto_runs/run_<timestamp>/checkpoints/best/*`
- `output/ditto_runs/run_<timestamp>/metrics.json`
- `output/ditto_runs/run_<timestamp>/predictions.csv`
- `output/ditto_runs/LATEST_RUN`

## Notes vs original `ditto-master`

Implemented parity-critical behavior for EM training:
- Ditto-style pair serialization (`COL ... VAL ...`)
- Model aliases (`roberta`, `distilbert`) and direct HF model names
- Validation threshold tuning for best F1, then applying that threshold on test/eval
- Optional weighted pseudo-label training data
- Ditto text summarization pipeline
- Domain-knowledge injectors (`product` / `general`)
- MixDA augmentation path (`--da` + `--alpha-aug`)
