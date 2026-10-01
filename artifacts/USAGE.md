# Usage

How to build training sets and train and evaluate the students. Run every command from the repository root.

## Environments

Ditto, XGBoost, and the labeling workflows run in one conda environment (Python 3.10, CUDA PyTorch 2.4.1):

```bash
conda env create -f configs/ditto/environment.yml
conda activate ditto-modern
python -m spacy download en_core_web_sm
```

`configs/ditto/environment.lock.yml` lists every package version of the environment used for the paper. Without conda, `requirements.txt` installs the same main packages.

The Qwen3 students need a separate Python 3.12 environment with Unsloth. `scripts/archive/qwen_internal/requirements.lock.txt` lists its package versions, and `scripts/archive/qwen_internal/setup_env.sh` installs CUDA PyTorch first and the rest afterwards.

The teachers are called through the OpenAI API (GPT-5.2, and GPT-5-mini for the review prompt) and OpenRouter (Qwen 3.6 Plus, Kimi K2.6). Put the keys into a `.env` file in the repository root:

```bash
OPENAI_API_KEY=...
OPEN_ROUTER_API_KEY=...
```

## Build a training set

The released sets in `artifacts/training_data/` are the exact inputs of the paper. Rerunning a workflow calls the teacher again and produces a new set.

`configs/labeling/benchmarks_active.yaml` defines the source tables, attributes, embeddings, and label targets of each benchmark (`benchmarks_active_kimi.yaml`: Kimi K2.6 as teacher). The embeddings are precomputed in `benchmarks/*/embeddings/`, and `scripts/labeling/build_embeddings.py` rebuilds them.

**Similarity search** labels the nearest candidates of each record:

```bash
python scripts/labeling/similarity_search.py \
  --config configs/labeling/benchmarks_active.yaml \
  --benchmarks abt-buy \
  --profiles all \
  --model gpt-5.2 \
  --dry-run
```

Drop `--dry-run` to call the teacher. The run folder under `output/` contains the labels and a Ditto training file per profile.

**AL-ML** trains a feature-based committee on a teacher-labeled seed set and asks the teacher for the pairs the committee disagrees on most:

```bash
python scripts/labeling/active_learning_ml.py \
  --left-csv benchmarks/abt-buy/abt-buy-train-left.csv \
  --right-csv benchmarks/abt-buy/abt-buy-train-right.csv \
  --embeddings-dir benchmarks/abt-buy/embeddings \
  --left-emb abt-buy_left_embeddings.npy \
  --right-emb abt-buy_right_embeddings.npy \
  --left-schema-map '{"id":"id","title":"title","description":"description","price":"price"}' \
  --right-schema-map '{"id":"id","title":"title","description":"description","price":"price"}' \
  --model gpt-5.2 \
  --target-size 6000 \
  --seed-size 100 --seed-positives 30 --seed-max-calls 400 \
  --labels-per-iteration 500 --active-candidates 20000 \
  --llm-concurrency 10 \
  --output-root output/active_learning_ml \
  --run-name abt-buy_alml \
  --preview-k 5
```

`--preview-k 5` prints the first candidate pairs without labeling. Drop it for a full run, and add `--resume` with the same `--run-name` to continue an interrupted run.

**AL-Ditto** uses the same seed and feature-based phase and then a bagged committee of Ditto models:

```bash
python scripts/labeling/active_learning_ditto.py \
  --left-csv benchmarks/abt-buy/abt-buy-train-left.csv \
  --right-csv benchmarks/abt-buy/abt-buy-train-right.csv \
  --embeddings-dir benchmarks/abt-buy/embeddings \
  --left-emb abt-buy_left_embeddings.npy \
  --right-emb abt-buy_right_embeddings.npy \
  --left-schema-map '{"id":"id","title":"title","description":"description","price":"price"}' \
  --right-schema-map '{"id":"id","title":"title","description":"description","price":"price"}' \
  --model gpt-5.2 \
  --target-size 6000 \
  --seed-size 100 --seed-positives 30 --seed-max-calls 400 \
  --phase2-target-size 1000 \
  --phase3-batch-size 500 --phase3-candidates 20000 --phase3-ensemble-mode ditto_only \
  --llm-concurrency 10 \
  --output-root output/active_learning_ditto \
  --run-name abt-buy_alditto
```

For Qwen 3.6 Plus or Kimi K2.6 as teacher, pass `--api-base-url https://openrouter.ai/api/v1 --api-key-env-var OPEN_ROUTER_API_KEY` and the OpenRouter model name, for example `--model moonshotai/kimi-k2.6`.

## Post-process a training set

`scripts/post_processing/relabel_three_phase_generated_labels_batch.py` relabels every pair of a training set with the review prompt (`artifacts/prompts/review_system_prompt.txt`) through the Batch API. `build_cleaning_variants.py` then builds the relabel and relabel-drop variants from the original and the reviewed set:

```bash
python scripts/post_processing/build_cleaning_variants.py \
  --original-profile /path/to/original/profiles/all \
  --reviewed-profile /path/to/reviewed/profiles/all \
  --output-root output/cleaning/abt-buy
```

The closure variants of Table 5 use `consistent_closure.filter_rows`, which drops positive bridge edges and then negatives inside the remaining positive components, and `closure_combinations.combine`, which intersects or unites these drops with the review disagreements. `scripts/post_processing/README.md` describes the scripts.

## Train and evaluate a student

**Ditto.** `train_ditto.py` reads a YAML config that names the training, validation, and test files per benchmark. To train on a released set, point `train` to it:

```yaml
defaults:
  model_name: roberta-base
  batch_size: 64
  max_len: 256
  epochs: 50
  lr: 5.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  early_stopping_patience: 10
  seed: 42
  fp16: true
  da: del
  alpha_aug: 0.8
  max_field_len: 350
  exclude_valid_from_train: true
benchmarks:
  abt-buy:
    train: artifacts/training_data/abt-buy/gpt-5.2/active_learning_ditto/abt-buy_train.json.gz
    valid: benchmarks/abt-buy/abt-buy-valid.csv
    valid_pair_id_only: true
    valid_lookup_train: benchmarks/abt-buy/abt-buy-train.json
    test: benchmarks/abt-buy/abt-buy-gs.json.gz
    fields: [title, description, price]
    field_aliases: {title: [name]}
```

```bash
python scripts/training/train_ditto.py --config my_config.yaml --run-name abt-buy_alditto_seed42
```

The run folder holds `benchmark_report.json` with the test F1 and the resolved splits in `splits/`. `configs/ditto/benchmarks_training.yaml` trains the benchmark references and lists the validation and test files and attributes of all eight benchmarks. The other configs in `configs/ditto/` are those of the paper runs. Their training paths name the original run folders, and the released file of every run is listed in the `training_file` column of `results/*.csv`. The paper reports seeds 42, 52, and 62.

**XGBoost** trains on the released sets of Table 6 and evaluates on the test sets:

```bash
python scripts/training/train_xgboost.py --seeds 42,52,62
python scripts/training/train_xgboost.py --combos abt-buy:alditto,abt-buy:benchmark
```

**Qwen3** fine-tunes with LoRA in the Qwen environment. Convert a training set and the benchmark splits into chat format, train, and evaluate:

```bash
python scripts/archive/qwen_internal/convert_wdc_to_sft.py \
  --train-json-gz artifacts/training_data/abt-buy/gpt-5.2/active_learning_ditto/abt-buy_train.json.gz \
  --valid-json-gz <run folder>/splits/valid.json.gz \
  --test-json-gz benchmarks/abt-buy/abt-buy-gs.json.gz \
  --fields title,description,price --max-field-len 350 \
  --output-dir output/qwen/abt-buy_alditto/sft
python scripts/training/train_qwen.py \
  --data-dir output/qwen/abt-buy_alditto/sft --output-dir output/qwen/abt-buy_alditto/model \
  --model-name Qwen/Qwen3-8B --num-train-epochs 3 --eval-steps 50 --save-steps 50 --save-total-limit 3 \
  --early-stopping-patience 10 --load-best-model-at-end --seed 42
python scripts/archive/qwen_internal/evaluate_lora.py \
  --data-path output/qwen/abt-buy_alditto/sft/test.jsonl \
  --model-path output/qwen/abt-buy_alditto/model/final_adapter \
  --base-model-name Qwen/Qwen3-8B --output-dir output/qwen/abt-buy_alditto/eval
```

The validation file is the one a Ditto run writes to `splits/valid.json.gz`. The paper trains Qwen3-8B for up to 3 epochs and Qwen3-0.6B and 1.7B for up to 10, with early stopping on the validation loss. `scripts/archive/qwen_internal/baseline_zero_shot_eval.py` evaluates a Qwen3 model zero-shot on the converted test file.
