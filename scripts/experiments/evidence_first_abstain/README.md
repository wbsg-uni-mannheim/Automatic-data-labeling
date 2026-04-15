# Evidence-First Abstain Runner

This experiment runner labels benchmark pairs in two stages:

1. Stage 1 extracts structured matching evidence.
2. Stage 2 converts that evidence into `match`, `non_match`, or `abstain`.

Outputs are written under:

`output/experiments/evidence_first_abstain/<benchmark>/<split>/<model-slug>/`

Artifacts:

- `run_manifest.json`
- `results.jsonl`
- `summary.json`
- `predictions.csv`

`summary.json` reports:

- `accuracy`, `precision`, `recall`, `f1` on non-abstained decisions only
- `coverage` and `abstain_rate`
- `all_pair_accuracy`, which treats abstentions as unresolved rather than silently correct

## Example: `abt-buy` with GPT-5 mini

```bash
python scripts/experiments/evidence_first_abstain/run_benchmark.py \
  --benchmark abt-buy \
  --split test \
  --model gpt-5-mini \
  --workers 8 \
  --resume
```

## Dry run

```bash
python scripts/experiments/evidence_first_abstain/run_benchmark.py \
  --benchmark abt-buy \
  --split test \
  --model gpt-5-mini \
  --dry-run
```

## Notes

- The default model string is `gpt-5-mini`.
- If you have a different exact model alias available, pass it with `--model`.
- `abstain` is tracked separately in `summary.json` and excluded from precision/recall/F1.

## Single-Pass First Pass

This is the cheaper first-pass runner for the workflow you described:

- it uses the original tribunal-style prompt format
- it defaults to `agent_precision`
- it asks the model to directly choose `match`, `non_match`, or `abstain`
- it does not use a hand-tuned confidence threshold to force abstentions

Outputs are written under:

`output/experiments/single_pass_abstain/<benchmark>/<split>/<prompt-template>/model_decides_abstain/<model-slug>/`

Example:

```bash
python scripts/experiments/evidence_first_abstain/run_single_pass_abstain.py \
  --benchmark abt-buy \
  --split test \
  --model gpt-5.4-mini \
  --prompt-template agent_precision \
  --workers 8 \
  --resume
```

This first-pass output is directly compatible with the second-pass reviewer.

## Positive-Only Skeptical Review

This post-processor applies two checks to first-pass positive matches:

- a skeptical second-pass LLM review

The final rule is simple:

- keep a first-pass `match` only if the skeptical reviewer also says `match`
- otherwise convert that positive to `abstain`
- keep first-pass `non_match` and `abstain` unchanged

Outputs are written under:

`<input-dir>/positive_only_skeptical_review/<prompt-template>/<review-model-slug>/`

Example:

```bash
python scripts/experiments/evidence_first_abstain/run_positive_only_skeptical_review.py \
  --input-dir output/experiments/single_pass_abstain/abt-buy/test/agent_precision/model_decides_abstain/gpt-5-4-mini \
  --review-model gpt-5.4-mini \
  --prompt-template agent_variant_skeptic \
  --workers 8 \
  --resume
```

## Dense Binary Dual Review

This is the dense end-to-end version:

- first pass: binary `match|non_match` with `agent_precision`
- positive review: binary skeptical review on first-pass `match`
- optional negative review: binary high-recall review on first-pass `non_match`
- final output is always dense, never `abstain`

Recommended default:

```bash
python scripts/experiments/evidence_first_abstain/run_dense_binary_dual_review.py \
  --benchmark abt-buy \
  --split test \
  --model gpt-5.4-mini \
  --first-pass-prompt agent_precision \
  --positive-review-prompt agent_variant_skeptic \
  --workers 8 \
  --resume
```

If you also want to try false-negative recovery:

```bash
python scripts/experiments/evidence_first_abstain/run_dense_binary_dual_review.py \
  --benchmark abt-buy \
  --split test \
  --model gpt-5.4-mini \
  --first-pass-prompt agent_precision \
  --positive-review-prompt agent_variant_skeptic \
  --review-non-matches \
  --negative-review-prompt agent_recall \
  --workers 8 \
  --resume
```

## Second Pass Reviewer

This reviewer supports multiple prompt templates and defaults to the best global single-prompt choice from the earlier sweep:

- `agent_precision` (default)
- `agent_balanced`
- `agent_variant_skeptic`
- `exact_batch`

It runs a second model pass and combines decisions like this by default:

- if first pass and second pass agree: keep the label
- if they disagree: set final decision to `abstain`
- existing first-pass abstains stay abstained

Example:

```bash
python scripts/experiments/evidence_first_abstain/run_second_pass_exact_prompt.py \
  --input-dir output/experiments/evidence_first_abstain/abt-buy/test/gpt-5-4-mini \
  --review-model gpt-5.4 \
  --prompt-template agent_precision \
  --selection scored \
  --workers 8 \
  --resume
```

Outputs are written under:

`<input-dir>/second_pass_review/<prompt-template>/<review-model-slug>/`
