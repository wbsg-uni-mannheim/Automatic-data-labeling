#!/usr/bin/env bash
set -euo pipefail

# Submits a zero-shot baseline eval on the ditto_baseline test split of each
# benchmark, one sbatch job per benchmark. Fields per benchmark match
# configs/ditto/benchmarks_training.yaml.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SBATCH_FILE="${REPO_ROOT}/cluster/slurm/qwen35_baseline_eval.sbatch"

declare -A BENCH_FIELDS=(
  [abt-buy]="title,description,price"
  [amazon-google]="title,manufacturer,price"
  [dblp-acm]="title,authors,venue,year"
  [dblp-scholar]="title,authors,venue,year"
  [walmart-amazon]="title,category,brand,modelno,price"
  [wdc]="title,brand,description,price,priceCurrency"
)

# Allow selecting a subset via BENCHMARKS env var, e.g.
# BENCHMARKS="wdc dblp-scholar" bash submit_qwen35_baseline_all.sh
BENCHMARKS="${BENCHMARKS:-abt-buy amazon-google dblp-acm dblp-scholar walmart-amazon wdc}"

for bench in ${BENCHMARKS}; do
  fields="${BENCH_FIELDS[$bench]:-}"
  if [[ -z "${fields}" ]]; then
    echo "skip: unknown benchmark '${bench}'" >&2
    continue
  fi
  echo "submit: ${bench} fields=${fields}"
  sbatch \
    --job-name="qwen35-baseline-${bench}" \
    --export="ALL,BENCHMARK=${bench},FIELDS=${fields}" \
    "${SBATCH_FILE}"
done
