# Runbook: new benchmarks + pool-construction sensitivity

Prepared 2026-09-01 for the paper revision. Everything here runs on the server from the repo root
inside the `ditto-modern` conda env with `OPENAI_API_KEY` set. No step below has been run against the
API yet — the data, configs and code are prepared, the embeddings and labeling are yours to launch.

## What was added

| Path | Purpose |
|---|---|
| `benchmarks/dn7-walmart-amazon/` | Walmart-Amazon re-blocked by Papadakis et al. (ICDE 2024, Zenodo 10.5281/zenodo.8164151, CC BY 4.0). 2,554 × 10,717 records, 26,050 train pairs, **1.8 % positive**. |
| `benchmarks/semi-heter/` | Machamp Semi-HETER (Wang, Li, Hirota, CIKM 2021). Book domain, heterogeneous schemas canonicalized to title/authors/publisher/year/isbn/pages/price (+`extra`). 994 × 1,049 records, 1,240 train pairs, 38.2 % positive. |
| `benchmarks/billiger-de/`, `billiger-en/` | billiger.de Products, mirroring the WDC setup (80 % corner cases, large train, 100 %-unseen gold standard). 3,494 × 3,495 records, 26,571 train pairs. |
| `scripts/benchmarks/prepare_{dn,machamp,billiger}.py` | Converters that produced the directories above (re-runnable). |
| `scripts/labeling/pool_builders.py` | Candidate-pool variants: `embedding` (paper), `bm25`, `rrf` (size-controlled fusion), `union` (uncapped). |
| `scripts/labeling/build_embeddings.py` | text-embedding-3-small (256-d) embeddings + batch request files, same layout as the released benchmarks. |
| `scripts/labeling/pool_recall.py` | Pool size / positive-pair recall / hard-pair density per pool method. No API calls. |
| `--pool-method` on all three labeling runners | `embedding` is the default and is bit-identical to the previous behaviour. |
| `configs/labeling/pool_sensitivity.yaml` | wdc + dn7 under embedding / bm25 / rrf as separate keys. |
| `cluster/slurm/run_new_benchmark.sbatch`, `submit_new_benchmarks.sh` | Embeddings → pool recall → similarity-search labeling, one job per benchmark. |
| `cluster/slurm/run_al_ditto_new_benchmark.sbatch`, `submit_pool_sensitivity.sh` | Active learning (Ditto) per benchmark / pool method. |

## Order of operations

```bash
# 0. sanity, no API
python scripts/labeling/similarity_search.py --benchmarks dn7-walmart-amazon,semi-heter,billiger-de --profiles large --dry-run
python scripts/training/train_ditto.py --benchmarks dn7-walmart-amazon,semi-heter,billiger-de --dry-run

# 1. embeddings (≈ 21k records total, a few minutes, cents)
for b in dn7-walmart-amazon semi-heter billiger-de; do python scripts/labeling/build_embeddings.py --benchmark $b; done

# 2. the zero-cost check that decides how to frame the pool experiment
python scripts/labeling/pool_recall.py --benchmark dn7-walmart-amazon     # all four methods by default
python scripts/labeling/pool_recall.py --benchmark wdc                    # large train split via train_pairs; reproduces the paper's 84.93 %

# 3. labeling — similarity search first (fast), then AL (Ditto) on the GPU
bash cluster/slurm/submit_new_benchmarks.sh                                    # or per benchmark, see the sbatch header
BENCHMARK=dn7-walmart-amazon PROFILE=large sbatch cluster/slurm/run_al_ditto_new_benchmark.sbatch

# 4. pool sensitivity (wdc + dn7 × bm25/rrf; embedding = the existing runs)
bash cluster/slurm/submit_pool_sensitivity.sh

# 5. students: benchmark-label baseline + machine-labeled sets, as for the existing benchmarks
python scripts/training/train_ditto.py --benchmarks dn7-walmart-amazon,semi-heter,billiger-de
```

## Validated locally (2026-09-01, no API)

- `embedding` pool is bit-identical to the original FAISS builder (Abt-Buy, 20,500 pairs).
- `pool_recall.py` reproduces the archived figures: Abt-Buy 99.68 %, DBLP-ACM 99.92 %.
- `similarity_search.py --dry-run` and `train_ditto.py --dry-run` pass for all four new benchmarks
  (split counts and fields verified end to end).
- Local caveat: the `env/` venv on the Mac has a broken scipy wheel (`_spropack` dlopen error) that
  breaks *every* runner via scikit-learn, new and old alike. `pip install --force-reinstall scipy`
  inside `env/` should fix it; the cluster `ditto-modern` env is unaffected.

## Decisions baked into the configs (change if you disagree)

- **Dn7 profiles.** `large` = 26,050 pairs / 457 positives, i.e. size-matched to the benchmark at its own
  1.8 % balance. The pool cannot contain more true matches than exist (~460), so a 20 %-positive target
  is not reachable here. That is ~26k GPT-5.2 calls (≈ US$15–20 at the paper's observed rate); `medium`
  (8,000 / 300) is the cheaper first pass.
- **Semi-HETER profiles.** `large` = 1,240 / 474 is budget-matched to the benchmark; `xl` = 5,000 / 1,900
  is the full-pool point. Report both. Only `title` is shared across the two sides, so run this one
  with Ditto and the LLM students; the feature-based Active learning (ML) and XGBoost need aligned
  attributes. `isbn` is the constant 9780000000000 in the source data (precision lost upstream).
- **billiger.de profiles** copy WDC's (2,500 / 10,000 / 20,000) for a like-for-like comparison.
- **Pool sensitivity design.** Four pools: `embedding` (paper: 18 NN + 2 random per left record), `bm25`
  over the concatenated fields (same size), `rrf` — the two rankings fused by reciprocal rank fusion and
  cut to the same size (size-controlled), and `union` — embedding top-18 ∪ BM25 top-18 + 2 random with no
  cut (recall-maximizing, pool grows to ≤38 per left record; size is reported). Same teacher, same
  budgets, same students; report pool size, positive recall, hard-pair density, downstream F1.
  **Why both rrf and union:** at fixed size, RRF re-ranks and can drop embedding candidates for lexical
  ones — on WDC that *lowers* positive recall (81.99 % vs 84.93 %), while on DBLP-ACM it lifts it to 100 %.
  The uncapped union cannot lose recall: on WDC it reaches **89.88 %** (+5 points) for an 18 % larger pool
  (106,896 vs 90,557 pairs), and 100 % on Abt-Buy and DBLP-ACM. Report both and let the numbers argue.
  WDC is the existing benchmark to use because its pool recall is 84.93 %, not ~99.9 % like the other
  four — there is headroom for the union to show something. `similarity` in the pool CSV is the
  method's ranking score and is only comparable within a query.

## Paper text to fix while at it

`sections/03_system.tex` §2.1 currently says the pools for all five benchmarks contain 99.56–99.92 % of
the training positives. The archived `scripts/archive/post_processing/analysis/compute_pool_recall.py`
records **WDC at 84.93 %** (the other four are 99.56–99.92). This is the figure that was lost to the
stale-editor overwrite noted in June; restore it.
