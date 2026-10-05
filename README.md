# Labeling Training Data for Entity Matching Using Large Language Models

This repository is the code and data release for the paper:

> **Labeling Training Data for Entity Matching Using Large Language Models**  
> Aaron Steiner and Christian Bizer, Data and Web Science Group, University of Mannheim.  
> Preprint: [arXiv:2606.28823](https://arxiv.org/abs/2606.28823).

It contains the benchmarks, the machine-labeled training sets, the prompts, the code to build training sets and to train and evaluate the students, and the per-run results behind every table and figure of the paper.

## Abstract

Large language models (LLMs) achieve strong performance on entity matching tasks without requiring task-specific training data, but applying them to large sets of candidate pairs is slow and costly. Matchers built on pretrained language models (PLMs), such as BERT, offer substantially faster inference, but require task-specific training data. This paper is a systematic experimental study of knowledge-distillation workflows for entity matching in which an LLM teacher labels training pairs that train a smaller student matcher. We vary the pair selection strategy, labeling budget, teacher model, correspondence post-processing method, and student model. We evaluate these workflows on eight entity matching benchmarks, including ones with hard non-matches, unseen entities, German-language offers from a broad product range, and multi-source records, and compare students trained on machine-labeled data with models trained on the original benchmark training sets. Our experiments show that in most cases training PLM-based matchers on the LLM-labeled training data performs similarly to training them on the original benchmark training sets. The pair selection strategy matters most for small labeling budgets, where active learning is often most effective. Open-weight teachers also train competitive students, so distillation needs no closed-weight models. The compact PLM-based student competes with much larger LLM students on most tasks, while requiring 34 to 459 times less inference time than direct LLM matching in our setup. On unseen entities, it falls well behind its teacher, as does the same student trained on benchmark data, so label quality alone does not explain the gap. Under GPT-5.2 pricing, LLM labeling costs per training set are on average $5.86 to $8.11. These findings support knowledge distillation as a practical approach to reduce the effort of labeling task-specific training data while enabling efficient inference.

## What the experiments cover

Eight entity matching benchmarks (see [`benchmarks/`](benchmarks/)):

- Abt-Buy, Walmart-Amazon, DBLP-ACM, and DBLP-Scholar from the DeepMatcher suite.
- Dn7 Walmart-Amazon, a rebuilt Walmart-Amazon task with hard non-matches.
- WDC Products with 100% unseen test entities.
- billiger.de, German product offers from thirteen categories with 50% unseen test entities.
- Semi-HETER from Machamp, semi-structured book records from five sources.

The experiments vary:

- **Pair selection:** similarity search, feature-based active learning (AL-ML), and Ditto-based active learning (AL-Ditto), at several label budgets.
- **Teacher model:** GPT-5.2, Qwen 3.6 Plus, and Kimi K2.6.
- **Post-processing:** relabeling, relabel-drop, closure-based dropping, and their intersection and union.
- **Student model:** Ditto, XGBoost, and Qwen3 (0.6B, 1.7B, 8B).

## Repository map

| Path | Contents |
|---|---|
| [`benchmarks/`](benchmarks/) | The eight benchmarks: record tables, splits, and precomputed embeddings. |
| [`artifacts/training_data/`](artifacts/training_data/) | The machine-labeled training sets, with a manifest. |
| [`artifacts/prompts/`](artifacts/prompts/) | The teacher labeling prompt and the review prompt used for relabeling. |
| [`artifacts/USAGE.md`](artifacts/USAGE.md) | Environments and commands for every step. |
| [`results/`](results/) | Per-run results behind each table and figure. |
| [`error_anlysis/`](error_anlysis/) | Human audit: annotations, sampling manifest, and error rates. |
| [`scripts/benchmarks/`](scripts/benchmarks/) | Converters that build Dn7, billiger.de, and Semi-HETER from their original releases. |
| [`scripts/labeling/`](scripts/labeling/) | Candidate pool and the three pair-selection workflows. |
| [`scripts/post_processing/`](scripts/post_processing/) | Relabeling and closure-based filtering. |
| [`scripts/training/`](scripts/training/) | Student training and evaluation for Ditto, XGBoost, and Qwen3. |
| [`configs/`](configs/) | Labeling and training configurations. |
| [`scripts/archive/`](scripts/archive/) | Helpers that the entry points call, and the scripts that cut the acquired sets to the reported sizes. |
| [`third_party/ditto_modern/`](third_party/ditto_modern/) | The Ditto implementation used for all Ditto students. |

## Paper results and their files

| Paper item | Training data | Results |
|---|---|---|
| Table 1: benchmark sizes | [`benchmarks/`](benchmarks/) | |
| Table 2: training sources (GPT-5.2) | `artifacts/training_data/*/gpt-5.2/{similarity_search,active_learning_ml,active_learning_ditto}/` | [`results/table2_training_sources.csv`](results/table2_training_sources.csv) |
| Table 3: benchmark pairs with GPT-5.2 labels | `artifacts/training_data/*/gpt-5.2/benchmark_pairs_relabeled/` | [`results/table3_benchmark_pairs_gpt52_labels.csv`](results/table3_benchmark_pairs_gpt52_labels.csv) |
| Figure 2: label budget | `artifacts/training_data/*/gpt-5.2/learning_curve/` | [`results/figure2_label_budget.csv`](results/figure2_label_budget.csv) |
| Table 4: teacher models | `artifacts/training_data/*/{gpt-5.2,qwen-3.6+,kimi-k2.6}/active_learning_ditto*/` | [`results/table4_teachers.csv`](results/table4_teachers.csv) |
| Table 5: post-processing | `artifacts/training_data/*/gpt-5.2/active_learning_ditto_*/` | [`results/table5_post_processing.csv`](results/table5_post_processing.csv) |
| Table 6: student models | as Table 2 | [`results/table6_student_models.csv`](results/table6_student_models.csv) (Ditto rows in Table 2) |
| Table 7: direct teachers | | [`results/table7_direct_teachers.csv`](results/table7_direct_teachers.csv), per-pair predictions in [`results/table7_direct_teachers/`](results/table7_direct_teachers/) |
| Table 8: labeling cost | | [`results/table8_labeling_costs.csv`](results/table8_labeling_costs.csv) |
| Table 9: inference time | | [`results/table9_inference_time.csv`](results/table9_inference_time.csv) |
| Table 10: training-set composition | as Tables 2 and 6 | [`results/table10_composition.csv`](results/table10_composition.csv), computed by `scripts/compute_composition.py` |
| Table 11: human audit | | [`error_anlysis/`](error_anlysis/) |

Each per-seed results file lists the test F1 of every student run, its training file, and the mean and SD printed in the paper. [`results/README.md`](results/README.md) describes the columns.

## Reproduction

Set up the environment for Ditto, XGBoost, and the labeling workflows:

```bash
conda env create -f configs/ditto/environment.yml
conda activate ditto-modern
```

The Qwen3 students use a separate environment (`scripts/archive/qwen_internal/setup_env.sh`). [`artifacts/USAGE.md`](artifacts/USAGE.md) gives the commands to build a training set with each selection strategy, to post-process it, and to train and evaluate Ditto, XGBoost, and Qwen3 students on the released sets.

## Citation

If you use the code, prompts, or machine-labeled training sets, please cite the paper:

```bibtex
@misc{steiner2026labeling,
  title         = {Labeling Training Data for Entity Matching Using Large Language Models},
  author        = {Steiner, Aaron and Bizer, Christian},
  year          = {2026},
  eprint        = {2606.28823},
  archiveprefix = {arXiv},
  primaryclass  = {cs.CL}
}
```

A `CITATION.cff` file is also included in the repository root.
