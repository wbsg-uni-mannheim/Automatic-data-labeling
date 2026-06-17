# Labeling Training Data for Entity Matching Using Large Language Models

This repository is the code and data release for the paper:

> **Labeling Training Data for Entity Matching Using Large Language Models**
> Aaron Steiner and Christian Bizer, Data and Web Science Group, University of Mannheim.

It contains the pipeline code, prompts, configuration, and the machine-labeled training sets used in the paper, so the experiments can be inspected and rerun without repeating the LLM labeling.

## Abstract

Entity matching research has shown that large language models (LLMs) achieve strong zero-shot performance, allowing them to perform matching tasks without task-specific training data. However, applying them to large candidate-pair sets remains relatively slow and expensive. Smaller matchers such as traditional machine learning methods or small language models (SLMs) like BERT are faster at inference time, but require task-specific labeled training data.

This paper evaluates knowledge-distillation workflows in which an LLM teacher labels training pairs and a smaller student matcher is trained on the resulting machine-labeled set. We compare pair-selection strategies, teacher models, post-processing variants, and student models on Abt-Buy, Walmart-Amazon, WDC Products, DBLP-ACM, and DBLP-Scholar, and compare each student with the same model trained on the benchmark labels.

With Ditto as the student and GPT-5.2 as the teacher, the best machine-labeled set stays within 1.78 F1 of the benchmark training set on every benchmark. On the three product benchmarks, its differences to the benchmark are +0.23 to +1.59 F1. The same conclusion holds with Qwen 3.6 Plus and the open-weight Kimi K2.6 as teachers.

Beyond label quality, pair selection shapes the training sets: active learning raises the positive rate by up to 12.74 percentage points and selects more hard positives than the benchmark set on four of five benchmarks.

Under GPT-5.2 pricing, LLM labeling costs \$28.31 to \$40.88 across all five datasets, compared with an estimated 470 hours of manual labeling for the benchmark training sets. At inference time, Ditto is 41.5 to 534 times faster than direct LLM matching in our measurements.

These results indicate that current LLMs can be used as teacher models to construct training data for faster downstream entity matchers while reducing manual-labeling effort.

## Artifacts

Start in **[`artifacts/`](artifacts/)**. It is the entry point for everything released with the paper:

- **[`artifacts/training_data/`](artifacts/training_data/)** — all 50 machine-labeled training sets (5 benchmarks × teacher × construction method), with a [`MANIFEST.csv`](artifacts/training_data/MANIFEST.csv).
- **[Key scripts](artifacts/README.md#scripts)** — links to the runners for the three training-set construction workflows and the student-training scripts.
- **[`artifacts/prompts/`](artifacts/prompts/)** — the teacher-labeling and post-processing prompts.
- **[`artifacts/USAGE.md`](artifacts/USAGE.md)** — worked Abt-Buy commands for each construction workflow.

The benchmark inputs and precomputed embeddings used by the runners are under **[`benchmarks/`](benchmarks/)**.

## What the experiments cover

Five entity-matching benchmarks: Abt-Buy, Walmart-Amazon, WDC Products, DBLP-ACM, and DBLP-Scholar. These benchmarks are publicly available, and the splits and embeddings used here are included under [`benchmarks/`](benchmarks/).

The experiments vary four dimensions:

- **Pair selection:** similarity search, feature-based active learning, and Ditto-based active learning.
- **Teacher model:** GPT-5.2, Qwen 3.6 Plus, and Kimi K2.6.
- **Post-processing:** relabeling, relabel-drop, closure-drop, and combined closure/relabel variants.
- **Student model:** Ditto, XGBoost, and Qwen3 fine-tuning.

## Repository map

| Path | Contents |
|---|---|
| [`artifacts/`](artifacts/) | Release front door: training data, prompts, and usage guide. |
| [`scripts/labeling/`](scripts/labeling/) | Three public training-set construction workflows. |
| [`scripts/training/`](scripts/training/) | Student-model training entry points for XGBoost, Ditto, and Qwen. |
| [`scripts/post_processing/`](scripts/post_processing/) | LLM relabeling and post-filter variant construction. |
| [`scripts/archive/`](scripts/archive/) | Historical utilities and implementation internals retained for provenance. |
| [`benchmarks/`](benchmarks/) | Benchmark datasets and precomputed embeddings read by the runners. |
| `configs/labeling/` | Benchmark labeling configuration. |

## Reproduction

[`artifacts/USAGE.md`](artifacts/USAGE.md) walks through constructing the Abt-Buy training set with all three pair-selection workflows. For command-line help on the main runners:

```bash
python scripts/labeling/similarity_search.py --help   # similarity search
python scripts/labeling/active_learning_ml.py --help    # feature-based active learning
python scripts/labeling/active_learning_ditto.py --help # Ditto-based active learning
python scripts/training/train_xgboost.py --help         # XGBoost student training
python scripts/training/train_ditto.py --help           # Ditto student training
python scripts/training/train_qwen.py --help            # Qwen student training
```

## Citation

If you use the code, prompts, or machine-labeled training sets, please cite the paper:

```bibtex
@inproceedings{steiner2027labeling,
  title     = {Labeling Training Data for Entity Matching Using Large Language Models},
  author    = {Steiner, Aaron and Bizer, Christian},
  booktitle = {Proceedings of the 30th International Conference on Extending Database Technology (EDBT)},
  year      = {2027}
}
```
