# Labeling Training Data for Entity Matching Using Large Language Models

This repository is the code and data release for the paper:

> **Labeling Training Data for Entity Matching Using Large Language Models**
> Aaron Steiner and Christian Bizer, Data and Web Science Group, University of Mannheim.

It contains the pipeline code, prompts, configuration, and the machine-labeled training sets used in the paper, so the experiments can be inspected and rerun without repeating the LLM labeling.

## Abstract

Recent large language models (LLMs) achieve strong performance on entity matching without requiring task-specific training data. However, applying these models to large sets of candidate pairs remains slow and costly. In contrast, entity matchers using traditional machine learning methods or small language models (SLMs), such as RoBERTa, offer much faster inference but require task-specific training data.

Our paper investigates whether the need to provide task-specific training data can be avoided by using knowledge-distillation workflows, in which an LLM serves as a teacher model to label training pairs that are subsequently used to train a smaller student model. We investigate knowledge distillation for entity matching along the following dimensions:  pair-selection strategy, teacher model, label post-processing method, and student model. We evaluate the workflows using the Abt-Buy, Walmart-Amazon, WDC Products, DBLP-ACM, and DBLP-Scholar benchmarks, and compare the performance of student models trained with machine-labeled data to the performance of the same models trained using the benchmark training sets.

Our experiments show that student models trained using the machine-labeled sets perform approximately on par with models trained on the benchmark training sets, with the remaining differences in both directions staying below two F1 points. Using GPT-5.2 to label the training sets for all five benchmarks costs US$28.31 to US$40.88, whereas manually labeling the same training sets is estimated to require 470 hours of work. At inference time, Ditto is 41.5 to 534 times faster than directly using an LLM to perform the matching tasks.

These results indicate that current LLMs, when combined with a suitable pair-selection method, can substantially reduce or even eliminate the manual effort required to label use case-specific training data for entity matching.

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
@inproceedings{steiner2026labeling,
  title     = {Labeling Training Data for Entity Matching Using Large Language Models},
  author    = {Steiner, Aaron and Bizer, Christian},
  booktitle = {let's see},
  year      = {2026}
}
```
