# Training Data

The 50 machine-labeled training sets used in the paper, laid out as:

```text
{benchmark}/{teacher}/{method}/{benchmark}_train.json.gz
```

Each file trains a downstream student (Ditto, Qwen3, or XGBoost). Row counts match the training-set composition table in the paper.

## Coverage

| Teacher | Similarity search | AL (ML) | AL (Ditto) | Post-processing variants |
|---|---|---|---|---|
| **GPT-5.2** | 5 benchmarks | 5 | 5 | 5 variants × 5 benchmarks |
| **Qwen 3.6+** | — | — | 5 | — |
| **Kimi K2.6** | — | — | 5 | — |

Qwen and Kimi label only the AL (Ditto) setting, for the teacher-comparison experiment. The post-processing variants run on the GPT-5.2 AL (Ditto) labels.

## Methods

| Method directory | Description |
|---|---|
| `similarity_search` | Top-k similarity-blocked pairs, labeled in one pass |
| `active_learning_ml` | Active learning with a feature-based student |
| `active_learning_ditto` | Three-phase active learning with a Ditto student |
| `active_learning_ditto_relabel` | Relabels disagreements with gpt-5-mini |
| `active_learning_ditto_relabel_drop` | Relabels, then drops the changed rows |
| `active_learning_ditto_closure_drop` | Drops rows that fail the positive-edge bridge check |
| `active_learning_ditto_closure_and_relabel` | Drops rows flagged by both the bridge check and relabeling |
| `active_learning_ditto_closure_or_relabel` | Drops rows flagged by either signal |

## Manifest

[`MANIFEST.csv`](MANIFEST.csv) records the benchmark, teacher, method, row count, and path of every file. 50 files in total, about 78 MB compressed.
