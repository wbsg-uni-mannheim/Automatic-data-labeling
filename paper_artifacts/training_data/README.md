# Training Data Used in the Experiments

Structure: `paper_artifacts/training_data/{benchmark}/{labeler}/{method}/{benchmark}_train.json.gz`

Each file is a materialized training set used for downstream classifiers including Ditto, Qwen3 student models, and XGBoost. Counts match the paper table on training-set composition by benchmark and construction method.

## Coverage matrix

| Labeler | Similarity search | AL (ML) | AL (Ditto) | Post-processing variants |
|---|---|---|---|---|
| **GPT-5.2** | yes (5 benchmarks) | yes (5) | yes (5) | yes, 5 variants across 5 benchmarks |
| **Qwen 3.6+** | not used | not used | yes (5) | not used |
| **Kimi K2.6** | not used | not used | yes (5) | not used |

Qwen and Kimi were only used as labelers for AL-Ditto in the LLM-labeler-quality comparison. Post-processing variants were run on GPT-labeled AL-Ditto output.

## Methods

| Method dir | Description |
|---|---|
| `similarity_search` | Top-k similarity-blocked pairs labeled in one shot |
| `active_learning_ml` | Iterative active learning with a logistic-regression / ML student |
| `active_learning_ditto` | Three-phase active learning with a Ditto student (base run) |
| `active_learning_ditto_relabel` | + Re-label disagreements with gpt-5-mini |
| `active_learning_ditto_relabel_drop` | + Re-label and drop the changed rows |
| `active_learning_ditto_closure_drop` | + Drop rows broken by transitive-closure bridge check |
| `active_learning_ditto_closure_and_relabel` | + Closure and relabel: drop only rows flagged by both signals |
| `active_learning_ditto_closure_or_relabel` | + Closure or relabel: drop rows flagged by either signal |

## Manifest

`MANIFEST.csv` records the benchmark, labeler, method, row count, and artifact path for each materialized file.

## Total

50 training files (5 benchmarks × 10 method/labeler combinations); ~78 MB compressed.
