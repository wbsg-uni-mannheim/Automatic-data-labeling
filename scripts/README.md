# Scripts

| Path | Purpose |
|---|---|
| `labeling/` | The three pair-selection workflows that build machine-labeled training sets. |
| `post_processing/` | Relabeling with the review prompt and closure-based filtering. |
| `training/` | Training and evaluation of the XGBoost, Ditto, and Qwen3 students. |
| `benchmarks/` | Converters that build Dn7, billiger.de, and Semi-HETER from their original releases. |
| `archive/` | Helpers that the entry points call, and the scripts that cut the acquired sets to the reported sizes. |
| `compute_composition.py` | Computes the training-set composition of Table 10. |
