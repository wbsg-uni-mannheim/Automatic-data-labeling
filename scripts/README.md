# Script Organization

The script tree exposes a small public workflow surface.

| Directory | Purpose |
|---|---|
| `labeling/` | Three workflows for constructing machine-labeled training sets. |
| `training/` | Student-model training entry points for XGBoost, Ditto, and Qwen. |
| `post_processing/` | LLM relabeling and post-filter variant construction. |
| `archive/` | Historical scripts, one-off utilities, and implementation internals kept for provenance. |

New users should start with `labeling/`, then use `training/`, and consult `post_processing/` only for relabeling and post-filter variants.
