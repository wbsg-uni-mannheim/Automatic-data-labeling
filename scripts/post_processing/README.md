# Post-Processing

This directory contains the public post-processing workflow for the released training sets.

## Entry Points

| Script | Purpose |
|---|---|
| `relabel_three_phase_generated_labels_batch.py` | Relabels generated labels with the LLM review prompt and materializes relabeled profiles. |
| `build_drop_changed_profiles.py` | Creates the variant that drops pairs whose label changes under LLM relabeling. |
| `build_closure_bridge_profiles.py` | Creates closure-based variants, including closure-only, closure-and-relabel, and closure-or-relabel filtering. |
