# Post-Processing

Build the post-processing variants of the released training sets.

| Script | What it does |
|---|---|
| `relabel_three_phase_generated_labels_batch.py` | Relabels generated labels with the review prompt and writes relabeled profiles. |
| `build_drop_changed_profiles.py` | Drops pairs whose label changes under relabeling. |
| `build_closure_bridge_profiles.py` | Builds the closure variants: closure-only, closure-and-relabel, and closure-or-relabel. |
