# Post-Processing

Build the post-processing variants of the released training sets.

| Script | What it does |
|---|---|
| `relabel_three_phase_generated_labels_batch.py` | Relabels generated labels with the review prompt and writes relabeled profiles. |
| `build_drop_changed_profiles.py` | Drops pairs whose label changes under relabeling. |
| `build_closure_bridge_profiles.py` | Builds the closure variants: closure-only, closure-and-relabel, and closure-or-relabel. |
| `build_cleaning_variants.py` | Builds the variants from one frozen profile and one completed review. |
| `consistent_closure.py` | Drops positive bridge edges and then negatives inside the remaining positive components (closure variant of Table 5). |
| `closure_combinations.py` | Intersects or unites the closure drops with the review disagreements (Table 5). |

For a new cleaning comparison, review the selected source profile once, then run:

```bash
python scripts/post_processing/build_cleaning_variants.py \
  --original-profile /path/to/original/profiles/all \
  --reviewed-profile /path/to/reviewed/profiles/all \
  --output-root /path/to/new/cleaning-comparison
```

Use the exact frozen training profile for both inputs. The review must contain
every original pair exactly once. This command does not call any label provider
or train any models. It keeps the original text and order, uses positive bridges
in components of at least three nodes, and writes five `train.json.gz` exports.
The intersection/union variants drop pairs using the original labels; only
`v_relabel` changes retained labels. Existing nonempty output directories are
rejected. The manifest identifies identical ordered training exports; reuse a fit
only if its seed, preprocessing, training settings and evaluation splits also match.
