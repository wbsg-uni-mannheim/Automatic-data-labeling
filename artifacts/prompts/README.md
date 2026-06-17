# Prompt Materials

The workflow uses prompts for two stages: initial teacher labeling and post-processing review.

## Initial Teacher Labeling

- `artifacts/prompts/entity_matching_labeling_prompt.txt`

This prompt asks the teacher model to decide whether a record pair denotes the same real-world entity and to return a JSON decision.

## Post-Processing Relabeling

The relabeling step reviews each labeled pair with a precision-oriented prompt:

- [`entity_matching_relabel_prompt.md`](entity_matching_relabel_prompt.md)

`scripts/archive/review_workflows/experiments/evidence_first_abstain/prompts/` holds the other review prompts (recall, balanced, skeptic, and abstain variants) that we tried during development.
