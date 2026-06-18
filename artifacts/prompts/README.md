# Prompts

The workflow uses two prompts: one to label pairs with the teacher, one to relabel them in post-processing.

## Teacher labeling

[`entity_matching_labeling_prompt.txt`](entity_matching_labeling_prompt.txt) asks the teacher whether a record pair refers to the same real-world entity and returns a JSON decision.

## Post-processing relabeling

[`entity_matching_relabel_prompt.md`](entity_matching_relabel_prompt.md) reviews each labeled pair with a precision-oriented prompt. The conservative decision drives the relabel and drop variants.

`scripts/archive/review_workflows/experiments/evidence_first_abstain/prompts/` holds the recall, balanced, skeptic, and abstain variants tried during development.
