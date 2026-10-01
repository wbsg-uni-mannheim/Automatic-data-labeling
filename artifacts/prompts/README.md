# Prompts

The workflow uses two prompts: one to label pairs with the teacher, one to relabel them in post-processing.

## Teacher labeling

[`entity_matching_labeling_prompt.txt`](entity_matching_labeling_prompt.txt) asks the teacher whether a record pair refers to the same real-world entity and returns a JSON decision.

## Post-processing relabeling

[`entity_matching_relabel_prompt.md`](entity_matching_relabel_prompt.md) reviews each labeled pair with a precision-oriented prompt. The conservative decision drives the relabel and drop variants. [`review_system_prompt.txt`](review_system_prompt.txt) is the same prompt as the plain text file that `scripts/post_processing/relabel_three_phase_generated_labels_batch.py` reads.
