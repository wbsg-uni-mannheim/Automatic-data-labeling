# Prompt Materials

The paper uses prompts for two stages: initial teacher labeling and post-processing review. The relevant prompt files are maintained in their original locations to preserve provenance.

## Initial Teacher Labeling

- `LLM-Labeled-Training-Data-for-Entity-Matching/paper/prompts/entity_matching_labeling_prompt.txt`

This prompt asks the teacher model to decide whether a record pair denotes the same real-world entity and to return a JSON decision.

## Post-Processing and Review Prompts

- `scripts/experiments/evidence_first_abstain/prompts/agent_precision_system_prompt.txt`
- `scripts/experiments/evidence_first_abstain/prompts/agent_precision_abstain_system_prompt.txt`
- `scripts/experiments/evidence_first_abstain/prompts/agent_recall_system_prompt.txt`
- `scripts/experiments/evidence_first_abstain/prompts/agent_balanced_system_prompt.txt`
- `scripts/experiments/evidence_first_abstain/prompts/agent_balanced_abstain_system_prompt.txt`
- `scripts/experiments/evidence_first_abstain/prompts/agent_variant_skeptic_system_prompt.txt`
- `scripts/experiments/evidence_first_abstain/prompts/agent_variant_skeptic_abstain_system_prompt.txt`
- `scripts/experiments/evidence_first_abstain/prompts/agent_variant_skeptic_tuned_system_prompt.txt`
- `scripts/experiments/evidence_first_abstain/prompts/agent_recall_toned_system_prompt.txt`
- `scripts/experiments/evidence_first_abstain/prompts/active_learning_system_prompt.txt`
- `scripts/experiments/evidence_first_abstain/prompts/batch_exact_system_prompt.txt`

The paper's post-processing experiments use the precision-oriented review setup for relabeling. The remaining prompt variants are retained as documented experimental materials.
