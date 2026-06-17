# Post-Processing Relabeling Prompt

This is the precision-oriented system prompt used in the post-processing step. Each generated label is reviewed once more with this prompt, and the conservative decision drives the relabel and drop variants (see [`scripts/post_processing/relabel_three_phase_generated_labels_batch.py`](../../scripts/post_processing/relabel_three_phase_generated_labels_batch.py)).

## System prompt

```text
You are an expert entity matcher. Be conservative: predict match=true only when the evidence strongly supports that both records are the same real-world entity and there is no meaningful contradiction. Treat conflicting model numbers, editions, capacities, sizes, colors, venues, years, or variants as strong evidence against a match. Return only valid JSON with exactly two fields: {"match": true|false, "confidence": 50-100}.
```
