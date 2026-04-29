# EDBT Paper Draft

This directory contains the starting scaffold for an EDBT-oriented paper on selecting and labeling entity matching training examples with LLM-based systems.

## Target Format

Checked on 2026-04-21:

- EDBT 2026 research papers use the latest ACM double-column SIG conference proceedings format.
- Submissions are non-anonymous.
- Regular research and Experiments & Analysis papers: 12 pages plus unlimited references.
- Short research and Vision papers: 6 pages plus unlimited references.
- The paper must include an `Artifacts` section immediately before references.
- Current EDBT 2026 research rounds have already passed; this scaffold should be adapted to the next active EDBT cycle unless there is a special track or later venue target.

Sources:

- https://edbticdt2026.github.io/?contents=EDBT_CFP.html
- https://edbt.org/

## Proposed Outline

1. Introduction: why EM training data is the bottleneck and why the contribution is a selection-plus-labeling system.
2. Background and problem setting: EM, LLM labeling, and quality risks.
3. Training-set construction system: candidate pool construction, example selection, LLM labeling, and materialization.
4. Selected-and-labeled-vs-official training-set comparison: overlap, novelty, coverage, class balance, difficulty, proxy noise.
5. Experimental design: official vs selected-and-labeled training sets, fixed downstream matcher, shared test splits.
6. Results: selected-and-labeled vs official downstream performance, selection-strategy effects, and training-set composition analysis.
7. Discussion: why selected-and-labeled data can work, when it fails, positioning against prior LLM-for-EM work.
8. Related work.
9. Conclusion.
10. Artifacts.

## Build

From this directory:

```sh
make
```

The scaffold copies `acmart.cls` and `ACM-Reference-Format.bst` from `/Users/aaronsteiner/Downloads/acmart-primary`.

## Next Writing Tasks

- Decide whether the paper is a regular research paper or an Experiments & Analysis submission.
- Add the main official-vs-selected-and-labeled downstream F1 table.
- Refresh all selected-and-labeled-vs-official training-set analysis numbers from one canonical script.
- Add manually reviewed error examples.
- Consolidate Ditto training results into one final benchmark table.
- Create a stable artifact bundle without credentials or machine-local paths.
