# Writing Guidelines (Steiner & Bizer group)

**Purpose:** point Claude (or a co-author) to this file at the start of any academic-writing or paper-editing session so the style is consistent from the first edit, without re-explaining preferences each time.

**TL;DR for Claude:** Write like a scientist, not an LLM. Concise, declarative, precise. No rhetorical questions, no filler, no semicolons, no overclaiming. Take the reader by the hand with explicit goals and table pointers — but do it with concrete signposting, never throat-clearing. Verify every method description against the code. Compile and check page count after edits.

---

## 1. Voice & tone

- **Declarative, not interrogative.** Never frame analysis as a question. Avoid "Here we ask whether X can Y", "We test whether…", "asks whether…", "the question is…". State what a section *does* with verbs like **investigates, examines, evaluates, compares, quantifies, measures, reports, analyzes**.
  - Bad: *"This section asks whether machine-labeled sets can replace the benchmark sets."*
  - Good: *"This section compares the fitness for use of the machine-labeled sets with the benchmark sets."*
- **Lead with the claim.** No throat-clearing or filler signposting.
  - Bad: *"Three patterns emerge from the table…"*, *"It is noteworthy that…"*, *"Interestingly,…"*
  - Good: state the finding directly.
- **No hype / editorializing.** Drop "noteworthy", "remarkably", "clearly demonstrates", "a real lever, but it does not make or break…". These read as LLM-like.
- **Don't write verdicts in analysis sections.** A results/analysis paragraph should not read like a conclusion.

## 2. Mechanics

- **No semicolons in prose.** Split into two sentences, or use a comma + conjunction. (Semicolons in TikZ/code/listings are fine.)
- **No "big" words:** avoid *comprehensive, sophisticated, leverage (as verb), delve,* etc. Use plain words.
- **No negation-then-contrast.** Avoid "We do not X, but keep Y." State it positively.
  - Bad: *"We do not add the transitive edges, but keep the original match edges and look for bridges."*
  - Good: *"Operating directly on the match edges, we look for bridges."*
- **Don't restate the same point** two or three ways in one paragraph. Say it once.
- **Grammar:** keep parallelism (gerund lists), fix a/an, avoid comma splices.

## 3. Claims & rigor (do not overclaim)

- **Prefer "comparable / on par / within X F1"** over "outperforms / beats" when the gap is within run-to-run variance. Reserve strong verbs for clear, supported gaps.
- **Apples-to-apples only.** Compare on the same axis. E.g., do not claim "reaches benchmark performance with half the labels" if the benchmark curve was never swept — compare against what was actually measured (e.g., similarity search on the same budget axis).
- **Understate limitations of our own methods — with numbers, not adjectives.** "trails the others by 19 to 35 F1 on four of five benchmarks" — not "falls off markedly" (too vague) and not "fails / is not fit for use" (overstated).
- **Hedge honestly:** "may", "could", "a possible explanation is" for interpretations we don't prove.
- **Cite exact numbers**, not vague summaries.
  - Bad: *"captures most of the teacher's performance."*
  - Good: *"trails the direct LLM by 4.53 F1 on Abt-Buy and 3.29 F1 on Walmart-Amazon."*
- **Method text must match the code.** Verify against the implementation before describing any method. Describe what we actually do, precisely (e.g., a *lexicographic* ranking, not a "blended score"; positive-edge *bridges*, not a "transitive closure").
- **The "what's the number?" rule (the most common reviewer complaint).** Every word of degree — *matches, competitive, large, sharp, much, closely, well, approaches, falls off, drops* — is a flag. Three tests for any sentence:
  1. **Replace the word with the number.** If you measured it, state it: "matches it" → "within 1.4 F1 of it"; "stays competitive" → "the gap is just 1.4 F1"; "approaches them only on X" → "within 1.6 F1 on X".
  2. **An evaluative word is allowed only (a) as a topic sentence the paragraph then proves with numbers, or (b) with the number pinned to it** — "competitive (within 1.4 F1)". A bare evaluative word in the middle of a results description is the bad case.
  3. **Check it isn't smuggling a false claim.** "matches" can imply a tie when one side is actually ahead or behind — the vague word hides it, the number does not.

## 4. Structure & reader guidance (Chris's feedback, verbatim)

> "Was noch fehlt ist finales fine-tuning des Text, da teilweise einfache Dinge für Außenstehende zu kompakt dargestellt werden und es schön wäre den Leser stärker an die Hand zu nehmen (z.B. mehr Verweise auf Tabellen, öfters explizierte Nennung des Ziels der jeweiligen Analyse)."

Translation / action items:
- **Each analysis section opens with one plain, declarative goal sentence.** (Reconcile with §1: state the goal, never as a question.)
- **Reference tables/figures often and actively** — point to the specific table/row when citing a number, not just once at the top of a section. Every float must be referenced where it is discussed.
- **Define terms for outsiders at first use**, then reuse: *fitness for use, teacher/student, corner cases, hard positives/negatives, blocking.* Define once.
- **Don't present simple things too compactly.** Unpack a step a non-expert would stumble on.
- **Don't pre-announce or spoil results in the Method section.** Method = what we do; Results = the outcome. Keep setup details in the results section where the point is discussed, not duplicated in the experimental setup.
- **End each analysis with a one-line "what this shows"** tied back to the paper's argument.

## 5. Headings

- Use **established, descriptive headings**, not catchy/"plakativ" question headings (e.g. *"What is the effect of the labeler?"*). A question heading implies we comprehensively answer it, which our experiment set rarely does.
- **Do not use a bold finding as a heading.** Name the *dimension* being analyzed (e.g., "Effect of the Teacher Model", "Analysis of the Training Set Composition").

## 6. Tables & figures

- **Number tables in order of first reference** in the text.
- **Captions:** keep them compact; describe the plot. **Do not surface diverging run counts in captions** — just describe what is shown.
- **Round to sensible precision** (2 decimals is the default).
- Bold = best per row/benchmark, underline = second best (state this in the caption).

## 7. Terminology (this project — adapt per project)

- **"machine-labeled"** (not "auto-labeled").
- **"workflow"** for our system (not "pipeline"; "pipeline" is fine only as a generic field term like "data integration pipelines").
- **"inference time" / "runtime"** (not "inference cost" — we measure time, not money).
- **Name the models** (Ditto, Qwen3-8B, …). Do not write "neural students".
- Avoid framing well-known ideas as our finding (e.g., don't label the work a "selection-and-labeling problem").

## 8. Venue / format (EDBT example — update per venue)

- EDBT: **single-blind**; **12 pages body + unlimited references**; ACM `sigconf` (`acmart`).
- The **body must stay ≤ 12 pages**; references may spill beyond.
- Float placement (`[!b]`, `dblfloatfix`, last-page `\balance`) is allowed — it changes layout, not margins/fonts/spacing. **Do not** use spacing hacks (`\vspace{-…}`, shrinking `\textheight`, smaller fonts) to gain room.
- Compile cycle: `pdflatex → bibtex → pdflatex → pdflatex`; confirm **0 errors** and that all `\ref`/citations resolve.

## 9. Working with Claude (process)

- At session start, point to this file.
- After edits: **compile, then report LaTeX errors, undefined refs, and page count.**
- When describing a method or quoting a number, **check the code/data first**.
- Default to concise. When cutting, remove fluff and redundancy — but keep the reader-guidance items in §4 (goals + table pointers + first-use definitions).
- Make targeted edits; re-read a file before editing if an editor/linter may have changed it.
