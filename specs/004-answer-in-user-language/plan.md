# Implementation Plan: Answer in the User's Language

**Branch**: `feat/answer-in-user-language` | **Date**: 2026-09-10 | **Spec**: [spec.md](./spec.md)

## Summary

Pass `detected_language` to the answer prompt as its own variable, in the two
profiles that ignore it. Nothing else moves.

The constraint that decides the design: **`input` must stay exactly as it is**,
because `create_retrieval_chain` hands it to the retriever, and `HybridRetriever`
hands it to the query expander. Both must keep seeing the English rephrasing and
nothing else. That is what rules out #140's mechanism and what makes this change
provably free of retrieval risk — the retrieval query is byte-identical before and
after, which a test can assert directly rather than measure statistically.

## Technical Context

**Language/Version**: Python 3.12

**Primary Dependencies**: `langchain-core` prompts; no new dependency.

**Testing**: pytest. `bin/retrieval_baseline` is *not* needed — see below.

**Constraints**: English questions must take exactly today's path (FR-007), with no
added prompt content and no extra model call.

**Scale/Scope**: Two prompt templates, two call sites, one shared instruction.

## Constitution Check

| Article | How this plan satisfies it |
|---|---|
| I — verify the user path | The exit criterion is asking the assembled chain a French question and reading the answer, which is how the bug was found. |
| II — measure, don't argue | Retrieval is unchanged **by construction**: `input` is not touched. A test asserts the retrieval query is byte-identical, which is stronger than a baseline diff and does not spend an hour of API time. |
| III — characterization tests pin behaviour | A test pins that an English question produces no language instruction at all. |
| IV — fail loudly | Nothing to fail: an absent language falls back to today's behaviour, which is correct English output. |
| V — derive from the source of truth | The language comes from `BaseState`, where the detector already put it. Nothing re-detects. |
| VI — bias to doing over filing | Two contributed PRs are resolved rather than left open. |

**No violations.**

## Implementation Stages

### Stage 1 — React-to-Me answers in the detected language

Add a language instruction to the reactome answer prompt as a template variable, and
pass `state["detected_language"]` at the call site in `generate_answer`.

The instruction carries #140's nomenclature rule, which is the part of that PR worth
keeping: gene symbols, protein names, pathway names, `R-HSA-*` identifiers and URLs
stay in English.

**What an English question pays.** No extra model call, and a byte-identical
retrieval query — but it does gain the instruction in its answer prompt, saying to
answer in English.

An earlier draft of this plan claimed English "costs nothing", which contradicted the
spec's own FR-007 and was simply untrue: a sentence added to every prompt is a
change, however small. FR-007 has been corrected to say what is actually guaranteed.
The cost is one sentence against roughly 3,200 tokens of retrieved context; the
alternative is a branch whose common path only non-English users exercise.

**Exit criteria**: a French question is answered in French with nomenclature intact;
the retrieval query is byte-identical to today's; an English question still gets a
well-formed English answer.

### Stage 2 — Plant Reactome, identically

The same change to `plantreactome/prompt.py` and its call site. It is a separate
stage only because it is a separate deployment and can be verified separately.

**Exit criteria**: same three, against the plantreactome profile.

### Stage 3 — Close #125 and #140

Close both with credit, saying specifically what each contributed: #140's target and
nomenclature rule, #125's mechanism. Say plainly why #140's mechanism was not taken,
with the measured number — half the retrieved context changes.

#125's hallucination-grading and web-search work is **not** resolved by this and must
not be described as such; it belongs to #123.

**Exit criteria**: both closed with credit; the spec records the outcome.

## Complexity Tracking

| Decision | Simpler alternative rejected | Why |
|---|---|---|
| A prompt variable | Append to `input`, as #140 does | Measured: about half the fused documents change. The instruction reaches the retriever and the query expander, neither of which should see it. |
| Instruct the generator | Translate the finished answer | D1. An extra model call and its latency on every non-English message, plus translation errors over a scientific answer, to gain a nomenclature protection the prompt already achieves. |
| Always pass the language | Branch on "is it English" | A branch means two paths, one of which is rarely exercised. Passing "English" is the same code doing the same thing. |
| No baseline run | `bin/retrieval_baseline` before and after | The retrieval query is unchanged by construction; a test asserting that is exact, where a baseline diff would only show noise and cost an hour. |

## Out of Scope

- Verifying the answer really is in the requested language.
- The corpus, embeddings, or the detector.
- #125's hallucination grader and web search — #123's decision.
