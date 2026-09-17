# Implementation Plan: Searching Only the Collections a Question Needs

**Branch**: `009-collection-routing` | **Date**: 2026-09-17 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/009-collection-routing/spec.md`

## Summary

Every collection in the bundle is searched for every question, and each contributes
a fixed ten documents whether or not it had anything to say. Adding a fifth cost
**+2,376 context tokens and +3.4s** on a question about CDK5 that has nothing to do
with variants.

The intent classifier already makes one LLM call per question and returns a source.
It will also return which collections to search. That adds no call and no latency,
and the per-collection descriptions it needs already exist in
`reactome_descriptions_info` -- written for routing, currently read only by
`bin/retrieval_baseline`.

## Technical Context

**Language/Version**: Python 3.12

**Primary Dependencies**: langchain 1.x, langchain-chroma, chromadb 0.6.3, pydantic 2

**Storage**: Chroma collections on disk under `embeddings/<model>/reactome/<release>/`

**Testing**: pytest. `bin/retrieval_baseline` for retrieval diffs, `bin/answer-sweep`
for end-to-end answers

**Target Platform**: Linux container behind FastAPI/Chainlit

**Project Type**: single project

**Performance Goals**: reduce context tokens and retrieval latency on questions that
do not need every collection. Today's baseline, measured: 5 collections = 50 docs,
9,437 tokens, 14.9s; 4 collections = 40 docs, 7,061 tokens, 11.5s

**Constraints**: no additional LLM call; a classifier failure must degrade to
today's behaviour, never to a narrower search

**Scale/Scope**: 5 collections today, 112,000 documents; the feature exists because
that number will grow

## Constitution Check

| Principle | Gate | How this plan satisfies it |
|---|---|---|
| I. Verify the path a user takes | Tests must go through the agent, not only the retriever | The sync retriever is not the served path (`aretrieve_documents` is). Acceptance runs `bin/answer-sweep`, which drives the whole graph |
| II. Measure retrieval changes | `bin/retrieval_baseline` capture before and after, diff reported | Phase 1 captures the baseline **before** any code changes, so the comparison exists to be made |
| III. Characterization tests | Current behaviour pinned before changing it | A test asserting all collections are searched today, so switching to selection is a deliberate edit of test and code together |
| IV. Fail loudly | Config that cannot be honoured stops rather than substitutes | A classifier naming a collection that is not in the bundle is a bug in the prompt or the bundle. It is logged at WARNING and the selection falls back to all collections -- never silently dropped, never a narrower search |
| V. Derive from the source of truth | No hand-synchronised lists | The selectable collections are derived from `list_chroma_subdirectories` on the live bundle, not a literal. `reactome_descriptions_info` is the one place a collection is described |

**Gate result**: pass. No violations to justify.

### The gate that is not yet met

The spec records it: thirteen sweep questions cannot cover five collections, and
**four of the five have no question that fails if routing stops searching them**.
Principle I is only satisfied once they do. That is task work in Phase 1, before the
routing change lands, not after.

## Project Structure

### Documentation (this feature)

```
specs/009-collection-routing/
├── spec.md
├── plan.md            # this file
├── research.md        # Phase 0
├── data-model.md      # Phase 1
├── contracts/
│   └── intent_classifier.md
├── quickstart.md
└── tasks.md           # /speckit-tasks
```

### Source Code (repository root)

```
src/
├── agent/tasks/intent_classifier.py   # gains the collection selection
├── agent/profiles/react_to_me.py      # threads the selection to the RAG
├── retrievers/
│   ├── csv_chroma.py                  # HybridRetriever filters by selection
│   └── reactome/metadata_info.py      # descriptions become a serving input
└── evaluation/answer_sweep.py         # per-collection questions

tests/
├── agent/test_intent_classifier_sources.py
├── retrievers/test_collection_selection.py   # new
└── retrievers/test_sync_async_equivalence.py # both paths must filter alike
```

## Complexity Tracking

| Addition | Current need | Why the simpler option is insufficient |
|---|---|---|
| Collection names in the classifier's structured output | Selection must cost no extra latency | A second LLM call doubles the routing cost, which is what the feature exists to reduce |
| Selection passed through `ReactToMeState` | The retriever is built once at startup, per profile | Rebuilding a retriever per question would re-read every BM25 index on every message |

## Phase 0: Research

See [research.md](./research.md).

## Phase 1: Design

See [data-model.md](./data-model.md), [contracts/](./contracts/),
[quickstart.md](./quickstart.md).

### Post-design constitution re-check

Unchanged: pass. The design adds no new LLM call, derives the collection list from
the bundle rather than a literal, and fails toward the wider search.
