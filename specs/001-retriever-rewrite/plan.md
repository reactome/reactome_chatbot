# Implementation Plan: Retriever Rewrite

**Branch**: `plan/retriever-rewrite` | **Date**: 2026-09-08 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-retriever-rewrite/spec.md`

## Summary

Replace `HybridRetriever`, which subclasses `MultiQueryRetriever` and reaches into
LangChain internals in five places, with a plain `BaseRetriever` that owns its own
fusion. Swap `SelfQueryRetriever` for plain semantic search (D1), keep BM25, and
make the context budget an argument rather than a module constant (D2, D3).

The work is deliberately staged so that a behaviour change and a structural change
are never in the same commit. Stage 1 is a pure refactor that must produce
byte-identical retrieval; only Stage 2 changes what the model sees.

## Technical Context

**Language/Version**: Python 3.12

**Primary Dependencies**: `langchain-core` (BaseRetriever, Document), `langchain-chroma`,
`langchain-community` (BM25Retriever), `rank-bm25`, `nltk`. The point of the rewrite
is to depend on `langchain-core` contracts only.

**Storage**: Chroma collections on disk, from an installed embeddings bundle;
sibling `csv_files/*.csv` for BM25.

**Testing**: pytest. `tests/retrievers/test_hybrid_retriever.py` already pins the
fusion behaviour, including the tie-break and the `[1/n]*n` weighting that cannot
affect ordering. `bin/retrieval_baseline` for before/after measurement.

**Target Platform**: Linux server (container), and developer machines.

**Project Type**: Library within a single application repository.

**Performance Goals**: Retrieval makes exactly one LLM call per message, down
from 21 (FR-009). No wall-clock target; the LLM answer dominates.

**Constraints**: Sync and async must agree (FR-007). Behaviour must be measurable
with the committed question set before and after (FR-008).

**Scale/Scope**: 4 Reactome collections, ~121k documents total; `userguide` has its
own retriever and is out of scope.

## Constitution Check

*GATE: must pass before implementation, re-checked after.*

| Article | How this plan satisfies it |
|---|---|
| I — verify the user path | Every stage ends with the full RAG chain answering a real question through **both** `invoke` and `ainvoke`, not just the retriever in isolation. This is the failure mode that recurred most. |
| II — measure, don't argue | `bin/retrieval_baseline capture` before Stage 1 and after each stage; Stage 1 must show **zero** difference. |
| III — characterization tests pin behaviour | The existing retriever tests must pass unchanged through Stage 1. Any test that must change in Stage 2 changes together with the code and says why. |
| IV — fail loudly | Missing bundle raises with an actionable message rather than passing `None` into a `Path` (FR-006), removing the `B008` suppression and four mypy baseline entries. |
| V — derive from the source of truth | Budget and over-fetch stay single named parameters; the harness keeps importing them from the module rather than re-declaring. |
| VI — bias to doing over filing | Stages are small enough to land individually; nothing here is deferred to an issue that could be done now. |

**No violations.** The one judgement call is Stage 3 (budget as an argument), which
could be deferred — it is kept because the interface should be settled while the
file is open rather than bolted on later, per spec User Story 2.

## Project Structure

### Documentation (this feature)

```text
specs/001-retriever-rewrite/
├── spec.md      # what and why, with D1–D4 settled
└── plan.md      # this file
```

No `research.md`, `data-model.md` or `contracts/` are generated. There are no
unresolved unknowns — D1–D4 are decided and the evidence is in the spec — and the
feature introduces no new data model or external contract. Generating empty
scaffolding would be ceremony.

### Source Code

```text
src/retrievers/
├── csv_chroma.py           # HybridRetriever -> plain BaseRetriever; owns RRF
├── reactome/rag.py         # bundle becomes an explicit argument
├── uniprot/rag.py          # same
├── plantreactome/rag.py    # same
└── userguide/             # already a plain BaseRetriever; the model to follow

tests/retrievers/
└── test_hybrid_retriever.py  # extended, not rewritten

bin/retrieval_baseline        # already mirrors the pipeline
```

## Implementation Stages

### Stage 1 — Structural only, zero behaviour change

Replace the `MultiQueryRetriever` subclass with a `BaseRetriever` that implements
`_get_relevant_documents` / `_aget_relevant_documents`, and vendor Reciprocal Rank
Fusion into this repository (FR-001, FR-002).

Query expansion stays (D4) and `SelfQueryRetriever` stays for now — this stage
changes *how the code is arranged*, not what it returns.

RRF is copied from `EnsembleRetriever.weighted_reciprocal_rank` as it behaves
today: `weight / (rank + 60)` with rank counted from 1, de-duplicated on
`page_content`, ties resolved by first appearance. Those properties are already
pinned by tests.

**Exit criteria**: existing tests pass unchanged; `retrieval_baseline compare`
shows **zero** differences; the full chain answers a real question through both
`invoke` and `ainvoke`.

### Stage 2 — D1: plain semantic search replaces SelfQuery

Swap `SelfQueryRetriever` for `vectordb.as_retriever(...)` (FR-009). Retrieval
drops to one LLM call per message.

`metadata_info.py` stays: `evaluator.py` and `bin/retrieval_baseline` still
construct SelfQuery for comparison, so deleting it would break the tool that
measures this change.

**Exit criteria**: `retrieval_baseline compare` shows differences confined to the
vector side; a test asserts retrieval makes exactly one LLM call; the chain
answers real questions in both paths.

### Stage 3 — D2/D3: the caller supplies the budget

`MAX_DOCUMENTS_PER_COLLECTION` becomes a constructor argument with today's value
as the default (FR-005). The bundle becomes an explicit argument, removing the
`B008` suppression and the four `retrievers.*.rag` mypy baseline entries (FR-006,
SC-006).

**Exit criteria**: two callers with different budgets in one process get different
amounts of context; the four baseline entries are deleted; gates green.

## Complexity Tracking

| Decision | Simpler alternative rejected | Why |
|---|---|---|
| Vendor RRF (~15 lines) | Keep calling `EnsembleRetriever.weighted_reciprocal_rank` | The method is public, but reaching it requires `EnsembleRetriever(retrievers=[])` — an empty retriever built solely to borrow a method. Ranking is the product's core quality; a library upgrade should not be able to reorder results silently. |
| Three stages | One commit | A behaviour change and a structural change in the same diff cannot be attributed when results move. Same reason the rewrite precedes the LangChain upgrade. |
| Keep `metadata_info.py` | Delete 339 dead lines | It is not dead: the evaluator and the baseline harness still construct SelfQuery. Deleting it would break the tool that measures Stage 2. |
| Keep multi-query expansion | Remove it too | D4. Removing two things at once makes any quality change unattributable. Measured separately afterwards. |

## Out of Scope

- The LangChain upgrade itself. This unblocks it; it does not perform it.
- Answer quality evaluation (ragas). This plan changes what reaches the model;
  whether answers improve is measured separately.
- `userguide` retriever — already a plain `BaseRetriever`.
- Token-based budgeting (D3 caveat) and removing expansion (D4 follow-up).
