# Feature Specification: Retriever Rewrite

**Feature Branch**: `spec/retriever-rewrite`

**Created**: 2026-09-08

**Status**: Decisions settled — D1 by Helia, D2–D4 recorded below. Ready for `/speckit-plan`.

**Input**: Rewrite the Reactome retriever: replace the `HybridRetriever` that subclasses `MultiQueryRetriever` with a plain `BaseRetriever`, decide whether `SelfQueryRetriever` is replaced by plain semantic search, and make the context budget caller-supplied.

## Why now

Three things converge on this component:

1. **The LangChain upgrade is blocked by it.** `HybridRetriever` reaches into LangChain internals in five places (below). Those are not public API and do not survive a major version.
2. **Three open PRs are queued behind it** — #116 (FlashRank reranker), #133 (EmbeddingsFilter), and the config work in #112/#151 — because none should be built on a class about to change shape.
3. **The upcoming surfaces need different context budgets.** Chat, chat-alongside-search-results, and analysis summarisation each want a different amount of retrieved material, and the budget is currently a module constant.

## What exists today

Per user message, `HybridRetriever`:

1. **expands** the question into 4 alternates via one LLM call (5 queries with `include_original`)
2. for each of **4 Chroma collections** (`reactions`, `summations`, `complexes`, `ewas`), for each query: BM25 over the collection CSV, and a `SelfQueryRetriever` over the Chroma collection — the latter being one LLM call each, so **20 LLM calls per message** for query construction
3. **fuses** BM25 and vector results as separate ranked lists via LangChain's RRF, per collection
4. **caps** at 10 documents per collection, giving 40 documents (~6,300 tokens)

Steps 3 and 4 were fixed in the week of 2026-09-04 (issues #169, #170); steps 1 and 2 are unchanged.

### The five reaches into LangChain internals

| location | what it does |
|---|---|
| `csv_chroma.py:143` | subclasses `MultiQueryRetriever` |
| `csv_chroma.py:59,144` | overrides a required field to `None` via `SkipJsonSchema` |
| `csv_chroma.py:192-194` | builds a throwaway `MultiQueryRetriever` to steal its `.llm_chain` |
| `csv_chroma.py:200` | assigns `_retrievers` outside pydantic |
| `csv_chroma.py:206` | instantiates `EnsembleRetriever(retrievers=[])` — an empty retriever — to borrow one method |

`retrieve_documents` / `aretrieve_documents` are also overrides of internal methods, not documented extension points.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Retrieval survives a LangChain major version (Priority: P1)

A developer upgrades LangChain. The retriever continues to work, because it depends only on `BaseRetriever`, `Document` and the Runnable interface — the contracts LangChain has held stable across every version — rather than on internals.

**Why this priority**: This is the blocker. The lockfile carries 43 advisories, concentrated in the LangChain line, and they cannot be addressed while the retriever depends on internals.

**Independent Test**: Upgrade LangChain in a branch. The suite passes and `bin/retrieval_baseline compare` shows no change attributable to the retriever's own logic.

**Acceptance Scenarios**:

1. **Given** the rewritten retriever, **When** `mypy` and `pytest` run, **Then** no code references `MultiQueryRetriever`, `EnsembleRetriever` internals, or `SkipJsonSchema` field overrides.
2. **Given** a question, **When** retrieval runs through both `invoke` and `ainvoke`, **Then** both return the same documents in the same order for the same query set.
3. **Given** the RRF fusion, **When** documents tie on score, **Then** the tie-break is defined by the implementation rather than inherited from a library's iteration order.

---

### User Story 2 — The caller states how much context it wants (Priority: P2)

A surface asks for retrieval and specifies its own budget. The chat surface asks for more; a chat answer displayed beside full search results asks for less, because the search results are the comprehensive part; analysis summarisation asks for something different again.

**Why this priority**: Independently valuable and independently testable, but the interface should be settled during the rewrite rather than bolted on. Recorded in #172.

**Independent Test**: Call the retriever twice with different budgets and assert the returned document counts differ accordingly, with no change to module-level state.

**Acceptance Scenarios**:

1. **Given** a caller-supplied budget of N documents per collection, **When** retrieval runs, **Then** at most N documents per collection are returned.
2. **Given** no budget supplied, **When** retrieval runs, **Then** a documented default applies and the behaviour matches today's.
3. **Given** two callers with different budgets in the same process, **When** both retrieve, **Then** neither affects the other.

---

### User Story 3 — Retrieval cost per message is understood and reducible (Priority: P3)

An operator can see, and change, how many LLM calls a single message costs in retrieval.

**Why this priority**: Real but not blocking. Worth settling while the code is open, since the answer determines whether `SelfQueryRetriever` stays.

**Independent Test**: Count LLM calls for one message before and after; assert the number matches what the configuration implies.

**Acceptance Scenarios**:

1. **Given** a question, **When** retrieval runs, **Then** the number of LLM calls is derivable from the number of collections, query variants, and whether self-querying is enabled.
2. **Given** self-querying is disabled, **When** retrieval runs, **Then** no LLM call is made for query construction.

### Edge Cases

- No embeddings bundle installed — must fail with an actionable message, not `None` flowing into a `Path` parameter (current `B008` / mypy baseline).
- A collection whose documents carry no `st_id` — de-duplication must fall back rather than raise.
- A bundle whose collections were built with different embedding models — already reported by `resolve_embedding_model`; retrieval must not silently proceed.
- Zero results from one retriever for a query — fusion must not divide by zero or drop the other retriever's results.
- A budget of 0 or larger than the corpus.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The retriever MUST implement `langchain_core.retrievers.BaseRetriever` via `_get_relevant_documents` / `_aget_relevant_documents` and MUST NOT subclass `MultiQueryRetriever`.
- **FR-002**: Reciprocal Rank Fusion MUST be owned by this repository rather than borrowed from `EnsembleRetriever`, and its constant, tie-break and de-duplication key MUST be explicit and tested.
- **FR-003**: BM25 and plain semantic results MUST be fused as separate ranked lists
  (preserving the #170 fix), and the BM25-vs-vector weighting MUST be adjustable --
  today it is `[1/n]*n`, which cannot affect ordering at all.
- **FR-004**: Results MUST be de-duplicated per Reactome entity, falling back to page content where no `st_id` exists (preserving the #169 fix).
- **FR-005**: The context budget MUST be supplied per call, counted in documents per
  collection (D2, D3), with a documented default matching today's behaviour.
- **FR-006**: The embeddings bundle MUST be an explicit argument, not resolved at import time in a default argument.
- **FR-007**: Synchronous and asynchronous paths MUST return identical results for identical inputs, and MUST be tested together.
- **FR-008**: Behaviour MUST be measured with `bin/retrieval_baseline` before and after,
  on the committed question set.
- **FR-009**: The vector retriever MUST NOT make an LLM call (D1). Retrieval MUST make
  exactly one LLM call per message, for query expansion (D4).

### Key Entities

- **Collection** — one Chroma directory plus its BM25 source CSV (`reactions`, `summations`, `complexes`, `ewas`, and `userguide` on its own retriever).
- **Document** — a `langchain_core.documents.Document`, identified by `st_id` where present.
- **Budget** — how many documents a caller wants, per collection.

## Decisions

### D1 — `SelfQueryRetriever` is replaced by plain semantic search — DECIDED (Helia)

BM25 stays; the vector side becomes `vectordb.as_retriever(search_kwargs={"k": ...})`
with no LLM in the loop.

**Consequences, stated plainly because this is not a free simplification:**

- **LLM calls in retrieval drop from 21 to 1 per message** — one expansion call,
  down from one expansion plus 4 collections x 5 query variants of self-querying.
- **It removes the component most likely to break on LangChain 1.x.** SelfQuery
  depends on `lark` and on structured-output behaviour that has moved between
  versions.
- **Retrieval results change materially.** #171 measured SelfQuery as *stable run
  to run*, agreeing with plain vector on 0.48 of documents and 0.19 of positions.
  This is a real change in what reaches the model, not the removal of noise.
- **Metadata filtering is lost as a capability**, not merely as code. SelfQuery
  translated a question into a Chroma metadata filter; nothing replaces that.
- **339 lines across the `metadata_info.py` files become dead in the retrieval
  path**, still referenced only by `src/evaluation/evaluator.py` and
  `bin/retrieval_baseline`, which construct SelfQuery for comparison. They should
  not be deleted while those still need them.

**Verification**: recapture #171's numbers against current `main` first. They
predate the #169 and #170 fixes, which changed what the vector side returns, so
the 0.48/0.19 figures describe a retriever that no longer exists.

### D2 — The budget is per collection — DECIDED

Not global. The decisive reason is technical rather than editorial: each
collection is fused independently, so an RRF score of 0.03 in `reactions` and
0.03 in `summations` are not comparable quantities. A global top-N would sort by
a comparison that has no meaning.

The secondary reason is coverage: `reactions` holds mechanism, `summations` holds
prose description, `complexes` composition, `ewas` protein identity. A global cap
could return 40 reactions and no descriptions. Measured today, the per-collection
cap yields an even 10/10/10/10 split.

Revisit only if the collections are ever fused into one ranked list, at which
point the scores would be comparable and this reasoning would no longer hold.

### D3 — The budget is counted in documents — DECIDED, with a caveat

Tokens are the better unit in principle, and #139 was right to say so. But the
measurement does not currently justify the complexity: 40 documents span
5,796-9,531 tokens, a 1.6x spread, at roughly 7% of the context window. A token
ceiling would tighten a bound that is not binding.

Documents are also the unit a caller can reason about when asking for "less"
alongside search results.

**Caveat**: keep it a single named parameter so a token ceiling can be added as a
second bound without changing the interface. `tiktoken` is already in the
lockfile via `langchain-openai`, so that costs no new dependency when it is
wanted.

### D4 — Multi-query expansion stays, for now — DECIDED

Keep it, and measure it separately afterwards.

The reason is not that it is known to be valuable — its value has never been
measured — but that D1 already changes retrieval materially. Removing expansion
in the same change would make any quality difference impossible to attribute.
This is the same principle as rewriting before upgrading LangChain: change one
variable at a time.

It is also cheap in relative terms once D1 lands: with self-querying gone,
expansion is the *only* LLM call in retrieval. Its real cost is multiplying
retrieval work by five, which is local Chroma and BM25 work rather than API
calls.

**Follow-up**: measure expansion on its own after D1, using
`bin/retrieval_baseline` with and without it.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: No reference to LangChain internals remains in the retriever; the LangChain upgrade proceeds without further retriever changes.
- **SC-002**: `bin/retrieval_baseline compare` before and after shows only differences attributable to a decision recorded above, not to incidental refactoring.
- **SC-003**: Sync and async return identical documents for the same query set, asserted by a test.
- **SC-004**: Two callers with different budgets get different amounts of context in the same process.
- **SC-005**: Retrieval makes exactly one LLM call per message, down from 21, asserted
  by a test that counts them.
- **SC-006**: The four mypy baseline entries for `retrievers.*.rag` -- reactome, uniprot,
  plantreactome and userguide -- are deleted, not carried forward. All four share the
  same cause: `EmbeddingEnvironment.get_dir()` resolved in a default argument (FR-006).

## Assumptions

- The bundle format is unchanged: `<provider>/<model>/<database>/<version>` containing per-collection Chroma directories and a sibling `csv_files/`.
- BM25 stays. Helia's proposal keeps it, it is deterministic, and it costs no LLM call.
- Answer quality is judged by the ragas work, not by this spec. This spec changes *what reaches the model*; whether that improves answers is measured separately.
- The rewrite happens before the LangChain upgrade, so that a behaviour change and an upgrade change cannot be confused for one another.
