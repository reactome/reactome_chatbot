# Feature Specification: Retriever Rewrite

**Feature Branch**: `spec/retriever-rewrite`

**Created**: 2026-09-08

**Status**: Historical record of the retriever rewrite, 2026-09-08 to 2026-09-09 (#182, #183, #185). The decisions, requirements and scope below were set for that work and are not requirements for later changes. Where the code has changed since, the section says so. Current behaviour is in `src/retrievers/csv_chroma.py`. The outcome, including two predictions that were wrong, is recorded at the end of this file.

**Input**: Rewrite the Reactome retriever: replace the `HybridRetriever` that subclasses `MultiQueryRetriever` with a plain `BaseRetriever`, decide whether `SelfQueryRetriever` is replaced by plain semantic search, and make the context budget caller-supplied.

## Why the rewrite was done (2026-09-08)

Three things converge on this component:

1. **The LangChain upgrade is blocked by it.** `HybridRetriever` reaches into LangChain internals in five places (below). Those are not public API and do not survive a major version.
2. **Three open PRs are queued behind it** — #116 (FlashRank reranker), #133 (EmbeddingsFilter), and the config work in #112/#151 — because none should be built on a class about to change shape.
3. **The upcoming surfaces need different context budgets.** Chat, chat-alongside-search-results, and analysis summarisation each want a different amount of retrieved material, and the budget is currently a module constant.

## The retriever before the rewrite (2026-09-08)

Per user message, the pre-rewrite `HybridRetriever`:

1. **expands** the question into 4 alternates via one LLM call (5 queries with `include_original`)
2. for each of **4 Chroma collections** (`reactions`, `summations`, `complexes`, `ewas`), for each query: BM25 over the collection CSV, and a `SelfQueryRetriever` over the Chroma collection — the latter being one LLM call each, so **20 LLM calls per message** for query construction
3. **fuses** BM25 and vector results as separate ranked lists via LangChain's RRF, per collection
4. **caps** at 10 documents per collection, giving 40 documents (~6,300 tokens)

Steps 3 and 4 were fixed in the week of 2026-09-04 (issues #169, #170); steps 1 and 2 were unchanged at the time.

### The five reaches into LangChain internals (line numbers refer to the file before #182)

| location | what it does |
|---|---|
| `csv_chroma.py:143` | subclasses `MultiQueryRetriever` |
| `csv_chroma.py:59,144` | overrides a required field to `None` via `SkipJsonSchema` |
| `csv_chroma.py:192-194` | builds a throwaway `MultiQueryRetriever` to steal its `.llm_chain` |
| `csv_chroma.py:200` | assigns `_retrievers` outside pydantic |
| `csv_chroma.py:206` | instantiates `EnsembleRetriever(retrievers=[])` — an empty retriever — to borrow one method |

`retrieve_documents` / `aretrieve_documents` are also overrides of internal methods, not documented extension points.

## What the rewrite set out to do

Three goals, in priority order: retrieval that depends only on `BaseRetriever`,
`Document` and the Runnable interface, so a LangChain major upgrade does not break
it; a context budget the caller sets; and a per-message LLM-call cost that is
understood and reducible. The Outcome section records how each landed.

## Requirements of the rewrite, and where they stand (checked 2026-10-21)

| ID | Required by the rewrite | Today |
|---|---|---|
| FR-001 | Plain `BaseRetriever`, not a `MultiQueryRetriever` subclass | Holds (`csv_chroma.py:387`) |
| FR-002 | RRF owned here; constant, tie-break and de-dup key explicit and tested | Holds: `reciprocal_rank_fusion`, `RRF_K = 60`, ties by first appearance, keyed on `page_content` (`csv_chroma.py:218-250`) |
| FR-003 | BM25 and vector fused as separate lists; BM25-vs-vector weighting adjustable | Separate lists hold (`csv_chroma.py:610-611`). Adjustable weighting was not built: `weighted_reciprocal_rank` always uses equal weights (`csv_chroma.py:570-574`) |
| FR-004 | De-duplicate per entity, falling back to page content | Holds: `dedupe_by_entity` (`csv_chroma.py:199`) |
| FR-005 | Budget supplied per call, in documents per collection | Built as a constructor argument, `max_documents_per_collection` (default 10), not per call. No current caller passes one, so every retriever uses the default (`csv_chroma.py:366-379`) |
| FR-006 | Bundle passed explicitly | Holds (`embeddings_directory` parameter) |
| FR-007 | Sync and async identical, tested together | Holds (`tests/retrievers/test_sync_async_equivalence.py`) |
| FR-008 | Measured with `bin/retrieval_baseline` before and after | A process requirement for the rewrite itself |
| FR-009 | No LLM call on the vector side; exactly one per message, for expansion | The vector side holds. One call per message by default; none when `QUERY_EXPANSION_ALTERNATES=0`, added 2026-09-20 (see D4) |

### Key Entities

- **Collection** — one Chroma directory plus its BM25 source CSV. The retriever
  searches whatever collections the installed bundle has. At the time of the
  rewrite, the Reactome bundle had `reactions`, `summations`, `complexes` and
  `ewas`; `disease_variants` was added later (spec 008). `userguide` has its own
  retriever.
- **Document** — a `langchain_core.documents.Document`, identified by `st_id` where present.
- **Budget** — how many documents a caller wants, per collection.

## Decisions

### D1 — `SelfQueryRetriever` replaced by plain semantic search (decision for the rewrite, 2026-09-08)

BM25 was kept; the vector side became `vectordb.as_retriever(search_kwargs={"k": ...})`,
with no LLM in the loop.

**Consequences, stated plainly because this is not a free simplification:**

- **LLM calls in retrieval drop from 21 to 1 per message** — one expansion call,
  down from one expansion plus 4 collections x 5 query variants of self-querying.
- **It removes the component most likely to break on LangChain 1.x.** SelfQuery
  depends on `lark` and on structured-output behaviour that has moved between
  versions.
- **Retrieval results change materially.** Recaptured 2026-09-08 against current
  `main`, with both retrievers at `k = 10 x VECTOR_OVERFETCH` collapsed to 10
  distinct entities -- the configuration the pipeline actually uses:

  | collection | set overlap | rank agreement |
  |---|---|---|
  | complexes | 0.49 | 0.18 |
  | ewas | 0.54 | 0.17 |
  | reactions | 0.58 | 0.14 |
  | summations | 0.57 | 0.24 |
  | **overall** | **0.55** | **0.18** |

  SelfQuery and plain semantic search return **about half the same documents, in
  almost entirely different order**. SelfQuery is stable run to run, so this is a
  real change in what reaches the model, not the removal of noise.
- **Metadata filtering is lost as a capability**, not merely as code. SelfQuery
  translated a question into a Chroma metadata filter; nothing replaces that.
  Still the case: the vector retriever is configured with `k` only
  (`csv_chroma.py:484`).
- **`metadata_info.py` left the retrieval path.** At the time, `src/evaluation/evaluator.py`
  and `bin/retrieval_baseline` still used it to build SelfQuery for comparison.
  Since then, the evaluator calls the shipping pipeline (cf9ad6a), and
  `reactome_descriptions_info` has become a serving input: the intent classifier
  builds its collection list from it (`src/agent/tasks/intent_classifier.py:55-67`),
  and so does `bin/routing-probe`. `bin/retrieval_baseline` still builds SelfQuery
  from it.

**Verification**: done. The figures above supersede the 0.48/0.19 in #171, which
predated the #169 and #170 fixes and described a retriever that no longer exists.
Set overlap moved 0.48 -> 0.55 once both sides were measured at the pipeline's
real configuration; rank agreement was unchanged at ~0.18. The conclusion is
unmoved: this is a substantial change to retrieval, and it should be judged on
answer quality rather than on overlap, which cannot say which set is better.

### D2 — Budget counted per collection (decision for the rewrite)

Not global. The decisive reason is technical rather than editorial: each
collection is fused independently, so an RRF score of 0.03 in `reactions` and
0.03 in `summations` are not comparable quantities. A global top-N would sort by
a comparison that has no meaning.

The secondary reason is coverage: `reactions` holds mechanism, `summations` holds
prose description, `complexes` composition, `ewas` protein identity. A global cap
could return 40 reactions and no descriptions. Measured 2026-09-08, the
per-collection cap yielded an even 10/10/10/10 split.

**Later change (38f8885, 2026-09-18).** The per-collection lists are now also fused
with each other by RRF, so the final list is ranked across collections instead of
grouped by collection. The per-collection cap still applies first: each searched
collection contributes at most `max_documents_per_collection` documents, and the
fused list is cut to that cap times the number of collections searched. The change
therefore reorders documents without dropping any (`csv_chroma.py:612-633`). A
global top-N, which the reasoning above argued against, has not been introduced.

### D3 — Budget counted in documents (decision for the rewrite)

Tokens are the better unit in principle, and #139 was right to say so. But the
measurement did not justify the complexity: 40 documents spanned 5,796-9,531
tokens (measured 2026-09-08, four collections), a 1.6x spread, at roughly 7% of
the context window. A token ceiling would have tightened a bound that was not
binding.

Documents are also the unit a caller can reason about when asking for "less"
alongside search results.

The rewrite kept the budget as one named parameter so that a token ceiling could
be added later as a second bound without changing the interface. None has been
added. `tiktoken` is in the lockfile.

### D4 — Multi-query expansion kept during the rewrite

The rewrite left expansion unchanged and left measuring it to a separate change.

The reason is not that it is known to be valuable — its value has never been
measured — but that D1 already changes retrieval materially. Removing expansion
in the same change would make any quality difference impossible to attribute.
This is the same principle as rewriting before upgrading LangChain: change one
variable at a time.

With self-querying gone, expansion became the *only* LLM call in retrieval. Its
other cost is multiplying retrieval work by five, which is local Chroma and BM25
work rather than API calls.

**Follow-up, done 2026-09-20 (spec 010, T020).** Over six tracked questions, the
expansion call cost about 1.27s whatever it returned, so fewer alternates save
only fan-out (table in `csv_chroma.py:89-112`). The count became configurable
through `QUERY_EXPANSION_ALTERNATES` (default 4; 0 skips the call). The default
stayed at 4. That measurement covered the tracked questions only and does not
establish that recall is unaffected in general.

**How expansion works now:** one LLM call asks for exactly four alternates
(`multi_query_prompt`, `csv_chroma.py:131`). The retriever uses the first N, where
N is the configured count, and the Reactome retriever appends the original
question last (`include_original=True`, `csv_chroma.py:377`; `_queries`, `:536`).
A count above four does not add queries.

## Assumptions at the time (2026-09-08)

- The bundle format is unchanged: `<provider>/<model>/<database>/<version>` containing per-collection Chroma directories and a sibling `csv_files/`.
- BM25 was kept: it is deterministic and costs no LLM call.
- Answer quality is judged by the ragas work, not by this spec. This spec changes *what reaches the model*; whether that improves answers is measured separately.
- The rewrite was to precede the LangChain upgrade, so a behaviour change and an upgrade change could not be confused.


---

## Outcome (2026-09-09)

All three stages shipped. `HybridRetriever` is a plain `BaseRetriever`;
`SelfQueryRetriever` is gone from the pipeline; the bundle and the context budget
are constructor arguments. Retrieval makes **one LLM call per message, down from
21**. The five reaches into LangChain internals are zero, so the upgrade is
unblocked — which was the point.

Suppressions deleted rather than annotated: four `retrievers.*.rag` mypy baseline
entries and four `B008` noqa comments.

| stage | PR | what landed |
|---|---|---|
| 1 | #182 | plain `BaseRetriever`, RRF vendored, byte-identical results |
| 2 | #183 | plain semantic search replaces SelfQuery |
| 3 | #185 | `require_dir` for the bundle, budget as a constructor argument |

### Two predictions in the plan were wrong

Recorded because the point of this file is that the next person does not repeat
them.

1. **"Stage 1 must show *zero* difference"** was unfalsifiable as written. Chroma's
   ANN search has a noise floor — two identical runs already differed on 2 of 8
   question-collections — so "zero" could never have been observed, and a real
   regression of that size would have been invisible against it. An exit criterion
   has to be stated against measured noise, not against an ideal.

2. **"Stage 2 becomes deterministic"** did not happen. Removing the LLM from the
   vector side removed *that* source of variance; the ANN variance underneath it
   stayed. Removing one of two causes does not make an effect go away, and the plan
   asserted it would without checking which cause dominated.

### One thing this rewrite did not do

It changed what reaches the model four times over — over-fetch, de-duplication,
genuine BM25/vector fusion, and plain semantic search — and **none of those changes
has been evaluated for answer quality**. The rewrite's plan said so under Out of
Scope, which made it a known gap rather than an oversight, but it is still a gap.

At the time, `src/evaluation/evaluator.py` could not measure the shipping pipeline:
it built its own `SelfQueryRetriever` + `EnsembleRetriever` + `MergerRetriever`. It
was changed on 2026-09-09 to call `create_reactome_rag` (cf9ad6a; see
`src/evaluation/README.md`). This repository does not record whether an
answer-quality evaluation of the four retrieval changes has been run since.

### Alternatives rejected during the rewrite

| Choice | Simpler alternative rejected | Why |
|---|---|---|
| Vendor RRF (~15 lines) | Keep calling `EnsembleRetriever.weighted_reciprocal_rank` | The method is public, but reaching it required `EnsembleRetriever(retrievers=[])` -- an empty retriever built solely to borrow a method. Ranking is the product's core quality; a library upgrade should not be able to reorder results silently. |
| Three stages | One commit | A behaviour change and a structural change in the same diff cannot be attributed when results move. Same reason the rewrite preceded the LangChain upgrade. |
| Keep `metadata_info.py` | Delete it | The evaluator and the baseline harness still built SelfQuery from it, so deleting it would have broken the tool that measured Stage 2. D1 records how its use has changed since. |
| Keep multi-query expansion | Remove it too | D4. Removing two things at once makes any quality change unattributable. |
