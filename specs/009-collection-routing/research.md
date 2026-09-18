# Phase 0 Research: Collection Routing

## R1. How does a per-question selection reach a retriever built once?

**The problem, from the code rather than assumed.** `ReactToMeProfile.__init__` builds
the RAG chain once and holds it in `self.rags["reactome"]`. Inside,
`create_retrieval_chain(retriever=..., combine_docs_chain=...)` closes over a single
`HybridRetriever` instance. The retriever's entry point is
`_get_relevant_documents(query: str, *, run_manager)` -- a string, with no room for
"and search only these collections".

Rebuilding per question is not an option: `HybridRetriever.from_subdirectory` reads
every collection's CSV and constructs a `BM25Retriever` over it. For `reactions` alone
that is 17,004 documents tokenised with `word_tokenize`. That is startup work, not
per-message work.

**Options considered**

| Option | How | Why not / why |
|---|---|---|
| A. `RunnableConfig["configurable"]` | The profile calls `rag.ainvoke(..., config)`; the retriever reads `config["configurable"]["collections"]` | `config` already flows to the retriever -- `postprocess` reads `config["configurable"].get("enable_postprocess")` today, so the mechanism is in use in this codebase already |
| B. `ConfigurableField` | Declare the field configurable and bind per call | Works, but pydantic-configurable retrievers re-validate on every bind; A gets the same result with machinery already present |
| C. contextvar | Set around the call | Invisible coupling, and wrong under concurrency if it ever leaks across tasks |
| D. Encode in the query string | Prefix the question | Reaches BM25 and the query expander as text. The codebase already carries a comment warning against exactly this for `detected_language` |

**Decision: A.** It reuses a path this repo already relies on and adds no new
LangChain surface.

**Rationale.** `src/agent/profiles/base.py` reads `config["configurable"]` in
`postprocess`, so configurable state is established practice here, and the failure
mode is a missing key -- which defaults to "all collections" and is therefore safe.

## R2. Can the classifier name collections reliably?

Unknown, and the plan does not depend on it being perfect. The measurement decides.

What is known: the same call already routes `reactome` / `userguide` / `live`
correctly for the thirteen sweep questions, and sharpening one rule on 2026-09-17
fixed the variant questions without breaking the live ones. A per-collection
description already exists for each collection in `reactome_descriptions_info`.

**Decision**: ask for collections in the same structured output, and treat an empty
or unparseable list as "all". Measure with `bin/retrieval_baseline` before deciding
whether it is good enough.

## R3. What happens when the classifier names a collection that does not exist?

It means the prompt and the bundle disagree -- a new collection added without a
description, or a renamed one. Principle IV says configuration that cannot be
honoured must not substitute something plausible.

**Decision**: log at WARNING naming the unknown collection, and fall back to searching
**all** collections for that question.

**Rationale.** The alternatives are worse. Dropping the unknown name silently narrows
the search for a reason nobody can see. Raising takes the chatbot down for what is a
recoverable prompt/bundle mismatch. Falling back to all is the only option whose
failure mode is today's behaviour.

The selectable set is derived from `list_chroma_subdirectories()` on the live bundle,
so "does not exist" is decided against what is actually installed, not a literal list
that would itself drift (Principle V).

## R4. What is the baseline, and when is it captured?

Captured **before** any code change, or there is nothing to compare against.

Already measured on 2026-09-17, on *"What does CDK5 phosphorylate in Alzheimer
disease?"*, a question needing no variant data:

| | docs | context tokens | retrieval |
|---|---|---|---|
| 4 collections | 40 | 7,061 | 11.5s |
| 5 collections | 50 | 9,437 | 14.9s |

And the inversion: `disease_variants` contributed **15%** of context to that question
and **5%** to the ABCA1 question it exists for, because every collection gets ten
documents regardless of quality.

**Decision**: `bin/retrieval_baseline capture` over the fixed question set on the
Release97 bundle is task 1, before anything else.

## R5. Which path must be measured?

The served path is `aretrieve_documents`; `bin/retrieval_baseline` drives
`retrieve_documents`. They are separate implementations and were verified equivalent
on 2026-09-17 (PR #227), so measuring the sync path is currently valid.

**Decision**: the equivalence test is a prerequisite of this work, not a nice-to-have.
Both paths must filter by the same selection, and the existing test must be extended
to assert that -- otherwise the measurement stops describing what users get.

## The config route the data model describes does not exist

**Measured 2026-09-18** against langchain-core 0.2.14, the version pinned here.

data-model.md routes the selection as
`RunnableConfig["configurable"]["collections"]` into
`HybridRetriever.retrieve_documents`. That is not reachable:

- config is **not** passed to `_get_relevant_documents` as a kwarg — a retriever
  defined with `**kwargs` receives an empty dict
- `var_child_runnable_config` is **unset** inside a retriever run, so the ambient
  config cannot be read either

Two alternatives were rejected. A per-request attribute on the retriever races,
because it is built once at startup and shared by every request. Rebuilding a
filtered retriever per request throws away the constructed BM25 indexes.

**Decision: a module-level `ContextVar`**, set around the retrieval call. Under
asyncio each task gets a copy of the context, so one request's narrowing cannot
narrow another's — pinned by a test that runs a narrow and a wide request
concurrently and asserts neither sees the other.

The data model's diagram is left as the intent; this is how it is carried.

