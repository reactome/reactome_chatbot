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

## What narrowing actually costs, measured 2026-09-19

Two measurements, and only the second means anything.

**Citation overlap is arithmetic, not signal.** Narrowing to two collections
keeps 5-7 of the full top-12 citations; narrowing to one keeps 2. That looks like
a large loss, but reciprocal rank fusion interleaves the five per-collection
lists, so a top-12 draws about 2.4 from each. Keeping two lists predicts ~4.8,
and 5-7 is what was observed. The number measures the interleave, not whether
anything useful was lost. Do not use it as a quality measure.

**Whether answers survive is the measurement that counts.** Re-measured
2026-09-19 through `answer_sweep.run()` itself, after the first attempt
reimplemented the matching and got the number wrong (see the correction below).
Two skips on this host, both `needs_live`, in both arms.

| | all five collections | narrowed to `reactions` + `summations` |
|---|---|---|
| passed | **13 / 13** | **10 / 13** |
| failed | 0 | 3 |
| skipped | 2 | 2 |

Two of the three failures are real losses, and they name their collection:

| question | needs | what the narrowed answer said |
|---|---|---|
| "What is the UniProt accession for the TP53 protein" (`P04637`) | `ewas` | "not explicitly provided in the context searched" |
| "Which diseases involve variants of the PTEN gene" | `disease_variants` | pathway-level prose, no variant named |

**The third failure was not a loss -- the check was wrong.** Narrowed, and with
`disease_variants` excluded, the ABCA1 question named all six curated variants
(`C1417R`, `Q537R`, `S1446L`, `N935S`, `W590S`, `R587W`) with the OMIM id. The
expectation failed it because the pattern required `ABCA1 C1417R` adjacency and
the answer rendered them as a numbered list of `**C1417R**`. The pattern is fixed
and pinned by a test; the answer was correct all along.

That has a consequence for this feature: **ABCA1's variants are reachable from
`summations` prose**, so that question does not guard `disease_variants`. Only
the PTEN question does.

### The correction

The first measurement reported 12/15. It was produced by a throwaway script that
reimplemented the sweep's matching, and it was wrong twice over: it never read
the `must` field, which nine of the fifteen expectations carry, and it counted
the two `needs_live` questions as passes where the sweep skips them. Both errors
push the number up. The re-run uses `run()` and `report()` from
`src/evaluation/answer_sweep.py` and records which collections retrieval actually
searched, so the run proves its own precondition -- the first void re-run
searched nothing at all and would otherwise have been read as a result.

### What this settles

**Collections are not interchangeable, and the mapping is legible.** Accession
questions need `ewas`; PTEN variant questions need `disease_variants`. That is
the signal a classifier can be prompted on.

It is a weaker result than the first measurement suggested, and the weakening is
the useful part: content is duplicated across `summations` more than the guard
table assumed. `reactions` was already known to be unguardable by answer because
`summations` covers `Pathway OR ReactionLikeEvent`; `disease_variants` now joins
it for at least one question.

**One excluded collection is untested by construction.** Narrowing excluded
`complexes`, `ewas` and `disease_variants`. No tracked question guards
`complexes` (T005 is still open), so its exclusion could not have produced a
failure. The absence of a fourth failure is not evidence.

**It still justifies the fail-wide rule.** Where a narrow selection does lose the
answer, it removes it rather than degrading it -- the TP53 answer says the
accession is not there, and the PTEN answer falls back to pathway prose with no
variant. Widening on uncertainty costs latency; narrowing wrongly costs the
answer.

**And it sets the acceptance bar.** Routing is worth shipping only if the sweep
stays at the control's score with the classifier choosing -- 13/13 here, 15/15 in
the container where MCP is configured. The TP53 and PTEN questions are the ones
to watch, because they fail loudly and specifically. A classifier that never
routes to `complexes` would still score full marks, which is why T005 matters.
