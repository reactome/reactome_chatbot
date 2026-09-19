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

## What routing actually buys, measured 2026-09-19

The classifier is wired up and choosing. Three measurements, and the second one
is not the result the feature was pitched on.

### It holds the acceptance bar

`answer-sweep` with the classifier selecting collections: **13/13, the control
score**, two `needs_live` skips. Five of six retrievals were narrowed, and the
routing is the mapping the earlier measurement predicted:

| question | routed to |
|---|---|
| UniProt accession for TP53 | `ewas` |
| Selective autophagy summary | `summations` |
| ABCA1 variants | `disease_variants` |
| PTEN variant diseases | `disease_variants` |
| What CDK5 phosphorylates | `ewas`, `summations` |
| How TP53 regulates PTEN transcription | *all five* -- left empty, correctly |

Only six of the thirteen questions retrieve at all: the refusals never do, and
the userguide questions use a different bundle. So the routing evidence is six
questions, which is a small base and should be said rather than glossed.

### It does not reduce the prompt

This is the correction. Narrowing was expected to cut context; it does not.

| question | docs, all | docs, narrowed | chars, all | chars, narrowed |
|---|---|---|---|---|
| TP53 accession | 10 | 10 | 2,436 | 2,436 |
| ABCA1 variants | 10 | 10 | 6,163 | 5,846 |
| Selective autophagy | 10 | 10 | 22,114 | 21,797 |

**Ten documents either way, context within 2%.** The chain caps the fused list
at ten regardless of how many collections fed it, so narrowing changes *which*
documents arrive, not how many. Any claim that this feature reduces prompt cost
is wrong, and T003's framing -- "context tokens for a question needing one
collection fall" -- was the wrong thing to measure.

### What it does reduce is retrieval work

Median of three runs per question, timed inside `aretrieve_documents`:

| | all five | narrowed |
|---|---|---|
| median retrieval | **1.82s** | **1.43s** |

About 21%, or roughly 0.4s against a first token near ten seconds. Real, and
modest. Five collections mean ten sub-retrievals (BM25 and vector each); one
collection means two.

### So the honest case for this feature

It did not fix a failing question -- the control already passed 13/13. It buys
a fifth off retrieval time and it puts the right documents in a fixed-size
context, which should matter most where the fixed ten are currently crowded out
by the wrong collection. That last part is plausible and **not** measured here.

Against that, a wrong narrow removes an answer rather than degrading it, which
is why the prompt leans hard on leaving the selection empty and why every
failure path in `resolve_collections` widens. The sweep shows correct routing
on six questions; it does not show that the classifier is right in general.

`complexes` still has no guard question (T005), so a classifier that never
routes to it would score full marks.

### Adversarial check: nine questions the sweep never asks

Six retrieving questions is a thin base to ship a router on, so the classifier
was probed with questions chosen to be awkward -- broad ones that must *not*
narrow, and cross-collection ones where a single choice loses half the answer.

| question | chose |
|---|---|
| What is apoptosis? | *all* |
| Explain the role of TP53 in the cell cycle | *all* |
| Tell me everything Reactome knows about ferroptosis | *all* |
| What complexes contain TP53? | `complexes` |
| Which proteins are in the MCM complex and what do they do? | `complexes`, `summations` |
| What is the UniProt ID for BRCA1? | `ewas` |
| Which variants of BRCA1 cause disease, and the mechanisms? | `disease_variants`, `summations` |
| Inputs and outputs of the CDK1/MCM2 phosphorylation | `reactions` |
| How does Reactome describe the Wnt signalling pathway? | `summations` |

Nine of nine as intended: every broad question left open, every specific one
narrowed to the collection that holds the answer, and both cross-collection
questions took `summations` alongside their primary -- which is what the prompt
asks for and the reason the mechanism half of those questions survives.

This is nine single classifications, not nine end-to-end answers, so it shows
the routing decision is sound and not that the answers are.

**A hallucinated collection name is safe by construction**, and deliberately so:
`collections` is `list[str]` rather than an enum of the five names. A name that
is not in the bundle is widened to all by `resolve_collections` with a WARNING,
which is the failure direction this feature requires. Constraining the schema
would instead make structured output reject the response, turning a harmless
mistake into a failed answer.

## T005: `complexes` cannot be guarded by asking a question either

Attempted properly on 2026-09-19 rather than left as "no candidate found". It
failed, and the reason is structural and worth writing down, because it is the
same reason `reactions` failed and it was not obvious the second time.

**Method.** Removing `complexes` from what `resolve_collections` sees is an
exact simulation of a bundle without it -- the name then widens to all the
others, as it would there. This replaced copying a 3.3G bundle.

**A false start worth recording.** The first attempt set the
`selected_collections` ContextVar around the call. That no longer works now the
classifier is wired: `generate_answer` sets it from the classifier's choice and
overwrites anything set outside, so *both arms searched only `complexes`* and
the result was void. It was caught by instrumenting what retrieval actually
searched, which is the second time in one day that check has saved a
measurement.

**Composition questions are too variable to build a guard on.** Asking which
proteins make up a complex requires naming four to seven things, and the model
names a different subset each run. The same configuration, three runs:

| question | complexes only | complexes+summations | all five |
|---|---|---|---|
| Nup107 components (of 4) | 2, 1, 1 | 2, 1, 2 | 2, 1, 1 |
| U7 snRNP subunits (of 7) | 2, 1, 1 | 2, 2, 2 | 6, 2, 2 |

An earlier four-run pass appeared to show `complexes` making the Nup107 answer
*worse* -- 1,1,1,1 against 3,3,3,3 without it. It did not survive being run
again. That claim was under-powered and is withdrawn; the variance is larger
than any difference between the arms, which is exactly the trap
[[probabilistic-bugs-need-sized-tests]] describes.

Every guard that works is a **single stable token**: `P04637`, `Tangier`,
`lysosom`. So the question was inverted to ask for one -- name the components,
ask which complex holds them:

| question | with `complexes` | without |
|---|---|---|
| Which complex contains CYBA and CYBB? | 0/4 | 0/4 |
| Which complex has NUP133, NUP160, NUP37? | 4/4 | **4/4** |
| Which complex is made of LSM10, LSM11, SNRPB? | 3/4 | **4/4** |

**Answered just as well without it.** Complex *names* appear throughout
`reactions`, as the names of inputs and outputs, and throughout `summations`
prose. So any question naming or seeking a complex is answerable from those,
and the only thing structurally unique to `complexes` -- the component list --
is the thing whose answers are too variable to assert on.

### What this changes

`complexes` joins `reactions`: a collection whose content is duplicated
elsewhere cannot be guarded by asking a question, however the question is
worded. T005 should become what T007 already is -- a retrieval-level assertion
that the collection was searched -- rather than a hunt for a better candidate,
which is now two failed hunts and a structural explanation of why.

Until that exists, **the sweep cannot detect a classifier that never routes to
`complexes`**, and that remains the strongest reason not to deploy routing on
the strength of a green sweep alone.
