# Feature Specification: The Answer Cascade

**Feature Branch**: `spec/answer-cascade`

**Created**: 2026-09-14

**Status**: Draft. One decision (D1) for the team, and it is the hard one.

**Input**: Detect when a question wants an analysis. Otherwise check the embeddings;
if they have no match, search the Reactome Search API for the term; if that finds
nothing, fall back to Tavily.

## The shape

```
                    ┌─ analysis question? ──> MCP analysis agent
question ──> detect ┤
                    └─ otherwise ──> embeddings ──> Search API ──> Tavily
                                       (no match)     (0 entries)
```

Two different mechanisms, and the distinction matters: **analysis is a branch taken
up front**, because it is a different kind of question. The rest is a **cascade**,
each step tried only when the one before it found nothing.

## Half of it already exists

| step | today |
|---|---|
| classify the question | `intent_classifier` — routes `reactome` / `userguide` |
| embeddings | the RAG chain |
| "was that good enough?" | `completeness_grader` |
| Tavily | `tavily_wrapper`, run in `postprocess` |
| **Reactome Search API** | **nothing** |
| **analysis routing** | **nothing** |

So the cascade's skeleton is real: retrieval, a grader that decides whether the
answer was good enough, and a web search when it was not. What is being asked for is
one new step inserted in the middle, and one new branch at the top.

## The hard part is the signal, not the order

Each arrow needs an answer to "did this find anything?", and the three are not
equally easy.

### Search API — free and unambiguous

Measured against `ContentService/search/query`:

| query | entries |
|---|---|
| `PALB2` | 39 |
| `Impaired BRCA2 binding to PALB2` | 89 |
| `zzzznotathing` | **0** |

The result count *is* the signal. No model call, no threshold to tune, no judgement.
This step is the cheapest in the cascade and the easiest to get right.

### Embeddings — the genuinely hard one

"If it does not find a match" has no obvious definition, because similarity search
*always* returns its k nearest documents. There is no empty result; there is only a
result that happens to be irrelevant.

Two candidate signals, and they behave differently:

**A distance threshold.** Free and instant, but the number is arbitrary and drifts
with the embedding model. A threshold tuned on Release95 and `text-embedding-3-large`
means nothing after either changes, and nothing warns you.

**The completeness grader**, which already exists and already does this job for the
Tavily step. It judges the generated answer rather than the retrieved distance, which
is the question actually being asked — but it costs a model call and requires
generating an answer before discovering it was not worth generating.

This is D1, and it is the only decision here that is not obvious.

### Analysis — a classifier, and there is one already

Detecting "this wants an analysis" is classification, and `intent_classifier` already
classifies. It should gain the destination rather than acquire a sibling, which is
the mistake #142 makes: a second LLM classification call on the same question, in a
pipeline where cutting calls from 21 to one was the point of spec 001.

Analysis questions look different enough to be tractable — they carry a list of
identifiers, and ask what is enriched or over-represented.

## What it costs

Worst case, a question that falls all the way through: classification, retrieval and
generation, a completeness judgement, a Search API call, and a Tavily call. On a
surface already at ~22s per question, each step must earn its place.

Two mitigations are inherent to the design. Most questions stop at the embeddings, so
the deep path is rare. And the two new steps are the cheap ones — the Search API
measured at well under a second, and analysis at **0.2s** measured through
`reactome-mcp`.

## Observed, 2026-09-14

Two questions asked on beta by a Reactome developer. Beta was running `e398a37`
— production's commit, which predates both the intent classifier and the user
guide source — so this is not a report on current `main`. It is recorded because
it is what the cascade is for, asked by a real user in their own words.

### "I want to run a gsea with my list of genes"

> I'm sorry, but the information regarding how to perform a Gene Set Enrichment
> Analysis (GSEA) is not currently available in the Reactome Knowledgebase.

Then three external links: `fgsea` in R, bigomics.ch, and a YouTube tutorial.

Reactome's own analysis service was never consulted. The failure is not that the
answer was empty — it is that a Reactome user asking to analyse a gene list was
sent to somebody else's tools. **This is User Story 1**, in the user's phrasing
rather than the spec's.

### "what tools are available for me to run on here"

> The Reactome Knowledgebase does not currently provide specific information
> about tools available for running analyses on the Human Reactome (HRE).

This one is worse, and in a way worth naming precisely: it is **confidently
wrong**, not unhelpful. Analysis is a flagship Reactome feature. A retrieval
miss was reported as a fact about Reactome, because nothing distinguishes "the
vector store did not have this" from "Reactome does not have this". That is the
"user cannot tell which source answered" edge case, showing up as an assertion
rather than as a gap.

It is also a user guide question. `main` has a user guide source and routes to
it — but the user guide bundle is not installed on the beta host, so `main`
would log `User guide embeddings not configured; routing will use reactome only`
and fall back the same way. **Routing is necessary and not sufficient**; the
source has to be present, and its absence currently degrades silently.

### Reactome does do GSEA — through a service nothing here talks to

An earlier draft of this spec asserted that Reactome performs
over-representation analysis and not GSEA, and that the honest reply was to say
so. **That was wrong**, and it is worth recording because the spec came within
one merge of telling a future implementer to say something false to users.

There are two analysis services:

| | | |
|---|---|---|
| `AnalysisService` | over-representation over an identifier list | what `reactome-mcp` wraps |
| **ReactomeGSA** (`gsa.reactome.org`) | **PADOG, Camera, ssGSEA, terapadog** | **nothing here talks to it** |

Camera is described by the service itself as *"a gene set analysis algorithm
similar to the classical GSEA algorithm"*. Verified 2026-09-14 against
`GET https://gsa.reactome.org/0.1/methods`, and `reactome.org/gsa/` is a live
page.

So the answer to "I want to run a GSEA" is not "Reactome does ORA instead". It
is ReactomeGSA — a Reactome product — and the reason the beta chatbot sent a
user to `fgsea` and a YouTube tutorial is that **no part of this system knows
ReactomeGSA exists**.

That is a gap in `reactome-mcp` before it is a gap in the cascade: routing the
question correctly cannot help while there is no tool behind the route.

### The rephrase turns "do this" into "how do I do this"

Measured 2026-09-14, three runs per question. Production classifies the
**rephrased** question, not the raw one — `preprocess` rephrases first and
passes `rephrased_input` to the classifier — and the rephrase systematically
converts an action request into a how-to question:

| asked | rephrased to |
|---|---|
| I want to run a gsea with my list of genes | How **can I perform** a Gene Set Enrichment Analysis (GSEA) … |
| Analyse these genes for over-representation: TP53 BRCA1 EGFR | How **can I analyze** the over-representation of the genes … |

"How can I…" is a user-guide shape, and it routes accordingly. One question
changes destination because of it: *Analyse these genes for over-representation*
classifies as `reactome` when raw and `userguide` after rephrasing.

This matters for FR-001. The analysis branch cannot be built on the classifier
alone while the step in front of it is rewriting requests-to-act into
questions-about-how — by the time the classifier sees the question, the
signal it needs is gone. **FR-011.**

### What `main`'s classifier actually does with them

Measured 2026-09-14, `gpt-4o-mini`, **three runs per question, classifying the
rephrased form as production does**. Every question routed identically on all
three runs, so the classifier is stable here and single samples below are not
hiding variance:

| question | routes to | wanted |
|---|---|---|
| I want to run a gsea with my list of genes | `userguide` | analysis |
| Which pathways are enriched in TP53, BRCA1, EGFR, MYC, CDKN1A? | `reactome` | analysis |
| Analyse these genes for over-representation: TP53 BRCA1 EGFR | `userguide` (`reactome` if not rephrased) | analysis |
| What do TP53, BRCA1 and EGFR have in common? | `reactome` | `reactome` ✓ |
| **what tools are available for me to run on here** | **`userguide`** | **`userguide` ✓** |
| How do I run a gene list analysis on the Reactome website? | `userguide` | `userguide` ✓ |
| How do I use the pathway browser? | `userguide` | `userguide` ✓ |
| What does CDK5 phosphorylate in Alzheimer's disease? | `reactome` | `reactome` ✓ |
| How does TP53 regulate PTEN transcription? | `reactome` | `reactome` ✓ |
| What is R-HSA-9613829? | `reactome` | `reactome` ✓ |
| What is the current price of a Nature subscription? | `userguide` | neither |

Three things follow.

**The second beta failure is already routed correctly.** "what tools are
available for me to run on here" classifies as `userguide` on `main` today. The
classifier is not the problem for that question; the missing bundle is. Install
it on the beta host and that failure goes away without any cascade work — which
is a cheaper fix than this spec, and should be done first.

**The analysis questions land wherever the two existing destinations allow.**
Two go to `reactome`, one to `userguide` — not wrong given the choices, simply
unable to express the right answer. That is the case for FR-001, measured
rather than argued.

**A question that belongs nowhere is forced somewhere.** The Nature subscription
question routes to `userguide`, because with two destinations and a required
choice there is no way to say "neither". Today the completeness grader catches
this downstream and Tavily answers it, so the outcome is acceptable by accident.
Adding destinations makes the guess worse, not better: the more sources, the
more confident the misrouting. FR-010.

### Where these questions live now

`tests/golden/cascade-questions.txt`, alongside the cases for every other branch
of the cascade — including two that must be answered from the embeddings exactly
as they are today, and one identifier that should stop at the Search API rather
than reach Tavily.

They are kept out of `tests/golden/questions.txt` deliberately: that file is a
fixed baseline for comparing retrieval captures over time, and adding to it
would invalidate every capture taken before.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — A gene list gets an analysis (Priority: P1)

A researcher pastes identifiers and asks what is enriched. The question is routed to
analysis rather than retrieval, and comes back with pathways and p-values.

**Independent Test**: a gene set with a known enrichment; the answer names it.

**Acceptance Scenarios**:

1. **Given** a list of identifiers and an enrichment question, **Then** analysis runs,
   not retrieval.
2. **Given** an ordinary question, **Then** it is not routed to analysis.
3. **Given** analysis is unavailable, **Then** the chatbot says so and does not answer
   from the vector store as though it had analysed anything.

---

### User Story 2 — A term the embeddings miss is still found (Priority: P1)

A user asks about something in Reactome that retrieval does not surface — a recently
added pathway, or an exact identifier. The Search API finds it.

**Why this priority**: this is the step being added, and the gap it closes is real —
the bundle is a snapshot, while the Search API is live.

**Acceptance Scenarios**:

1. **Given** retrieval found nothing useful, **When** the Search API returns entries,
   **Then** they inform the answer.
2. **Given** the Search API returns **0 entries**, **Then** the cascade proceeds to
   Tavily.
3. **Given** retrieval succeeded, **Then** no Search API call is made.

---

### User Story 3 — Nothing in Reactome, so look outside (Priority: P2)

Neither the embeddings nor the Search API has it. Tavily results appear, clearly
marked as external.

**Acceptance Scenarios**:

1. **Given** both Reactome sources found nothing, **Then** Tavily runs.
2. **Given** Tavily results, **Then** they appear as links beside the answer, not
   blended into it — the existing `SearchResults` element.

### Edge Cases

- **Everything fails.** The honest answer is "Reactome does not cover this", not a
  confident answer assembled from nothing.
- **The user cannot tell which source answered.** Retrieval, live search and the open
  web have very different standing. An answer that mixes them silently is a new class
  of wrong.
- **A misrouted analysis question.** A gene list sent to retrieval gets a plausible
  pathway answer that is not an analysis, and nothing says so.
- **Latency on the deep path.** Four steps before an answer appears.

## Requirements *(mandatory)*

- **FR-001**: Analysis questions MUST be routed to analysis, by extending the existing
  classifier rather than adding a second one.
- **FR-002**: The cascade MUST stop at the first step that finds something.
- **FR-003**: The Search API step MUST use its own result count as the match signal —
  no model call, no tuned threshold.
- **FR-004**: A step that finds nothing MUST fall through, never fabricate.
- **FR-005**: The source of an answer MUST be distinguishable: bundle, live Reactome,
  or the open web.
- **FR-006**: Tavily results MUST remain links beside the answer, not generated into
  it.
- **FR-007**: With the new steps disabled, behaviour and call count MUST be exactly as
  today.
- **FR-008**: An analysis request MUST be answered with the Reactome service
  that performs it, and MUST name which was used. Over-representation
  (`AnalysisService`) and gene set analysis (ReactomeGSA: PADOG, Camera, ssGSEA,
  terapadog) are different analyses; satisfying a request for one with the other
  without saying so is the failure this spec exists to prevent, not relocate.
  A request naming GSEA MUST NOT be answered "Reactome does not do that" — it
  does, at `gsa.reactome.org`.
- **FR-009**: A source that is configured but unavailable MUST be
  distinguishable from a source that had no answer. The user guide falls back to
  `reactome` with only a log line when its bundle is missing, so a routing
  success and a missing bundle produce the same reply.
- **FR-010**: The classifier MUST be able to decline. It currently returns one
  of a closed set and cannot say "none of these" — measured 2026-09-14, "What is
  the current price of a Nature subscription?" routes to `userguide`. The
  downstream grader rescues that today; adding destinations without adding a way
  to decline makes misrouting more confident, not less.
- **FR-011**: The signal the analysis branch routes on MUST survive the
  rephrase, or be taken before it. Measured 2026-09-14: the rephrase rewrites
  "I want to run a gsea with my list of genes" as "How can I perform a Gene Set
  Enrichment Analysis…", which is a user-guide shape. An intent to *act* becomes
  a question about *how to act*, and the classifier cannot recover what the
  rephrase removed.

## Success Criteria *(mandatory)*

- **SC-001**: A gene-list question yields a real analysis or an explicit refusal.
- **SC-002**: A question about content the bundle lacks but Reactome has is answered.
- **SC-003**: A question with nothing anywhere gets an honest "not covered".
- **SC-004**: A question answered from the embeddings costs the same as today.
- **SC-005**: A user can tell which source answered.

## Decisions for the team

### D1 — What counts as "the embeddings found no match"?

| option | what it means |
|---|---|
| **A. Reuse the completeness grader** (recommended) | It exists, it already gates Tavily, and it judges the answer rather than a distance — which is the real question. Costs a model call and requires generating first. |
| B. A distance threshold | Free and instant, but the number is arbitrary, drifts with the embedding model and the release, and nothing tells you when it has stopped meaning anything. |
| C. Threshold first, grader second | Cheap rejection of the obviously-irrelevant, grader for the rest. Two signals to reason about and two ways to be wrong. |

**Recommendation: A.** It is already there, already trusted for the Tavily decision,
and reusing it makes the cascade one consistent idea rather than two. If its latency
proves to be the problem, B becomes a tuning exercise on top — but that should be
driven by a measurement, not a guess.

## Assumptions

- Most questions stop at the embeddings, so the deep path is rare. Worth measuring
  once the cascade exists; if most questions fall through, the bundle is the problem.
- The Search API stays fast and unauthenticated.
- Analysis arrives over hosted MCP ([spec 006](../006-mcp-hosting/spec.md)), not a
  spawned subprocess.

## Out of Scope

- Rebuilding the bundle. That is the Release 98 work and reduces how often the
  cascade goes deep, but it is not this.
- Which MCP tools beyond analysis to expose.
- Presenting analysis results in the UI.
