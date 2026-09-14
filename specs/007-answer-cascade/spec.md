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
