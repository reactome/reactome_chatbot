# Feature Specification: Live Reactome Data and Analysis

**Feature Branch**: `spec/mcp-live-data`

**Created**: 2026-09-10

**Status**: Draft. Three contributed PRs; two decisions (D1, D2) for the team.

**Input**: #127, #137 and #142 from @GovindhKishore, integrating the `reactome-mcp`
server so the chatbot can query live Reactome APIs and run enrichment analysis.

## Two problems, wrongly bundled as one

The PRs are motivated by a single sentence: MCP tools *"query Reactome APIs directly,
meaning they always return current data regardless of when embeddings were built."*
That is true, and it conflates two problems with very different costs.

### Problem 1 — the answers are two releases out of date

Measured today:

| | |
|---|---|
| installed bundle | **Release95** |
| bundle built | 2026-09-02, eight days ago |
| `reactome.org/ContentService/data/database/version` | **97** |

The bundle is two releases behind, and it was already two behind on the day it was
built. So this is not drift from age — nothing rebuilt it against a current release.

**This does not need MCP.** It needs the bundle rebuilt, and rebuilt on a schedule.
That is `bin/embeddings_manager` and a cron entry, against a capability the
repository already has.

### Problem 2 — the chatbot cannot analyse anything

`reactome-mcp` exposes enrichment analysis, pathway traversal and entity lookup.
None of that is expressible as similarity search over stored text. No amount of
rebuilding fixes it, because it is not a retrieval problem: the user gives a gene
list and wants a computation.

**This is the part only MCP can do**, and it is the honest reason to take the work.

Separating them matters because Problem 1 is most of the stated benefit and the
cheapest fix, while Problem 2 is the smaller-sounding benefit that actually requires
the architecture.

## What the three PRs do

| PR | adds | lines |
|---|---|---|
| #127 | an MCP client speaking JSON-RPC over stdio to a spawned subprocess | +161 |
| #137 | five MCP tools wrapped as LangChain `StructuredTool`s, wired to React-to-Me | +317/−20 |
| #142 | an LLM router choosing between RAG, MCP search and MCP analysis | +450/−20 |

They build on each other in order and are the work of one contributor. Taken
together they are a coherent design, and the sequencing is right: client, then
tools, then routing.

## What taking them costs

**A prototype dependency.** `reactome/reactome-mcp` is Reactome's own repository,
which is the good case — but its description reads *"This is just a prototype for
now"* and it was last pushed 2026-07-01, over two months ago. The chatbot would take
a runtime dependency on it.

**A subprocess in the container.** #127 uses `asyncio.create_subprocess_exec` with
stdio pipes, so the image must carry the MCP server and its runtime, and each chat
process spawns and supervises a child. The container now runs as a non-root user
(#198), which that must work under. Nothing today spawns a process; this would be
the first.

**A second router.** #142 adds `create_query_router` alongside the existing
`create_intent_classifier`, which already routes between `reactome` and `userguide`.
Two LLM classification calls on the same question, in a pipeline where removing
LLM calls was the point of spec 001 — retrieval went from 21 calls to one. The
existing classifier should gain the new destinations rather than acquire a sibling.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Answers reflect the current release (Priority: P1)

A user asks about a pathway added in Release 96 or 97 and gets it.

**Why this priority**: It is the largest part of the stated benefit and by far the
cheapest to deliver. It is P1 *and* it is not what the PRs build.

**Independent Test**: Rebuild the bundle against Release97, ask about content added
since 95, compare against today.

**Acceptance Scenarios**:

1. **Given** a rebuilt bundle, **When** a user asks about recent content, **Then**
   the answer includes it.
2. **Given** a release cadence, **When** a new release ships, **Then** rebuilding is
   a scheduled operation rather than a thing someone remembers.

---

### User Story 2 — The chatbot can run an enrichment analysis (Priority: P2)

A user pastes a gene list and asks which pathways are over-represented. The chatbot
runs the analysis and explains the result.

**Why this priority**: Genuinely new capability, and impossible without something
like MCP. P2 below staleness only because staleness affects every question and this
affects a class of question we do not serve at all today.

**Acceptance Scenarios**:

1. **Given** a gene list, **When** analysis is requested, **Then** a real Reactome
   analysis runs and its result is explained.
2. **Given** the MCP server is unavailable, **When** analysis is requested, **Then**
   the chatbot says so plainly rather than answering from the vector store as if it
   had analysed anything.

---

### User Story 3 — Routing costs one classification, not two (Priority: P2)

A question is classified once, into one of the destinations available.

**Acceptance Scenarios**:

1. **Given** MCP is enabled, **When** a question arrives, **Then** exactly one
   classification call is made.
2. **Given** MCP is disabled, **When** a question arrives, **Then** behaviour and
   call count are exactly as today.

### Edge Cases

- **The MCP server dies mid-conversation.** A supervised subprocess needs a defined
  answer: restart, degrade to RAG, or fail. Silently degrading to RAG is the worst
  option, because an analysis question would get a retrieval answer.
- **The prototype changes its tool surface.** Five tools are wrapped by name; a
  rename upstream breaks them at call time, not at start-up.
- **Live and stored data disagree.** The vector store says Release95, the API says
  97. An answer that mixes both without saying so is a new class of wrong.
- **Analysis latency.** Enrichment is not a sub-second call, and the chat surface
  already runs at 22s per question.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The installed bundle MUST be rebuildable against the current release
  without code changes.
- **FR-002**: Rebuilding MUST be schedulable, not a remembered manual step.
- **FR-003**: The deployed release version MUST be visible — an operator must be able
  to see which release is being answered from.
- **FR-004**: If MCP is adopted, it MUST be optional: with it disabled, behaviour and
  LLM call count are exactly as today.
- **FR-005**: If MCP is adopted, questions MUST be classified exactly once, by
  extending the existing intent classifier rather than adding a second.
- **FR-006**: An unavailable MCP server MUST produce an explicit failure for
  analysis questions, never a silent fall back to retrieval.
- **FR-007**: Where an answer draws on live API data rather than the bundle, that
  MUST be distinguishable.

## Success Criteria *(mandatory)*

- **SC-001**: The gap between the deployed bundle's release and Reactome's current
  release is at most one.
- **SC-002**: An operator can determine the answering release without reading code.
- **SC-003**: With MCP disabled, a question costs the same LLM calls as today.
- **SC-004**: A gene-list question produces a real analysis or an explicit refusal,
  never a retrieval answer dressed as one.

## Decisions for the team

### D1 — Rebuild the bundle now, independently of MCP?

Recommended: **yes, and first.** Two releases behind is the larger share of the
stated benefit, it needs no new dependency, no subprocess and no router, and it can
ship this week. It also makes the MCP decision honest by removing staleness from its
justification, leaving analysis — which is the real case.

### D2 — Adopt MCP for analysis?

| option | what it means |
|---|---|
| **A. Not yet** | Rebuild bundles, revisit when `reactome-mcp` is past prototype. Costs nothing; the capability gap remains. |
| **B. Adopt behind a flag** | Take #127 and #137, fold routing into the existing classifier rather than #142's second one, ship disabled by default. Real capability, contained blast radius, a prototype dependency that cannot affect anyone who has not enabled it. |
| **C. Adopt fully** | Everything the PRs propose, on by default. Fastest to the capability, and puts a self-described prototype and a supervised subprocess in the path of every user. |

**Recommendation: B**, contingent on D1 being done first. It buys the irreplaceable
part while the dependency is still a prototype, and FR-004 means the cost of being
wrong is a flag nobody turned on.

## Assumptions

- `reactome-mcp` staying a Reactome project. If it were third-party the answer would
  be different; a prototype from the same organisation is a shared risk, not an
  external one.
- Rebuilding bundles is routine. If it is not, that is the finding, and it makes
  D1 more urgent rather than less.

## Out of Scope

- Which MCP tools to wrap beyond the five #137 chose.
- Replacing retrieval with MCP. Retrieval over curated text is what the product is;
  MCP adds computation beside it.
- Analysis result presentation in the UI.
