# Feature Specification: Live Reactome Data and Analysis

**Feature Branch**: `spec/mcp-live-data`

**Created**: 2026-09-10

**Status**: Historical record of the review of three contributed PRs, 2026-09-10 to
2026-09-14. Drafted as a proposal; this file records no approval of its requirements.
Afterwards, the client and tools from #127 and #137 were taken in, reworked (dbd02cc,
PR #212, 2026-09-14); #142 was not merged; a `live` destination was added to the
existing intent classifier (4c4e8a7, PR #216, 2026-09-14), off unless an MCP server is
configured. Enrichment analysis in the chat was later built without MCP. The
requirements and decisions below were set for that review, not for later changes;
their status is in "Requirements, and where they stand". Current behaviour is in
`src/reactome_mcp/session.py`, `src/agent/tasks/intent_classifier.py` and
`src/agent/profiles/react_to_me.py`.

**Input**: #127, #137 and #142, contributed PRs integrating the `reactome-mcp`
server so the chatbot can query live Reactome APIs and run enrichment analysis.

## Two problems, wrongly bundled as one

The PRs are motivated by a single sentence: MCP tools *"query Reactome APIs directly,
meaning they always return current data regardless of when embeddings were built."*
That is true, and it conflates two problems with very different costs.

### Problem 1 — the answers were two releases out of date (measured 2026-09-10)

Measured 2026-09-10:

| | |
|---|---|
| installed bundle | **Release95** |
| bundle built | 2026-09-02, eight days ago |
| `reactome.org/ContentService/data/database/version` | **97** |

The bundle was two releases behind, and it was already two behind on the day it was
built. So this was not drift from age — nothing had rebuilt it against a current release.

**This did not need MCP.** It needed the bundle rebuilt, and rebuilt on a schedule:
`bin/embeddings_manager` and a cron entry, against a capability the repository
already had. What happened next is under D1.

### Problem 2 — the chatbot could not analyse anything (2026-09-10)

`reactome-mcp` exposes enrichment analysis, pathway traversal and entity lookup.
None of that is expressible as similarity search over stored text. No amount of
rebuilding fixes it, because it is not a retrieval problem: the user gives a gene
list and wants a computation.

**At the time this looked like the part only MCP could do**, and it was the honest
reason to take the work.

Separating them mattered because Problem 1 was most of the stated benefit and the
cheapest fix, while Problem 2 was the smaller-sounding benefit that seemed to require
the architecture. In the event, the chat's analysis was built against Reactome's
Analysis Service and ReactomeGSA directly, without MCP (see D2).

## What the three PRs did (as submitted)

| PR | adds | lines |
|---|---|---|
| #127 | an MCP client speaking JSON-RPC over stdio to a spawned subprocess | +161 |
| #137 | five MCP tools wrapped as LangChain `StructuredTool`s, wired to React-to-Me | +317/−20 |
| #142 | an LLM router choosing between RAG, MCP search and MCP analysis | +450/−20 |

They built on each other in order and were the work of one contributor. Taken
together they were a coherent design, and the sequencing was right: client, then
tools, then routing.

Outcome: the client and tools from #127 and #137 were taken in, reworked, as
`src/reactome_mcp/` (dbd02cc, PR #212, 2026-09-14). #142 was not merged.

## What taking them would cost (assessment, 2026-09-10)

**A dependency described as a prototype.** `reactome/reactome-mcp` is Reactome's own
repository, which is the good case. Its repository description read *"This is just a
prototype for now"*, and it was last pushed 2026-07-01. Spec 006 checked the code on
2026-09-14 and found the description stale (see its "A correction to spec 005").
The chatbot would take a runtime dependency on it.

**A subprocess in the container.** #127 used `asyncio.create_subprocess_exec` with
stdio pipes, so the image would have to carry the MCP server and its Node runtime,
and each chat process would spawn and supervise a child, under the non-root user the
container runs as since #198. Nothing spawned a process at the time. Outcome: the
stdio subprocess is kept for developer machines only (`REACTOME_MCP_SERVER`,
`src/reactome_mcp/process.py:39`); deployments connect over Streamable HTTP
(`REACTOME_MCP_URL`, c02219a, 2026-09-15; `src/reactome_mcp/session.py:91-104`), and
the image has no Node (`Dockerfile:3`).

**A second router.** #142 added `create_query_router` alongside the existing
`create_intent_classifier`, which then routed between `reactome` and `userguide`.
Two LLM classification calls on the same question, in a pipeline where removing
LLM calls was the point of spec 001 — retrieval went from 21 calls to one. The
review's position was that the existing classifier should gain the new destinations
instead. Outcome: #142 was not merged, and `live` was added to the existing
classifier (`src/agent/tasks/intent_classifier.py:10`, 4c4e8a7).

## User stories as proposed (2026-09-10)

Proposed, not recorded as approved. What became of each is in "Requirements, and
where they stand" below.

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

**Why this priority**: Genuinely new capability, and at the time thought impossible
without something like MCP (it was later built against Reactome's services directly;
see D2). P2 below staleness only because staleness affected every question and this
affected a class of question not served at all at the time.

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
- **The server changes its tool surface.** Five tools are wrapped by name; a
  rename upstream breaks them at call time, not at start-up.
- **Live and stored data disagree.** The vector store says Release95, the API says
  97. An answer that mixes both without saying so is a new class of wrong.
- **Analysis latency.** Enrichment is not a sub-second call, and the chat surface
  then ran at about 22s per question.

## Requirements, and where they stand (checked 2026-10-21)

Proposed by this review; no approval is recorded. Where one is not built, whether it
is still wanted is not recorded.

| ID | Proposed | Status |
|---|---|---|
| FR-001 | Bundle rebuildable against the current release without code changes | `bin/embeddings_manager make` exists and needs a Neo4j connection passed explicitly (`deploy/rebuilding-embeddings.md`). Whether a full rebuild against a current release has run without code changes is not recorded |
| FR-002 | Rebuilding schedulable, not a remembered manual step | Not built: no scheduled rebuild exists in this repository |
| FR-003 | Operator can see which release is being answered from | Partly: the answer endpoint reports `release` in its `start` event (`src/api/answer.py:130`), read once from the bundle directory name (`bin/chat-fastapi.py:59`; `src/util/embedding_environment.py:67-79`). The name is the only record and can disagree with the data inside (see D1). The chat UI does not show it |
| FR-004 | MCP optional; disabled means behaviour and LLM call count unchanged | Holds: `live` is offered only when `REACTOME_MCP_URL` or `REACTOME_MCP_SERVER` is set (`src/agent/profiles/react_to_me.py:86-88`; `src/reactome_mcp/session.py:59-60`); otherwise the classifier gets the two-source prompt (`src/agent/tasks/intent_classifier.py:110-141, 189-206`; `tests/agent/test_intent_classifier_sources.py:21-33`) |
| FR-005 | Classified once, by extending the existing classifier | Holds: `live` is a value of the existing classifier's `source` (`intent_classifier.py:10, 151-154`) |
| FR-006 | Unavailable MCP gives an explicit failure for analysis, never a silent fallback | Analysis does not use MCP; the chat calls the Analysis Service directly and says so when that fails (`bin/chat-chainlit.py:455-458`; `src/analysis/gene_list.py:585-589`). A `live`-routed question with the MCP unavailable falls back to retrieval with only a log warning (`react_to_me.py:219-243`) |
| FR-007 | Answers drawing on live data distinguishable | Not built as a label. Live answers carry no citations, because no retriever ran (d3740aa) |
| SC-001 | Deployed bundle at most one release behind | Not established; see D1 |
| SC-002 | Answering release visible without reading code | As FR-003 |
| SC-003 | MCP disabled costs the same LLM calls | As FR-004 |
| SC-004 | A gene list gets a real analysis or an explicit refusal | In the chat: a recognised gene list is offered an over-representation analysis, run on confirmation (`bin/chat-chainlit.py:732`; `src/analysis/gene_list.py:419`), with an explicit message on failure. The search page has no analysis path |

## Decisions put to the team (2026-09-10)

### D1 — Rebuild the bundle now, independently of MCP?

The review recommended **yes, and first**: two releases behind was the larger share
of the stated benefit, and it needed no new dependency, subprocess or router. It
would also remove staleness from MCP's justification, leaving analysis as the real
case.

Recorded answer, 2026-09-14: Release 98 was nearly done, and the plan was to generate
new embeddings and move them to production then. On that basis MCP was to be judged
on analysis alone.

A readiness check on 2026-09-14 found three things not ready, each of which would fail
partway through a long job: disk (3.4 G bundle, 5.2 G free), S3 (no credentials on
that machine, so `push` could not reach production), and Neo4j (needed by `make`,
configured nowhere). Written up in
[`deploy/rebuilding-embeddings.md`](../../deploy/rebuilding-embeddings.md).

What the repository shows since (checked 2026-10-21): by 2026-09-17 beta served a
bundle directory named `Release97` (spec 010; spec 009's `before.json`). Its four
Neo4j-backed collections were built from Release95 data and its `disease_variants`
from the Release 97 download (`src/data_generation/disease_variant/__init__.py:152-156`).
No Release 98 bundle is recorded here, and whether that rebuild happened is not
recorded.

### D2 — Adopt MCP for analysis? (options as proposed, 2026-09-10)

| option | what it means |
|---|---|
| **A. Not yet** | Rebuild bundles, revisit when `reactome-mcp` is past prototype. Costs nothing; the capability gap remains. |
| **B. Adopt behind a flag** | Take #127 and #137, fold routing into the existing classifier rather than #142's second one, ship disabled by default. Real capability, contained blast radius, a prototype dependency that cannot affect anyone who has not enabled it. |
| **C. Adopt fully** | Everything the PRs propose, on by default. Fastest to the capability, and puts a self-described prototype and a supervised subprocess in the path of every user. |

The review recommended B, contingent on D1 being done first, on the grounds that it
bought the irreplaceable part while the dependency was described as a prototype, and
FR-004 meant the cost of being wrong was a flag nobody turned on.

Outcome: approval of an option is not recorded. What was built resembles B, but for
live database lookups rather than analysis: #127 and #137 reworked (dbd02cc), routing
folded into the existing classifier (4c4e8a7), off unless an MCP server is
configured. The MCP's analysis tool is among the five wrapped
(`src/reactome_mcp/tools.py:74`), so it is reachable from `live`-routed questions, but
the classifier describes `live` as questions about the database itself, not analysis
(`src/agent/tasks/intent_classifier.py:29-36`). Analysis in the chat was built
without MCP: gene lists go to the Analysis Service (`src/analysis/client.py:23, 252`;
3b2c792, 2026-09-25) and expression matrices to ReactomeGSA (`src/gsa/client.py:50`;
f51283a, 2026-09-22).

## Assumptions at the time (2026-09-10)

- `reactome-mcp` staying a Reactome project. If it were third-party the answer would
  be different; a prototype from the same organisation is a shared risk, not an
  external one.
- Rebuilding bundles is routine. If it is not, that is the finding, and it makes
  D1 more urgent rather than less.

## Out of scope for the review

- Which MCP tools to wrap beyond the five #137 chose.
- Replacing retrieval with MCP. Retrieval over curated text is what the product is;
  MCP adds computation beside it.
- Analysis result presentation in the UI.
