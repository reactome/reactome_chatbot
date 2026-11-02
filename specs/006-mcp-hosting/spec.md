# Feature Specification: Hosting reactome-mcp

**Feature Branch**: `spec/mcp-hosting`

**Created**: 2026-09-14

**Status**: Historical record of the review of hosting `reactome-mcp`, 2026-09-14.
Drafted as a proposal; this file records no approval of its requirements. D3 and D4
were recorded as settled and deferred on 2026-09-14. Since then, reactome-mcp gained a
Streamable HTTP transport and the chatbot connects to it when `REACTOME_MCP_URL` is
set (c02219a, PR #219, 2026-09-15); where the server runs (D2) is not recorded in this
repository. The findings, requirements and decisions below were set for that review,
not for later changes; their status is in "Requirements, and where they stand".
Current client behaviour is in `src/reactome_mcp/session.py` and
`src/reactome_mcp/http_client.py`.

**Input**: Is `reactome/reactome-mcp` the right design, does it have the features we
want against the alternatives, and where should it run? At the time, analysis was to
work in the app and direct Neo4j access was not wanted.

## A correction to spec 005 (2026-09-14)

[Spec 005](../005-mcp-live-data/spec.md) called `reactome-mcp` "just a prototype".
That was its GitHub *description* field, not its state. Reading the code says
otherwise, and the difference matters because it was an argument against adopting it.

As read on 2026-09-14, it registered **over 40 tools and 10 resources** across Reactome's Content Service and
Analysis Service: enrichment analysis, full-text search with faceting and spellcheck,
pathway hierarchy traversal, entity and complex lookup, PSICQUIC interactors, diagram
and SBGN/SBML export, species and disease annotation, and external ID mapping. Its
then most recent release (1.4.0) fetched the graph schema live via APOC and cached it
per session — not prototype behaviour. (`bin/mcp-probe` later counted 53 tools on
1.4.0; `src/reactome_mcp/README.md:34-37`.)

## Does it have what we want? (assessment, 2026-09-14)

**Yes, and specifically the thing that justifies the whole exercise.** Analysis in the
app means submitting a gene or protein list and getting over-representation results
with p-values, FDR and found/not-found identifiers. That is `reactome-mcp`'s first
listed feature, wrapping the Analysis Service directly.

It cannot come from the embeddings bundle at any freshness, because it is a
computation over a user's list rather than a lookup of stored text. This was the
capability gap spec 005 identified, and this server could close it.

In the event, analysis in the chat was built against Reactome's services directly,
not through this server: gene lists go to the Analysis Service
(`src/analysis/client.py:23, 252`) and expression matrices to ReactomeGSA
(`src/gsa/client.py:50`). The server's analysis tool is among the five the chatbot
wraps (`src/reactome_mcp/tools.py:74`) and is reachable only from `live`-routed
questions.

### Direct Neo4j access stays off, verifiably (as read in reactome-mcp, 2026-09-14)

Not a configuration convention — the tools are never registered:

```typescript
// Graph database tools — only when NEO4J_URI is set
if (isNeo4jConfigured()) {
  registerCypherTools(server);
}
```

`isNeo4jConfigured()` is `Boolean(NEO4J_URI)`. With the variable unset the Cypher
tools do not exist in the server's tool list, so the model cannot see or call them.
That satisfies "no direct Neo4j for now" without a fork or a patch, and it is a
property to re-check rather than assume if the server is ever upgraded.

## The alternatives (surveyed 2026-09-14)

Six Reactome MCP servers existed. Ours was the most recently maintained and the only
one Reactome controls.

| repository | language | last push | notes |
|---|---|---|---|
| **reactome/reactome-mcp** | TypeScript | **2026-07-01** | ours; 40+ tools; Content + Analysis |
| tc2fh/reactome-mcp | Python | 2026-06-22 | enrichment; mentions SSE and stdio |
| tc2fh/reactome-db-mcp | Python | 2026-06-22 | direct SQL to a local Reactome MySQL |
| ron-42/reactome-mcp-python | Python | 2026-03-16 | Analysis Service, stdio |
| openpharma-org/reactome-mcp-server | JavaScript | 2026-03-10 | self-described production-ready |
| Augmented-Nature/Reactome-MCP-Server | JavaScript | 2025-12-21 | nine months stale |

Depending on a third party for the interface to our own knowledgebase would be the
strange choice, and none is better positioned on features. `tc2fh/reactome-mcp` is
worth reading for its transport work; `tc2fh/reactome-db-mcp` goes direct to MySQL,
which is the coupling we are explicitly avoiding.

**The review recommended keeping `reactome/reactome-mcp`**, and it is the server the
chatbot uses. The question it asked was not which to adopt but what the others did
better, and the one answer was transport.

## Where it should run — the blocker at the time (2026-09-14)

`reactome-mcp` was **stdio-only**: `src/index.ts` constructed a
`StdioServerTransport` and nothing else — no SSE listener, no HTTP server.

MCP over stdio is a local subprocess protocol — one client, one spawned process,
JSON-RPC over pipes. It cannot be publicly accessible, and it cannot serve two
clients. So "host it somewhere" was not yet a deployment question; it was a change to
`reactome-mcp`.

**Resolved by 2026-09-15.** reactome-mcp gained Streamable HTTP (run as
`node dist/http-server.js` with `MCP_HTTP_PORT`; `src/reactome_mcp/README.md:77-84`),
and the chatbot connects to it when `REACTOME_MCP_URL` is set (c02219a, PR #219;
`src/reactome_mcp/session.py:54-56, 91-104`).

### The change was small and supported

The MCP TypeScript SDK ships `server/streamableHttp` — Streamable HTTP is the
current standard transport for remote MCP servers. `reactome-mcp` already depended on
`@modelcontextprotocol/sdk` at `^1.12.0`, which then resolved to 1.30.0, so the
transport was available without changing the pin.

### Why hosting it is better than spawning it

The alternative — what #127 proposed — was for the chatbot to spawn the server as a
subprocess. Hosting removes three costs at once:

| | subprocess | hosted over HTTP |
|---|---|---|
| Node.js in the Python image | required | not needed |
| process supervision, restarts | ours to write | none |
| non-root container can spawn it | must be made to work | not applicable |
| serving other clients | one per chatbot process | any client, including Claude directly |

That last row is the one that makes it worth doing properly rather than expediently:
a hosted `reactome-mcp` is useful to anyone with an MCP client, not only to this
chatbot. That is a Reactome service, not a chatbot implementation detail — which is
the instinct behind running it alongside the website.

## User stories as proposed (2026-09-14)

Proposed, not recorded as approved. What became of each is in "Requirements, and
where they stand" below.

### User Story 1 — A user gets an enrichment analysis (Priority: P1)

A researcher pastes a gene list and asks which pathways are over-represented. The
chatbot runs a real Reactome analysis and explains the result.

**Independent Test**: a known gene list with a known enriched pathway; the answer
names it with a p-value.

**Acceptance Scenarios**:

1. **Given** a gene list, **When** analysis is requested, **Then** a real Analysis
   Service result is returned, with identifiers it could not find.
2. **Given** the MCP server is unreachable, **When** analysis is requested, **Then**
   the chatbot says so, and does not answer from the vector store as though it had
   analysed anything.

---

### User Story 2 — The server is reachable over HTTP (Priority: P1)

`reactome-mcp` serves MCP over Streamable HTTP at a URL, to any client.

**Why this priority**: nothing else here is possible first, and it is the piece that
lives in a different repository.

**Acceptance Scenarios**:

1. **Given** a running server, **When** a client connects over HTTP, **Then** it can
   list tools and call one.
2. **Given** `NEO4J_URI` is unset, **When** a client lists tools, **Then** no Cypher
   tool appears.
3. **Given** two clients, **When** both connect, **Then** both work.

---

### User Story 3 — The chatbot uses it without a subprocess (Priority: P2)

The chatbot reaches the hosted server over HTTP; the image gains no Node runtime and
supervises no child process.

**Acceptance Scenarios**:

1. **Given** MCP is configured off, **Then** behaviour and LLM call count are exactly
   as today.
2. **Given** MCP is configured on, **Then** the Docker image contains no Node.js.

### Edge Cases

- **The hosted server is down.** Analysis must fail visibly. A retrieval answer to an
  analysis question is the worst outcome, because it looks like success.
- **A public server is abusable.** Analysis Service calls are not free; an open
  endpoint needs rate limiting, and that is a hosting decision rather than an MCP one.
- **Tool surface drift.** Tools are called by name; a rename upstream fails at call
  time, not at startup.
- **Analysis latency.** Enrichment is not sub-second, on a surface then at about 22s
  per question.

## Requirements, and where they stand (checked 2026-10-21)

Proposed by this review; no approval is recorded. Rows about reactome-mcp itself or
its deployment cannot be checked from this repository.

| ID | Proposed | Status |
|---|---|---|
| FR-001 | reactome-mcp serves MCP over Streamable HTTP | Done in reactome-mcp by 2026-09-15; the chatbot's HTTP client was verified against a real server (c02219a) |
| FR-002 | Cypher/Neo4j tools unregistered unless configured; re-checked on upgrade | A property of reactome-mcp; not checked from this repository since 2026-09-14 |
| FR-003 | Chatbot reaches the server over HTTP, no subprocess, no Node in its image | Holds when `REACTOME_MCP_URL` is set (`src/reactome_mcp/session.py:91-104`, `http_client.py`); the image is `python:*-slim` with no Node (`Dockerfile:3`). The stdio subprocess remains for developer machines via `REACTOME_MCP_SERVER` (`session.py:105-112`; `process.py:39`) |
| FR-004 | MCP disabled leaves behaviour and LLM call count unchanged | Holds; see spec 005, FR-004 |
| FR-005 | Unreachable server gives an explicit failure for analysis, never a silent fallback | Analysis does not go through MCP; the chat reports a failed Analysis Service call (`bin/chat-chainlit.py:455-458`). A `live`-routed question with the MCP unreachable falls back to retrieval with only a log warning (`src/agent/profiles/react_to_me.py:219-243`); a failed start is retried after 60 s (`session.py:44-45`) |
| FR-006 | The deployment rate-limits | Concerns the MCP deployment, which is not in this repository; not established. The chatbot bounds what one question sends — at most 6 tool calls per round, 2 rounds (`src/reactome_mcp/answer.py:31-32`) — not what the server accepts |
| SC-001 | A gene list gets a real analysis with p-values, or an explicit refusal | In the chat, via the Analysis Service directly (spec 005, SC-004) |
| SC-002 | A second MCP client can use the same server concurrently | Not recorded. Within one chatbot process, overlapping calls on one session were fixed in 42c505b (PR #313) |
| SC-003 | No Cypher tool listed when Neo4j is unconfigured | As FR-002 |
| SC-004 | MCP off costs the same LLM calls | As FR-004 |
| SC-005 | Chatbot image contains no Node.js | Holds (`Dockerfile`) |

## Decisions put to the team (2026-09-14)

### D1 — Who adds Streamable HTTP to reactome-mcp? (resolved by 2026-09-15)

It was a change in that repository, not this one, and it blocked everything else; the
SDK already provided the transport. reactome-mcp had gained Streamable HTTP by
2026-09-15, when the chatbot's HTTP client was verified against it (c02219a, PR #219).

### D2 — Where does it run? (no decision recorded)

| option | what it means |
|---|---|
| **A. Alongside the website** (recommended) | Public, one deployment, useful to any MCP client. Treats it as a Reactome service rather than a chatbot dependency. Needs rate limiting and a public URL. |
| B. Beside the chatbot, private | Smaller blast radius, no public surface, no rate-limit worry. Wastes the main advantage: nobody else can use it. |
| C. Both | A public instance and a private one for the chatbot. Honest about differing reliability needs; two deployments to maintain. |

The review recommended A, which is where this started: a hosted MCP endpoint for
Reactome is a product in its own right, and the chatbot would then simply be its first
client.

No decision is recorded. What this repository shows (checked 2026-10-21): no
deployment file here configures an MCP server — `compose.yaml`, `env_template` and
`deploy/beta/env.beta.template` have no `REACTOME_MCP_*` variable. Commit messages and
code comments record MCP configured on beta from 2026-09-18 (d3740aa;
`src/agent/graph.py:511`), reached at an internal host name
(`src/reactome_mcp/answer.py:156`), which matches option B there. Whether a public
instance exists, and what production uses, is not recorded.

### D3 — Does it stay in its own repository? *(recorded as settled 2026-09-14: yes)*

It has stayed in its own repository. Hosting did not change this, because **HTTP is
an additional transport rather than a replacement**. The proposal was for `main()` to
pick a transport by environment variable; as built, the HTTP server is its own entry
point, `dist/http-server.js`, beside the stdio `dist/index.js`
(`src/reactome_mcp/README.md:30, 81`). Same codebase, same tools, both modes.

That matters because people already clone it and point Claude at it. Nothing about
hosting takes that away.

The two modes serve different users and neither obsoletes the other:

| mode | who | why that mode |
|---|---|---|
| stdio, cloned | researchers with Claude; curators | works offline, and is the only way to reach the Cypher tools, which need a local Neo4j |
| HTTP, hosted | the chatbot; any remote client | no clone, no install, no Node, no build |

And a separate repository is right independently of hosting: a different toolchain
(TypeScript, against a Python chatbot and an Angular site), a different audience —
its consumers are individual researchers, this chatbot, and whoever else adopts it —
and a different cadence, tracking Reactome's API surface rather than any one client's
features.

**Deployment configuration is not a reason to merge.** The manifest that runs it
belongs to whoever operates it and points at a built image; one does not vendor
nginx's source to deploy nginx.

### D4 — npm publishing *(recorded as deferred 2026-09-14)*

On 2026-09-14 `reactome-mcp` was not on npm — `registry.npmjs.org/reactome-mcp`
returned 404. Publishing it would replace clone-plus-build with `npx reactome-mcp` for
the people already using it locally, which is a real ergonomic gain and independent of
hosting.

Recorded as deferred, to be done once across `reactome-mcp`, the website and the
other repositories together, because publishing one in isolation sets a precedent the
others then have to match. Whether it has been published since is not recorded here.

## Assumptions at the time (2026-09-14)

- The Analysis Service can absorb the traffic. If not, that is a rate-limiting
  parameter rather than a design change.
- `reactome-mcp` stays maintained. It is Reactome's own, so this is a staffing
  question rather than a third-party risk.

## Out of scope for the review

- Which tools to expose to the chatbot's model beyond analysis.
- Replacing retrieval. MCP adds computation beside it.
- The Cypher tools, deliberately, at the time.
