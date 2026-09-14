# Feature Specification: Hosting reactome-mcp

**Feature Branch**: `spec/mcp-hosting`

**Created**: 2026-09-14

**Status**: Draft. Findings settled; two decisions (D1, D2) for the team.

**Input**: Is `reactome/reactome-mcp` the right design, does it have the features we
want against the alternatives, and where should it run? Analysis must work in the
app; direct Neo4j access is not wanted for now.

## A correction first

[Spec 005](../005-mcp-live-data/spec.md) called `reactome-mcp` "just a prototype".
That was its GitHub *description* field, not its state. Reading the code says
otherwise, and the difference matters because it was an argument against adopting it.

It registers **over 40 tools and 10 resources** across Reactome's Content Service and
Analysis Service: enrichment analysis, full-text search with faceting and spellcheck,
pathway hierarchy traversal, entity and complex lookup, PSICQUIC interactors, diagram
and SBGN/SBML export, species and disease annotation, and external ID mapping. The
most recent release (1.4.0) fetches the graph schema live via APOC and caches it per
session — not prototype behaviour.

## Does it have what we want?

**Yes, and specifically the thing that justifies the whole exercise.** Analysis in the
app means submitting a gene or protein list and getting over-representation results
with p-values, FDR and found/not-found identifiers. That is `reactome-mcp`'s first
listed feature, wrapping the Analysis Service directly.

It cannot come from the embeddings bundle at any freshness, because it is a
computation over a user's list rather than a lookup of stored text. This is the
capability gap spec 005 identified, and this server closes it.

### Direct Neo4j access stays off, verifiably

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

## The alternatives

Six Reactome MCP servers exist. Ours is the most recently maintained and the only one
Reactome controls.

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

**Recommendation: keep `reactome/reactome-mcp`.** The question worth asking is not
which to adopt but what the others do better, and the one answer is transport.

## Where it should run — the actual blocker

`reactome-mcp` is **stdio-only**. `src/index.ts` constructs a
`StdioServerTransport` and nothing else: no SSE listener, no HTTP server.

MCP over stdio is a local subprocess protocol — one client, one spawned process,
JSON-RPC over pipes. It cannot be publicly accessible, and it cannot serve two
clients. So "host it somewhere" is not a deployment question yet; it is a change to
`reactome-mcp`.

### The change is small and supported

The MCP TypeScript SDK ships `server/streamableHttp` — Streamable HTTP is the
current standard transport for remote MCP servers. `reactome-mcp` already depends on
`@modelcontextprotocol/sdk` at `^1.12.0`, which resolves to 1.30.0, so the transport
is available today without changing the pin.

### Why hosting it is better than spawning it

The alternative — what #127 proposes — is for the chatbot to spawn the server as a
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

## User Scenarios & Testing *(mandatory)*

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
- **Analysis latency.** Enrichment is not sub-second, on a surface already at 22s per
  question.

## Requirements *(mandatory)*

- **FR-001**: `reactome-mcp` MUST serve MCP over Streamable HTTP.
- **FR-002**: The Cypher/Neo4j tools MUST remain unregistered unless explicitly
  configured, and this MUST be re-verified on any upgrade.
- **FR-003**: The chatbot MUST reach the server over HTTP, spawning no subprocess and
  adding no Node.js to its image.
- **FR-004**: With MCP disabled, chatbot behaviour and LLM call count MUST be exactly
  as today.
- **FR-005**: An unreachable server MUST produce an explicit failure for analysis
  questions, never a silent fall back to retrieval.
- **FR-006**: The deployment MUST rate-limit, since the server fronts a shared
  Reactome service.

## Success Criteria *(mandatory)*

- **SC-001**: A gene list produces a real analysis with p-values, or an explicit
  refusal.
- **SC-002**: A second MCP client can use the same server concurrently.
- **SC-003**: No Cypher tool is listed when Neo4j is unconfigured.
- **SC-004**: With MCP off, a question costs the same LLM calls as today.
- **SC-005**: The chatbot image contains no Node.js.

## Decisions for the team

### D1 — Who adds Streamable HTTP to reactome-mcp?

It is a change in that repository, not this one, and it blocks everything else. The
SDK already provides the transport. This needs an owner and is the critical path.

### D2 — Where does it run?

| option | what it means |
|---|---|
| **A. Alongside the website** (recommended) | Public, one deployment, useful to any MCP client. Treats it as a Reactome service rather than a chatbot dependency. Needs rate limiting and a public URL. |
| B. Beside the chatbot, private | Smaller blast radius, no public surface, no rate-limit worry. Wastes the main advantage: nobody else can use it. |
| C. Both | A public instance and a private one for the chatbot. Honest about differing reliability needs; two deployments to maintain. |

**Recommendation: A**, which is where this started. A hosted MCP endpoint for
Reactome is a product in its own right, and the chatbot is then simply its first
client.

### D3 — Does it stay in its own repository? *(settled 2026-09-14: yes)*

Hosting does not change this, because **HTTP is an additional transport rather than
a replacement**. `main()` picks a transport; branching on an environment variable
gives stdio when a client spawns it and HTTP when it runs as a server. Same codebase,
same tools, both modes — which is why the SDK ships both transports side by side.

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

### D4 — npm publishing *(deferred 2026-09-14)*

`reactome-mcp` is not on npm — `registry.npmjs.org/reactome-mcp` returns 404. Publishing
it would replace clone-plus-build with `npx reactome-mcp` for the people already using
it locally, which is a real ergonomic gain and independent of hosting.

**Deferred deliberately**, to be done once across `reactome-mcp`, the website and the
other repositories together. Publishing one in isolation sets a precedent the others
then have to match, so the batching is the point rather than the delay.

## Assumptions

- The Analysis Service can absorb the traffic. If not, that is a rate-limiting
  parameter rather than a design change.
- `reactome-mcp` stays maintained. It is Reactome's own, so this is a staffing
  question rather than a third-party risk.

## Out of Scope

- Which tools to expose to the chatbot's model beyond analysis.
- Replacing retrieval. MCP adds computation beside it.
- The Cypher tools, deliberately, for now.
