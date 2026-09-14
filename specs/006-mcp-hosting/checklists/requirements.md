# Specification Quality Checklist: Hosting reactome-mcp

**Purpose**: Validate completeness and quality before planning
**Created**: 2026-09-14
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Edge cases identified
- [x] Scope clearly bounded
- [x] Dependencies and assumptions identified

## Adversarial review of this specification

| claim | how it was checked | verdict |
|---|---|---|
| "`reactome-mcp` is stdio-only" | `src/index.ts` constructs `StdioServerTransport`; no SSE, express or `listen(` anywhere in `src/` | **verified** |
| "the SDK ships Streamable HTTP" | unpacked `@modelcontextprotocol/sdk@1.30.0`: `server/streamableHttp.js` present | **verified** |
| "available without changing the pin" | `reactome-mcp` pins `^1.12.0`, which resolves to 1.30.0 | **verified** |
| "Cypher tools are not registered when Neo4j is unset" | `if (isNeo4jConfigured()) registerCypherTools(server)`, and `isNeo4jConfigured()` is `Boolean(NEO4J_URI)` | **verified** — not registered, so not merely disabled |
| "40+ tools, Content + Analysis" | its README, corroborated by `src/tools/` | **verified** |
| "six Reactome MCP servers exist" | GitHub search; ours most recently pushed | **verified** |

### The claim spec 005 got wrong

It called `reactome-mcp` "just a prototype", taken from the repository's
one-line description field. The code contradicts it. That was an argument against
adopting the server, built on a stale sentence nobody had updated — corrected at the
top of this specification rather than quietly.

The lesson is the same one spec 004 recorded: check the thing, not the description
of the thing.

### Not verified

**The server has still never been run from here.** Tool latency, Analysis Service
behaviour under load, and what a failure looks like to a client are all unobserved.
Every number in this document is a count of code, not a measurement of behaviour. A
plan should open by running it once.
