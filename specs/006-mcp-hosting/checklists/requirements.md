# Claim Checks: Hosting reactome-mcp

**Created**: 2026-09-14
**Feature**: [spec.md](../spec.md)

Historical. How the specification's claims were checked when it was written. The
template's completion checkboxes are in git history up to 75ca531.

## Adversarial review of this specification (2026-09-14)

| claim | how it was checked | verdict |
|---|---|---|
| "`reactome-mcp` is stdio-only" | `src/index.ts` constructs `StdioServerTransport`; no SSE, express or `listen(` anywhere in `src/` | **verified** on 2026-09-14; reactome-mcp gained Streamable HTTP by 2026-09-15 (c02219a) |
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

### Not verified at the time (2026-09-14)

When this was written the server had not been run from here, so every number in the
specification is a count of code, not a measurement of behaviour. Later the same day
it was run: spec 007 measured an analysis at 0.2s over stdio, and `bin/mcp-probe`
recorded per-tool latencies (`src/reactome_mcp/README.md:34-43`). The HTTP client was
verified end to end against a real server on 2026-09-15 (c02219a). Analysis Service
behaviour under load has not been recorded here.
