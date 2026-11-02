# Claim Checks: Live Reactome Data and Analysis

**Created**: 2026-09-10
**Feature**: [spec.md](../spec.md)

Historical. How the specification's claims were checked when it was written. The
template's completion checkboxes are in git history up to 75ca531.

## Adversarial review of this specification (2026-09-10)

Every load-bearing claim was checked rather than reasoned about, after spec 004
shipped a wrong one.

| claim | how it was checked | verdict |
|---|---|---|
| "the bundle is two releases behind" | `cat embeddings/current` → Release95; `reactome.org/ContentService/data/database/version` → 97 | **verified** |
| "it was already stale when built" | bundle mtime 2026-09-02, eight days ago | **verified** — so this is not drift from age |
| "`reactome-mcp` is a prototype" | its own repository description: *"This is just a prototype for now"*, last push 2026-07-01 | the description said so; **wrong about the code**, as spec 006 found on 2026-09-14 |
| "#127 spawns a subprocess" | `asyncio.create_subprocess_exec` with stdio pipes, in its diff | **verified** |
| "#142 duplicates the existing router" | it adds `create_query_router`; `create_intent_classifier` already exists and routes reactome/userguide | **verified** |

### The claim this specification does not make

That MCP fixes staleness. It would, but so would rebuilding the bundle, and the
rebuild needs no dependency, no subprocess and no router. Stating it that way is the
whole contribution of this document: the PRs' own justification is mostly a cheaper
problem wearing the expensive problem's clothes.

### Not verified at the time (2026-09-10)

Nobody had run `reactome-mcp` from this repository; its 53 tools, their latency and
their behaviour under failure were taken from #127's description. It was run from
here on 2026-09-14: `bin/mcp-probe` (dbd02cc) connected to reactome 1.4.0, which
exposed 53 tools (output in `src/reactome_mcp/README.md:34-43`).
