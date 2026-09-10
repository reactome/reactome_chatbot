# Specification Quality Checklist: Live Reactome Data and Analysis

**Purpose**: Validate specification completeness and quality before planning
**Created**: 2026-09-10
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] No implementation details beyond what the contributed PRs already fix
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] All acceptance scenarios are defined
- [x] Edge cases are identified
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

## Adversarial review of this specification

Every load-bearing claim was checked rather than reasoned about, after spec 004
shipped a wrong one.

| claim | how it was checked | verdict |
|---|---|---|
| "the bundle is two releases behind" | `cat embeddings/current` → Release95; `reactome.org/ContentService/data/database/version` → 97 | **verified** |
| "it was already stale when built" | bundle mtime 2026-09-02, eight days ago | **verified** — so this is not drift from age |
| "`reactome-mcp` is a prototype" | its own repository description: *"This is just a prototype for now"*, last push 2026-07-01 | **verified** |
| "#127 spawns a subprocess" | `asyncio.create_subprocess_exec` with stdio pipes, in its diff | **verified** |
| "#142 duplicates the existing router" | it adds `create_query_router`; `create_intent_classifier` already exists and routes reactome/userguide | **verified** |

### The claim this specification does not make

That MCP fixes staleness. It would, but so would rebuilding the bundle, and the
rebuild needs no dependency, no subprocess and no router. Stating it that way is the
whole contribution of this document: the PRs' own justification is mostly a cheaper
problem wearing the expensive problem's clothes.

### Not verified, and worth knowing

Nobody has run `reactome-mcp` from this repository. Its 53 tools, their latency and
their behaviour under failure are taken from #127's description, not observed. Any
plan built on this should start by running it once.

## Notes

Deviations from the template, deliberate:

1. **Three contributed PRs are reviewed in the spec body.** They are the starting
   material and the reason to take or reject each is inseparable from reading them.
2. **The spec argues against most of its own feature's justification.** That is the
   finding, not a digression.
