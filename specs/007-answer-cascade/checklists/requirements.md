# Specification Quality Checklist: The Answer Cascade

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
| "the Search API returns 0 for nonsense" | `ContentService/search/query`: `PALB2` → 39 entries, `Impaired BRCA2 binding to PALB2` → 89, `zzzznotathing` → **0** | **verified** — the count is the signal |
| "no Search API client exists" | grep across `src/` for ContentService / search endpoints | **verified** — nothing |
| "the cascade skeleton already exists" | `completeness_grader` gates `perform_web_search` in `tools/external_search/workflow.py` | **verified** |
| "a classifier already exists" | `intent_classifier`, routing `reactome` / `userguide` | **verified** |
| "analysis is fast" | driven over stdio against `reactome-mcp`: **0.2s** for eight genes, 108 pathways | **measured** |

### The claim this specification is careful not to make

That the embeddings have a "no match" state. **They do not** — similarity search
always returns its k nearest documents, so there is no empty result, only an
irrelevant one. The phrase "if it does not find a match" describes an intent, not a
mechanism, and D1 exists because turning it into a mechanism is a real choice with
no obviously right answer.

Writing it as though the signal were obvious would have been the easy mistake, and
would have hidden the only hard decision in the document.

### Not verified

Nobody has measured how often a question would actually fall through to the Search
API. That number decides whether this cascade is worth building or whether the
bundle is simply stale — and the Release 98 rebuild may change it substantially.
Worth measuring before, not after.
