# Implementation Plan: Summarise analysis results

**Branch**: `011-summarise-analysis-results` | **Date**: 2026-09-18 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `specs/011-summarise-analysis-results/spec.md`

## Summary

Take an analysis token, fetch the completed result from beta's Analysis Service,
and produce a short readable account of what it says — which pathways came out on
top, whether they survive multiple-testing correction, why identifiers went
unmatched, and what the numbers mean. Cite every pathway by stable id. Store the
summary against `(token, release, tier)` so the same token yields the same text,
and say plainly that it was generated.

The feature never runs an analysis. It never sends the user's identifiers unless
they choose that, and never their filename, sample name or column labels at all
under the default choice.

## Technical Context

**Language/Version**: Python 3.12, as the rest of the service
**Primary Dependencies**: the existing LangGraph answer surface, caller-token
verification, `AnchorStripper`, and the SSE response shape — all reused, none
rebuilt
**Storage**: summaries keyed `(token, release, tier)`. **No durable store exists on
beta today** — `POSTGRES_LANGGRAPH_DB` is unset there and LangGraph already falls
back to `MemorySaver`. First increment may hold summaries in process memory;
persistence is follow-up work (research D4)
**Testing**: pytest, with the suite's existing rule that it passes with no API keys
set. The disclosure guarantee is tested by recording outbound requests, not by
inspecting output
**Target Platform**: the same container that serves `/chat` and `/api/answer`
**Project Type**: single service
**Performance Goals**: comparable to the answer endpoint — first token in about
ten seconds. A stored summary returns immediately, which is most requests after
the first
**Constraints**: read from beta, never production; browser-like `User-Agent` on
outbound calls or the site's automation blocking returns 403 HTML; never state a
statistic the result does not contain
**Scale/Scope**: four analysis types, of which three are summarised and one
(ReactomeGSA) is recognised and declined

## Constitution Check

| Principle | Status | Note |
|---|---|---|
| I — verify the path a user takes | **Pass, with a named risk** | The endpoint must be exercised over HTTP on the real mounted app, as spec 010's was. The captcha middleware and the new human-presence bar interact only on the served path, and that is exactly where spec 010's route check found a problem that isolated tests could not. |
| II — measure retrieval changes, do not argue | **Not applicable** | This feature retrieves nothing from the vector store. It reads a completed analysis result. |
| III — characterization tests pin behaviour | **Pass** | The behaviours worth pinning are negative: what is *not* sent under the aggregate tier, that `gone` is distinct from `not_found`, and that nothing is said when nothing passes correction. |
| IV — fail loudly, never quietly differently | **Pass, and it cuts both ways** | Misconfiguration must stop the process, as the caller-token key already does. But a runtime failure must be quiet to the caller — a terminal state, never an HTTP error — because an analysis page must not break because this service did. |
| V — derive from the source of truth | **Pass, and it drives a decision** | The release is read from `GET /database/version`, not hardcoded or copied from the embeddings bundle. That matters because the Analysis Service deletes results on a release change, so the release is also the cache-invalidation key. |
| VI — bias to doing over filing | **Pass** | The blocker (how human presence is asserted) is recorded as a task with a named counterpart, not filed as a question and left. |
| VII — parked is not dead | **Pass** | ReactomeGSA is explicitly deferred with the reason, not silently omitted. |

**No violations requiring justification.**

## Project Structure

### Documentation (this feature)

```
specs/011-summarise-analysis-results/
├── spec.md
├── plan.md              # this file
├── research.md          # D1-D8, measured against beta's API
├── data-model.md        # what is read, what is produced, and the disclosure tiers
├── quickstart.md        # how to validate it with a real analysis token
├── checklists/
│   └── requirements.md
└── contracts/
    └── summary_endpoint.md
```

### Source Code (repository root)

```
src/
├── api/
│   ├── answer.py             # existing; patterns reused, not modified
│   └── analysis_summary.py   # new: the endpoint
├── analysis/                 # new
│   ├── client.py             # fetch a result by token from beta; 404 vs 410
│   ├── disclosure.py         # the field allow-list that defines the aggregate tier
│   ├── summarise.py          # build the prompt input from a result
│   └── store.py              # summaries keyed (token, release, tier)
└── util/
    └── caller_token.py       # existing; extended if human presence rides the token

tests/
├── api/
│   └── test_analysis_summary.py
└── analysis/
    ├── test_disclosure.py    # what must never be sent
    ├── test_client.py        # 404, 410, and the release
    └── test_store.py         # stability, and invalidation on release change
```

`src/analysis/` is separate from `src/agent/` because none of this touches the
graph or retrieval. It reads an external result and shapes it for a prompt.

## Complexity Tracking

One thing here is more complex than it first appears, and it is worth naming
rather than discovering.

**The disclosure tier is a field allow-list, not a field denial.** The obvious
implementation — "strip the identifiers" — misses `summary.fileName`,
`summary.sampleName` and `expression.columnNames`, all of which are user-supplied
free text that can carry a lab's unpublished filename or a patient sample label. A
denial list is wrong by default whenever the Analysis Service adds a field; an
allow-list is only ever wrong by omission, which costs a missing sentence rather
than a disclosure.

Everything else is deliberately unambitious: one external call, one prompt, one
store keyed by three values.
