# Implementation Plan: Chatbot Answers in the Website Search Results

**Branch**: `010-search-page-answers` | **Date**: 2026-09-17 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/010-search-page-answers/spec.md`

## Summary

Build the endpoint first, slow, and let the website integrate against something real
while the latency work happens behind it. Their side is blocked on nothing but this
service's absence; ours is blocked on nothing but time.

Three things do not exist: an HTTP answer endpoint, any streaming surface, and
token verification. The latency budget (FR-005) is deliberately **not** a gate on the
first increment -- an endpoint that answers correctly in fifteen seconds unblocks the
other repo; an endpoint that does not exist blocks it completely.

## Technical Context

**Language/Version**: Python 3.12

**Primary Dependencies**: FastAPI, langchain 1.x, langgraph, Chainlit (mounted on the
same app)

**Storage**: none new. The graph, bundle and MCP are as they are

**Testing**: pytest; `bin/answer-sweep` for answer equivalence

**Target Platform**: the existing `biochat_beta_guest` container behind nginx

**Project Type**: single project

**Performance Goals**: FR-005 is first token ≤2s, complete ≤10s. Today p50 is 15.2s,
p90 22.4s, with no first-token concept at all

**Constraints**: the endpoint and the chat UI must share one graph (SC-003); a
failure must render as "no panel", never a broken one (FR-006)

**Scale/Scope**: one endpoint, one streaming surface, one verification step

## Constitution Check

| Principle | Gate | How this plan satisfies it |
|---|---|---|
| I. Verify the path a user takes | Test through the served path | The endpoint is tested over HTTP with a real client, not by calling the handler. The served path is async and mounted alongside Chainlit, which is where mounting order and middleware interact |
| II. Measure retrieval changes | Before-and-after on real questions | This adds no retrieval change. If the latency work later touches retrieval, that measurement belongs to it, not here |
| III. Characterization tests | Pin current behaviour first | The captcha middleware currently intercepts every path under `CHAINLIT_URI`. A test pins that before a new route is added beside it |
| IV. Fail loudly | No plausible substitutes | An unverified request is refused, not answered anonymously. A missing signing key stops the endpoint from starting rather than accepting everything |
| V. Source of truth | No hand-synchronised constants | Citations come from retrieved documents' `st_id` metadata, not from parsing anchors out of the model's prose |

**Gate result**: pass.

### The one that needs care

Principle IV cuts both ways here. FR-006 says fail **invisibly** -- any error renders
as "no panel" -- and Principle IV says fail **loudly**. They are not in conflict but
the line matters: *misconfiguration* stops the process (no signing key, no bundle),
while a *runtime* failure answering one question returns `state: failed` and is
logged. The first is loud because an operator must fix it; the second is quiet
because a search page must not break.

## Key design decisions, from reading the code

**Citations come from retrieval, not from the prose.** The prompt currently instructs
the model to emit `<a href="...">` anchors inline. Parsing those back out of a token
stream would be fragile and would couple the contract to prompt wording. The
retrieved documents already carry `st_id` in metadata, so citation events are emitted
from the retrieval result. This is why the contract specifies stable IDs rather than
HTML.

**Streaming needs a surface that does not exist.** `AgentGraph` exposes `ainvoke` and
nothing else; Chainlit gets its streaming from `AsyncLangchainCallbackHandler` with
`final_stream`, not from a graph-level API. Two options, resolved in research.md.

**The FastAPI app does not hold a graph.** `AgentGraph` is constructed in
`bin/chat-chainlit.py`, which Chainlit mounts. The endpoint needs one, and building a
second would double 51.5s of startup and risk the two answering differently, which
SC-003 forbids.

## Project Structure

```
specs/010-search-page-answers/
├── spec.md, plan.md, research.md
├── contracts/answer_endpoint.md
├── quickstart.md
└── tasks.md

src/
├── api/                      # new: the endpoint, its models, SSE framing
├── agent/graph.py            # gains a streaming surface
└── util/caller_token.py       # new: signature verification only

bin/chat-fastapi.py           # mounts the router; middleware ordering matters
tests/api/                    # new
```

## Complexity Tracking

| Addition | Current need | Why the simpler option is insufficient |
|---|---|---|
| A second answer surface beside Chainlit | The search page cannot use a websocket chat UI | Chainlit's socket protocol is not a documented API and is not something another repo should couple to |
| Streaming on `AgentGraph` | FR-002, and 10s is only survivable if text appears early | Returning a complete answer means a blank panel for fifteen seconds |

## Phases

Phase 0 research: [research.md](./research.md). Phase 1 design: the contract already
exists and was agreed with the website session; `quickstart.md` covers validation.
