# Tasks: Chatbot Answers in the Website Search Results

**Feature**: 010-search-page-answers | **Plan**: [plan.md](./plan.md) | **Spec**: [spec.md](./spec.md)

Test tasks included: this adds a public surface with an authentication boundary, and
Principle III wants current behaviour pinned before a route is added beside the
captcha middleware.

**MVP is Phase 3.** An endpoint that answers correctly and slowly unblocks the
website repo entirely. Speed is Phase 5 and does not gate the handover.

## Phase 1: Setup

- [ ] T001 Create branch `010-search-page-answers`; confirm `.specify/feature.json` points at specs/010-search-page-answers
- [ ] T002 [P] Record today's baseline in specs/010-search-page-answers/quickstart.md: p50 15.2s, p90 22.4s over the 15 sweep questions, so Phase 5 has a before

## Phase 2: Foundational (blocks the endpoint)

- [ ] T003 Pin the captcha middleware's current behaviour in tests/api/test_middleware_scope.py: every path under CHAINLIT_URI redirects to the captcha page except the existing allowlist
- [ ] T004 Give `AgentGraph` one home the FastAPI app can reach, in bin/chat-fastapi.py via lifespan, so the endpoint and Chainlit share one instance (SC-003) rather than paying 51.5s of startup twice
- [ ] T005 Add a streaming surface to src/agent/graph.py using the compiled graph's `astream_events`, yielding answer tokens and retrieved documents
- [ ] T006 Test that T005 yields more than one token for a real question — not that it returns 200. A surface that completes without streaming is the failure this must catch

## Phase 3: User Story 1 — a verified person sees an answer forming (P1)

**Independent test**: post a question with a valid token; assert the first event
arrives, tokens stream, citations resolve to real stable IDs, and `done` carries a
state.

- [x] T007 [P] [US1] Add src/util/human_token.py: verify signature and expiry only, stateless, no consumption tracking
- [x] T008 [P] [US1] Test human_token in tests/util/test_human_token.py: valid, expired, wrong key, tampered payload, absent — every failure refuses
- [x] T009 [US1] Refuse at startup in src/util/human_token.py when the verifying key is missing or unreadable, rather than accepting everything (Principle IV)
- [x] T010 [US1] Add the SSE endpoint in src/api/answer.py implementing contracts/answer_endpoint.md: start, token, citation, done
- [x] T011 [US1] Emit citations from retrieved documents' `st_id` metadata, deduplicated — never by parsing anchors out of the model's prose
- [x] T012 [US1] Mount the router in bin/chat-fastapi.py and let the captcha middleware pass /chat/api/ through, since the endpoint verifies its own caller
- [x] T013 [US1] Test over HTTP with a real client in tests/api/test_answer_endpoint.py, not by calling the handler — mounting order and middleware only interact in the served path (Principle I)
- [ ] T014 [US1] Assert the answer matches the chat UI's for the same question (SC-003); two surfaces that can disagree is a defect
- [x] T014a [US1] Give each request its own checkpointer thread in src/api/answer.py; `id(body)` put 192 of 200 requests on a shared thread, and `chat_history` is checkpointed state the rephraser reads (PR #236)
- [x] T014b [US1] Send `release` on start and `seconds` on done per contracts/answer_endpoint.md; both were promised to the website and neither was implemented (PR #236)

## Phase 4: User Story 2 — no answer without a verified person (P1)

- [x] T015 [US2] Refuse missing, expired, malformed and wrongly-signed tokens before any model call, in src/api/answer.py
- [x] T016 [US2] Test that no model call happens for a refused request in tests/api/test_answer_endpoint.py, by asserting on a patched graph rather than on timing (SC-002)
- [ ] T017 [P] [US2] Rate limit per token as a backstop; the budget is the website's, enforced before the call reaches here (FR-008)
- [x] T018 [US2] Return `state: failed` with no partial answer on any internal error, so the page renders no panel (FR-006)
- [x] T018a [US2] Bound the answer at 120s in src/api/answer.py; FR-006 names timeout and only the LLM client's 360s-per-call limit existed, so a stuck upstream could hold a connection for over half an hour (PR #236)

## Phase 5: Latency (does NOT gate the handover)

- [ ] T019 Measure first-token and completion separately across the tracked questions; publish the distribution, not one question. Baseline for one question, 2026-09-17: first ANSWER token at 36.1s, not the 3.0s a naive stream-the-first-token measurement reports
- [ ] T020 Reduce query expansion from 5 variants, measuring recall with bin/retrieval_baseline — its own call plus a 5x retrieval fan-out
- [ ] T020b Establish whether the four preprocessing calls must be sequential, and whether a search-page question needs all of them. They cost ~16s before retrieval starts and produce 36 tokens between them — the largest block in front of the first answer token
- [ ] T021 Land spec 009 collection routing and re-measure
- [ ] T022 Re-assess FR-005 against the result and say plainly whether 2s/10s is reachable

## Phase 6: Handover

- [ ] T023 Tell the website session the endpoint exists, with a curl that streams, and what it does not yet do
- [ ] T024 [P] Update specs/010-search-page-answers/spec.md status and record the measured first-token and completion times

## Dependencies

```
Phase 1 -> Phase 2 -> Phase 3 -> Phase 4 -> Phase 6
                                   Phase 5 runs alongside 4 and 6
```

T004 blocks T005; T005 blocks T010; T007 blocks T012; T010 blocks T013, T014, T015.

## Parallel opportunities

- T007 and T008 alongside T004-T006: token verification touches nothing the graph does
- T017 and T024: different files

## Implementation strategy

Ship Phases 1-4 as one PR: a correct, slow, verified, streaming endpoint. That is the
handover. Phase 5 follows on its own, with its own measurements, and the website repo
is not waiting on it.
