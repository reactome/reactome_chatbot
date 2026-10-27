# Search-Page Answers: Task-ID Index

Historical. The step-by-step task list for this work (T001-T028) is in git history
up to 3a4bc94. Kept here are the IDs that code comments and this folder refer to.
The ID T023 was used twice; code that cites T023 means the live-path failure signal.

| ID | Task | Outcome |
|---|---|---|
| T002 | Record the baseline before building | 2026-09-17: whole answer p50 15.2s, p90 22.4s; in research.md, R6 |
| T003 | Pin the captcha middleware before adding a route beside it | `tests/util/test_captcha_scope.py`; the rule is in `src/util/captcha_scope.py` |
| T004 | One graph shared by Chainlit and the endpoint | `src/agent/registry.py`, built in the FastAPI lifespan |
| T005, T006 | A streaming surface, tested for streaming rather than for a 200 | `AgentGraph.astream_answer` (`src/agent/graph.py:461`) |
| T007-T009, T015, T016 | Caller-token verification, refusal before any model call, no start without a key | `src/util/caller_token.py`, `src/api/answer.py`; tests in `tests/util/test_caller_token.py`, `tests/api/test_answer_endpoint.py`. T007 said "signature and expiry only"; audience and other checks were added later (T025; research R4) |
| T010-T013, T011a, T013a | The SSE endpoint, citations from retrieval metadata, mounting, an HTTP test, link stripping, a first end-to-end run | `src/api/answer.py`. First run 2026-09-17: release 97, answered in 19.5-44.2s, 12 citations, no anchors |
| T014 | SC-003: the two surfaces stay configured the same | `tests/api/test_answer_matches_chat.py` (PR #237). Answer equality was dropped as the criterion: same surface twice 0.331 similarity, endpoint vs chat 0.356 (2026-09-18) |
| T014a, T014b | A checkpointer thread per request; `release` on `start` and `seconds` on `done` | PR #236 |
| T017 | Backstop rate limit | `src/util/rate_limit.py` (PR #237) |
| T017a | Stop running a web search nobody receives | `enable_postprocess=False` (`src/api/answer.py:158-164`, PR #237) |
| T018 | `state: failed` on any internal error (FR-006) | Built. The task said "with no partial answer", but tokens already sent stay sent and the stream then ends `done` `failed` (contract, Known gaps) |
| T018a | Bound an answer at 120s | `ANSWER_TIMEOUT_SECONDS` (`src/api/answer.py:59-67`, PR #236) |
| T019, T019a | Measure the latency distribution; run preprocessing in two rounds | 2026-09-18, PR #238; figures in spec.md |
| T020 | Measure query expansion and make the count configurable | research.md, "Query expansion, measured 2026-09-20"; `QUERY_EXPANSION_ALTERNATES`, default unchanged |
| T020c | Does a search-page question need all four preprocessing calls? | research.md, 2026-09-20 |
| T022 | Is 2s/10s reachable? | research.md, 2026-09-20: complete met, first token not |
| T023 | Carry a live-tool failure out of the graph as a fact, not prose | `LiveReport` → `live_tool_failed`, which `src/evaluation/answer_sweep.py` retries on (81d24c3, 2026-09-20). The no-tools fallback deliberately does not set it (`src/agent/profiles/react_to_me.py:219-236`) |
| T023 (second use) | Hand the endpoint over to the website | Done 2026-09-18; recorded as: they built against it and a token they minted verified |
| T024 | Record the measured times in spec.md | Done 2026-09-18 |
| T025, T026 | Enforce `aud`; rename `human_token` to `caller_token` (D1) | PR #240 (39b73be), finished in 6d517a0 |
| T027 | Does a client hang-up cancel the work? | Yes, measured 2026-09-18 (PR #240) |
| T028 | Cite user-guide pages by `url` | Recorded as approved 2026-09-18; built in 0780086 |

Not done when the list was retired, and no record of whether they are still wanted:

- T020a, deciding what to do about non-reproducible retrieval: three runs of one
  question shared only 4 of 19 citations (Jaccard 0.26), because query expansion is
  itself a model call. It affects what FR-007 could cache. No decision is recorded
- [ ] T021 Land spec 009 collection routing and re-measure. **Routing landed 2026-09-19 (0fe7d88), after this task was written; no re-measurement is recorded.** When written on 2026-09-18, what had landed was *source* routing (`resolve_active_sources` picks reactome / userguide / live), not collection routing.
