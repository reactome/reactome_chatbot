---

description: "Task list for summarising analysis results"
---

# Tasks: Summarise analysis results

**Input**: Design documents from `specs/011-summarise-analysis-results/`
**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/summary_endpoint.md](./contracts/summary_endpoint.md)

Tests are requested for this feature. Three properties get them before their
implementation, because each is a claim about something *not* happening and none
of them fails visibly: what is never sent, that a dead token is distinguished from
a deleted one, and that the same token returns the same text.

## Phase 1: Setup

- [x] T001 Create `src/analysis/` with an `__init__.py`, separate from `src/agent/` because nothing here touches the graph or retrieval
- [x] T002 [P] Create `tests/analysis/` **without** an `__init__.py`. The task said to add one "alongside the existing `tests/api/`" and that premise was wrong: no test directory in this repository has one. Adding it put `tests/analysis` on `sys.path` as the top-level `analysis` package, shadowing `src/analysis`, and every import failed. Test basenames must also be unique for the same reason: `test_client.py` collided with `tests/reactome_mcp/test_client.py`, so this one is `test_analysis_client.py`
- [x] T003 Record the beta Analysis Service base URL in configuration rather than a literal, defaulting to beta and never production, in `src/analysis/client.py`

## Phase 2: Foundational (blocking)

**These block every user story. Nothing below Phase 2 can be built without them.**

- [x] T004 Write the disclosure allow-list in `src/analysis/disclosure.py`: name every field of an `AnalysisResult` that may be sent under the `aggregate` tier, as an allow-list rather than a denial, so a new field from the Analysis Service is excluded by default
- [x] T005 [P] Test the allow-list in `tests/analysis/test_disclosure.py` by **recording the outbound request body** and asserting `summary.fileName`, `summary.sampleName` and `expression.columnNames` never appear — not by reading the summary and seeing nothing alarming. These three are user-supplied free text and are the reason the tier is an allow-list (research D5)
- [x] T006 Fetch a result by token in `src/analysis/client.py`, with a browser-like `User-Agent`, because the site's automation blocking returns a 403 with an HTML body to library user-agents and it looks exactly like an auth failure
- [x] T007 [P] Test in `tests/analysis/test_client.py` that 404 yields `not_found`, **410 yields `gone`** and **500 also yields `not_found`**, as measured: a malformed token returns 500, which the OpenAPI does not document, and treating it as a service fault would produce a `failed` state or a retry loop against a service that will answer identically every time (research D2)
- [x] T008 Read the current release from `GET /database/version` in `src/analysis/client.py`, deriving it rather than hardcoding it, because it is both the reported release and the cache-invalidation key (Principle V). **That endpoint answers text/plain and rejects `Accept: application/json` with 406** -- found by calling beta, after every mocked test passed
- [x] T009 Detect a ReactomeGSA result from `gsaMethod`/`gsaToken` in `src/analysis/client.py` and return `unsupported`, so a result we do not model is never summarised confidently (research D8)

## Phase 3: User Story 1 — What does my result say? (P1)

**Goal**: A reader with an analysis token gets a readable account of what the result says.

**Independent test**: Submit a known token; the summary names the pathways the result ranks highest, describes significance the result supports, and cites each by stable id.

- [x] T010 [US1] Build the prompt input from an aggregate result in `src/analysis/summarise.py`: top pathways with their `found`/`total`/`ratio`/`pValue`/`fdr`, the analysis type, species, and the service's own `warnings`
- [x] T011 [P] [US1] Test in `tests/analysis/test_summarise.py` that a result where nothing passes FDR produces prompt input that says so, so the model is never handed a "top pathway" framing for a null result (FR-004)
- [x] T012 [US1] Emit pathway citations as `st_id` events reusing the answer endpoint's citation shape, in `src/api/analysis_summary.py`
- [x] T013 [US1] Add the SSE endpoint in `src/api/analysis_summary.py` per [contracts/summary_endpoint.md](./contracts/summary_endpoint.md): `start` with release and analysis type, `token`, `citation`, `done`
- [x] T014 [US1] Mount the router in `bin/chat-fastapi.py` and add its prefix to the captcha exemption, as the answer endpoint's is
- [x] T015 [US1] Test over HTTP on the real mounted app in `tests/api/test_analysis_summary.py`, not by calling the handler — mounting order and middleware interact only on the served path (Principle I), which is where spec 010's route check found what isolated tests could not
- [x] T016 [P] [US1] Test that every `st_id` a summary cites appears in that result's `pathways[]`, mechanically rather than by reading, in `tests/api/test_analysis_summary.py`, so an invented or mismatched identifier fails (SC-003)

## Phase 4: User Story 2 — Why were my identifiers not found? (P2)

**A constraint measured 2026-09-20 that changes scenario 1.** The aggregate
result carries `identifiersNotFound` and **nothing about how many identifiers
were submitted** — the denominator lives only behind `/token/{token}/found/all`,
which returns the reader's own identifiers and is therefore the disclosing
tier. So the proportion the scenario asks for is not derivable at the default
tier, and asking for one would invent a statistic the way D9 describes. The
count is reported instead; the reader knows what they submitted.

**Goal**: A reader learns why identifiers went unmatched and whether the result can be trusted.

**Independent test**: Submit a token from an analysis with a deliberate identifier mismatch; the summary reports the proportion and names the likely cause.

- [x] T017 [US2] Include `identifiersNotFound`, `pathwaysFound` and `resourceSummary` in the aggregate prompt input in `src/analysis/summarise.py`, which together explain most mismatches without disclosing anything
- [x] T018 [US2] Add the `identifiers` tier in `src/analysis/disclosure.py`, fetching `GET /token/{token}/notFound` only when the request asked for it
- [x] T019 [P] [US2] Test (in tests/api/test_analysis_summary.py, on the served path) that the `identifiers` tier is never reached without an explicit request, by asserting the not-found call is not made under the aggregate tier
- [x] T020 [US2] Test that a result with every identifier found produces a summary that says so rather than inventing a problem, in `tests/analysis/test_summarise.py` (spec US2 scenario 2)

## Phase 5: User Story 3 — What do these numbers mean? (P3)

**Fragility is computed in code, not asked for in the prompt.** A hit resting
on fewer than five matched entities is flagged per pathway, because a model
handed a tiny p-value and a tiny count describes the p-value. The threshold is
on the *count*, not the ratio: 2 of 3 looks perfect by ratio and is nearly
meaningless, 40 of 200 looks poor and is real evidence.

**No per-pathway request parameter was added.** US3's independent test implies
one, but the website has not asked for it and every shown pathway already
carries its own counts, so the explanation is per-pathway without new contract
surface.

**Goal**: The statistics are explained using the reader's own numbers.

**Independent test**: Ask about one pathway in a result; the explanation uses that pathway's counts, not generic definitions.

- [x] T021 [US3] Carry per-pathway counts into the prompt input for a named pathway in `src/analysis/summarise.py`
- [x] T022 [P] [US3] Test that a pathway significant by p-value but not by FDR is described as distinguishing the two, in `tests/analysis/test_summarise.py` (spec US3 scenario 2)
- [x] T023 [P] [US3] Test that a pathway with very few found entities is described as fragile, using its actual counts, in `tests/analysis/test_summarise.py` (spec US3 scenario 1)

## Phase 6: User Story 4 — Readings specific to the analysis type (P4)

**The type enum, read from the API on 2026-09-20 rather than guessed**:
`SPECIES_COMPARISON`, `OVERREPRESENTATION`, `EXPRESSION`, `GSA_REGULATION`,
`GSA_STATISTICS`, `GSVA`. Three are summarised; the three GSA ones are
declined.

That reading also closed a gap in T009: `is_gsa` recognised ReactomeGSA by
its `gsaMethod` field alone, so a result carrying one of those types
*without* that field would have been summarised confidently — the outcome D8
exists to prevent. Both signals are checked now.

**Goal**: An expression result and a species comparison each get the reading that fits them.

**Independent test**: Submit one of each; neither summary describes the other's kind of result.

- [x] T024 [US4] Branch the prompt input on `summary.type` in `src/analysis/summarise.py`, and for `EXPRESSION` carry `entities.exp[]` and the value range **without** `expression.columnNames`, which is user-supplied text
- [x] T025 [US4] For `SPECIES_COMPARISON`, state in the prompt input that findings are inferred by orthology in `src/analysis/summarise.py`, so the summary cannot present them as observed
- [x] T026 [P] [US4] Test that an expression result's summary refers to behaviour across columns and a species comparison's does not, and vice versa, in `tests/analysis/test_summarise.py`

## Phase 7: Stability and transparency (FR-014, FR-015)

- [ ] T027 Store summaries keyed `(token, release, tier)` in `src/analysis/store.py`; an aggregate summary and a disclosing one are different artefacts and must not be interchanged
- [ ] T028 [P] Test in `tests/analysis/test_store.py` that the same token returns byte-identical text on a second request, and that a release change discards the stored summary — the second half matters because the Analysis Service deletes the underlying result on a release (research D2, D3)
- [ ] T029 Report `cached` on the `start` event in `src/api/analysis_summary.py`, so the interface can say a summary was reused rather than implying the generator is deterministic (FR-015)
- [ ] T030 Record in [research.md](./research.md) that the first increment's store is in-process and lost on deploy, and open follow-up work for a durable store — beta sets no `POSTGRES_LANGGRAPH_DB` today (research D4)

## Phase 8: Human presence (FR-013) — UNBLOCKED 2026-09-19

Agreed with the website session; the shape is in [research.md](./research.md) D6.
Still do not let the plain caller token satisfy this by default: spec 010's D1
settled that it asserts service identity and says nothing about humanity. The
claims below are additional, and their absence must refuse.

**One dependency remains, and it is theirs**: their proxy mints caller tokens
for the answer route only. A summary route must exist before any claim can be
carried. Everything in this phase can be built and tested before that lands.

- [x] T031 Agree with the website session how human presence is asserted — `human`, `human_iat` and `subject` claims on the caller token, minted only when their Turnstile-backed identity cookie validated. They proposed gating on the analysis token instead and withdrew it: a token proves an analysis happened, not that a person is present, and tokens travel in pasted URLs
- [x] T031a Verify `human_iat` against a **30-minute** freshness bound in `src/util/caller_token.py`, refusing an older one. They refuse to mint past the same bound, so it fails at both ends rather than relying on either alone. **Compared in whole seconds** (`int(now) - int(issued)`), matching their `nowSeconds - floor(solvedAt/1000)`: with a float clock the inclusive bound is unreachable, because a claim issued exactly 1800s ago is 1800.0003s old when checked. Caught by the test pinning the edge
- [x] T031b Key the rate limiter on `human_sub` when present, falling back to the caller identity, in `src/api/analysis_summary.py` — per-person throttling rather than per-proxy-address. Also limit per analysis token: twenty summaries of one analysis is not a scientist
- [x] T031c Test that a `human` claim with a stale `human_iat` is refused with **zero model calls**, in `tests/api/test_analysis_summary.py` — the freshness bound is the half most likely to be dropped, because the claim being present looks like success
- [x] T032 [US1] Verify the assertion in `src/util/caller_token.py` -- `human_presence_reason` checks `human` and a 30-minute `human_iat`, refusing before any model call
- [ ] T033 [P] Test that a request without the assertion is refused and makes **zero model calls**, counted on a patched graph rather than inferred from timing, in `tests/api/test_analysis_summary.py` (SC-004)

## Phase 9: Polish

- [ ] T034 [P] Bound the summary in `src/api/analysis_summary.py` as the answer endpoint is, so a stuck upstream cannot hold a connection
- [ ] T035 [P] Log an abandoned summary stream in `src/api/analysis_summary.py`, as the answer endpoint does, so a caller that starts summaries it does not want is visible
- [ ] T036 Run the [quickstart](./quickstart.md) scenarios against beta with a real analysis token and record the outcome, including first-token timing
- [ ] T037 Tell the website session the endpoint exists, what it does not yet do, and the `gone` outcome they must handle — only once it is live on beta, not when it merges

## Dependencies

- **Phase 2 blocks everything.** T004 (the allow-list) blocks any task that sends result content anywhere.
- T006 blocks T010, T017, T021, T024 — nothing can be summarised before a result can be fetched.
- T008 blocks T027: the release is part of the storage key.
- T013 blocks T015, T029, T034, T035.
- **T031 is agreed (2026-09-19); T031a-c and T032-T033 follow from it, and block nothing else** — every other story can be built and tested behind a refusing gate. The website adding a summary route to its proxy is the only external dependency left.
- US1 is independent. US2, US3 and US4 each build on US1's prompt-input path but are separately testable.

## Parallel opportunities

- T005, T007 and T009 touch different files and can run together once T004 and T006 exist.
- T011, T016, T019, T022, T023, T026 are all tests in distinct files.
- T034 and T035 are independent polish on one file and should be done together.

## Implementation strategy

**MVP is Phase 1, 2 and 3** — an aggregate summary of what a result says, refusing
where human presence is not asserted. That is useful on its own: a reader with a
token gets a readable account, and nothing of theirs is disclosed.

Then US2, which is the question users actually ask most, followed by stability
(Phase 7) before the remaining stories, because a summary that changes on reload
undermines trust faster than a missing type-specific reading.

Phase 8 can be agreed in parallel with all of it and must land before the feature
is offered to anyone.
