# Feature Specification: Chatbot Answers in the Website Search Results

**Feature Branch**: `010-search-page-answers`

**Created**: 2026-09-17

**Status**: Historical record of the search-page answer endpoint work (this repo's
half only), 2026-09-17 to 2026-09-20. The endpoint, streaming, caller-token
verification, the backstop rate limit and release attribution were implemented
2026-09-17 to 2026-09-18 (212c80e, 39b73be; PRs #236, #237, #240); `answer_id` was
added 2026-09-25 by spec 013. Not built: the latency target (FR-005a, first token
not met), choosing which searches get an answer (D2, User Story 3) and caching
(D3); whether these are still wanted is not recorded. The requirements and
decisions below were set for that work, not for later changes. Current endpoint
behaviour is in [contracts/answer_endpoint.md](./contracts/answer_endpoint.md) and
`src/api/answer.py`.

**Input**: *"a feature we want is for the chatbot responses to be integrated into
the website search page results in a google like style ... this would have to be a
coordinated effort with the website repo ... the functionality should only be
available to proven to be human users."*

## The latency constraint this work was designed around (2026-09-17 to 2026-09-18)

The first measurement, 2026-09-17, was on beta, Release97 bundle, `gpt-4o-mini`,
warm process; its figures were later replaced. The table is the 2026-09-18
re-measurement, two runs of each tracked question. The latest recorded measurement
is 2026-09-20: first token p50 5.57s, complete p50 8.90s (T022 in
[research.md](./research.md)).

| | |
|---|---|
| First answer token | min **2.8s**, p50 **9.6s**, p90 **12.2s**, max **14.0s** (n=26) |
| Whole answer | min **3.4s**, p50 **10.4s**, p90 **18.1s**, max **21.8s** (n=30) |
| Preprocessing | 2.6s, in two rounds of two calls |
| Query expansion and retrieval | 2.3s |
| Retrieval end to first answer token | 6.1s -- now the largest single block |
| Graph construction | ~70s, once at startup |

*This table has been wrong twice. It first quoted "25.4s and 32.2s", which was two
runs of one question near the maximum. It then quoted p50 15.2s for the whole answer
and ~36s to the first token; neither reproduces now. Preprocessing went from four
sequential calls to two rounds; the rest of the difference is not fully explained
(see the contract). Collection routing (spec 009) had not landed when this was
written. Re-measure after any retrieval or preprocessing change rather than
quoting these.*

A Google AI overview is generally reported to arrive in one to two seconds -- an
assumption here, not a measurement of ours. At a p50 of just under ten seconds to the
first token (2026-09-18; 5.57s on 2026-09-20, T022 in research.md), a panel on the
search results page is still empty long after a person
has read the ordinary results. The gap has closed by more than half since this spec
was written and the conclusion still holds, which is the point worth keeping: the
requirement was never about a specific number.

This was treated as deciding whether the feature works, not as polish, so it was
stated up front rather than discovered during implementation.

Two consequences ran through the rest of this spec.

The answer was made to **stream** (FR-002), on the expectation that something would
appear in the first second or two. It did not: the first answer token cannot arrive
before preprocessing and retrieval finish (next section).

And retrieval was then the largest single component for the questions that matter
here -- but **collection routing was not the only lever on it, and might not be the
biggest**.
Retrieval cost scales with *queries x collections*. Query expansion turns one
question into **five** queries at a cost of 2.4s, and every one of them is run
against every collection. Cutting five collections to two saves about as much as
cutting five queries to two, and at the time only the first had a spec (the query
count became configurable on 2026-09-20, T020). Measured 2026-09-17:

| | |
|---|---|
| 5 queries x 5 collections, served async path | 12.5s |
| 1 query x 5 collections, served async path | ~1.5s |

When this was written (2026-09-17), [spec 009](../009-collection-routing/spec.md)
had not been built; it landed 2026-09-19. The claim that it alone was on the
critical path did not survive this measurement.

## The latency target, and why it is not a requirement yet

This spec was written with a first-token budget of two seconds, on the assumption that
streaming was the answer to a slow completion. Building the streaming surface
disproved that, and the requirement changed rather than the measurement.

Measured 2026-09-17 by streaming `astream_events` and grouping by `run_id`:

| phase | seconds at the median |
|---|---|
| preprocessing -- rephrase \| language, then safety \| intent | 2.6 |
| query expansion and retrieval | 2.3 |
| retrieval end to the answer's first token | **6.1** |
| first answer token | **9.6** |

**Nothing of the answer exists until preprocessing and a retrieval have finished**,
but that is now under five seconds rather than the twenty this spec previously
recorded. On these 2026-09-17 figures the largest remaining block was the answer
model's own time to first token against a retrieved context. The 2026-09-20
measurement (T022 in research.md) puts that at 0.84s and retrieval at 3.15s
instead; the two have not been reconciled.

A naive measurement reports 3.0s, because the first streamed token of any kind
belongs to the rephraser. That number is wrong in the most flattering direction, and
is the one a casual check would have produced.

**So the first increment made no latency promise**, and none has been made since. It
was correct, verified and streaming, at the speed of the day, so the website repo
could build the panel, the token handshake and every failure path against it, none
of which depend on how fast it is.

What would have had to change for FR-005a, in the order the 2026-09-17 measurement
suggested. The 2026-09-20 measurement (T022 in research.md) reorders it: retrieval
first at 3.15s, then preprocessing at 1.58s, with the answer model at 0.84s, a
figure not reconciled with the 6.1s below.

1. **The answer model's time to first token**, 6.1s against a retrieved context and
   the largest block left. Nothing in these specs addresses it; a smaller or faster
   model, or a shorter context, are the obvious handles
2. **Preprocessing**, now 2.6s in two rounds. Whether a search-page question needs
   all four calls at all is still open -- the rounds only removed the waiting
3. **Query expansion and retrieval**, 2.3s together, measured before
   [spec 009](../009-collection-routing/spec.md)'s collection routing landed, with
   every collection searched

Stating a budget the code could not meet would have baked it into a contract another
repo builds against, which is why FR-005 makes no latency promise and FR-005a is a
target.

## User scenarios as written 2026-09-17, and where they stand (checked 2026-10-21)

### User Story 1 - A visitor searches and sees an answer forming (Priority: P1)

As written: someone searches reactome.org and, above the ordinary results, an answer
begins appearing within about two seconds and completes in under ten, with citations
into Reactome. The timing was the FR-005a target; the first-token half was not met
(T022 in research.md).

**Acceptance as written, and status**
1. Tokens stream: more than one token event. Holds (`src/api/answer.py:166-171`; T006)
2. The first token belongs to the **answer**, not to the rephraser or the query
   expander. Holds: answer tokens are those from the `model` node after a retriever
   completes, or after preprocessing on the live-lookup path
   (`src/agent/graph.py:500-583`; research R5c)
3. "Every factual claim carries a citation resolvable at
   `reactome.org/content/detail/<stId>`." Not as built: citations are separate events
   listing what retrieval found, not per claim (research R3); a user-guide page is
   cited by `url` (T028); a live-lookup answer has none; and links in the prose are
   stripped (contract, "What `token` text contains")
4. A question the knowledgebase cannot answer says so rather than inventing. Asked of
   the model by the answer prompt (`src/retrievers/reactome/prompt.py:17`); such an
   answer ends `answered`, not `nothing_found`

### User Story 2 - A caller without a valid token gets no answer (Priority: P1)

Written as "someone who has not been verified gets no answer", expecting HTTP
401/403. D1 (2026-09-18) changed the premise: the search path has no human check, so
what is verified is a token the website mints for each call, not a person.

**Why P1**, as written: a cost and abuse control, not a feature toggle. Every search
reaching the model is a bill, and search pages are crawled.

**As built**: a missing, invalid, expired or wrong-audience token, or a caller over
the backstop limit, gets HTTP 200 with a single `done` event, `state: refused`, and no
model call (`src/api/answer.py:79-113`; `tests/api/test_answer_endpoint.py:153-176`).
Not 401/403: a refusal is a well-formed stream so the website handles one shape
(`src/api/answer.py:79-85`).

**Acceptance as written, and status**
1. No valid token → refused before any model call. Holds
2. "A token is bound to the person, expires, and cannot be replayed from elsewhere."
   Partly: tokens expire (`exp` is required; the website mints at +120s) and must
   carry this service's audience, so one minted for another consumer is refused. They
   are not bound to a person, and a token can be presented again until it expires:
   nothing records use (`src/util/caller_token.py:34-35`)
3. Refusal is cheap and does not consume a rate-limit slot. Holds: token refusals
   happen before the limiter, and a rate-limited request is not counted
   (`src/util/rate_limit.py:66-76`)

### User Story 3 - Not every search gets an answer (Priority: P2)

As written: a search for a single gene name, or a navigational query, returns ordinary
results with no AI panel; only questions an answer would serve get one, tested
against queries labelled should-answer / should-not.

**Not built.** The endpoint answers every request that passes the token and rate
checks; nothing classifies a search as not worth answering (D2). Whether this is
still wanted is not recorded.

### Edge cases as written, and status

- **Retrieval finds nothing**: say so; never fill the gap with general knowledge.
  Asked of the model by the prompt (`src/retrievers/reactome/prompt.py:14-17`)
- **The model is slow or the upstream is down**: the panel disappears rather than
  hanging. This service ends the stream with `done` `failed` after 120s or on an error
  (`src/api/answer.py:67`, `:219-233`); what the page shows is the website's
- **The person navigates away mid-stream**: the request is cancelled. Holds, measured
  2026-09-18 (T027; `src/api/answer.py:192-218`)
- **The same question twice**: was to be served from cache (D3). Not built: every
  request is answered afresh
- **A question that is unsafe or off-topic**: the safety checker refuses it, and the
  refusal must not render as a panel. Holds: the refusal comes from a different graph
  node and is not streamed, so the stream ends `nothing_found` with no tokens
  (`src/agent/profiles/react_to_me.py:101-110`; `src/agent/graph.py:580-583`)

## Requirements of this work, and where they stand (checked 2026-10-21)

| ID | Requirement as written | Status |
|---|---|---|
| FR-001 | An HTTP endpoint accepting a question and returning an answer with citations (none existed; Chainlit owned the conversation over websockets) | Holds: `POST <CHAINLIT_URI>/api/answer` (`src/api/answer.py:96`, `bin/chat-fastapi.py:73-78`) |
| FR-002 | The response streams | Holds: server-sent events (`src/api/answer.py:129-264`) |
| FR-003 | Refuse any request without a valid caller token, before any model call | Holds (`src/api/answer.py:98-115`) |
| FR-004 | Citations are Reactome stable IDs, so the website styles links itself | Superseded 2026-09-18 (T028): a citation carries `st_id` or, for a user-guide page, `url` (`src/agent/graph.py:219-261`). Live-lookup answers carry none |
| FR-005 | The caller can render progressively; no latency guarantee in the first increment (see "The latency target, and why it is not a requirement yet") | Holds; no latency guarantee has been made since |
| FR-005a | *Target, not a requirement*: first token within 2s and complete within 10s, at p50 | Last measured 2026-09-20 (T022): complete 8.90s p50, met; first token 5.57s p50, not met. Not re-measured since |
| FR-006 | Fail invisibly: any error, timeout or refusal gives a response the website can render as "no panel" | Holds within the stream: errors, the 120s timeout and refusals end with a non-`answered` `done` (`src/api/answer.py:59-67`, `:79-93`, `:219-233`). Two limits: tokens sent before a failure stay sent, and a malformed request body gets HTTP 422 JSON, not a stream |
| FR-007 | Answers attributable to a Reactome release, so a cached answer can be identified after one | Holds: `release` on `start`, read at startup (`bin/chat-fastapi.py:55-59`); `null` when unknown. No cache uses it (D3) |
| FR-008 | Rate limited per verified person, independently of the chat UI | Built as a per-caller backstop: 30 requests per 600s by default, keyed on the token's `sub`, else `jti`, else a hash of the token (`src/util/rate_limit.py:30-49`, `:94-103`). It counts callers, not verified people (D1) |

### Key Entities

- **Question**: the search string. The release is the one the service started with,
  not something the caller sends
- **Answer**: streamed text, a list of citations (a stable ID or a URL, and a display
  name), and a completion state -- `answered`, `nothing_found`, `refused` or `failed`
- **Caller token** (called "human token" until 2026-09-18): a short-lived token the
  website mints for each call. It asserts the call came from the website, not that a
  person was verified (D1). This service's Turnstile check guards the chat UI, not
  this endpoint

## Success criteria, and where they stand (checked 2026-10-21)

- **SC-001**: the endpoint streams -- more than one token event for a question the
  knowledgebase can answer -- and its first-token and completion times are measured
  and published as a distribution over the tracked question set. A number going down
  is the goal; a number being met is not, until FR-005a's blocker is removed.
  *Done 2026-09-18 (T019) and re-measured 2026-09-20 (T022, research.md)*
- **SC-002**: Zero model calls for requests without a valid token, measured by
  counting calls under a load of unauthenticated requests. *Pinned by tests that
  assert a patched graph is never called (`tests/api/test_answer_endpoint.py:153-176`,
  T016); no load measurement is recorded*
- **SC-003**: The answer sweep stays green, and the two surfaces stay
  *configured* the same -- same profile, same shared graph, no difference that can
  reach the answer. Textual equality is explicitly **not** the criterion, because
  it is not achievable: measured 2026-09-18 on one graph at temperature 0, the
  same question asked twice through the *same* surface produced answers 0.331
  similar, while endpoint-versus-chat scored 0.356. The surfaces differ from each
  other no more than either differs from itself. The sweep is the right mechanism
  precisely because it matches patterns rather than literals. *Pinned by
  `tests/api/test_answer_matches_chat.py` (T014): one profile, one shared graph. The
  one configured difference, the endpoint skipping the post-answer web search
  (`src/api/answer.py:158-164`), cannot change the answer text*
- **SC-004**: No search-page request can make the search page itself slower or fail;
  verified by taking the service down and confirming the page still renders. *The
  website's side; this repository has no record of that check*

## Known: retrieval is not reproducible (measured 2026-09-18, not re-measured)

Measured 2026-09-18, the same question asked three times returned **12 citations
each time but only 4 pathways common to all three**, a union of 19 and a Jaccard
of 0.26 between two runs. Query expansion is itself a model call, so each run
expands the question differently and retrieves different documents.

This matters beyond wording. A reader who reloads the panel sees different
sources, and FR-007's cache invalidation assumes an answer is a stable artifact
of a release. Deciding what to do about it -- seeding or caching the expansion,
or dropping it for this path -- was left open (T020a), and no decision is recorded.
"Continue in chat" keeps the answer as it was shown rather than regenerating it, for
this reason (spec 013, `src/api/answer_store.py:3-6`).

## Decisions

### D1 -- what does the caller present? (recorded as decided with the website side, 2026-09-18; implemented in 39b73be)

This was "what proves a person is human?", and the question had a false premise.
The website session checked before answering: there is **no human gate on the
search path**, and there is not going to be one -- nobody solves a captcha to run
a search. Their only hCaptcha belongs to the contact form, gating that form and
spent on submit. Every search is anonymous and ungated by design.

So the token cannot honestly assert humanity, and asking it to would have meant
inventing a claim. **It asserts caller identity instead**: minted server-side by
the website, per request, EdDSA, with `iss`, `aud: reactome-chatbot`, `iat`,
`exp` at +120s, and `sub` -- described then as an opaque per-visit id of 128 random
bits, not derived from anything about the reader, held in an HttpOnly cookie. No
address, no IP hash; neither side was to hold personal data. *(Corrected 2026-09-19
in `src/util/caller_token.py:15-22`: for a reader who has passed the website's
Turnstile challenge, `sub` is that identity cookie's subject and lasts as long as the
cookie; the per-visit id is the fallback for readers who have not.)*

This service verifies signature, expiry and **audience**, and keys its backstop
limit on `sub`. The request field was renamed from `human_token` to `caller_token`
in the code and in this folder (PR #240, finished in 6d517a0).

Abuse control moved with the premise, to the website. As the website side described
it on 2026-09-18, the panel is opt-in behind a click, so a crawled search never
reaches a model, and its proxy rate limits by address. Neither is visible from this
repository.

The decision kept Turnstile where it was, guarding the chat UI only. It was never
part of this path, and the decision was not to merge the two checks.

### D2 -- which searches get an answer? (open; not built)

Not all of them, per User Story 3. The recommendation recorded on 2026-09-17 was to
extend the existing intent classifier rather than add a second one, which would cost
a call, and the question was left open pending
[spec 007](../007-answer-cascade/spec.md) (which does not itself mention the search
page). No decision is recorded. The endpoint answers every request that passes the
token and rate checks.

### D3 -- caching (open; not built)

Search traffic was expected to repeat in a way chat traffic does not, so a cache
keyed by question and release would cut both latency and cost; FR-007 exists to make
invalidation possible. It was left open for want of a repeat-rate measurement on real
search traffic, which the website has and this repository does not. No decision is
recorded and no cache exists. Retrieval not being reproducible (above) bears on what
such a cache could store.

## Scope of this work (as set 2026-09-17)

**In (this repo)**: the answer endpoint, streaming, citation shape, token
verification, rate limiting, release attribution, and latency work towards FR-005a.

**Out (the website repo)**: the panel, its visual design, where it sits on the page,
how citations render, and the UX of the chat's deeper integration. This spec
committed to a contract, not a look.

**Out (both, at the time)**: the broader "chat integrated into the site's flow and
feel" work, left for its own spec once this contract existed.

## Assumptions made 2026-09-17, and how they held

- The website can verify a human before calling this service. Did not hold: the
  search path has no human check, and the token asserts caller identity instead (D1)
- Beta is the integration target first; the MCP pointed at `beta.reactome.org` as of
  2026-09-17. Beta serves the endpoint at `/chat/guest/api/answer`
  (`deploy/beta/env.beta.template:21`, `deploy/beta/README.md` section 4)
- The same graph serves the endpoint and the chat UI. Holds: one shared instance
  (`src/agent/registry.py`); SC-003 says what "the same" can and cannot mean
