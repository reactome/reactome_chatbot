# Contract: the analysis summary endpoint

**Status**: Describes the endpoint as implemented in `src/api/analysis_summary.py`
(checked 2026-10-21). Built 2026-09-19 to 2026-09-20, live on beta from
2026-09-21. The request shape, the refusal shape and how human presence is
asserted were agreed with the website repo on 2026-09-19. Two points of that
agreement are not implemented; see "Known gaps" below.

## Request

```
POST {CHAINLIT_URI}/api/analysis-summary
Content-Type: application/json

{ "token": "MjAyNjA5MTgxMjM0NTY",
  "caller_token": "<the website's EdDSA JWT, as for /api/answer>",
  "disclosure": "aggregate" }
```

`{CHAINLIT_URI}/api` is the router prefix; with `CHAINLIT_URI` unset it is
`/chat/api` (`bin/chat-fastapi.py:77-83`).

**The analysis token is passed straight through, never validated by the
caller.** This service distinguishes 404, 410 and the undocumented 500 a
malformed token returns (research D2). A second validator on the website
would be a second thing that believes it knows what a good token looks like,
and two validators disagree eventually. This service checks the token's shape
itself: a token that is not 1-200 of `A-Z a-z 0-9 _ -` followed by up to two
`=` or `%3D` gets `not_found` without any request to the Analysis Service
(`src/analysis/client.py:35-51`, `:127-132`).

`disclosure` is `aggregate` or `identifiers`, and is **required** — there is no
default, because a default is not a choice. `aggregate` never transmits the user's
identifiers, filenames, sample names or expression column labels. It does pass
on up to five of the Analysis Service's own `warnings`, each cut to 200
characters. They are free text the service composes, and this service cannot
check them for quoted user input (`src/analysis/disclosure.py:65-76`).
`identifiers` additionally fetches up to 50 of the reader's unmatched
identifiers (`GET /token/{token}/notFound`, first page) and sends them to the
model (`src/analysis/client.py:199-237`).

### Request validation

`token` must be 1-256 characters and `disclosure` one of the two values above.
A body that fails this, or omits either field, gets **HTTP 422** with FastAPI's
validation error body, not an event stream (`SummaryRequest`,
`src/api/analysis_summary.py:85-89`; `tests/api/test_analysis_summary.py:285-292`).
`caller_token` may be omitted: that is a refusal (`no_caller`) inside an HTTP 200
stream, as below.

### Human presence

**This endpoint requires it and `/api/answer` does not.** A caller token with
no presence claim gets a full answer from `/api/answer` and `no_human` from
here. That asymmetry is deliberate and load-bearing — it is why the stricter
gate can be applied to one route and not the other — and it was not written
down until the website measured it on 2026-09-21.

The reason is what each endpoint discloses. `/api/answer` returns public
pathway text, and the search path it serves has no human gate at all; spec
010's D1 settled that its token asserts *service identity* and deliberately
says nothing about a person. This endpoint sends a user's own uploaded
analysis to a model provider, and at the disclosing tier their submitted
identifiers with it. The choice of what to disclose is only meaningful if a
person made it.


`caller_token` carries three additional claims, minted only when the website's
Turnstile-backed identity cookie validated on that request:

| claim | meaning |
|---|---|
| `human` | `true`. **Absent otherwise, never `false`** -- a missing claim and a failed check are indistinguishable here |
| `human_iat` | epoch seconds, when the challenge was solved |
| `human_sub` | the cookie's random 16-byte subject, for per-person rate limiting |

**Freshness is 30 minutes and the bound is inclusive**: `now - human_iat <= 1800`,
so a challenge solved exactly 1800s ago is accepted and 1801s is not.

Stated in whole seconds on purpose. The bound was first agreed as "1800.000
accepted, 1800.001 refused", which is not a distinction this claim can carry:
`human_iat` is epoch **seconds**, and it is derived as the cookie's expiry
minus a constant TTL, so it arrives already rounded to a second. A sub-second
edge would be a boundary neither side can actually be on, tested against a
clock finer than the value. Effective precision is one second, and a challenge
solved 1800.4s ago presents as 1800 and is accepted.

Both sides enforce it, by agreement --
the website refuses to mint past it, this service refuses to accept past it
(`HUMAN_MAX_AGE_SECONDS`, `src/util/caller_token.py:228`) -- so neither is a
single point of failure, and changing the bound needs a change on both sides.

How this service reads the claims (`human_presence_reason`,
`src/util/caller_token.py:280-309`):

- `human` anything other than JSON `true` → `refused` / `no_human`
- `human_iat` missing or not a number → `no_human`
- more than 1800 whole seconds old → `stale_human`
- more than 60 seconds in the future → `no_human`
- `human_sub` is not checked here; it is used only for rate limiting (see "Limits")

**`human_sub`, not `sub`.** The claim was named separately when `sub` was
understood here to be an opaque *per-visit* id, used by `identity_of` to key the
answer endpoint's backstop limiter; reusing `sub` for the per-browser cookie
subject would have changed that limiter's meaning. The website corrected that
understanding on 2026-09-19: their `callerSubject()` prefers the identity
cookie's subject whenever the reader has passed a challenge and falls back to a
per-visit value only when they have not. So `sub` is browser-scoped for every
gated reader, and the answer endpoint's limiter counts across visits for them.

As a result, while `callerSubject()` prefers the verified identity, `sub` and
`human_sub` carry the **same value** whenever both are present. They remain
separate claims because they mean different things and would diverge if that
preference changed; nothing should test that they agree.

**How `human_sub` is held.** It is used as received, not hashed, in a
rate-limiter key (`human:<value>`) in this process's memory
(`src/api/analysis_summary.py:157-161`). A key is dropped at the limiter's next
sweep once it has had no request for a full window (`src/util/rate_limit.py:78-91`).
The 2026-09-19 agreement said "hashed, in memory only"; see "Known gaps".

## Response: Server-Sent Events

The same framing as `/api/answer`, for the same reasons — a summary takes
comparable time, and a caller must be able to ignore it safely.

```
event: start
data: {"release": 97, "analysis_type": "OVERREPRESENTATION", "cached": false, "disclosure": "aggregate"}

event: citation
data: {"st_id": "R-HSA-109581", "display_name": "Apoptosis"}

event: token
data: {"text": "Of the pathways shown, four remain significant after "}

event: done
data: {"state": "summarised", "seconds": 6.2}
```

Event order (`src/api/analysis_summary.py:166-328`):

- **Summarised**: one `start`; then one `citation` per pathway sent to the
  model (the first 12 of the result, `src/analysis/disclosure.py:89`), all
  before any text; then `token` events; then `done`. A newly generated summary
  streams as many `token` events. A stored summary arrives as **one** `token`
  event carrying the whole text.
- **Refused**: only `done`, with `seconds: 0.0` and a `reason`. No `start`.
- **`not_found`, `gone`, `unsupported`**: only `done`. No `start`.
- **`failed`**: `done`, with or without a preceding `start` and partial `token`
  events, depending on where it failed. A failure before `start` (the Analysis
  Service unreachable, timing out or answering unexpectedly) sends only `done`.

`done` carries `state` and `seconds` (rounded to 0.1), and `reason` only when
`state` is `refused`. `reason` is one of:

- `no_caller`: no caller token, or one that is invalid, expired or too long
- `no_human`
- `stale_human`: the only one a reader can fix, by passing the challenge again
- `rate_limited`

The handler also has an `unsupported_tier` refusal
(`src/api/analysis_summary.py:154-155`). It cannot occur today, because an
unknown `disclosure` value is rejected with a 422 first.

Citations are every pathway the model was given, not only the ones the text
discusses. They come from the result rather than the model's prose, so each is
a pathway present in the result (SC-003).

`state` is one of `summarised`, `not_found`, `gone`, `unsupported`, `refused`,
`failed`.

> **These are not `/api/answer`'s state names, and the two are easy to
> conflate.** That endpoint's success state is `answered` and its empty state
> is `nothing_found`; this one's are `summarised` and `not_found`. The event
> shapes are otherwise nearly identical, so a consumer building both panels
> from one mental model will map a *successful* summary onto an unrecognised
> state — which, if the fallback is `failed`, renders a complete and correct
> summary as a truncated failure above and below the text. That happened on
> 2026-09-21.
>
> `release` is a **number** on both, though it reaches this one as text from
> the Analysis Service and is parsed here. It is null when unparseable or when
> the release lookup fails, which is a legitimate value on both endpoints.

Anything but `summarised` means render no summary. Every outcome above arrives
as an HTTP 200 event stream, so a failure here does not become an error status
on the analysis page. The exception is a malformed request body, which gets
HTTP 422 (see "Request validation").

**No prose source list.** Citations arrive as `citation` events, as on
`/api/answer`. The model is told not to end the text with a list of them
(`SYSTEM_PROMPT` rule 6, `src/api/analysis_summary.py:76-77`). Unlike
`/api/answer`, which strips one with `SourcesSectionStripper`, this endpoint
does not remove a list if the model writes one anyway.

The text is model output, passed through as generated; this endpoint does not
sanitise markup in it.

**Expression summaries are told to refer to columns as `column 1`, `column 2`**,
numbered from one in the order the values appear, and in no other form
(`TYPE_INSTRUCTION["EXPRESSION"]`, `src/analysis/summarise.py:179-195`). This is
a prompt instruction, not checked in code. It held on the one expression result
checked on beta on 2026-09-21. The labels are user-supplied text this service
receives from the Analysis Service but never sends to the model, so the caller
is the only side that can name them -- and can only substitute them into the
prose if the wording is fixed. Anything else ("the first column", "the
leftmost sample") reaches the reader as written.

**`disclosure` on `start` is the tier the summary was actually built from**,
which is not always the one requested. If `identifiers` was asked for and the
unmatched identifiers could not be retrieved, the summary is the aggregate one
and this says `aggregate`. A stored aggregate summary for the token is reused if
there is one.

Without it the reader chooses to disclose, the lookup fails, and they are
handed the other summary with nothing to distinguish it -- disclosure with no
benefit, and no way for the interface to say so. A caller should surface the
difference rather than silently presenting an aggregate summary as the
disclosing one.

Nothing to disclose is **not** a failed disclosure: a result where every
identifier matched reports `identifiers`, because the tier was honoured and
there was simply nothing to retrieve.

`cached` on `start` says whether this text was generated now or reused. It exists
because the interface must not imply determinism it does not have: a reader who
regenerates may get different wording, and `cached: false` is when that happens.

## What a caller must handle

**`gone` is not `not_found`.** The Analysis Service deletes results on a new
release and says so with 410. A reader whose token returns `gone` should be told
their analysis predates the current release and to run it again — that is an
action, where `not_found` is a dead end.

**Summaries are reused per `(token, release, disclosure)`**, by storage rather
than by the generator being deterministic (`src/analysis/store.py`). The same
request returns the same text, with `cached: true`, while all of these hold:

- the same process is still running. The store is in memory, so a restart or
  deploy loses every stored summary (research D4)
- the summary has not been evicted. The store keeps the 512 most recently used
  summaries (`src/analysis/store.py:35`)
- the Analysis Service still reports the same release. The release is part of
  the key, so after a release change the old summary is no longer found. It is
  not deleted; it ages out of the store
- the release lookup succeeded. When it fails, `release` is null and the
  summary is generated without being stored or reused

`disclosure` in the key is the tier the summary was built from (`disclosure` on
`start`). Every request still fetches the result and the release from the
Analysis Service first, so a deleted result reports `gone` even when a summary
of it is stored.

Concurrent requests for the same summary wait for the first one's generation
and are served its text, with `cached: true` (`src/api/analysis_summary.py:199-212`).
A stored summary is never replaced by a later generation
(`src/analysis/store.py:76-82`).

"Continue in chat" (spec 013, `POST /api/handoff`) can hand off only a summary
that is still stored here under the current release and the same tier
(`stored_summary`, `src/api/analysis_summary.py:61-67`; `src/api/handoff.py:142-157`),
so the same conditions limit it.

**An aggregate summary and a disclosing one are different artefacts.** They are
stored separately and must not be presented as the same summary at different
detail.

## What this endpoint will not do

- Run, re-run or re-implement an analysis. It summarises a result that exists.
- Summarise a ReactomeGSA result. It recognises one by `gsaMethod`/`gsaToken`
  or by type `GSA_REGULATION`, `GSA_STATISTICS` or `GSVA`, and returns
  `unsupported` (`src/analysis/client.py:84-109`).

The model is instructed not to state a statistic the result does not contain,
and not to describe the lowest p-value as a finding when nothing passes
correction. Whether anything passes is computed in code and given to it
(`src/analysis/summarise.py:51-66`, `:146-153`, `:266-276`). These are
instructions to a model, not guarantees.

Results are read from beta's Analysis Service by default; `ANALYSIS_BASE_URL`
points the service elsewhere (`src/analysis/client.py:22-23`, `:63-64`).

## Limits

| what | value | where |
|---|---|---|
| Whole request, from result lookup to the last token | 120 s, then `failed` | `SUMMARY_TIMEOUT_SECONDS`, `src/api/analysis_summary.py:47` |
| Model request | 90 s | `src/api/analysis_summary.py:351` |
| Each Analysis Service request | 20 s | `src/analysis/client.py:33` |
| Requests per person | 30 per 600 s by default (`ANSWER_RATE_LIMIT`, `ANSWER_RATE_WINDOW_SECONDS`). Keyed on `human_sub`, else the caller token's `sub`, else `jti`, else a hash of the token. Counted separately from `/api/answer`'s limit, which reads the same settings. Every request that passes the presence check counts, including ones served from the store | `src/api/analysis_summary.py:51`, `:157-164`; `src/util/rate_limit.py:30-49`, `:94-103` |
| Pathways sent to the model and cited | the first 12, in the result's order | `src/analysis/disclosure.py:89` |
| Unmatched identifiers sent (`identifiers` tier) | 50 | `src/analysis/client.py:200` |
| Caller token | at most 4,096 characters; `EdDSA` or `RS256`; `exp` and `aud` required; `aud` must equal `CALLER_TOKEN_AUDIENCE`, default `reactome-chatbot`; `iat`/`nbf` up to 30 s in the future tolerated | `src/util/caller_token.py:48-53`, `:109-121`, `:124-200` |

## Known gaps between the 2026-09-19 agreement and the code

Recorded here, not resolved. Which side should change has not been decided.

- **`human_sub` is not hashed.** The agreement says the identifier is held
  "hashed, in memory only". It is held in memory, as received
  (`src/api/analysis_summary.py:159-161`).
- **No per-analysis-token limit.** Research D6 records adopting the website's
  suggestion to rate limit per analysis token as well as per caller. Only the
  per-caller limit exists (`src/api/analysis_summary.py:157-164`).

## Questions raised for the website side (2026-09-18)

1. How human presence is asserted. Settled 2026-09-19 (above, and research D6).
2. Where the disclosure choice is presented, and how the two options are
   described so a user can tell what they are trading.
3. Whether the analysis page shows a summary automatically once consent
   exists, or asks each time. The spec requires opt-in per request (FR-011).
   This endpoint keeps no record of consent: it summarises whenever it is
   called with a valid request.

This repository does not record answers to 2 and 3.
