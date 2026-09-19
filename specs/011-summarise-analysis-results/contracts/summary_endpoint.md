# Contract: the analysis summary endpoint

Not implemented. **Agreed with the website repo on 2026-09-19** -- the request
shape, the refusal shape and how human presence is asserted are settled, and
they have the presence claims built on a branch. Nothing here is open.

## Request

```
POST {CHAINLIT_URI}/api/analysis-summary
Content-Type: application/json

{ "token": "MjAyNjA5MTgxMjM0NTY",
  "caller_token": "<the website's EdDSA JWT, as for /api/answer>",
  "disclosure": "aggregate" }
```

**The analysis token is passed straight through, never validated by the
caller.** This service already distinguishes 404, 410 and the undocumented 500
a malformed token returns (research D2). A second validator on the website
would be a second thing that believes it knows what a good token looks like,
and two validators disagree eventually.

`disclosure` is `aggregate` or `identifiers`, and is **required** — there is no
default, because a default is not a choice. `aggregate` never transmits the user's
identifiers, filenames, sample names or expression column labels.

### Human presence

`caller_token` carries three additional claims, minted only when the website's
Turnstile-backed identity cookie validated on that request:

| claim | meaning |
|---|---|
| `human` | `true`. **Absent otherwise, never `false`** -- a missing claim and a failed check are indistinguishable here |
| `human_iat` | epoch seconds, when the challenge was solved |
| `human_sub` | the cookie's random 16-byte subject, for per-person rate limiting |

**Freshness is 30 minutes and the bound is inclusive**: a challenge solved
exactly 1800.000s ago is accepted, 1800.001s is not. Both sides enforce it --
the website refuses to mint past it, this service refuses to accept past it --
so neither is a single point of failure, and if it is ever changed it is
changed in both places in one change.

**`human_sub`, not `sub`.** `sub` already means something here: an opaque
*per-visit* id, which `identity_of` uses to key the answer endpoint's backstop
limiter. The cookie subject is also 16 bytes but is per-*browser* and lives as
long as the cookie. Putting it in `sub` would silently change that limiter from
"this visit" to "this browser", on an endpoint neither repo is otherwise
touching. A separate claim also makes it explicit that a persistent
pseudonymous identifier is being held -- hashed, in memory only.

**And the model of `sub` above was itself wrong.** The website corrected it on
2026-09-19: their `callerSubject()` prefers the identity cookie's subject
whenever the reader has passed a challenge, falling back to the per-visit value
only when they have not. So `sub` has *already* been browser-scoped for every
gated reader, and the answer endpoint's backstop limiter has been counting
across visits since the gate shipped. That is the stronger throttle and it is
kept; the source comments describing it as per-visit are fixed.

A consequence to write down: while `callerSubject()` prefers the verified
identity, `sub` and `human_sub` carry the **same value** whenever both are
present. They are still separate claims, because they mean different things and
would diverge the moment that preference changed -- and because agreement
between them is not a signal anything should test.

## Response: Server-Sent Events

The same framing as `/api/answer`, for the same reasons — a summary takes
comparable time, and a caller must be able to ignore it safely.

```
event: start
data: {"release": 97, "analysis_type": "OVERREPRESENTATION", "cached": false}

event: token
data: {"text": "Of the 312 pathways hit, four remain significant after "}

event: citation
data: {"st_id": "R-HSA-109581", "display_name": "Apoptosis"}

event: done
data: {"state": "summarised", "seconds": 6.2}
```

`state` is one of `summarised`, `not_found`, `gone`, `unsupported`, `refused`,
`failed`. Anything but `summarised` means render no summary. Always HTTP 200 —
never an error code, so the analysis page cannot be broken by this service.

`cached` on `start` says whether this text was generated now or reused. It exists
because the interface must not imply determinism it does not have: a reader who
regenerates may get different wording, and `cached: false` is when that happens.

## What a caller must handle

**`gone` is not `not_found`.** The Analysis Service deletes results on a new
release and says so with 410. A reader whose token returns `gone` should be told
their analysis predates the current release and to run it again — that is an
action, where `not_found` is a dead end.

**Summaries are stable per `(token, release, disclosure)`**, by storage rather
than by the generator being deterministic. The same request returns the same text
until the release changes. After a release the stored summary is discarded,
because the result it described has been too.

**An aggregate summary and a disclosing one are different artefacts.** They are
stored separately and must not be presented as the same summary at different
detail.

## What this endpoint will not do

- Run, re-run or re-implement an analysis. It summarises a result that exists.
- State a statistic the result does not contain.
- Describe the lowest p-value as a finding when nothing passes correction.
- Summarise a ReactomeGSA result. It recognises one and returns `unsupported`.
- Read from production. Results come from beta's Analysis Service.

## Open questions for the website side

1. How human presence is asserted (above) — the blocker.
2. Where the disclosure choice is presented, and how the two options are described
   so a user can tell what they are trading.
3. Whether the analysis page shows a summary automatically once consent exists, or
   requires the ask each time. The spec requires opt-in per request; if that is
   wrong for the page, it is a spec change rather than an implementation one.
