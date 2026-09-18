# Contract: the analysis summary endpoint

Draft. Not implemented, and not yet agreed with the website repo. The parts
marked **open** need their agreement before anything is built against them.

## Request

```
POST {CHAINLIT_URI}/api/analysis-summary
Content-Type: application/json

{ "token": "MjAyNjA5MTgxMjM0NTY",
  "caller_token": "<the website's EdDSA JWT, as for /api/answer>",
  "disclosure": "aggregate" }
```

`disclosure` is `aggregate` or `identifiers`, and is **required** — there is no
default, because a default is not a choice. `aggregate` never transmits the user's
identifiers, filenames, sample names or expression column labels.

**Open**: how the caller demonstrates a person is present. Spec 010's D1 settled
that `caller_token` asserts service identity and says nothing about humanity, so
it cannot carry this on its own. The likely shape is an additional claim minted
after the Turnstile check the website already performs on the chat, but that is
theirs to agree.

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
