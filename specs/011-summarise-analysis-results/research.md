# Research: summarising analysis results (2026-09-18 to 2026-09-21)

Findings from the analysis-summary work. Measured against beta's Analysis
Service (`/AnalysisService/v3/api-docs`, release 97) from 2026-09-18; later
sections give their own dates. Every claim below came from the API or from this
repository, not from recollection. "Decision" in a section means a decision for
that work. Corrections added on 2026-10-21 are marked. Current behaviour is in
[contracts/summary_endpoint.md](./contracts/summary_endpoint.md).

## D1 — Results are addressed by a token, and that is the whole input

**Decision**: The feature takes an analysis token. It never accepts an identifier
list and never runs an analysis.

**Rationale**: `GET /token/{token}` returns the completed `AnalysisResult`. The
analysis has already happened; re-running it here would duplicate the Analysis
Service and violate Principle V. It also means this endpoint never receives the
user's submitted list.

*(Corrected 2026-10-21.)* At the `identifiers` tier the endpoint does fetch up to
50 of the user's unmatched identifiers from the Analysis Service and sends them to
the model (D5; `src/analysis/client.py:199-237`). Separately, the chat's
gene-list feature (3b2c792, 2026-09-25) submits identifiers to the Analysis
Service through `submit_identifiers` in the same module. This endpoint does not
use it.

**Alternatives considered**: accepting identifiers and running the analysis
ourselves — rejected outright, it is another service's job and would put user data
through us needlessly.

## D2 — A result can be *gone*, and the two ways differ

**Decision**: Distinguish the two failure codes in what the user is told.

**Rationale**: the API defines exactly two error responses:

| code | meaning |
|---|---|
| 404 | no result corresponds to the token |
| **410** | **result deleted due to a new data release** |

These are not the same to a reader. 404 is "we cannot find that"; 410 is "that
analysis was run against an earlier release and has been discarded — run it
again". Collapsing them into one message wastes information the service went to
the trouble of giving us.

**And a third code the API does not document.** Measured against beta:

| token | response |
|---|---|
| well-formed but unknown (`MjAyNjA5MTgxMjM0NTY`) | 404, as documented |
| malformed (`x`, `%20`) | **500** — not in the OpenAPI at all |

So the client must treat 500 as a negative outcome too, not as a service fault to
retry or surface. FR-009 says an unknown, expired *or malformed* token is a normal
negative outcome; without this measurement the implementation would have handled
the two documented codes and let a malformed token become a `failed` state, or
worse a retry loop against a service that will answer the same way every time.

**Alternatives considered**: treating any non-200 as "no summary" — simpler, but
it would leave a user re-pasting a token that will never work again, which is why
410 stays distinct even though 500 does not.

**Later (2026-09-19).** A token outside the expected base64url-style shape is now
rejected before any request and reported as `not_found`, because a
caller-supplied `<token>/notFound` reached the identifier endpoint
(`src/analysis/client.py:35-51`).

## D3 — Stability comes from storing the summary, keyed by token *and release*

**Decision for this work**: store a generated summary against `(token, release)`
and serve the stored one thereafter. *(As built, 2026-09-20: the key is
`(token, release, tier)`, the tier being the one the summary was built from. A
release change is handled by the key, not by deleting anything: the old entry
is no longer looked up and ages out of the bounded store. `src/analysis/store.py`.)*

**Rationale**: FR-014 wants the same token to yield the same summary. Generation
cannot provide that — measured on this repo (spec 010, on the answer surface rather
than on summaries), the same question through the same surface twice scores 0.33
similarity. Storage can, because an analysis result is a
fixed artefact.

The release must be part of the key because D2 says the Analysis Service *deletes
results on a new release*. Without it, a stored summary outlives the result it
describes and we would serve a confident account of an analysis that no longer
exists. The current release is readable at `GET /database/version` (beta: `97`),
which is the same invalidation signal the answer endpoint already publishes as
`release`.

**Alternatives considered**: seeding the model for determinism — measured and
rejected, seeded runs still scored 0.47 and 0.14 similarity. Caching by token
alone — rejected by D2.

## D4 — No durable store on beta; summaries are kept in process

**Decision for this work (2026-09-18)**: keep summaries in process memory for the
first increment, relying on FR-015's transparency to make regeneration visible,
and leave a durable store as follow-up work.

**Rationale**: beta set no `POSTGRES_LANGGRAPH_DB` (2026-09-18), so LangGraph fell
back to `MemorySaver` and nothing on that host persisted across a restart. An
in-memory store satisfies FR-014 within a process lifetime and loses summaries on
deploy — which is honest only because FR-015 makes regeneration visible rather
than surprising.

**Alternatives considered**: requiring Postgres on beta before shipping anything —
rejected as a blocker disproportionate to the first increment; noting it as
follow-up work is enough.

**Built 2026-09-20, and the follow-up is now open rather than implied.**
`SummaryStore` holds summaries in process, bounded at 512 entries, evicting
the least recently used. So:

- a summary survives a reload and not a deploy
- beta deploys often, so most readers will regenerate at some point
- `cached` on `start` is what makes that honest rather than surprising, and
  is the reason FR-015 exists alongside FR-014

**The durable store was not built.** As of 2026-10-21 the store is still in
process (`src/analysis/store.py`), so summaries are lost on deploy by design.
"Continue in chat" (spec 013) loses the ability to hand them off at the same
time. When this was written, `POSTGRES_LANGGRAPH_DB` was unset on beta, so there
was no existing home to reuse; this repository does not record beta's current
setting. Whether a durable store is still wanted is not recorded.

Two decisions inside the store worth not reversing by accident. An **empty
summary is never stored**: a failed or abandoned generation leaves no text,
and storing it would serve the emptiness back forever as though it were the
answer, indistinguishable from a result with nothing to say. And eviction
**drops the oldest rather than refusing the newest**, because a reader whose
summary was evicted simply regenerates, where refusing new entries would
quietly stop the feature working for everyone after the first few hundred.

**Added 2026-10-03 (d153fbf).** A stored summary is never overwritten, and a
second request for a summary that is being generated waits for it. So two
readers, or a reader and a later "Continue in chat", see the same text
(`src/analysis/store.py:76-82`; `src/api/analysis_summary.py:199-212`).

## D5 — What "an option that discloses no identifiers" means, precisely

**Decision**: Two disclosure tiers, defined by field rather than by intention.

**Aggregate tier — no user data leaves the service.** Everything needed for user
stories 1, 3 and 4 is already free of user content:

- `summary.type`, `species`, `speciesName`, `projection`, `interactors`,
  `includeDisease`
- `pathways[]`: `stId`, `name`, `species`, and `entities`/`reactions` statistics —
  `found`, `total`, `ratio`, `pValue`, `fdr`, `curatedFound`, `interactorsFound`
- `resourceSummary`, `speciesSummary`, `pathwaysFound`, `identifiersNotFound`
  (a count), `warnings`

*(Corrected 2026-10-21.)* As built, the allow-list (`src/analysis/disclosure.py:24-63`)
keeps `entities` statistics only, not `reactions`. It adds `resource` and `exp`
per pathway and `inDisease`, keeps only the first 12 pathways and
`expression.min`/`max`, and keeps at most five `warnings` cut to 200 characters
each. The allow-list bounds what *may* be sent. What the model is actually given
is narrower (`prompt_input`, `src/analysis/summarise.py:51-141`): the analysis
type; per pathway `stId`, `name`, `found`, `total`, `pValue`, `fdr` and `exp`;
`identifiersNotFound`, `resourceSummary`, `speciesSummary`, `warnings` and the
expression range; and values computed from them. `species`, `speciesName`,
`projection`, `interactors`, `includeDisease`, `pathwaysFound`, `ratio`,
`curatedFound`, `interactorsFound`, `resource` and `inDisease` are allow-listed
but not passed on.

**Identifier tier — only on explicit request.** `GET /token/{token}/notFound` and
`/token/{token}/found/entities/{pathway}` return the user's own identifiers. As
built, the `identifiers` tier uses only the first: up to 50 unmatched
identifiers (`src/analysis/client.py:199-237`).

**The trap, and it is not the gene list.** Three fields in the *aggregate* result
are user-supplied free text and must be excluded from the aggregate tier:

- `summary.fileName` — e.g. `smith_lab_unpublished_2026.txt`
- `summary.sampleName`
- `expression.columnNames` — e.g. `Patient_001_tumour`

A tier defined as "don't send the gene list" would pass all three straight
through. This is why the tier is defined as a field allow-list, not a denial of
one obvious field.

**Consequence worth stating**: the aggregate tier answers user story 1 fully, and
user story 2 *partially* — it can report the resource mismatch, which is usually
the cause, but cannot name which identifiers failed. That is a real and
explainable difference for the user to choose between.

*(Corrected 2026-10-21, from a finding of 2026-09-20.)* This originally said the
aggregate tier could report the *proportion* unmatched. It cannot. The
aggregate result carries `identifiersNotFound` and nothing about how many
identifiers were submitted, which is only available from
`/token/{token}/found/all`, an identifier-tier endpoint. The summary reports
the count and is told never to state a proportion
(`src/analysis/summarise.py:155-171`).

## D6 — Proving a person is present is the website's to assert, not ours to infer

**Decision**: Require the caller to assert human presence explicitly; do not infer
it from the existing caller token.

**Rationale**: D1 of spec 010 settled that the caller token asserts *service
identity* and deliberately says nothing about humanity — there is no human gate on
the search path. The chat is now Turnstile-gated (2026-09-18), so the website can
demonstrate presence there. What it cannot do is let the search-page path silently
satisfy a requirement that path was never designed to meet.

**Agreed with the website session, 2026-09-19.** It rides as claims on the
caller token, minted only when their Turnstile-backed identity cookie validated
on the request:

| claim | meaning |
|---|---|
| `human` | `true`, set only when a valid, unexpired identity cookie was presented. **Absent otherwise, never `false`**, so a missing claim and a failed check are indistinguishable to us |
| `human_iat` | when the challenge was solved, epoch **seconds**. Derived from the cookie's expiry minus their identity TTL, so it arrives rounded to a second -- the bound is whole-second, and a sub-second edge is not a state this claim can represent |
| `human_sub` | the cookie's random 16-byte identifier, for per-identity rate limiting. Carries nothing about the person |

**Named `human_sub`, not `sub`.** At the time `sub` was understood here to be an
opaque *per-visit* id that `identity_of` uses for the answer endpoint's
backstop limiter. The cookie subject is per-*browser* and lives as long as the
cookie, so reusing the claim would have changed that limiter's meaning on an
endpoint neither repo was touching. *(Corrected 2026-10-21, from the website's
correction of 2026-09-19 recorded in the contract: `sub` is already the
identity cookie's subject for any reader who has passed the challenge, and
per-visit only for one who has not. The claims stayed separate.)*

**Freshness is 30 minutes, inclusive in whole seconds (`now - human_iat <= 1800`), enforced at both ends.** We refuse a `human_iat`
older than that, and they refuse to mint the claim past it, so neither side is
a single point of failure. Long enough that reading a result, choosing a
disclosure tier and requesting a summary is never re-challenged; short enough
that a stolen cookie is not a durable pass. The claim's lifetime is deliberately
shorter than the token's, because an HMAC cookie is itself a bearer credential.

**They proposed gating on the analysis token instead, and withdrew it.** The
argument was that a token proves real work already happened, so it is decent
evidence somebody meant it — sound for abuse resistance, but this requirement is
about consent, not cost. A token proves an analysis happened; it does not prove
a person is present, nor that the person present is the one who ran it. Analysis
tokens travel in URLs that people paste into tickets and papers, so
token-as-authorization lets a forwarded link send someone else's identifier list
to a model provider. And FR-011/FR-012 make summarising opt-in with a choice of
disclosure tier — a choice a bot holding a link can make is a consent mechanism
that consents on the user's behalf, which is worse than no choice because it
looks like one.

**Their caveat, recorded as adopted**: rate limit per analysis token *and* per
caller regardless. A token asking for twenty summaries of one analysis is not a
scientist. That is a throttle, not evidence of a person, and it does not
substitute for the claim. *(2026-10-21: only the per-caller half was built. The
limiter keys on `human_sub`, falling back to the caller token's identity; there
is no per-analysis-token limit (`src/api/analysis_summary.py:157-164`). The
contract lists this as a known gap.)*

**Dependency at the time (2026-09-19)**: their proxy minted caller tokens for the
answer route only, so a summary route had to be added before any claim could be
carried. The contract records the website calling this endpoint by 2026-09-21.
Nothing here assumes the browser calls us directly — it cannot, since the cookie
is same-site to their origin.

**Alternatives considered**: re-verifying a Turnstile token ourselves — rejected,
it would put a second captcha secret and a second verification path in this
service for a check the website has already done.

## D7 — Reuse the answer endpoint's shape, not its endpoint

**Decision**: A separate surface that reuses caller verification, progressive
delivery, citation events and terminal-state failure.

**Rationale**: the shapes fit — a summary takes comparable time, cites pathways by
stable id, and must be safe for a caller to ignore. But the inputs differ (a token,
not a question), the authorisation bar differs (D6), and the output is stored
(D3). Overloading one endpoint with both would make the stricter requirement
apply to neither or to both.

## D8 — ReactomeGSA is recognised, not summarised

**Decision**: Detect `gsaMethod`/`gsaToken` and decline to summarise, saying why.

**Rationale**: GSA is a separate service on a different host with its own result
shape. Recognising it costs one field check and prevents the worst outcome — a
confident summary of a result we do not actually model.

**Extended 2026-09-20.** Detection also checks the analysis type:
`GSA_REGULATION`, `GSA_STATISTICS` and `GSVA` are declined even without
`gsaMethod` (`src/analysis/client.py:84-109`).

## D9 -- a truncated result invites a statistic the result does not contain

Found on 2026-09-19 by calling the real model with a real result, after every
stubbed test passed.

The payload is bounded to the top twelve pathways of a possible 1,280. The
first prompt input reported `pathways_significant: 12` beside
`pathways_total: 1280`, and the model wrote:

> "The analysis identified a total of 12 significant pathways out of 1280
> pathways assessed."

Which is false. Twelve were *sent*, all twelve passed, so the true count is at
least twelve and unknown above. FR-002 says never state a statistic the result
does not contain, and the result does not contain this one -- the truncation
created the false impression, not the model.

**Decision**: the prompt input distinguishes `pathways_shown` from
`pathways_total`, reports `significant_among_shown`, and carries
`significant_count_is_exact`. The count is exact only when a non-significant
pathway appears among those shown -- which, in a list ordered by p-value,
means everything below it is non-significant too. **The ordering is checked
rather than assumed**, because relying on another service's default sort is
how a claim like this goes quietly wrong. When the count is a lower bound the
model is told so explicitly, and the system prompt forbids the claim outright.

Re-run against the real model on 2026-09-19: the sentence did not appear (the
number of runs is not recorded).

**Later (2026-10-03, d153fbf).** The lower-bound instruction ("every one of
them is significant") is now given only when every shown pathway is
significant. When the count is inexact for another reason, the model is told the
overall count cannot be determined (`src/analysis/summarise.py:244-260`, `:279-290`).

**The general shape is worth keeping.** Bounding a payload for cost is not
neutral -- it changes what the data appears to say, and a model will read the
appearance. Any future truncation here needs the same treatment: say what was
omitted, or say that the derived number is a bound.
