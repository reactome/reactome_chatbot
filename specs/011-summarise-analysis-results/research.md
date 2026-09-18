# Research: summarising analysis results

Measured against beta's Analysis Service (`/AnalysisService/v3/api-docs`,
release 97) on 2026-09-18. Every claim below came from the API or from this
repository, not from recollection.

## D1 — Results are addressed by a token, and that is the whole input

**Decision**: The feature takes an analysis token. It never accepts an identifier
list and never runs an analysis.

**Rationale**: `GET /token/{token}` returns the completed `AnalysisResult`. The
analysis has already happened; re-running it here would duplicate the Analysis
Service and violate Principle V. It also means we never hold the user's input.

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

## D3 — Stability comes from storing the summary, keyed by token *and release*

**Decision**: Store a generated summary against `(token, release)`. Serve the
stored one thereafter. Discard on a release change.

**Rationale**: FR-014 wants the same token to yield the same summary. Generation
cannot provide that — measured on this repo, the same question through the same
surface twice scores 0.33 similarity. Storage can, because an analysis result is a
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

## D4 — There is nowhere durable to store it yet

**Decision**: Treat the store as a required piece of work, not an assumption.
First increment may keep summaries in process memory, provided the transparency
requirement (FR-015) already tells the user a summary can be regenerated.

**Rationale**: beta sets no `POSTGRES_LANGGRAPH_DB`, so LangGraph already falls
back to `MemorySaver` and nothing on that host persists across a restart. An
in-memory store satisfies FR-014 within a process lifetime and loses summaries on
deploy — which is honest only because FR-015 makes regeneration visible rather
than surprising.

**Alternatives considered**: requiring Postgres on beta before shipping anything —
rejected as a blocker disproportionate to the first increment; noting it as
follow-up work is enough.

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

**Identifier tier — only on explicit request.** `GET /token/{token}/notFound` and
`/token/{token}/found/entities/{pathway}` return the user's own identifiers.

**The trap, and it is not the gene list.** Three fields in the *aggregate* result
are user-supplied free text and must be excluded from the aggregate tier:

- `summary.fileName` — e.g. `smith_lab_unpublished_2026.txt`
- `summary.sampleName`
- `expression.columnNames` — e.g. `Patient_001_tumour`

A tier defined as "don't send the gene list" would pass all three straight
through. This is why the tier is defined as a field allow-list, not a denial of
one obvious field.

**Consequence worth stating**: the aggregate tier answers user story 1 fully, and
user story 2 *partially* — it can report the proportion unmatched and the resource
mismatch, which is usually the cause, but cannot name which identifiers failed.
That is a real and explainable difference for the user to choose between.

## D6 — Proving a person is present is the website's to assert, not ours to infer

**Decision**: Require the caller to assert human presence explicitly; do not infer
it from the existing caller token.

**Rationale**: D1 of spec 010 settled that the caller token asserts *service
identity* and deliberately says nothing about humanity — there is no human gate on
the search path. The chat is now Turnstile-gated (2026-09-18), so the website can
demonstrate presence there. What it cannot do is let the search-page path silently
satisfy a requirement that path was never designed to meet.

**Open with the website repo**: the shape of the assertion — most likely an
additional claim in the caller token, minted only after a Turnstile verification
they already perform. This needs their agreement and is recorded as a task, not
decided here.

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
