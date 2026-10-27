# Data model: summarising analysis results

Drafted 2026-09-18; corrected 2026-10-21 to match the code (`src/analysis/`,
`src/api/analysis_summary.py`). Everything here is read from the Analysis
Service or produced by this feature. Nothing about an analysis is stored except
the summary we generate.

## Read from the Analysis Service

### AnalysisResult (retrieved by token)

Only allow-listed fields may leave this service (`src/analysis/disclosure.py:24-63`).
Of those, only the ones marked "yes" in the last column are given to the model
(`prompt_input`, `src/analysis/summarise.py:51-141`).

| field | used for | allow-listed (aggregate tier) | given to the model |
|---|---|---|---|
| `summary.type` | which reading applies: `OVERREPRESENTATION`, `EXPRESSION`, `SPECIES_COMPARISON`; `GSA_REGULATION`, `GSA_STATISTICS`, `GSVA` are declined (D8) | yes | yes, as `analysis_type` |
| `summary.gsaMethod`, `summary.gsaToken` | detect ReactomeGSA and decline (D8) | no; read only for detection | no |
| `summary.species`, `speciesName` | which organism was analysed | yes | no |
| `summary.projection`, `interactors`, `includeDisease` | caveats on the result | yes | no |
| `summary.fileName`, `sampleName` | nothing — user-supplied free text | no, under either tier | no |
| `pathways[]` | only the first 12, in the result's order | yes | yes |
| `pathways[].stId` | the citation; how a reader opens the pathway | yes | yes |
| `pathways[].name` | what the summary calls it | yes | yes |
| `pathways[].species`, `.inDisease` | — | yes | no |
| `pathways[].entities.found` / `.total` | how much of the pathway was hit; fewer than 5 found is flagged as fragile | yes | yes |
| `pathways[].entities.ratio` | — | yes | no |
| `pathways[].entities.pValue` / `.fdr` | significance, before and after correction (FDR ≤ 0.05) | yes | yes |
| `pathways[].entities.curatedFound` / `.interactorsFound` / `.resource` | — | yes | no |
| `pathways[].entities.exp[]` | expression values per column (story 4) | yes | yes |
| `pathways[].reactions.*` | — | no | no |
| `expression.columnNames` | nothing — user-supplied column labels | no, under either tier | no |
| `expression.min` / `.max` | the range values sit in | yes | yes |
| `resourceSummary` | which identifier resources matched, the usual clue to a mismatch | yes | yes |
| `speciesSummary` | species breakdown | yes | yes |
| `identifiersNotFound` | how many did not match (a count, not the identifiers) | yes | yes |
| `pathwaysFound` | how many pathways were hit at all | yes | no |
| `warnings` | what the service itself flagged; at most 5, each cut to 200 characters | yes | yes |

The model is also given values computed from these:

- a verdict (`empty`, `nothing_significant` or `has_findings`)
- per-pathway `significant` and `fragile`
- how many pathways the result has and how many were shown
- whether the significant count is exact (research D9)
- whether every identifier matched
- the number of expression columns

### Not-found identifiers (retrieved only on explicit request)

`GET /token/{token}/notFound` — the user's own unmatched identifiers. Identifier
tier. Fetched only when the user has chosen the disclosing option and no stored
summary can be reused, and never otherwise. The first 50 are fetched and sent to
the model (`src/analysis/client.py:199-237`).

### Release

`GET /database/version` — the release the Analysis Service is currently serving,
read on every request. Part of the stored summary's key (as the raw text),
because results are deleted on a release change. If the lookup fails, the
summary is still generated but neither stored nor reused, and `release` on
`start` is null.

## Produced by this feature

### Disclosure choice

What the user agreed to share, carried as the request's required `disclosure`
field (`aggregate` | `identifiers`). There is no default in the API, and no
consent record: this service keeps no timestamp or other state for the choice
(the drafted `asked_at` field was not built). The 2026-09-18 draft said only
`aggregate` may be pre-selected on a page and `identifiers` never; that is the
website's to do, and this service cannot enforce it.

### AnalysisSummary (ours, not the service's)

Held in `SummaryStore` (`src/analysis/store.py`), in process memory.

| field | meaning |
|---|---|
| `token` | the analysis summarised (part of the key) |
| `release` | the release it was generated against, as text (part of the key) |
| `tier` | the disclosure tier it was built from (part of the key) — an aggregate summary and a disclosing one are different artefacts and are not interchanged |
| `text` | the summary itself; an empty text is never stored |
| `citations` | `(st_id, display name)` for every pathway sent to the model |
| `generated_at` | when it was stored. Not sent to the caller |

**Key**: `(token, release, tier)`. A release change makes every summary for the
prior release unreachable, matching the service deleting the results
themselves. The old entries are not deleted but age out, as the store keeps the
512 most recently used. The first summary stored under a key is never
replaced. Everything is lost on restart.

### Citation

The answer endpoint's `st_id` form: `st_id` with a `display_name` (the pathway
name, or the `st_id` when there is none), which the caller links to
`reactome.org/content/detail/<st_id>`. The answer endpoint can also send a
`url` citation; this one never does. Citations are taken from the result, not
from the model's text, so a summary cites only pathways present in the result
it describes (SC-003).

## Outcomes

| outcome | when |
|---|---|
| `summarised` | a summary was produced |
| `not_found` | the token matches no result (404), the service answers 500 (what a malformed token gets), or the token fails this service's own format check |
| `gone` | the result was deleted by a release (410) — tell the user to re-run |
| `unsupported` | a ReactomeGSA result, recognised and declined |
| `refused` | no verified caller (`no_caller`), no fresh evidence of a person (`no_human`, `stale_human`), or rate limited (`rate_limited`); the reason is on `done` |
| `failed` | anything else, including an Analysis Service error, a model error or empty reply, and the 120 s timeout |

`gone` is deliberately distinct from `not_found`: one is a dead end, the other has
an action attached.
