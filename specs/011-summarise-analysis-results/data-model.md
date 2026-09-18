# Data model: summarising analysis results

Everything here is read from the Analysis Service or produced by this feature.
Nothing about an analysis is stored except the summary we generate.

## Read from the Analysis Service

### AnalysisResult (retrieved by token)

| field | used for | tier |
|---|---|---|
| `summary.type` | which reading applies: `OVERREPRESENTATION`, `EXPRESSION`, `SPECIES_COMPARISON` | aggregate |
| `summary.gsaMethod`, `summary.gsaToken` | detect ReactomeGSA and decline (D8) | aggregate |
| `summary.species`, `speciesName` | which organism was analysed | aggregate |
| `summary.projection`, `interactors`, `includeDisease` | caveats the summary must respect | aggregate |
| `summary.fileName`, `sampleName` | **never sent** — user-supplied free text | identifier |
| `pathways[].stId` | the citation; how a reader opens the pathway | aggregate |
| `pathways[].name` | what the summary calls it | aggregate |
| `pathways[].entities.found` / `.total` / `.ratio` | how much of the pathway was hit | aggregate |
| `pathways[].entities.pValue` / `.fdr` | significance, before and after correction | aggregate |
| `pathways[].entities.curatedFound` / `.interactorsFound` | whether a hit rests on curated data or inferred interactors | aggregate |
| `pathways[].entities.exp[]` | expression values per column (story 4) | aggregate |
| `expression.columnNames` | **never sent** — user-supplied column labels | identifier |
| `expression.min` / `.max` | the range values sit in | aggregate |
| `resourceSummary` | which identifier resources matched, the usual clue to a mismatch | aggregate |
| `speciesSummary` | species breakdown | aggregate |
| `identifiersNotFound` | how many did not match (a count, not the identifiers) | aggregate |
| `pathwaysFound` | how many pathways were hit at all | aggregate |
| `warnings` | what the service itself flagged | aggregate |

### Not-found identifiers (retrieved only on explicit request)

`GET /token/{token}/notFound` — the user's own unmatched identifiers. Identifier
tier. Fetched only when the user has chosen the disclosing option, and never
otherwise.

### Release

`GET /database/version` — the release the Analysis Service is currently serving.
Part of the stored summary's key, because results are deleted on a release change.

## Produced by this feature

### DisclosureChoice

What the user agreed to share, chosen per request.

| field | values | meaning |
|---|---|---|
| `tier` | `aggregate` \| `identifiers` | which fields may be sent |
| `asked_at` | timestamp | when the user chose; absent means no consent and no summary |

`aggregate` is the default and the only one that can be pre-selected. `identifiers`
is never a default.

### AnalysisSummary (ours, not the service's)

| field | meaning |
|---|---|
| `token` | the analysis summarised |
| `release` | the release it was generated against; with `token`, the storage key |
| `tier` | which disclosure tier produced it — an aggregate summary and a disclosing one are different artefacts and must not be interchanged |
| `text` | the summary itself |
| `citations` | the pathways discussed, by stable id |
| `generated_at` | when, so the interface can say how old it is |

**Key**: `(token, release, tier)`. A release change invalidates every summary for
the prior release, matching the service deleting the results themselves.

### Citation

Reuses the answer endpoint's shape exactly: `st_id` with a display name, resolving
to `reactome.org/content/detail/<st_id>`. A summary cites only pathways present in
the result it describes — an invented or mismatched id is the failure this pins.

## Outcomes

| outcome | when |
|---|---|
| `summarised` | a summary was produced |
| `not_found` | the token matches no result (404) |
| `gone` | the result was deleted by a release (410) — tell the user to re-run |
| `unsupported` | a ReactomeGSA result, recognised and declined |
| `refused` | no verified caller, or no evidence of a person |
| `failed` | anything else |

`gone` is deliberately distinct from `not_found`: one is a dead end, the other has
an action attached.
