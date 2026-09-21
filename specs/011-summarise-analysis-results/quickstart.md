# Quickstart: validating analysis summaries

How to check this feature actually works, using real analysis results rather than
fixtures. Nothing here needs production.

## Prerequisites

- A caller-token keypair, as for the answer endpoint.
- Network access to `beta.reactome.org`. **Use a browser-like `User-Agent`**: the
  site's automation blocking returns a 403 with an HTML body to library
  user-agents, which looks exactly like an auth failure and is not one.

## Get a real analysis token

Run an analysis against beta and keep the token. A small, deliberately mixed list
gives you both a result and unmatched identifiers to summarise:

```bash
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
printf 'TP53\nEGFR\nCDK5\nNOT_A_REAL_GENE\nALSO_NOT_REAL\n' > /tmp/ids.txt

curl -s -A "$UA" -H 'Content-Type: text/plain' --data-binary @/tmp/ids.txt \
  'https://beta.reactome.org/AnalysisService/identifiers/projection?pageSize=1&page=1' \
  | head -c 400
```

The response carries `summary.token`. That token is the only input this feature
takes.

## Check the result behaves as the contract assumes

```bash
# The result itself
curl -s -A "$UA" "https://beta.reactome.org/AnalysisService/token/$TOKEN?pageSize=5&page=1" | head -c 400

# The release the summary will be keyed against
curl -s -A "$UA" "https://beta.reactome.org/AnalysisService/database/version"

# The unmatched identifiers -- identifier tier, fetched only with consent
curl -s -A "$UA" "https://beta.reactome.org/AnalysisService/token/$TOKEN/notFound"
```

Expect `database/version` to match the release in the summary's `start` event.

## Scenarios that must pass

| # | setup | expected |
|---|---|---|
| 1 | a token whose top pathways pass FDR | summary names those pathways, cites each by stable id |
| 2 | a list that hits nothing significant | summary says nothing passes correction; no p-value presented as a finding |
| 3 | mostly unmatched identifiers | summary reports the **count** and points at identifier type or species. Not the proportion: the aggregate result has no submitted-total in it, so a percentage would be invented (Phase 4) |
| 4 | `disclosure: aggregate` | the user's identifiers, filename, sample name and column labels appear in no outbound request |
| 5 | same token requested twice | byte-identical summary, `cached: true` on the second |
| 6 | an expired token | `state: not_found` |
| 7 | a token from before a release | `state: gone`, and the user is told to re-run |
| 8 | a ReactomeGSA token | `state: unsupported` |
| 9 | no caller token / no human assertion | `state: refused`, and **zero model calls** |

Scenario 4 is the one to automate first and the one worth distrusting: it is an
assertion about what was *not* sent, so it should be checked by recording outbound
requests, not by reading the summary and seeing nothing alarming.

Scenario 5 is what makes FR-014 real. Test it across a restart too, and expect it
to fail there until a durable store exists (research D4) — that failure is known,
not a surprise.

## Checking a summary is honest

Mechanical checks first, because they do not need judgement:

- every `st_id` cited appears in that result's `pathways[]`
- no number in the text is absent from the result
- when `pathwaysFound` is 0, the text says nothing was found

Then a curator reads a handful spanning strong, weak and empty results and says
whether each conveys how much to trust it. That is SC-006 and it cannot be
automated.

## Run against beta, 2026-09-21

Image `cc4315c`, release 97, inside the container -- the deployed code, the
installed bundle and the real Analysis Service. The caller-token check is
exercised separately over HTTP, because this service holds only the public
half of the keypair by design and cannot mint one to drive the full route.

| # | outcome |
|---|---|
| 1 | **pass** — summarised, 12 citations, every one present in the result, first token **1.8s** |
| 3 | **pass** — count reported, no invented percentage |
| 4 | **pass** — three aggregate payloads checked, including the expression one carrying `Patient_001_tumour`; no forbidden field and no user label in any of them. The disclosing payload *does* carry the names, so the check is not passing because nothing was sent |
| 5 | **pass** — second request `cached: true` and byte-identical |
| 6 | **pass** — unknown token gives `not_found` |
| 7 | **pass** — a constructed pre-release token gives `gone` on the deployed build |
| 9 | **pass** — verified over HTTP on the deployed route: no caller token gives `refused` / `no_caller`, HTTP 200 |

Also checked, beyond the table: an `EXPRESSION` result referred to `column 1`,
`column 2`, `column 3` and no other form, and the `identifiers` tier named the
reader's unmatched identifiers.

**Three scenarios were not run, and the reason is that they cannot be
constructed here rather than that they were skipped.**

- **2, nothing significant** — needs an identifier list that hits no pathway
  past correction. Every list tried produced significant hits, and inventing
  one by editing a result would test the code against a fixture rather than
  the service. Covered by unit tests on the verdict instead.
- ~~**7, `gone`**~~ — **this turned out to be constructible, and now passes.**
  See below.
- **8, ReactomeGSA** — needs a GSA analysis, which is a different service.
  Detection is unit-tested against all three GSA type values.

Scenario 5 across a restart is expected to fail until a durable store exists
(research D4), and was not attempted for that reason.

**The first run of scenario 4 proved nothing**, and the correction is worth
keeping because the mistake is easy to repeat. The disclosure check ran
before the expression summary, so the payload carrying the column labels --
the strongest disclosure risk in the feature -- was not among those examined.
It passed by ordering rather than by evidence. The check now runs last, over
every aggregate payload produced, and asserts in the other direction too:
the disclosing tier's payload must contain the names, or a clean aggregate
check might only mean nothing was ever sent.

### Scenario 7 is constructible after all

An analysis token is base64 of `YYYYMMDDHHMMSS_counter` --
`MjAyNjA5MTkxODExNDJfMTE=` decodes to `20260919181142_11`. So a token for any
past moment can be made without the service having issued it.

Measured on 2026-09-21, the service distinguishes three ages rather than two:

| token timestamp | response |
|---|---|
| 2026-09-12 and earlier | 404 |
| **2026-09-14 onward** | **410** |

So there is a retention window in which a result deleted by a release is
still *remembered as deleted*, and anything older is simply unknown. A token
inside that window produces a real 410, and the deployed pipeline maps it to
`gone`:

```
python -c "import base64; print(base64.b64encode(b'20260915120000_1').decode())"
# -> MjAyNjA5MTUxMjAwMDBfMQ==   ->  outcome: gone
```

This matters beyond ticking the scenario off. The `gone` path would otherwise
have been first exercised for real during a release, which is the worst
moment to discover a handling bug in it -- and the window moves, so the
timestamp above will eventually fall out of it and start returning 404.
Anyone re-running this should find the current boundary rather than reuse
that token.