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
| 1 | **pass** — summarised, 12 citations, every one present in the result. First token **1.8s and 2.0s on two runs of one token**: two samples, not a distribution, and not comparable to the answer endpoint's figures, which are medians over several questions. Quote it as "about two seconds on this token" or measure properly |
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
| 2026-09-13 00:00:01 and earlier | 404 |
| **2026-09-13 12:00:00 onward** | **410** |

**It is a rolling cutoff measured in time, not a date boundary.** The two
readings above are the same calendar day: one second past midnight is 404 and
noon is 410, so the window is roughly the last eight days and it slides
forward continuously. The website session independently confirmed the same
behaviour on production, and found the boundary a day earlier than a first
sample here had suggested — which is the argument for deriving it rather than
quoting it.

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
moment to discover a handling bug in it.

**Bisect for the boundary at run time; never hardcode a token.** The cutoff
slides forward daily, so any fixed timestamp eventually falls out of the
window and starts returning 404 -- and a test that does that does not fail,
it quietly starts asserting `not_found` while still passing. That is the same
shape as everything else this feature has had to guard against: a check that
goes on passing after it has stopped testing the thing.

**Two people measured this and agreed, and both were wrong.** One read the
boundary as 09-13 and the other as 09-14; the answer is that there is no
boundary day at all. Neither had tested *within* a day, because both were
picturing a date cutoff and so both asked "which day" rather than "which
shape". The agreement felt like corroboration and was not — it was two
measurements taken the same way, and a shared assumption survives any number
of those. It took testing at a finer grain than the question assumed.

The assumption worth naming, because it cost more than the boundary did: this
scenario was recorded as untestable on the belief that only the service can
mint a token. That was never decided or written down anywhere, and it was
load-bearing for a plan in which `gone` would first be exercised by a curator
during a release. Nothing about it looked like a question.