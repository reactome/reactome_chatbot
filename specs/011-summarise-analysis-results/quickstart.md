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
| 3 | mostly unmatched identifiers | summary reports the proportion and points at identifier type or species |
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
