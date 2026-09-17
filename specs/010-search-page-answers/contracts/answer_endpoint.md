# Contract: the search-page answer endpoint

What this repo commits to providing, and what the website repo can build against.
Shapes are proposals until both sides agree; the *properties* below are the parts
worth arguing about.

## Request

```
POST /chat/api/answer
Content-Type: application/json

{ "question": "what does CDK5 phosphorylate in Alzheimer disease?",
  "human_token": "<evidence the caller verified a person>" }
```

`human_token` is a short-lived token minted by the **website's** server after it
verifies its own captcha, presented on a server-side proxied call (D1, agreed with
the website session on 2026-09-17). This repo verifies a signature; it never sees a
captcha, which is what makes it irrelevant that this repo uses Cloudflare Turnstile
and the search page uses hCaptcha.

Proposed, and this repo's to settle: **asymmetric signing** -- the website holds the
signing key, this service holds only a verifying key, so compromising this service
cannot mint tokens. A shared secret would let either side mint, which is a worse
blast radius for the side that is reachable from a search page.

The token carries a query budget, and the **website enforces it** before calling,
because it proxies every request. The rate limit here is a backstop, not the budget.

## Response: Server-Sent Events

Streaming rather than a single JSON body, because the whole answer currently takes
25-32s and a search page cannot wait. Streaming turns that into "something appears in
about two seconds".

```
event: start
data: {"release": 97, "answered": true}

event: token
data: {"text": "CDK5, when bound to p25, phosphorylates "}

event: citation
data: {"st_id": "R-HSA-8862803", "display_name": "Deregulated CDK5 triggers..."}

event: done
data: {"state": "answered", "seconds": 8.4}
```

`state` is one of `answered`, `nothing_found`, `refused`, `failed`.

### Why citations are separate events

So the website renders links in its own style. Returning prose with embedded HTML
anchors -- what the chat UI does today -- would force the search page to parse them
back out and re-style them. Stable IDs resolve at
`reactome.org/content/detail/<st_id>`.

## Properties worth holding to

**The search page must not wait for it.** The panel is optional and the results are
not; a search page that waits fifteen seconds for an optional answer is worse than
one with no answer. This is the website's to enforce, and it is written here because
it constrains the endpoint too -- there is no "just wait a bit longer" mode.

**Do not rely on the site's edge blocking.** `block-all-automation.conf` is on dev and
release and deliberately never on production, because it blocks Googlebot and would
deindex reactome.org. The edge blocks crawlers on beta and not where search traffic
actually lives, so an endpoint depending on it would test clean and ship naked. This
service gates itself.

**It must be safe to ignore.** Any failure, timeout, refusal or unverified caller
produces a `done` with a non-`answered` state. The website renders no panel. The
search page must never be slower or broken because this service is down (FR-006,
SC-004).

**No answer without a token.** Refused before any model call, not after (FR-003).
Search pages get crawled, and every crawled search reaching the model is a bill.

**Not every search gets a panel.** The caller may ask for every query; this service
decides, and may answer `nothing_found` because the question is navigational rather
than answerable (D2, User Story 3).

**The answer is tied to a release.** `start` carries it so a cached answer can be
invalidated after a release (FR-007).

**One brain.** The endpoint and the chat UI share a graph. If they can disagree about
the same question, that is a defect.

## Budget

| | target | today |
|---|---|---|
| first token | 2s | n/a -- no streaming endpoint exists |
| complete | 10s | p50 **15.2s**, p90 **22.4s**, max **31.5s** over the 15 tracked questions |

Retrieval dominates the heavy questions -- about 12.5s of a 27s answer -- and it
scales with *queries x collections*. Query expansion turns one question into five
queries (2.4s, one LLM call) and each runs against every collection, so cutting
queries is a lever of the same size as cutting collections. Only the latter has a
spec. See [009](../../009-collection-routing/spec.md), and note that it is one lever
rather than the whole of it.

One measurement worth keeping in view for anyone optimising this: the async
retrieval path is **not faster than the sync one** here -- 12.5s against 10.9s on the
same five queries. The concurrency in `aretrieve_documents` is not currently buying
throughput, so a plan that assumes it will is assuming something unmeasured.

## What the website side needs to decide

1. How a verified person is represented (D1) -- the security boundary
2. Whether it calls this service directly from the browser, or proxies server-side.
   Proxying makes the token question easier and keeps this service off the public
   internet
3. What it renders for each `state`, particularly `nothing_found` -- probably nothing
4. Whether repeat searches are cached at its layer or this one (D3)
