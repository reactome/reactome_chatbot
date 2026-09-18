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

`human_token` is D1 and **not yet decided** -- a signed cookie on the shared parent
domain, a short-lived minted token, or a server-side vouch. This repo verifies
evidence; it does not perform the check.

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

At most **12** citations are sent. Measured against the live endpoint this cap is
binding on ordinary questions, so treat it as the most relevant few rather than
the complete set.

### What `token` text contains

**Markdown, never HTML.** Headings and lists appear; anchors do not. The answer
prompt is the chat UI's and does emit inline `<a href=...>` links, so the endpoint
strips them -- across fragment boundaries, because the model streams one anchor as
twenty-odd pieces (`' <'`, `'a'`, `' href'`, `'="'`, `'https'`, ...). A caller that
rendered fragments as they arrived would otherwise show the raw tag before it
became a link.

The anchor's *text* is kept, since removing it would break any sentence with a
linked phrase in the middle. Where the model used a link as a trailing citation,
that leaves the pathway title as a bare clause -- cosmetic, and the structured
`citation` events are the reliable source for links.

## Properties worth holding to

**It must be safe to ignore.** Any failure, timeout, refusal or unverified caller
produces a `done` with a non-`answered` state. The website renders no panel. The
search page must never be slower or broken because this service is down (FR-006,
SC-004).

**The stream is bounded.** The server gives up after 120 seconds and sends `done`
with `state: failed`. A caller still needs its own timeout -- a dropped connection
sends nothing -- but the server will not hold one open indefinitely. 120s is about
twice the slowest complete answer measured; it is a ceiling, not a target.

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

Measured 2026-09-18 over the fifteen tracked sweep questions, two runs each, warm
process, `gpt-4o-mini` at temperature 0.

| | first increment promises | measured |
|---|---|---|
| first token | **nothing** | p50 **9.6s**, p90 12.2s, max 14.0s (n=26) |
| complete | **nothing** | p50 **10.4s**, p90 18.1s, max 21.8s (n=30) |
| streaming | **yes** | ~700-1,200 token events for one answer |

Through the served HTTP endpoint rather than the graph, on a four-question subset:
first token p50 10.8s, p90 12.2s.

**An earlier version of this contract said the first token arrives "around 36
seconds". That was wrong** -- it came from a single question, and it has not been
reproducible since. Two things changed underneath it: collection routing narrowed
retrieval to the sources a question actually needs, and the four preprocessing
calls now run in two rounds instead of four. Preprocessing is 2.6s of the total,
not the ~16s previously recorded.

**The first increment still makes no latency promise**, and ten seconds is still
far from the two an AI overview is reported to take. But design the panel for ten
seconds, not thirty: the gap between those is the difference between a brief wait
and an abandoned page.

Four of the thirty runs produce no answer token at all. Those are the two
deliberately unsafe questions, which return `nothing_found` in about 4.5s with an
empty body. A panel must handle "nothing, quickly" as a normal outcome.

A naive measurement reports 3.0s to first token. That token is the rephraser's.

**Where the time goes now**, at the median of a four-question repeated measure:

| phase | seconds |
|---|---|
| preprocessing (rephrase \| language, then safety \| intent) | 2.6 |
| query expansion and retrieval | 2.3 |
| retrieval finishing to the answer's first token | 6.1 |

The largest block is no longer retrieval or preprocessing: it is the answer model's
own time to first token with a retrieved context. Collection routing
([009](../../009-collection-routing/spec.md)) already did most of the work on
retrieval -- the 12.5s figure recorded here before it landed no longer reproduces.

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
