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

| | first increment promises | measured today |
|---|---|---|
| first token | **nothing** | ~20-36s: the answer starts only after four preprocessing calls and a retrieval |
| complete | **nothing** | p50 15.2s, p90 22.4s, max 31.5s end to end |
| streaming | **yes** | 1,168 token events for one answer |

**The first increment makes no latency promise, deliberately.** Two seconds was in an
earlier draft of this contract; building the streaming surface showed the answer's
first token arrives around 36 seconds, because nothing of it exists until the
rephrase, safety, language and intent calls and a retrieval have all finished.
Streaming improves the last part and does nothing about the first.

Design the panel for that: it must be able to show nothing for a long time, and to be
absent entirely. Do not build a spinner that implies an imminent answer, and do not
let the search results wait on it (FR-009).

A naive measurement reports 3.0s to first token. That token is the rephraser's.

Retrieval is one part and no longer the obvious first target. It dominates the heavy
questions -- about 12.5s of a 27s answer -- and scales with *queries x collections*. Query expansion turns one question into five
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
