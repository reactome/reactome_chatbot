# Contract: the search-page answer endpoint

What this repo commits to providing, and what the website repo can build against.
Shapes are proposals until both sides agree; the *properties* below are the parts
worth arguing about.

## Request

```
POST /chat/api/answer
Content-Type: application/json

{ "question": "what does CDK5 phosphorylate in Alzheimer disease?",
  "caller_token": "<a token the website minted for this call>" }
```

**`caller_token` is settled (D1, 2026-09-18), and it does not assert humanity.**
The field was `human_token` and the premise was wrong: there is no human gate on
the search path and there will not be one -- nobody solves a captcha to run a
search. What the token asserts is *caller identity*.

The website mints it server-side, per request, and verification here is:

| claim | required | checked against |
|---|---|---|
| signature | yes | the public key, EdDSA or RS256 -- never an HS* algorithm |
| `exp` | yes | now; they mint at +120s |
| `aud` | yes | `reactome-chatbot`, overridable with `CALLER_TOKEN_AUDIENCE` |
| `sub` | no, but expected | not validated; used as the rate-limit key |
| `iss` | no | not currently checked -- the signature already identifies the minter |

`sub` is an opaque per-visit id, 128 random bits, not derived from anything about
the reader. The backstop limit keys on it, so repeat questions in one reading
session count as one caller and a freshly minted token does not buy a fresh
allowance.

**A token with no `aud`, or one minted for a different audience, is refused.**
Worth stating because the earlier code refused *every* token carrying an `aud`
claim -- PyJWT rejects one when no audience is expected -- so this had to change
before the first real token could ever have been accepted.

Abuse control is the website's: the panel is opt-in behind a click, so a crawled
search never reaches a model, and their proxy rate limits by address.

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

### Some answers have no citations at all, by construction

A `citation` carries `st_id`, a Reactome stable identifier, and only the Reactome
and disease-variant collections have them. The userguide collection indexes
documentation pages whose metadata is a page URL, so a userguide-routed question
produces **zero citations** -- measured 2026-09-18: "How do I use the pathway
browser?" returned 0 citations and 452 tokens.

That is a good answer with no sources, not a failure. Two things follow for a
caller:

- Render nothing rather than an empty heading. The website's panel keeps its
  heading and list inside one conditional for this reason.
- It correlates with the **fastest** answers -- the userguide question is also
  the 3.3s-to-first-token one -- so it is common rather than exotic, and it is
  precisely what the four questions anyone would naturally test with fail to
  produce.

**Citations always precede prose**, and that is structural rather than
incidental: citations are emitted only on `on_retriever_end`, and token events
are gated on that same event, because retrieval completing is how the answer's
tokens are told apart from the query expander's. So sources are complete at the
moment the first token arrives, and a panel may render them before any prose.

#### Userguide pages are cited by `url` (implemented 2026-09-18)

A citation carries **exactly one** identifier, and the other key is absent rather
than empty:

```
event: citation
data: {"st_id": "R-HSA-8862803", "display_name": "Deregulated CDK5 triggers..."}

event: citation
data: {"url": "https://reactome.org/userguide/pathway-browser",
       "display_name": "The Pathway Browser"}
```

`st_id` resolves at `reactome.org/content/detail/<st_id>`; `url` is absolute and
used verbatim. A consumer that understands only `st_id` keeps working by skipping
what it does not recognise.

**No fabricated stable ids.** A made-up `R-` identifier would resolve to nothing
or, worse, to a real but wrong entity, and a reader cannot tell from the link
text. A userguide page is cited as what it is.

Measured after the change: "How do I use the pathway browser?" went from 0
citations to 4, and "How do I run a GSEA in Reactome?" to 2. Reactome questions
are unchanged.

Deduplicated by page, not by chunk -- the userguide bundle is 98 chunks across 10
pages, so chunk-level citations would repeat one page up to 27 times. The label is
the page title.

**Only `https://` and `http://` sources are cited.** `source` is a generic
LangChain metadata field and the CSV loaders set it to the file they read, so
without that check a Reactome document lacking an `st_id` cited a local
filesystem path. That was caught by running a real question, not by the unit
tests, which used fixtures too clean to contain one.

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

**Hanging up cancels the work.** Measured: when the caller closes the connection
mid-stream, `CancelledError` is raised inside the answer generator and nothing
further is produced -- 3 tokens read, 3 produced, then cancelled. So a reader who
navigates away does not cost a full model call, and there is nothing for the
proxy to cancel upstream beyond closing the connection.

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
