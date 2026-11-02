# Contract: the search-page answer endpoint

What the endpoint does, for the website repo to build against. Checked against the
code on 2026-10-21 (`src/api/answer.py`, `src/agent/graph.py`,
`src/util/caller_token.py`, `src/util/rate_limit.py`). The website built against an
earlier version of this file from 2026-09-18. Where the code differs from something
this file or the spec earlier promised, the difference is listed under "Known gaps"
at the end, not resolved here. Measurements are dated and describe the code of their
day.

## Request

```
POST <CHAINLIT_URI>/api/answer
Content-Type: application/json

{ "question": "what does CDK5 phosphorylate in Alzheimer disease?",
  "caller_token": "<a token the website minted for this call>" }
```

The path sits under the chat's mount point, so one nginx location covers both
(`bin/chat-fastapi.py:73-78`). On beta `CHAINLIT_URI=/chat/guest`, so the path is
`/chat/guest/api/answer` (`deploy/beta/env.beta.template:21`). `/chat/api/answer` is
used only when `CHAINLIT_URI` is unset.

- `question`: 1 to 2,000 characters. A body that is not valid JSON, or a missing,
  empty or overlong `question`, gets HTTP **422** with FastAPI's JSON error body, not
  an event stream (`src/api/answer.py:70-72`)
- `caller_token`: may be omitted; an absent or empty token is refused like any other
  invalid one

To watch it stream:

```bash
curl -N -X POST https://beta.reactome.org/chat/guest/api/answer \
  -H 'Content-Type: application/json' \
  -d '{"question":"what does CDK5 phosphorylate in Alzheimer disease?",
       "caller_token":"<token>"}'
```

`-N` matters: without it curl buffers, and the stream looks like one slow response.
(The 2026-09-18 handover mentioned an nginx rule answering some user agents with 403;
nginx configuration is not in this repository.)

No CORS headers are sent (`bin/chat-fastapi.py` adds no CORS middleware), so browser
script on another origin cannot read the response. The website calls the endpoint
from its own server (D1).

**`caller_token` is settled (D1, 2026-09-18), and it does not assert humanity.**
The field was `human_token` and the premise was wrong: there is no human gate on
the search path and there will not be one -- nobody solves a captcha to run a
search. What the token asserts is *caller identity*.

The website mints it server-side, per request, and verification here is:

| claim | required | checked against |
|---|---|---|
| signature | yes | the public key, EdDSA or RS256 -- never an HS* algorithm |
| `exp` | yes | now, with no leeway; they mint at +120s. No maximum lifetime is enforced here |
| `aud` | yes | `reactome-chatbot`, overridable with `CALLER_TOKEN_AUDIENCE` |
| `iat`, `nbf` | no | if present, must be numbers no more than 30s in the future (clock leeway) |
| `sub` | no, but expected | not validated; the rate-limit key, else `jti`, else a hash of the token |
| `iss` | no | not currently checked -- the signature already identifies the minter |

A token longer than 4,096 characters is refused without being parsed
(`src/util/caller_token.py:49-53`, `:134-137`). The reason for a refusal is logged
here and never returned to the caller.

`sub` is 128 random bits, not derived from anything about the reader. For a reader
who has passed the website's Turnstile challenge it is that identity cookie's
subject, so it lasts as long as the cookie, across visits; otherwise it is a
per-visit id (`src/util/caller_token.py:15-22`, corrected 2026-09-19). The backstop
limit keys on it, so a freshly minted token does not buy a fresh allowance.

**A token with no `aud`, or one minted for a different audience, is refused.**
Worth stating because the earlier code refused *every* token carrying an `aud`
claim -- PyJWT rejects one when no audience is expected -- so this had to change
before the first real token could ever have been accepted.

Abuse control is mainly the website's. As they described it on 2026-09-18, the panel
is opt-in behind a click, so a crawled search never reaches a model, and their proxy
rate limits by address. This service adds a backstop: 30 answers per caller per 600s
by default, set by `ANSWER_RATE_LIMIT` and `ANSWER_RATE_WINDOW_SECONDS`
(`src/util/rate_limit.py:94-103`). A request over it is refused like an invalid token
and is not counted.

## Response: Server-Sent Events

HTTP 200 and `text/event-stream` for every request that passes validation, including
refusals. Streaming rather than a single JSON body so text renders as it arrives. It
was chosen when a whole answer took 25-32s, in the hope that something would appear
in about two seconds. It does not, because the first answer token waits for
preprocessing and retrieval: last measured 2026-09-20, first token p50 5.57s and
complete p50 8.90s (see "Budget").

```
event: start
data: {"release": 97, "answered": true}

event: token
data: {"text": "CDK5, when bound to p25, phosphorylates "}

event: citation
data: {"st_id": "R-HSA-8862803", "display_name": "Deregulated CDK5 triggers..."}

event: done
data: {"state": "answered", "seconds": 8.4, "answer_id": "Ev3z3JDmIIUF4gkKTfrn9VF8"}
```

Events, in order:

- `start`, once, before any model work. `release` is the Reactome release of the
  bundle the service started with (an integer such as 97), or `null` if it cannot be
  determined (`bin/chat-fastapi.py:55-59`). `answered` is always `true` and predicts
  nothing; read the state on `done`
- `citation`, zero to 12 (below)
- `token`, zero or more
- `done`, once, last

A refusal is the exception. It is **only** a `done`, with no `start`:

```
event: done
data: {"state": "refused", "seconds": 0.0}
```

`state` is one of `answered`, `nothing_found`, `refused`, `failed`:

| state | when |
|---|---|
| `answered` | at least one answer token was produced |
| `nothing_found` | the stream ended without an answer token. The measured case (2026-09-18) is a question the safety check refused, whose reply is not streamed. A question retrieval cannot answer usually ends `answered`, with text saying it was not found |
| `refused` | the token was missing or invalid, or the caller is over the rate limit. No model call was made |
| `failed` | an error, or the 120s bound was reached. Tokens sent before it stay sent |

**`answer_id` appears only when `state` is `answered`** (added 2026-09-25,
spec 013), and not always then: it is omitted when the text left after stripping is
empty (`src/api/answer_store.py:47-59`). It is 24 URL-safe characters. It names the
answer exactly as this stream sent it -- the text after links and the sources list
were stripped, and the citations -- kept in process memory for an hour (at most
2,048 answers; lost on restart) so "Continue in chat" can open the chat on *that*
answer rather than a regenerated one (the same question scores ~0.33 similarity run
to run). Pass it to `POST <CHAINLIT_URI>/api/handoff`, under the same prefix as this
endpoint (`bin/chat-fastapi.py:84`), as `{"kind": "search", "answer_id": ...}`. It is
keyed per answer, not per question, so two readers of the same search never share
one.
Every other `done` has exactly `state` and `seconds`; a caller that ignores
unknown keys parses them all the same way.

> **This endpoint does not require a human-presence claim and
> `/api/analysis-summary` does.** A caller token without one gets a full
> answer here and `no_human` there. Deliberate: this returns public pathway
> text, that one returns a reader's own uploaded analysis.
>
> **`/api/analysis-summary` uses different names for the same ideas**:
> `summarised` for success and `not_found` for the empty case. The event
> shapes are nearly identical, so do not reuse one state list for both — a
> consumer did, and a correct summary rendered as a truncated failure. See
> `specs/011-summarise-analysis-results/contracts/summary_endpoint.md`.

### Why citations are separate events

So the website renders links in its own style. Returning prose with embedded links
-- what the chat UI shows, as HTML anchors until 2026-10-03 and markdown links since
(d71f6de) -- would force the search page to parse them back out and re-style them.
Stable IDs resolve at `reactome.org/content/detail/<st_id>`.

At most **12** citations are sent, in retrieval's fused order
(`src/agent/graph.py:200`, `:548-571`). They are what retrieval found, not what the
prose cited (research R3). On 2026-09-17 the cap was binding on ordinary questions;
that was before citations were taken from the fused result only (38f8885,
2026-09-18), and it has not been re-measured. Treat them as the most relevant few
rather than the complete set. `display_name` can be an empty string.

### Some answers have no citations at all

A question routed to the live lookup (the Reactome MCP tools) is answered from tool
calls with no retrieval, so it streams tokens and **no `citation` events**
(`src/agent/graph.py:500-532`; live-path streaming added 2026-09-18, d3740aa). When
the live tools are unavailable, the same question falls back to retrieval and is
cited as usual.

That is a good answer with no sources, not a failure. Render nothing rather than an
empty heading; the website's panel keeps its heading and list inside one conditional
for this reason.

*History: until 2026-09-18 user-guide answers had no citations either, because
user-guide documents carry no `st_id`. Measured 2026-09-18, "How do I use the pathway
browser?" returned 0 citations and 452 tokens, and was also the fastest answer (3.3s
to first token). They are now cited by `url` (next section).*

**On the retrieval path, citations precede prose**, and that is structural:
citations are emitted only on the fused retriever's `on_retriever_end`, and answer
tokens are gated on that same event, because retrieval completing is how the
answer's tokens are told apart from the query expander's. So sources are complete
when the first token arrives, and a panel may render them before any prose.

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

**Markdown without links.** Headings and lists appear; links do not. The answer
prompt is the chat UI's and asks for inline markdown links, `[display_name](URL)`
(`src/retrievers/reactome/prompt.py:20`; HTML `<a href=...>` anchors until
2026-10-03), so the endpoint strips both forms (`src/api/answer.py:133-144`,
`src/util/anchor_strip.py`) -- across fragment boundaries, because the model streams
one link as many pieces (one anchor arrived as twenty-odd: `' <'`, `'a'`, `' href'`,
`'="'`, `'https'`, ...). A caller that rendered fragments as they arrived would
otherwise show the raw link before it was complete. Text that might still become a
link is held back briefly; anything held at the end is sent as a final `token`.

The link's *text* is kept, since removing it would break any sentence with a
linked phrase in the middle. Where the model used a link as a trailing citation,
that leaves the pathway title as a bare clause -- cosmetic, and the structured
`citation` events are the reliable source for links.

**No trailing source list.** The answer prompts ask for a bullet list of every
citation at the end, which the chat UI renders. This caller gets its links from
the `citation` events, so that list is a duplicate -- and after the links come
off, a duplicate with no links in it. The endpoint drops the heading and
everything after it.

Callers **must not** pattern-match the heading themselves. Until 2026-09-19 the
prompts specified no heading at all, only "a bullet-point list of each unique
citation anchor", so the model invented one per answer: `## Sources`,
`## Most relevant sources`, `relevant references`, `Key sources`,
`Top citations`. The website was matching those and could not win, because it
was fitting samples from an unconstrained generator. The prompts now pin the
heading to exactly:

```
## Sources
```

and `SourcesSectionStripper` removes it on the served path. The stripper still
accepts the older variants, because a prompt is an instruction and not a
guarantee, but no caller needs to know that.

## Properties

**Safe to ignore, within the stream.** Every failure, timeout, refusal or unverified
caller ends with a `done` whose state is not `answered`, and the website renders no
panel (FR-006). Two cases fall outside that: a malformed request gets HTTP 422 JSON
rather than a stream, and a failure after tokens have started leaves those tokens
sent, followed by `done` `failed` (see Known gaps). Keeping the search page from
being slowed or broken when this service is down (FR-006, SC-004) is the website's
side.

**Hanging up cancels the work.** Measured 2026-09-18: when the caller closes the
connection mid-stream, `CancelledError` is raised inside the answer generator and
nothing further is produced -- 3 tokens read, 3 produced, then cancelled. So a reader
who navigates away does not cost a full model call, and there is nothing for the
proxy to cancel upstream beyond closing the connection (`src/api/answer.py:192-218`).

**The stream is bounded.** The server gives up after 120 seconds and sends `done`
with `state: failed` (`src/api/answer.py:67`). A caller still needs its own timeout
-- a dropped connection sends nothing -- but the server will not hold one open
indefinitely. 120s was set at about twice the slowest complete answer measured then;
it is a ceiling, not a target.

**No answer without a token.** Refused before any model call, not after (FR-003).
Search pages get crawled, and every crawled search reaching the model is a bill.

**Every request that passes the checks is answered.** Nothing decides that a search
is not worth answering; earlier versions said navigational questions might get
`nothing_found`, which was never built (D2, User Story 3).

**The answer is tied to a release.** `start` carries it so a cached answer can be
invalidated after a release (FR-007). It can be `null`, and no cache exists on this
side (D3).

**One graph.** The endpoint and the chat UI share one graph instance and profile
(`src/agent/registry.py`; `tests/api/test_answer_matches_chat.py`). The endpoint
skips the post-answer web search the chat runs (`src/api/answer.py:158-164`), which
does not change the answer text. The same question still gets differently worded
answers run to run on either surface (~0.33 similarity, 2026-09-18), so identical
text is not something either side can rely on.

## Budget

No latency is promised. The latest recorded measurement is 2026-09-20, on
`astream_answer` (the graph method this endpoint streams from), arms interleaved,
n=25: first token p50 **5.57s** (range 2.26s-11.44s), complete p50 **8.90s** (T022 in
[research.md](../research.md)). The rest of this section is the 2026-09-18 record
the website first built against, kept with its corrections.

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
reproducible since. Two things changed underneath it: source routing sent each
question to the source it needs (Reactome bundle, user guide or live lookup), and the four preprocessing
calls now run in two rounds instead of four. Preprocessing is 2.6s of the total,
not the ~16s previously recorded.

**The first increment made no latency promise**, and none has been made since. The
advice given on 2026-09-18 was to design the panel for about ten seconds, not
thirty; the 2026-09-20 figures above are within that, with a long tail on the first
token.

Four of the thirty runs produced no answer token at all. Those were the two
deliberately unsafe questions, which returned `nothing_found` in about 4.5s with an
empty body. A panel has to handle "nothing, quickly" as a normal outcome.

A naive measurement reports 3.0s to first token. That token is the rephraser's.

**Where the time went on 2026-09-18**, at the median of a four-question repeated measure:

| phase | seconds |
|---|---|
| preprocessing (rephrase \| language, then safety \| intent) | 2.6 |
| query expansion and retrieval | 2.3 |
| retrieval finishing to the answer's first token | 6.1 |

On these figures the largest block was the answer model's own time to first token
with a retrieved context. The 2026-09-20 measurement puts the answer model at 0.84s
and retrieval at 3.15s, the largest; the two records are not reconciled
(research.md, T022).

**A correction to an earlier version of this file**, which said collection routing
had landed and explained the improvement. It had not then: what had landed was
*source* routing. Collection routing ([009](../../009-collection-routing/spec.md)),
which selects among collections *within* the Reactome bundle, landed later, on
2026-09-19 (0fe7d88). The figures in this section were measured before it, with
retrieval searching all five collections.

So the honest position on why the earlier 12.5s and ~36s figures no longer
reproduce is that we do not fully know. Source routing accounts for the fast
user-guide answers (3.3s to first token). Running preprocessing in two rounds
accounts for about 2.5s. The rest may simply be that those figures came from one
question and were never representative.

One measurement worth keeping in view for anyone optimising this (2026-09-17, before
collection routing): the async retrieval path was **not faster than the sync one**
-- 12.5s against 10.9s on the same five queries. The concurrency in
`aretrieve_documents` was not buying throughput then, and it has not been
re-measured.

## Questions put to the website side (2026-09-17), and where they stand

1. How a caller is represented -- the security boundary. Settled 2026-09-18 (D1): a
   token the website mints per call, verified as above
2. Whether it calls this service from the browser or through a server-side proxy. It
   proxies: the token is minted on its server and its proxy rate limits (D1)
3. What it renders for each `state`, particularly `nothing_found`. The website's
   choice; not recorded here
4. Whether repeat searches are cached at its layer or this one (D3). Open; no cache
   exists on this side

## Known gaps

Differences between what this file or the spec earlier promised and what the code
does, as of 2026-10-21. Recorded, not resolved: which side should change has not been
decided.

- **First token in about two seconds.** Given as the reason for streaming; last
  measured at p50 5.57s (2026-09-20). FR-005a is an unmet target
- **Navigational searches answered `nothing_found`.** Earlier versions said this
  service might decline navigational queries (D2, User Story 3). Not built
- **"No panel" after tokens have started.** A failure or timeout after the first
  token leaves the streamed text with the caller, then sends `done` `failed` with no
  `answer_id` (`src/api/answer.py:219-262`). Task T018 described failure as "no
  partial answer". A panel that shows tokens as they arrive has to discard them on
  `failed` to show nothing
- **Malformed requests** get HTTP 422 JSON, not a `done` event
- **Caching by question and release** (FR-007, D3). `release` is sent; no cache exists
