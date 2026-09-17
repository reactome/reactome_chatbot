# Phase 0 Research: the answer endpoint

## R1. How does the endpoint stream, given `AgentGraph` has no streaming surface?

`AgentGraph` exposes `ainvoke` and nothing else. Chainlit's streaming comes from
`AsyncLangchainCallbackHandler` with `final_stream` -- a Chainlit object, not
something a plain HTTP handler can use.

| Option | How | Verdict |
|---|---|---|
| A. `astream_events` on the compiled graph | LangGraph's own event stream, filtered to the final LLM's tokens | **Chosen.** It is the framework's supported streaming API, and it exposes retrieval events too, which is where citations come from |
| B. A callback handler pushing to an `asyncio.Queue` | Mirrors what Chainlit does; the SSE response drains the queue | Works, and is what to fall back to if filtering A's event stream proves unreliable. Costs a bespoke handler and a lifetime to manage |
| C. Two graphs, one per surface | Simplest to write | Rejected. Doubles 51.5s of startup and lets the panel and the chat disagree, which SC-003 forbids |

**Decision: A, with B as the named fallback.** The risk in A is that the event
filtering depends on which node emits the final answer, so a graph change could
silently stop the stream. A test asserting that tokens actually arrive -- not merely
that the endpoint returns 200 -- is the guard, and that lesson is fresh: a 200 with
no content is exactly how the userguide shell fooled two sessions today.

## R2. Where does the graph live?

`bin/chat-chainlit.py` constructs `AgentGraph` at module scope; Chainlit is mounted
onto the FastAPI app in `bin/chat-fastapi.py`. The endpoint needs the *same*
instance, not another one.

**Decision**: construct it once in the FastAPI app's lifespan and let both surfaces
use it. Failing that, a module-level accessor the Chainlit app also imports.

**Rationale**: SC-003 requires the endpoint and the chat to answer identically, and
the cheapest way to guarantee that is one object. A second graph also pays 51.5s of
startup again and doubles the memory of the BM25 indexes.

## R3. How are citations produced?

Not by parsing the model's prose. The answer prompt instructs `<a href=...>` anchors,
and pulling those out of a token stream would couple the wire contract to prompt
wording -- prompt changes have happened twice this week.

**Decision**: emit citation events from the retrieved documents, whose metadata
already carries `st_id`. Deduplicate, and emit as each document set arrives.

**Consequence worth stating**: the citations are then "what retrieval found", which
is a superset of "what the answer used". The contract does not promise otherwise, and
a panel showing sources the prose did not cite is better than a panel that mis-parses
prose. If exactness is wanted later it needs the model to emit IDs, which is a prompt
change with its own measurement.

## R4. How is the human token verified?

Agreed with the website session on 2026-09-17: they mint after their own captcha,
asymmetric signing, they hold the signing key and this service holds only a verifying
key. The query budget is theirs, enforced before the call, because they proxy every
request.

**Decision here**: verification is signature + expiry, stateless, and nothing else.
No consumption tracking, no shared secret, no second counter that can disagree.

**Startup**: a missing or unreadable verifying key stops the process. Principle IV --
an endpoint that accepts everything because its key is absent is the worst outcome
available, and it would test clean.

## R5. Does the new route collide with the captcha middleware?

`verify_captcha_middleware` intercepts every path and returns early only for a small
allowlist under `CHAINLIT_URI`. A new route under `/chat/api/` would otherwise be
redirected to the captcha page.

**Decision**: the middleware must let the API path through, and the endpoint does its
own verification. A characterization test pins the middleware's current behaviour
first, so this change is deliberate rather than incidental.

This is also a reminder that the middleware is where a bug was found today: it read
`os.environ` for a secret that comes from `get_secret`, so a mounted Docker secret
silently disabled the captcha. Adding a route beside it warrants care.

## R5b. Streaming does not rescue the latency, and the reason is upstream

Measured 2026-09-17 by streaming `astream_events` from the compiled graph and
grouping by `run_id`. Six model calls run before and during one answer:

| at | tokens | node | what it is |
|---|---|---|---|
| 5.4s | 17 | preprocess | rephrase |
| 9.0s | 12 | preprocess | safety check |
| 12.6s | 1 | preprocess | language detection |
| 16.1s | 6 | preprocess | intent classifier |
| 19.8s | 80 | model | query expansion |
| **36.1s** | **1,189** | model | **the answer** |

**The answer's first token is at 36 seconds.** FR-005 asks for two. Streaming was the
plan's answer to a slow completion, and it does not help with this, because nothing
of the answer exists until four sequential preprocessing calls and a retrieval have
finished. A panel would show an empty box for half a minute and then fill quickly.

This reframes the latency work, which the spec had aimed at retrieval:

- **four preprocessing calls run in sequence before retrieval starts**, costing about
  16s between them for 36 tokens of output in total. Whether they must be sequential,
  or must all run for a search-page question, is now the biggest single question
- **query expansion** adds 80 tokens and its own call, then multiplies retrieval
- **retrieval** is what spec 009 addresses, and is no longer the obvious first target

A first-token budget cannot be met by making the answer stream. It needs fewer things
to happen before the answer starts.

## R5c. Which stream events are the answer?

Not separable by tags or metadata: the query expander and the answer both run at
`langgraph_node == "model"` with identical metadata keys, and non-`seq:` tags are
empty for both. The expander's first token is "Which" -- an expanded query, which
would render into the panel as though it were an answer.

**Decision**: treat `on_retriever_end` as the boundary. The expander runs *inside*
retrieval; the answer runs after it. So tokens from `node == "model"` count as answer
tokens only once a retriever has completed.

**Why not "the last model run"**: it is only knowable in hindsight, and a panel
cannot buffer until the end -- that would discard the streaming this is for.

## R6. What is the first increment?

An endpoint that answers correctly and slowly, streaming, with verification. Not fast.

**Rationale**: the website session cannot build the panel or the token handshake
against nothing, and is otherwise idle on this. p50 15.2s is too slow to ship to
users and entirely good enough to integrate against. The latency work is real and
separate, and spec 009 plus query-expansion reduction are its two known levers.
