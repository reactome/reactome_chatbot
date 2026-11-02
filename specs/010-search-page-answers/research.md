# Research: the search-page answer endpoint (2026-09-17 to 2026-09-20)

Findings from the work in [spec.md](./spec.md). "Decision" in a section means a
decision for that work. R1-R6 are the 2026-09-17 starting points; several were
corrected by later measurements, and the sections dated 2026-09-20 supersede them
where they conflict. Notes added on 2026-10-21 are marked. `src/agent/graph.py` cites
R3 by name.

## R1. How does the endpoint stream, given `AgentGraph` has no streaming surface?

`AgentGraph` exposes `ainvoke` and nothing else. Chainlit's streaming comes from
`AsyncLangchainCallbackHandler` with `final_stream` -- a Chainlit object, not
something a plain HTTP handler can use.

| Option | How | Verdict |
|---|---|---|
| A. `astream_events` on the compiled graph | LangGraph's own event stream, filtered to the final LLM's tokens | **Chosen.** It is the framework's supported streaming API, and it exposes retrieval events too, which is where citations come from |
| B. A callback handler pushing to an `asyncio.Queue` | Mirrors what Chainlit does; the SSE response drains the queue | Works, and is what to fall back to if filtering A's event stream proves unreliable. Costs a bespoke handler and a lifetime to manage |
| C. Two graphs, one per surface | Simplest to write | Rejected. Doubles 51.5s of startup and lets the panel and the chat disagree, which SC-003 forbids |

**Decision for this work: A, with B as the named fallback.** A was built
(`AgentGraph.astream_answer`, `src/agent/graph.py:461`). The risk in A is that the event
filtering depends on which node emits the final answer, so a graph change could
silently stop the stream. A test asserting that tokens actually arrive -- not merely
that the endpoint returns 200 -- is the guard, and that lesson is fresh: a 200 with
no content is exactly how the userguide shell fooled two sessions on 2026-09-17.

## R2. Where does the graph live?

`bin/chat-chainlit.py` constructs `AgentGraph` at module scope; Chainlit is mounted
onto the FastAPI app in `bin/chat-fastapi.py`. The endpoint needs the *same*
instance, not another one.

**Decision for this work**: construct it once in the FastAPI app's lifespan and let
both surfaces use it. Built as both: the lifespan builds it
(`bin/chat-fastapi.py:38-66`) and `src/agent/registry.py` is the accessor, which
also builds on demand when Chainlit runs without the FastAPI app.

**Rationale (2026-09-17)**: SC-003 then required the endpoint and the chat to answer
identically, and the cheapest way to guarantee that is one object. SC-003 was later
restated as "configured the same", because one surface does not answer identically
even to itself (spec.md, SC-003). A second graph would also pay the startup cost
again (51.5s as measured then) and double the memory of the BM25 indexes.

## R3. How are citations produced?

Not by parsing the model's prose. The answer prompt instructed `<a href=...>` anchors
(markdown links since 2026-10-03), and pulling those out of a token stream would
couple the wire contract to prompt wording -- prompt changes had happened twice that
week.

**Decision for this work**: emit citation events from the retrieved documents, whose
metadata carries `st_id`. Deduplicate, and emit as each document set arrives. *(As
built since 2026-09-18: only from the fused retriever result, not from each
sub-retriever (38f8885); at most 12; a user-guide page cited by `url` instead (T028).
`src/agent/graph.py:200-261`, `:534-571`.)*

**Consequence worth stating**: the citations are then "what retrieval found", which
is a superset of "what the answer used". The contract does not promise otherwise, and
a panel showing sources the prose did not cite is better than a panel that mis-parses
prose. If exactness is wanted later it needs the model to emit IDs, which is a prompt
change with its own measurement.

## R4. How is the human token verified?

Agreed with the website session on 2026-09-17: they mint after their own captcha,
asymmetric signing, they hold the signing key and this service holds only a verifying
key. The query budget is theirs, enforced before the call, because they proxy every
request. *(The captcha premise was withdrawn on 2026-09-18: the search path has no
human check, and the token asserts caller identity instead -- spec.md, D1. It was
renamed `caller_token`.)*

**Decision for this work (2026-09-17)**: verification is signature + expiry,
stateless, with no consumption tracking, no shared secret and no second counter.
*As built since: audience is also enforced (2026-09-18, T025); `iat`/`nbf` more than
30s in the future, and tokens over 4,096 characters, are refused (2026-10-03); and a
backstop rate limit keyed on `sub` was added (T017, `src/util/rate_limit.py`). It
remains stateless as to token use: a token is never recorded as spent
(`src/util/caller_token.py`).*

**Startup**: a missing or unreadable verifying key stops the process. Principle IV --
an endpoint that accepts everything because its key is absent is the worst outcome
available, and it would test clean.

## R5. Does the new route collide with the captcha middleware?

`verify_captcha_middleware` intercepts every path and returns early only for a small
allowlist under `CHAINLIT_URI`. A new route under `/chat/api/` would otherwise be
redirected to the captcha page.

**Decision for this work**: let the API path through the middleware and have the
endpoint do its own verification, with a characterization test pinning the
middleware first. Built: `src/util/captcha_scope.py` exempts `<CHAINLIT_URI>/api/`
(`bin/chat-fastapi.py:135-152`), pinned by `tests/util/test_captcha_scope.py` (T003).

The middleware was also where a bug was found on 2026-09-17: it read `os.environ`
for a secret that comes from `get_secret`, so a mounted Docker secret silently
disabled the captcha. It now uses the `get_secret` value (`bin/chat-fastapi.py:87`,
`:140-145`).

## R5b. Streaming does not rescue the latency, and the reason is upstream

*Note added 2026-10-21: these are one question's figures. Runs across the tracked
questions did not reproduce the 36.1s (T019, 2026-09-18), and preprocessing has run
in two rounds rather than four sequential calls since 2026-09-18 (T019a). The
structural point -- nothing of the answer exists until preprocessing and retrieval
finish -- still holds. Later figures: "T022" below.*

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

**On this run the answer's first token was at 36 seconds**, where FR-005 then asked
for two. Streaming was the plan's answer to a slow completion, and it does not help
with this, because nothing of the answer exists until preprocessing and a retrieval
have finished.

This reframed the latency work, which the spec had aimed at retrieval:

- **four preprocessing calls ran in sequence before retrieval started**, about 16s
  between them on this run for 36 tokens of output. They were moved to two rounds on
  2026-09-18 (T019a); whether all four are needed was examined in T020c below
- **query expansion** adds 80 tokens and its own call, then multiplies retrieval
  (measured in T020 below)
- **retrieval** was what spec 009 addressed; on 2026-09-20 it was again the largest
  block (T022 below)

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

*Built as decided (`src/agent/graph.py:534-583`). One exception added 2026-09-18
(d3740aa): a question routed to the live MCP lookup has no retriever, so the boundary
opens when preprocessing ends instead, provided live tools are available
(`src/agent/graph.py:500-532`).*

**Why not "the last model run"**: it is only knowable in hindsight, and a panel
cannot buffer until the end -- that would discard the streaming this is for.

## R6. What is the first increment?

An endpoint that answers correctly and slowly, streaming, with verification. Not fast.

**Rationale**: the website session cannot build the panel or the token handshake
against nothing, and is otherwise idle on this. p50 15.2s is too slow to ship to
users and entirely good enough to integrate against. The latency work is real and
separate, and spec 009 plus query-expansion reduction are its two known levers.

The baseline, measured 2026-09-17 across the 15 tracked sweep questions (T002):
whole answer p50 **15.2s**, p90 **22.4s**, min 6.7s, max 31.5s, with no first-token
concept -- the answer arrived whole or not at all. (Moved from quickstart.md,
2026-10-21.)

Why a separate HTTP endpoint rather than Chainlit (moved from plan.md, 2026-10-21):
the search page cannot use a websocket chat UI, and Chainlit's socket protocol is not
a documented API another repository should couple to.

## Query expansion, measured 2026-09-20 (T020)

The task said "reduce query expansion from 5 variants". The premise was
wrong, and the measurement is the useful part.

| alternates | expansion call | retrieval | total | documents kept |
|---|---|---|---|---|
| 4 (default) | 1.27s | 1.22s | **2.49s** | baseline |
| 2 | 1.26s | 0.65s | 1.92s | 84% |
| 1 | 1.38s | 0.57s | 1.95s | 77% |
| 0 | 0.00s | 0.31s | **0.31s** | 71% |

**The expansion call costs about 1.27s whatever it returns.** Trimming four
variants to two saves fan-out only -- about 0.57s of a 2.49s stage. The whole
cost goes away only by not making the call, which is a different change from
the one the task described.

With expansion off, `answer-sweep` passed **13/13 in 79s**, against roughly
150s with it on. That is the largest single latency lever found so far, and
it bears on T022.

### Two things about how this was measured

The first attempt varied the count by editing the prompt to ask for "exactly
N". The model obeyed at 2 and **ignored 1 and 0, producing four either way**,
so those rows silently re-measured the baseline -- visible only because the
number of queries actually asked was recorded alongside the timings. The
count is now enforced in code, which removes the model's obedience from the
experiment and is also how the feature is implemented.

Document overlap is not quality. 71% of baseline documents at zero alternates
says three-quarters of the retrieved set is unchanged; it says nothing about
whether the quarter that changed mattered.

### Why the default is unchanged

Thirteen tracked questions establish that *those* answers do not need
expansion. They do not establish that recall is unaffected in general, and
expansion exists for the questions nobody wrote a test for. So this ships as
`QUERY_EXPANSION_ALTERNATES` with the default at 4 -- a switch and a
measurement, not a verdict. The next step suggested then was to try 0 on beta, where
the sweep and the routing probe both run on every deploy; this repository has no
record of that being tried, and the default is still 4
(`src/retrievers/csv_chroma.py:85-86`).

## T022 -- is 2s/10s reachable? Measured 2026-09-20

Re-measured because the recorded number has been wrong three times. Taken on
`astream_answer`, the path the endpoint uses. The two arms are **interleaved**,
alternating which goes first, and the number of queries actually asked is
recorded so "expansion off" is shown rather than assumed. n=25 per arm.

| | default | expansion off |
|---|---|---|
| queries per retrieval | 5 | 1 |
| preprocess | 1.58s | 1.65s |
| retrieval | 3.15s | 0.29s |
| answer model (residual) | 0.84s | 0.79s |
| **first token, p50** | **5.57s** | **2.73s** |
| first token, min–max | 2.26s – 11.44s | 2.36s – 5.59s |
| **complete, p50** | **8.90s** | **7.16s** |

### The plain answer

**10s complete was met**, at 8.90s p50 on 2026-09-20 and 7.16s with expansion off.

**2s to first token was not met**, either way. Disabling query expansion got
p50 to 2.73s -- short of the target but the same order as it rather than
double. So FR-005a stayed a target with a named blocker.

### The tail is the finding, not the median

The default path's **best** case is 2.26s and its worst is 11.44s -- a fivefold
spread. Disabling expansion barely moves the best case (2.36s) and halves the
worst (5.59s).

So expansion does not make a typical request slow; it makes the slow requests
much slower. A target quoted at p50 would hide that entirely, and the panel
renders progressively, so the tail is the part a reader actually notices. The
recommendation recorded then: state any future latency requirement at a high
percentile.

Reported as min–max rather than p90: at n=25 a p90 is little more than the
second-highest sample, and the first attempt quoted one from n=15, where it
was barely more than the maximum.

### A number I cannot reconcile

The previous record has the answer model's time to first token at **6.1s**,
"the largest block left". This puts it at **0.84s**, with retrieval the
largest at 3.15s.

My figure is a *residual* -- first token minus preprocess minus retrieval --
so it absorbs anything unattributed, which biases it upward, not down. It is
also stable across both arms (0.84s and 0.79s) where the arms differ hugely in
retrieval, which is what a real per-call cost looks like rather than a
measurement artefact.

They disagreed by a factor of seven and only one could describe the code of the day.
No later measurement in this folder reconciles them.
Stated as unreconciled rather than quietly replacing the old number, because
"the answer model is the problem" has been steering this spec's priorities.

### What the interleaving showed

The first attempt ran the arms in sequence, default first, so warming would
have favoured expansion-off. Interleaving changed the conclusion not at all,
and the internal check for it is `preprocess`: it is 1.58s and 1.65s across
arms that differ tenfold in retrieval, and expansion cannot affect
preprocessing, so a sequencing confound would have shown up there and did not.

### What would have had to change, in measured order (2026-09-20)

1. **Retrieval, 3.15s** -- the largest block, and 2.86s of it is query
   expansion. `QUERY_EXPANSION_ALTERNATES=0` removes it and is one
   variable to revert. Its recall cost is unmeasured beyond thirteen tracked
   questions (T020).
2. **Preprocessing, 1.58s** -- whether a search-page question needs all four
   calls is still open (T020c). The two rounds removed the waiting, not the
   calls.
3. **The answer model, 0.84s** -- not worth attention on these numbers, which
   is exactly why the discrepancy above matters.

## T020c -- does a search-page question need all four preprocessing calls?

Measured 2026-09-20. The answer is yes for three of them, and "not for the
reason you would guess" for the fourth.

### The rephraser is not a no-op without history

It exists to resolve a follow-up against chat history, and the search page
sends one question on a fresh thread. The obvious conclusion is that it does
nothing there. Measured over the fifteen tracked questions with empty
history, it **changed 14 of 15**:

- "Which release of Reactome is this?" -> "Which version of Reactome is
  currently available?"
- "What does CDK5 phosphorylate in Alzheimer disease?" -> "What are the
  substrates that CDK5 phosphorylates in the context of..."

That is query normalisation rather than follow-up resolution: it feeds
retrieval and intent classification with different text than it was given.

**"Changed" is not "improved", and the two facts here point opposite ways.**
It rewrites 14 of 15 questions, *and* the sweep passes without it -- so for
the tracked set those rewrites are not load-bearing. What the measurement
supports is that removing it is a retrieval-*input* change rather than a
plumbing one, and therefore needs measuring on something other than the
thirteen questions that already pass either way. It does not support the
claim that the rewrites are valuable.

### With it bypassed, the sweep still passes -- and latency barely moves

`answer-sweep` was 13/13 with the rephraser replaced by a pass-through
(precondition asserted: bypassed 13 times).

Latency, both arms **interleaved in one window**, n=25 each:

| | first token p50 | min | max |
|---|---|---|---|
| expansion off | 2.94s | 2.35s | 5.94s |
| expansion off + rephrase off | **2.83s** | 2.04s | 8.12s |

About **0.11s**, which is the round arithmetic below rather than the ~0.8s the
call's own duration suggests.

A first version of this compared 2.97s against a 2.73s taken in an earlier
run and reported "no better, slightly worse" -- the cross-window comparison
that the T022 review had just corrected, repeated here a few hours later. The
direction was wrong; the conclusion that the saving is ~0.1s rather than ~0.8s
was not, and is now better supported.

### Why removing a 0.88s call saved 0.07s

The arithmetic that predicted ~0.8s was wrong, and the error is worth
recording because it is easy to repeat.

Preprocessing runs two rounds. Round one is `max(rephrase, language)` and
round two is `max(safety, intent)`. Median call times are rephrase 0.88s,
language 0.81s, safety 0.70s, intent 0.93s. So round one costs 0.88s with the
rephraser and **0.81s without it** -- language detection is nearly as slow, and
it was hiding behind the rephraser the whole time.

Bypassing the call does not remove the round. The saving needs the rounds
**restructured**: with no rephrasing, `rephrased_input` is just `user_input`,
so safety and intent no longer depend on round one and all three calls can run
together -- 0.93s instead of 1.74s. That is the ~0.8s, and it requires a code
change this measurement did not make. *(Not made as of 2026-10-21: preprocessing
still runs two rounds, `src/agent/profiles/react_to_me.py:143-172`.)*

### So, for FR-005a

**2s to first token is not reachable by removing these two calls.** Expansion
off gets p50 to 2.94s and adding rephrase-off reaches 2.83s -- the best single
observation dipped to 2.04s, and the median did not. Restructuring preprocessing into one round is worth
about 0.8s more on paper, which would put p50 near 2s -- on paper, and against
a tail that neither change addresses.

### What this does not establish

Both levers pass the same thirteen questions, and that is now being asked to
carry a lot. Each alone keeps the sweep green; together they keep it green;
none of that shows recall is unaffected, and removing *two* recall mechanisms
on one thin evidence base compounds the risk rather than adding to the
confidence. The rephraser rewriting 14 of 15 questions is the concrete reason
to be careful: whatever those rewrites are worth, the sweep is not what
measures it.
