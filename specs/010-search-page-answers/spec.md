# Feature Specification: Chatbot Answers in the Website Search Results

**Feature Branch**: `010-search-page-answers`

**Created**: 2026-09-17

**Status**: Draft. This repo's half only. Three decisions (D1-D3), and one
measured constraint that shapes all of them.

**Input**: *"a feature we want is for the chatbot responses to be integrated into
the website search page results in a google like style ... this would have to be a
coordinated effort with the website repo ... the functionality should only be
available to proven to be human users."*

## The constraint that shapes everything

Measured on 2026-09-17, beta, Release97 bundle, `gpt-4o-mini`, warm process:

Across all fifteen tracked questions, not one question twice:

| | |
|---|---|
| Whole answer | min **6.7s**, p50 **15.2s**, p90 **22.4s**, max **31.5s** |
| Retrieval step, heavy reactome question | ~12.5s of a 27s answer |
| Retrieval step, userguide question | a fraction of a 12.8s answer |
| Query expansion | 2.4s, one LLM call, producing 4 variants + the original |
| Graph construction | 51.5s, once at startup |

*An earlier draft of this spec quoted "25.4s and 32.2s" as the headline. That was two
runs of one question, and that question is near the maximum -- roughly twice the p50.
Corrected on 2026-09-17 after measuring the distribution.*

A Google AI overview is generally reported to arrive in one to two seconds -- an
assumption here, not a measurement of ours. At a p50 of fifteen seconds, a panel on
the search results page is still spinning long after a person has read the ordinary
results, so the conclusion survives the correction even though the number did not.

This is not a polish problem. It decides whether the feature works, so it is stated
here as a requirement rather than discovered during implementation.

Two consequences run through the rest of this spec.

The answer **must stream**, so something appears in the first second or two.

And retrieval is the largest single component for the questions that matter here --
but **collection routing is not the only lever on it, and may not be the biggest**.
Retrieval cost scales with *queries x collections*. Query expansion turns one
question into **five** queries at a cost of 2.4s, and every one of them is run
against every collection. Cutting five collections to two saves about as much as
cutting five queries to two, and only the first has a spec. Measured 2026-09-17:

| | |
|---|---|
| 5 queries x 5 collections, served async path | 12.5s |
| 1 query x 5 collections, served async path | ~1.5s |

[Spec 009](../009-collection-routing/spec.md) remains worth doing. The claim that it
alone is on the critical path does not survive the measurement.

## User Scenarios & Testing

### User Story 1 - A verified person searches and sees an answer forming (Priority: P1)

Someone searches reactome.org. Above the ordinary results, an answer begins appearing
within about two seconds and completes in under ten, with citations into Reactome.

**Why P1**: it is the feature. Everything else is a refinement of it.

**Independent test**: post a question to the answer endpoint with a valid human
token; assert the first token arrives inside the budget and citations resolve to real
stable IDs.

**Acceptance**
1. First streamed token within **2s** (FR-005)
2. Complete answer within **10s** (FR-005)
3. Every factual claim carries a citation resolvable at `reactome.org/content/detail/<stId>`
4. A question the knowledgebase cannot answer says so, rather than inventing

### User Story 2 - Someone who has not been verified gets no answer (Priority: P1)

An unverified visitor, or a script, gets the ordinary search results and no AI panel.
No LLM call is made.

**Why P1 and not P2**: this is a cost and abuse control, not a feature toggle. Every
search reaching the model is an unbounded bill, and search pages are crawled.

**Independent test**: post without a token, and with a forged one; assert HTTP 401/403
and that no LLM call was made.

**Acceptance**
1. No valid human token → refused before any model call
2. A token is bound to the person, expires, and cannot be replayed from elsewhere
3. Refusal is cheap and does not consume a rate-limit slot for real users

### User Story 3 - Not every search gets an answer (Priority: P2)

A search for a single gene name, or a navigational query, returns ordinary results
with no AI panel. Only questions that an answer would actually serve get one.

**Why P2**: the feature works without it, but the cost does not.

**Independent test**: a set of queries labelled should-answer / should-not; assert the
classifier's decision matches, and that no LLM answer call happens for the latter.

### Edge Cases

- **Retrieval finds nothing**: say so; never fill the gap with general knowledge
- **The model is slow or the upstream is down**: the panel disappears rather than
  hanging; the search page must not depend on this service being up
- **The person navigates away mid-stream**: the request is cancelled, not left running
- **The same question twice**: served from cache, not re-answered (see D3)
- **A question that is unsafe or off-topic**: the existing safety checker already
  refuses these, and its refusal must not render as an AI panel

## Requirements

### Functional Requirements

- **FR-001**: The service MUST expose an HTTP endpoint accepting a question and
  returning an answer with citations. No such endpoint exists today -- `chat-fastapi.py`
  serves only the captcha pages and a landing page, and Chainlit owns the conversation
  over websockets
- **FR-002**: The response MUST stream, so partial text can render before completion
- **FR-003**: The endpoint MUST refuse any request without a valid proof-of-human
  token, before any model call
- **FR-004**: Citations MUST be Reactome stable IDs, so the website can render links
  in its own style rather than parsing prose
- **FR-005**: First token within **2s** and complete within **10s**, at p50, for a
  question the knowledgebase can answer
- **FR-006**: The service MUST fail invisibly: any error, timeout or refusal returns
  a response the website can render as "no panel", never a broken panel
- **FR-007**: Answers MUST be attributable to a Reactome release, so a cached or
  stale answer can be identified after a release
- **FR-008**: The endpoint MUST be rate limited per verified person, independently of
  the chat UI's existing limits

### Key Entities

- **Question**: the search string, plus the release it was asked against
- **Answer**: streamed text, a list of citations (stable ID and display name), and a
  completion state -- answered, nothing-found, refused, or failed
- **Human token**: evidence that a person was verified, bound to them, time limited.
  Turnstile already does this in `chat-fastapi.py`; what is missing is a form the
  website can obtain and present

## Success Criteria

### Measurable Outcomes

- **SC-001**: p50 first token ≤ 2s, p50 complete ≤ 10s, measured on the tracked
  question set against a real bundle -- today p50 is 15.2s and p90 22.4s
- **SC-002**: Zero model calls for requests without a valid token, measured by
  counting calls under a load of unauthenticated requests
- **SC-003**: The answer sweep stays green: the endpoint and the chat UI give the
  same answer to the same question, because they share a graph
- **SC-004**: No search-page request can make the search page itself slower or fail;
  verified by taking the service down and confirming the page still renders

## Decisions

### D1 -- what proves a person is human?

Turnstile already exists here, and a bug in it was fixed on 2026-09-17: a deployment
mounting `CLOUDFLARE_SECRET_KEY` as a Docker secret had the captcha silently
disabled, because the middleware read `os.environ` directly while the value came from
`get_secret`.

What is missing is the handoff. The website needs something to present to this
service. Options: a signed cookie on the shared parent domain (what the chat uses
now); a short-lived token the website mints after its own Turnstile check; or the
website proxying the call and vouching server-side.

**Open.** It is a security boundary and belongs with whoever owns the website's
session model, not with this repo alone.

### D2 -- which searches get an answer?

Not all of them, per User Story 3. The intent classifier already decides what kind
of question it is, and [spec 007](../007-answer-cascade/spec.md) is about that
routing. Reusing it costs nothing extra; inventing a second classifier costs a call.

**Recommendation**: extend the existing classifier rather than add one. **Open**
pending 007.

### D3 -- caching

Search traffic repeats in a way chat traffic does not. A cache keyed by question and
release would cut both latency and cost, and FR-007 exists to make invalidation
possible.

**Open.** It needs a measurement of repeat rate on real search traffic, which this
repo does not have -- the website does.

## Scope

**In (this repo)**: the answer endpoint, streaming, citation shape, token
verification, rate limiting, release attribution, and the latency work needed to
meet FR-005.

**Out (the website repo)**: the panel, its visual design, where it sits on the page,
how citations render, and the UX of the chat's deeper integration. This spec commits
to a contract, not a look.

**Out (both, for now)**: the broader "chat integrated into the site's flow and feel"
work. That is a design effort that deserves its own spec once this contract exists,
and it is mostly not this repo's code.

## Assumptions

- The website can verify a human before calling this service; this repo verifies the
  evidence rather than performing the check
- Beta is the integration target first. The MCP already points at
  `beta.reactome.org` as of 2026-09-17
- The same graph serves the endpoint and the chat UI; two answer paths that could
  disagree would be a defect, not a feature
