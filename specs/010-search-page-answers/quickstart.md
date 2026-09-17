# Quickstart: validating the answer endpoint

## Baseline, before any of this is built

Measured 2026-09-17 across the 15 tracked sweep questions, so Phase 5 has a before:

- p50 **15.2s**, p90 **22.4s**, min 6.7s, max 31.5s
- no first-token concept: the answer arrives whole or not at all

## Once the endpoint exists

```bash
curl -N -X POST https://beta.reactome.org/chat/api/answer \
  -H 'Content-Type: application/json' \
  -d '{"question":"what does CDK5 phosphorylate in Alzheimer disease?",
       "human_token":"<token>"}'
```

`-N` matters. Without it curl buffers and the stream looks like a single slow
response, which is the thing being tested.

Expect `event: start` immediately, `event: token` repeatedly, `event: citation` as
retrieval resolves, and `event: done` with a state.

## What to check, and what not to conclude

**Tokens actually stream.** Not that the endpoint returns 200. A 200 that completes
without streaming is a pass on the wrong question -- the same shape as the userguide
shell that returned 200 and contained stylesheets. Count the events.

**A refused request makes no model call.** Assert on a patched graph, not on how fast
the refusal came back; a fast refusal and an expensive one look alike from outside.

**The endpoint and the chat agree.** Same question through `bin/answer-sweep` and
through the endpoint. Two surfaces that can disagree about the same question is a
defect, not a feature (SC-003).

**Failure renders as no panel.** Stop the graph, or send a malformed question, and
confirm a `done` with a non-`answered` state rather than a hang or a 500 body the
page would try to render.

## What this does not prove

Latency. The first increment is deliberately slow -- correct, verified and streaming,
at roughly today's p50. FR-005's 2s/10s is Phase 5, and the website repo is not
waiting on it.
