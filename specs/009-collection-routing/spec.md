# Feature Specification: Searching Only the Collections a Question Needs

**Feature Branch**: `spec/collection-routing`

**Created**: 2026-09-17

**Status**: Historical record of the collection-routing work, 2026-09-17 to
2026-09-19. Implemented 2026-09-19 (0fe7d88). The decisions and acceptance bar
below were set for that work, not for later changes. Current behaviour is in
[data-model.md](./data-model.md).

**Input**: *"the more tables we add the more tokens we use up and the longer the
responses take to return, it would be nice to search the disease variant table if
that is what they are asking about."*

## Acceptance bar for this work (agreed 2026-09-17)

- Q: When routing skips a collection that would have contributed a document, how much recall loss should reject the change? → A: Gate on answers, measure documents. No numeric document-loss threshold is set until there is one real measurement to set it from.

The answer sweep staying green is the pass/fail. `bin/retrieval_baseline` is captured
before and after and its diff reported, because Principle II requires the measurement
-- but a changed document set does not by itself block, since dropping documents is
what this change is for. A wrong answer is the defect; a dropped document is data.

Any threshold chosen today would be invented: there is no evidence yet on how often
the classifier picks wrong, and a number that looks rigorous and means nothing is the
failure this project keeps finding.

The weakness in that, stated rather than hidden: thirteen questions is a thin net, and
routing could break something nobody tracks. So growing the tracked set is part of
this work rather than a follow-up -- see "How the work was checked" below.

## The cost before routing (measured 2026-09-17)

Before routing, every collection in the bundle was searched for every question,
and each contributed up to `max_documents_per_collection` (10), regardless of
whether it had anything to say.

Adding `disease_variants` as a fifth collection, asked a question with nothing to do
with variants -- *"What does CDK5 phosphorylate in Alzheimer disease?"*:

| | docs | context tokens | retrieval |
|---|---|---|---|
| 4 collections | 40 | 7,061 | 11.5s |
| 5 collections | 50 | 9,437 | **14.9s** |

**+2,376 tokens (+34%) and +3.4s (+30%)**, all of it noise for that question. The
cost is per collection and per question, so, as an extrapolation rather than a
measurement, three more tables would be roughly +7K tokens and +10s on every
question anyone asks.

## The allocation is also inverted (measured 2026-09-17)

Worse than linear cost: `disease_variants` contributes **more** to questions it is
irrelevant to than to the one it exists for.

| question | tokens from `disease_variants` | share of context |
|---|---|---|
| What does CDK5 phosphorylate in Alzheimer disease? | 1,427 | **15%** |
| How does TP53 regulate PTEN transcription? | 1,135 | 8% |
| List the ABCA1 variants in Reactome | 616 | **5%** |

Each collection gets a fixed ten documents whether or not they are any good, and
variant documents are short (556 characters median) where pathway prose is long. So
the fixed allocation spends the most context on the collection with the least to
contribute, and the least on the one that answers the question.

## The approach: collections chosen in the existing classifier call

The intent classifier already runs **one LLM call per question** and returns a
source (`reactome`, `userguide`, `live`). Having it also name collections adds **no
latency and no call**.

The descriptions it would need already exist. `reactome_descriptions_info` in
`src/retrievers/reactome/metadata_info.py` holds a written description of every
collection. At the time it was read only by `bin/retrieval_baseline`. It is now
also the source of the classifier's collection list
(`src/agent/tasks/intent_classifier.py:55-67`) and of `bin/routing-probe`'s
coverage check.

Both retrieval paths now search only the collections `resolve_collections` returns
(`src/retrievers/csv_chroma.py:580`, `:641`).

## Decisions

*Both were left as recommendations after the Clarifications session, which settled
only the acceptance bar. `/speckit-analyze` caught that plan.md and data-model.md had
already built on them as though decided -- so they are recorded as decisions here,
on 2026-09-17, rather than left to be inferred from downstream documents.*

### D1 -- what happens when the classifier is unsure

**Decision for this work: select all collections.** Implemented: an empty or
unrecognised selection searches every collection (`resolve_collections`,
`src/retrievers/csv_chroma.py:320`). A wrong selection costs recall silently,
which is the failure this project keeps finding; a wrong *default* costs only what
we already pay today. So the change can only make things faster, never worse than
the current behaviour, unless the classifier actively picks a wrong subset.

The alternative -- always include a core set and gate only specialised collections
-- does not scale: the fifth collection is specialised today, but the reason for
this spec is that there will be a tenth.

### D2 -- fixed allocation, or relevance-weighted

Routing fixes *which* collections are searched. It does not fix the inversion above,
which is the fixed ten-per-collection cap.

Option (a): leave the cap alone. Simple, and routing alone removes most of the waste.

Option (b): give the 50-document budget out by score rather than by collection, so a
collection with nothing relevant contributes nothing even when it is searched.

**Decision for this work: (a).** They are independent, and doing both at once
makes it impossible to say which one moved the numbers. Option (b) was deferred
as a separate change. It has not been implemented, and this repository does not
record whether it is still wanted. The cross-collection fusion added in 38f8885
reorders documents but does not reallocate the budget by score; each searched
collection still contributes up to ten.

## How the work was checked

- Across two identical runs, BM25 was byte-identical on 80/80 question-collections,
  so a `bin/retrieval_baseline` diff reflects a real change rather than noise.
- The "before" capture is [before.json](./before.json) (2026-09-18). No "after"
  capture was made.
- Answer sweep: 13/13 with the classifier choosing, matching the control
  (research.md, 2026-09-19).
- Re-measurement: retrieval time was measured (research.md). The recorded
  context-size result conflicts with the code and is unverified; see research.md.
- Tracked questions were added for `ewas` and `summations`. `reactions` and
  `complexes` cannot be guarded by a question, so a retrieval-level test and the
  routing probe cover them instead.

## Finding: the collections overlap more than this spec assumed

Phase 2 set out to add one sweep question per collection, each failing if its
collection stopped being searched. Building them changed the picture.

Method: ask a candidate against the full bundle and against a copy with exactly one
collection removed. A question guards a collection only if it answers with it and
fails without it.

| collection | with | without | result |
|---|---|---|---|
| `ewas` | yes | no | **guards it** -- UniProt accessions live only there |
| `summations` | yes | no | **guards it** -- the curated prose lives only there |
| `disease_variants` (PTEN) | yes | no | **guards it** -- no variant is named without it |
| `disease_variants` (ABCA1) | yes | **yes** | all six variants came back from `summations` prose |
| `complexes` | -- | -- | no question found yet |
| `reactions` | yes | **yes** | answered with the collection removed entirely |

**No reaction name is unique to `reactions`.** The `summations` query covers
`Pathway OR ReactionLikeEvent`, so every event name appears in both collections by
construction. They overlap by design.

**And the overlap is wider than that.** Measured 2026-09-19 (research.md), the
ABCA1 question answered with all six curated variants while `disease_variants`
was excluded -- the summation prose for "Defective ABCA1 does not transport CHOL"
names them. So a collection can be partly reachable through `summations` too, and
whether a question guards a collection has to be measured per question, not
assumed from the collection it was written for.

Two consequences for this feature.

The recall risk is **lower** than this spec assumed. If a question can be answered
with a whole collection removed, routing away from that collection costs little --
which argues for the change rather than against it.

But the gate hole is **harder to close** than the plan implies, and for a reason the
plan had wrong. It is not that nobody wrote the questions; it is that a collection
whose content is duplicated elsewhere cannot be guarded by asking a question. For
`reactions` the guard would have to assert on retrieval -- which documents were
searched -- rather than on the answer.

Outcome: T007 (`reactions`) became a retrieval-level assertion. T005 (`complexes`)
was abandoned as a question and covered the same way (T005a), plus the routing
probe (T005b). See [tasks.md](./tasks.md) and research.md.

## Scope of this work

In: collection selection for the `reactome` source, defaulting to all; the
measurement above.

Out: the relevance-weighted allocation of D2 (deferred; see D2); `userguide`,
which has one collection; anything about the four Neo4j-backed collections being
Release95 while `disease_variants` came from the Release 97 download directory --
that is recorded in the bundle's `provenance.json` and was left to the
disease-variant work (spec 008).
