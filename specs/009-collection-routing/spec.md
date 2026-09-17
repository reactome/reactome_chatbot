# Feature Specification: Searching Only the Collections a Question Needs

**Feature Branch**: `spec/collection-routing`

**Created**: 2026-09-17

**Status**: Planned. D1, D2 and the acceptance bar are all decided and recorded
below; plan.md, data-model.md and tasks.md exist and depend on them.

**Input**: *"the more tables we add the more tokens we use up and the longer the
responses take to return, it would be nice to search the disease variant table if
that is what they are asking about."*

## Clarifications

### Session 2026-09-17

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
this work rather than a follow-up -- see Success Criteria.

## The cost is real, and it was measured

Every collection in the bundle is searched for every question. `retrieve_documents`
loops over all of them and each contributes up to `max_documents_per_collection`
(10), regardless of whether it had anything to say.

Adding `disease_variants` as a fifth collection, asked a question with nothing to do
with variants -- *"What does CDK5 phosphorylate in Alzheimer disease?"*:

| | docs | context tokens | retrieval |
|---|---|---|---|
| 4 collections | 40 | 7,061 | 11.5s |
| 5 collections | 50 | 9,437 | **14.9s** |

**+2,376 tokens (+34%) and +3.4s (+30%)**, all of it noise for that question. The
cost is per collection and per question, so three more tables is roughly +7K tokens
and +10s on every question anyone asks.

## The allocation is also inverted

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

## The proposal: decide collections in the call that already happens

The intent classifier already runs **one LLM call per question** and returns a
source (`reactome`, `userguide`, `live`). Having it also name collections adds **no
latency and no call**.

The descriptions it would need already exist. `reactome_descriptions_info` in
`src/retrievers/reactome/metadata_info.py` holds a written description of every
collection and is currently read only by `bin/retrieval_baseline`. It was written
for exactly this kind of routing and is otherwise unused in the serving path.

`HybridRetriever.retrieve_documents` then loops over the selected collections
instead of all of them -- a filter on `self.collection_retrievers.items()`.

## Decisions

*Both were left as recommendations after the Clarifications session, which settled
only the acceptance bar. `/speckit-analyze` caught that plan.md and data-model.md had
already built on them as though decided -- so they are recorded as decisions here,
on 2026-09-17, rather than left to be inferred from downstream documents.*

### D1 -- what happens when the classifier is unsure

**Decided: select all collections.** A wrong selection costs recall silently,
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

**Decided: (a) first, measured, then (b) separately.** They are independent,
and doing both at once makes it impossible to say which one moved the numbers.

## How we will know it worked

`bin/retrieval_baseline` exists precisely for this: it captures what each retriever
returns for a fixed question set and diffs two captures. Measured across two
identical runs BM25 is byte-identical on 80/80 question-collections, so a diff after
this change is a real difference in behaviour rather than noise.

- `capture` before and after; the diff names every question whose documents changed
- the answer sweep must stay at 13/13 -- including the two variant questions, which
  fail if routing sends them past `disease_variants`, and the species and release
  questions, which fail if it stops routing to `live`
- the token and latency numbers above, re-measured
- the tracked question set grown before the change lands, not after. Thirteen
  questions cannot cover five collections; each collection needs at least one
  question that fails if routing stops searching it. `disease_variants` has two
  already, and the other four have none.

That last one matters: the point of this change is a number going down, and it
should be reported as one.

## Scope

In: collection selection for the `reactome` source, defaulting to all; the
measurement above.

Out: the relevance-weighted allocation of D2, which is its own change; `userguide`,
which has one collection; anything about the four Neo4j-backed collections being
Release95 while `disease_variants` came from the Release 97 download directory --
that is recorded in the bundle's `provenance.json` and is spec 008's problem.
