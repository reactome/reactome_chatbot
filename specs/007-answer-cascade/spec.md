# Feature Specification: The Answer Cascade

**Feature Branch**: `spec/answer-cascade`

**Created**: 2026-09-14

**Status**: Historical record of the answer-cascade proposal and the observations
behind it, 2026-09-14. Drafted as a proposal; this file records no approval of its
requirements, and D1 was never recorded as decided. Of what it proposed, only FR-012
(questions about the database answered from live services) was built as written
(4c4e8a7, PR #216, 2026-09-14), off unless an MCP server is configured. Analysis was
built differently: the chat recognises gene lists and analysis requests before the
graph rather than through the classifier. The Search API step was not built. Whether
the unbuilt parts are still wanted is not recorded. The requirements below were set
for that proposal, not for later changes; their status is in "Requirements, and where
they stand". Current routing is in `src/agent/tasks/intent_classifier.py`,
`src/agent/profiles/react_to_me.py` and `handle_message` in `bin/chat-chainlit.py`.

**Input**: Detect when a question wants an analysis. Otherwise check the embeddings;
if they have no match, search the Reactome Search API for the term; if that finds
nothing, fall back to Tavily.

## The proposed shape

```
                    ┌─ analysis question? ──> MCP analysis agent
question ──> detect ┤
                    └─ otherwise ──> embeddings ──> Search API ──> Tavily
                                       (no match)     (0 entries)
```

Two different mechanisms, and the distinction matters: **analysis is a branch taken
up front**, because it is a different kind of question. The rest is a **cascade**,
each step tried only when the one before it found nothing.

## What existed on 2026-09-14, and since

| step | 2026-09-14 | since (checked 2026-10-21) |
|---|---|---|
| classify the question | `intent_classifier` — routes `reactome` / `userguide` | also `live` when an MCP server is configured (4c4e8a7), and picks collections (spec 009) |
| embeddings | the RAG chain | same role |
| "was that good enough?" | `completeness_grader` | still gates only the Tavily step (`src/tools/external_search/workflow.py:16-19, 73-79`) |
| Tavily | `tavily_wrapper`, run in `postprocess` | unchanged; chat only — the answer endpoint turns postprocess off (`src/api/answer.py:164`) |
| **Reactome Search API** | **nothing** | still no cascade step; the MCP's `reactome_search` tool is reachable only from `live`-routed questions |
| **analysis routing** | **nothing** | in the chat, by pattern matching before the graph (`bin/chat-chainlit.py:699-760`), not by the classifier |

So the cascade's skeleton was real: retrieval, a grader that decides whether the
answer was good enough, and a web search when it was not. What was asked for was
one new step inserted in the middle, and one new branch at the top.

## The hard part is the signal, not the order

Each arrow needs an answer to "did this find anything?", and the three are not
equally easy.

### Search API — free and unambiguous

Measured 2026-09-14 against `ContentService/search/query`:

| query | entries |
|---|---|
| `PALB2` | 39 |
| `Impaired BRCA2 binding to PALB2` | 89 |
| `zzzznotathing` | **0** |

The result count *is* the signal. No model call, no threshold to tune, no judgement.
This step is the cheapest in the cascade and the easiest to get right.

### Embeddings — the genuinely hard one

"If it does not find a match" has no obvious definition, because similarity search
*always* returns its k nearest documents. There is no empty result; there is only a
result that happens to be irrelevant.

Two candidate signals, and they behave differently:

**A distance threshold.** Free and instant, but the number is arbitrary and drifts
with the embedding model. A threshold tuned on Release95 and `text-embedding-3-large`
means nothing after either changes, and nothing warns you.

**The completeness grader**, which already exists and already does this job for the
Tavily step. It judges the generated answer rather than the retrieved distance, which
is the question actually being asked — but it costs a model call and requires
generating an answer before discovering it was not worth generating.

This is D1, and it is the only decision here that is not obvious.

### Analysis — a classifier, and there is one already

Detecting "this wants an analysis" is classification, and `intent_classifier` already
classified. The proposal was for it to gain the destination rather than acquire a
sibling, which was the mistake #142 made: a second LLM classification call on the same question, in a
pipeline where cutting calls from 21 to one was the point of spec 001.

Analysis questions look different enough to be tractable — they carry a list of
identifiers, and ask what is enriched or over-represented.

**What was built instead (checked 2026-10-21).** The classifier has no analysis
destination (`src/agent/tasks/intent_classifier.py:10`). The chat recognises gene
lists and analysis requests by pattern on the user's own message, before the graph
runs and so before the rephrase (`src/analysis/gene_list.py:41-80, 419`;
`bin/chat-chainlit.py:699-760`). That costs no LLM call; a recognised list is offered
an analysis and run only on confirmation. The search page has no analysis path.

## What it would cost (estimate, 2026-09-14)

Worst case, a question that falls all the way through: classification, retrieval and
generation, a completeness judgement, a Search API call, and a Tavily call. On a
surface then at ~22s per question, each step had to earn its place.

Two mitigations were expected. Most questions would stop at the embeddings, so the
deep path would be rare (assumed, never measured). And the two new steps were the
cheap ones — the Search API measured at well under a second, and analysis at **0.2s**
measured through `reactome-mcp`, both on 2026-09-14.

## Observed, 2026-09-14

Two questions asked on beta by a Reactome developer. Beta was running `e398a37`
— production's commit, which predates both the intent classifier and the user
guide source — so this is not a report on `main` as it then was. It is recorded because
it is what the cascade is for, asked by a real user in their own words.

### "I want to run a gsea with my list of genes"

> I'm sorry, but the information regarding how to perform a Gene Set Enrichment
> Analysis (GSEA) is not currently available in the Reactome Knowledgebase.

Then three external links: `fgsea` in R, bigomics.ch, and a YouTube tutorial.

Reactome's own analysis service was never consulted. The failure is not that the
answer was empty — it is that a Reactome user asking to analyse a gene list was
sent to somebody else's tools. **This is User Story 1**, in the user's phrasing
rather than the spec's.

### "what tools are available for me to run on here"

> The Reactome Knowledgebase does not currently provide specific information
> about tools available for running analyses on the Human Reactome (HRE).

This one is worse, and in a way worth naming precisely: it is **confidently
wrong**, not unhelpful. Analysis is a flagship Reactome feature. A retrieval
miss was reported as a fact about Reactome, because nothing distinguishes "the
vector store did not have this" from "Reactome does not have this". That is the
"user cannot tell which source answered" edge case, showing up as an assertion
rather than as a gap.

Later change (3ad234f, 2026-09-18): the Reactome answer prompt now tells the model to
say an answer was not found "in the Reactome pathway content searched", never that
Reactome lacks it (`src/retrievers/reactome/prompt.py:17-18`).

It is also a user guide question. `main` had a user guide source and routed to
it — but the user guide bundle was not installed on the beta host, so `main`
would log `User guide embeddings not configured; routing will use reactome only`
and fall back the same way. **Routing is necessary and not sufficient**; the
source has to be present. Its absence still degrades silently: a missing user guide
bundle is reported only by that INFO log line, and the classifier is then simply not
offered `userguide` (`src/agent/profiles/react_to_me.py:94-97, 120-125`).

### Reactome does do GSEA — through a service nothing here talked to (2026-09-14)

An earlier draft of this spec asserted that Reactome performs
over-representation analysis and not GSEA, and that the honest reply was to say
so. **That was wrong**, and it is worth recording because the spec came within
one merge of telling a future implementer to say something false to users.

There are two analysis services:

| | | |
|---|---|---|
| `AnalysisService` | over-representation over an identifier list | what `reactome-mcp` wraps |
| **ReactomeGSA** (`gsa.reactome.org`) | **PADOG, Camera, ssGSEA, terapadog** | **nothing here talked to it** on 2026-09-14; `src/gsa/` does since 2026-09-21 |

Camera is described by the service itself as *"a gene set analysis algorithm
similar to the classical GSEA algorithm"*. Verified 2026-09-14 against
`GET https://gsa.reactome.org/0.1/methods`, and `reactome.org/gsa/` is a live
page.

So the answer to "I want to run a GSEA" was not "Reactome does ORA instead". It
was ReactomeGSA — a Reactome product — and the reason the beta chatbot sent a
user to `fgsea` and a YouTube tutorial was that **no part of this system knew
ReactomeGSA existed**. This spec treated that as a gap in `reactome-mcp` first.

**What was built instead (checked 2026-10-21).** The chat talks to ReactomeGSA
itself, without MCP: an attached expression matrix runs a ReactomeGSA analysis
(`src/gsa/client.py:50`; `bin/chat-chainlit.py:699-701`), and a request to run one in
words is answered with how to do it in the chat (`src/gsa/chat.py:242-253`;
`bin/chat-chainlit.py:757`). A pasted gene list — identifiers without per-sample
measurements — is offered an over-representation analysis, never ReactomeGSA
(93827ea, 2026-10-02; `src/analysis/gene_list.py`).

### The rephrase turns "do this" into "how do I do this"

Measured 2026-09-14, three runs per question. Production classifies the
**rephrased** question, not the raw one — `preprocess` rephrases first and
passes `rephrased_input` to the classifier — and the rephrase systematically
converts an action request into a how-to question:

| asked | rephrased to |
|---|---|
| I want to run a gsea with my list of genes | How **can I perform** a Gene Set Enrichment Analysis (GSEA) … |
| Analyse these genes for over-representation: TP53 BRCA1 EGFR | How **can I analyze** the over-representation of the genes … |

"How can I…" is a user-guide shape, and it routes accordingly. One question
changes destination because of it: *Analyse these genes for over-representation*
classifies as `reactome` when raw and `userguide` after rephrasing.

This matters for FR-001. The analysis branch cannot be built on the classifier
alone while the step in front of it is rewriting requests-to-act into
questions-about-how — by the time the classifier sees the question, the
signal it needs is gone. **FR-011.**

In the chat this was resolved by detecting analysis requests on the raw message,
before the graph and its rephrase (`bin/chat-chainlit.py:732`).

### What `main`'s classifier did with them (2026-09-14)

Measured 2026-09-14, `gpt-4o-mini`, **three runs per question, classifying the
rephrased form as production does**. Every question routed identically on all
three runs, so the classifier is stable here and single samples below are not
hiding variance:

| question | routes to | wanted |
|---|---|---|
| I want to run a gsea with my list of genes | `userguide` | analysis |
| Which pathways are enriched in TP53, BRCA1, EGFR, MYC, CDKN1A? | `reactome` | analysis |
| Analyse these genes for over-representation: TP53 BRCA1 EGFR | `userguide` (`reactome` if not rephrased) | analysis |
| What do TP53, BRCA1 and EGFR have in common? | `reactome` | `reactome` ✓ |
| **what tools are available for me to run on here** | **`userguide`** | **`userguide` ✓** |
| How do I run a gene list analysis on the Reactome website? | `userguide` | `userguide` ✓ |
| How do I use the pathway browser? | `userguide` | `userguide` ✓ |
| What does CDK5 phosphorylate in Alzheimer's disease? | `reactome` | `reactome` ✓ |
| How does TP53 regulate PTEN transcription? | `reactome` | `reactome` ✓ |
| What is R-HSA-9613829? | `reactome` | `reactome` ✓ |
| What is the current price of a Nature subscription? | `userguide` | neither |

Three things follow.

**The second beta failure was already routed correctly.** "what tools are
available for me to run on here" classified as `userguide` on `main` at the time. The
classifier was not the problem for that question; the missing bundle was. Installing
it on the beta host was proposed as a cheaper first fix than any cascade work. This
repository does not record whether that was done.

**The analysis questions landed wherever the two existing destinations allowed.**
Two went to `reactome`, one to `userguide` — not wrong given the choices, simply
unable to express the right answer. That is the case for FR-001, measured
rather than argued.

**A question that belongs nowhere is forced somewhere.** The Nature subscription
question routed to `userguide`, because with two destinations and a required
choice there was no way to say "neither". The completeness grader caught this
downstream and Tavily answered it, so the outcome was acceptable by accident. The
argument was that adding destinations makes the guess worse, not better: the more
sources, the more confident the misrouting. FR-010. Since then `live` has been added
as a third destination, and there is still no way to decline
(`src/agent/tasks/intent_classifier.py:10, 151-154`). The grader and Tavily run only in
the chat, not on the search page's answer endpoint (`src/api/answer.py:164`).

### The strongest case is not the cascade at all

Measured 2026-09-14. "What species are included in the Reactome database?"
routes to `reactome` on all five runs, retrieves 40 documents, and is answered:

> The Reactome Knowledgebase primarily focuses on **Homo sapiens** (humans).
> There are no indications of other species being included.

Reactome covers **96 species**. `reactome_species` on the MCP returns all of
them with taxonomy IDs, in 1,988 characters.

Nothing malfunctioned. The routing was right, and those 40 documents really are
human pathways. **The corpus is a sample of the database's content, and the
question is about the database's scope.** No reranking, no better embedding and
no prompt change closes that gap, because the answer is not in the corpus at
any depth.

There is a class of these:

| question | retrieval | MCP |
|---|---|---|
| What species does Reactome cover? | infers from whatever was sampled | `reactome_species` — 96 |
| Which release is this? | cannot know; the bundle is a snapshot | `reactome_database_info` |
| Is pathway X in Reactome? | only if X was sampled | `reactome_search` |

All three tools were already wrapped in `src/reactome_mcp/tools.py`, chosen for
exactly this reason, and not wired to anything.

That made it the cheapest useful MCP work: it needed no answer to D1, no cascade,
and no new capability — only that these questions reach a tool instead of the
vector store. **Built the same day** as the `live` destination (4c4e8a7, PR #216;
FR-012), offered only when an MCP server is configured.

### Where these questions were recorded

`tests/golden/cascade-questions.txt`, alongside the cases for every other branch
of the proposed cascade — including two to be answered from the embeddings exactly
as before, and one identifier meant to stop at the Search API step, which was not
built. No code reads the file; it is a question list, not a test.

They are kept out of `tests/golden/questions.txt` deliberately: that file is a
fixed baseline for comparing retrieval captures over time, and adding to it
would invalidate every capture taken before.

## User stories as proposed (2026-09-14)

Proposed, not recorded as approved. What became of each is in "Requirements, and
where they stand" below.

### User Story 1 — A gene list gets an analysis (Priority: P1)

A researcher pastes identifiers and asks what is enriched. The question is routed to
analysis rather than retrieval, and comes back with pathways and p-values.

**Independent Test**: a gene set with a known enrichment; the answer names it.

**Acceptance Scenarios**:

1. **Given** a list of identifiers and an enrichment question, **Then** analysis runs,
   not retrieval.
2. **Given** an ordinary question, **Then** it is not routed to analysis.
3. **Given** analysis is unavailable, **Then** the chatbot says so and does not answer
   from the vector store as though it had analysed anything.

---

### User Story 2 — A term the embeddings miss is still found (Priority: P1)

A user asks about something in Reactome that retrieval does not surface — a recently
added pathway, or an exact identifier. The Search API finds it.

**Why this priority**: this was the step being proposed, and the gap it would close is real —
the bundle is a snapshot, while the Search API is live.

**Acceptance Scenarios**:

1. **Given** retrieval found nothing useful, **When** the Search API returns entries,
   **Then** they inform the answer.
2. **Given** the Search API returns **0 entries**, **Then** the cascade proceeds to
   Tavily.
3. **Given** retrieval succeeded, **Then** no Search API call is made.

---

### User Story 3 — Nothing in Reactome, so look outside (Priority: P2)

Neither the embeddings nor the Search API has it. Tavily results appear, clearly
marked as external.

**Acceptance Scenarios**:

1. **Given** both Reactome sources found nothing, **Then** Tavily runs.
2. **Given** Tavily results, **Then** they appear as links beside the answer, not
   blended into it — the existing `SearchResults` element.

### Edge Cases

- **Everything fails.** The honest answer is "Reactome does not cover this", not a
  confident answer assembled from nothing.
- **The user cannot tell which source answered.** Retrieval, live search and the open
  web have very different standing. An answer that mixes them silently is a new class
  of wrong.
- **A misrouted analysis question.** A gene list sent to retrieval gets a plausible
  pathway answer that is not an analysis, and nothing says so.
- **Latency on the deep path.** Four steps before an answer appears.

## Requirements, and where they stand (checked 2026-10-21)

Proposed by this spec; no approval is recorded. Where one is not built, whether it is
still wanted is not recorded. The measurements behind FR-010 to FR-012 are in
"Observed, 2026-09-14" above.

| ID | Proposed | Status |
|---|---|---|
| FR-001 | Analysis questions routed to analysis, by extending the existing classifier | Built differently. In the chat, gene lists and analysis requests are recognised by pattern before the graph (`src/analysis/gene_list.py:41-80, 419`; `bin/chat-chainlit.py:699-760`) and proposed, then run on confirmation. The classifier has no analysis destination (`src/agent/tasks/intent_classifier.py:10`). The search page has no analysis path |
| FR-002 | The cascade stops at the first step that finds something | Not built beyond the existing grader-then-Tavily step |
| FR-003 | The Search API step uses its own result count as the signal | Not built: nothing calls the Search API as a cascade step. The MCP's `reactome_search` tool is used only inside `live`-routed answers (`src/reactome_mcp/tools.py:54`) |
| FR-004 | A step that finds nothing falls through, never fabricates | No cascade to fall through. The prompts tell the model to say what was not found (`src/retrievers/reactome/prompt.py:17`; `src/reactome_mcp/answer.py:75-87`) |
| FR-005 | Answer source distinguishable: bundle, live Reactome, open web | Partly: Tavily results appear separately (FR-006); live answers carry no citations (d3740aa) but are not labelled |
| FR-006 | Tavily results remain links beside the answer | Holds in the chat (`src/agent/profiles/base.py:83-112`; `SearchResults` element, `src/util/chainlit_helpers.py:191`). The answer endpoint runs no web search (`src/api/answer.py:164`) |
| FR-007 | With the new steps disabled, behaviour and call count as before | Holds for `live`: without an MCP server it is not offered and the classifier gets the two-source prompt (`src/agent/tasks/intent_classifier.py:110-141, 189-206`; `tests/agent/test_intent_classifier_sources.py:21-33, 80-84`). The chat's analysis detection has no switch, and adds no LLM call |
| FR-008 | An analysis request is answered by the Reactome service that performs it, naming it; a GSEA request is never told "Reactome does not do that" | Holds in the chat: gene lists go to over-representation, named in the reply (`src/analysis/gene_list.py:520`); expression matrices go to ReactomeGSA (`src/gsa/`); a GSEA request in words is told how to run ReactomeGSA in the chat (`src/gsa/chat.py:242-253`). A gene list is never sent to ReactomeGSA (93827ea, 2026-10-02) |
| FR-009 | A configured-but-unavailable source distinguishable from one with no answer | Not built: a missing user guide bundle is an INFO log line (`src/agent/profiles/react_to_me.py:120-125`); an unavailable MCP falls back to retrieval with a WARNING log only (`react_to_me.py:219-243`) |
| FR-010 | The classifier can decline ("none of these") | Not built: `source` must be `reactome`, `userguide` or `live` (`src/agent/tasks/intent_classifier.py:10, 151-154`) |
| FR-011 | The analysis signal survives the rephrase, or is taken before it | Holds in the chat: detection runs on the raw message before the graph (`bin/chat-chainlit.py:732`) |
| FR-012 | Questions about the database itself (species, release, whether it contains something) answered from live services | Built (4c4e8a7, PR #216): the `live` destination (`src/agent/tasks/intent_classifier.py:29-36`; `src/agent/profiles/react_to_me.py:208-261`). Offered only when `REACTOME_MCP_URL` or `REACTOME_MCP_SERVER` is set; falls back to retrieval if the server is unavailable (FR-009) |
| SC-001 | A gene-list question yields a real analysis or an explicit refusal | In the chat; failure message at `src/analysis/gene_list.py:585-589` |
| SC-002 | Content the bundle lacks but Reactome has is answered | No cascade; `live` answers "is X in Reactome" questions where configured |
| SC-003 | Nothing anywhere gives an honest "not covered" | Through prompt wording only (`src/retrievers/reactome/prompt.py:17`) |
| SC-004 | A question answered from the embeddings costs the same | No cascade step was added to that path |
| SC-005 | A user can tell which source answered | As FR-005 |

## Decisions put to the team (2026-09-14)

### D1 — What counts as "the embeddings found no match"? (never decided)

| option | what it means |
|---|---|
| **A. Reuse the completeness grader** (recommended) | It exists, it already gates Tavily, and it judges the answer rather than a distance — which is the real question. Costs a model call and requires generating first. |
| B. A distance threshold | Free and instant, but the number is arbitrary, drifts with the embedding model and the release, and nothing tells you when it has stopped meaning anything. |
| C. Threshold first, grader second | Cheap rejection of the obviously-irrelevant, grader for the rest. Two signals to reason about and two ways to be wrong. |

The proposal recommended A: the grader already existed and was trusted for the Tavily
decision, and reusing it would make the cascade one consistent idea rather than two,
with B as a later tuning exercise if latency proved to be the problem. No decision is
recorded, and the step D1 was for was not built. The completeness grader still gates
only the Tavily step (`src/tools/external_search/workflow.py:16-19, 73-79`).

## Assumptions at the time (2026-09-14)

- Most questions stop at the embeddings, so the deep path is rare. Never measured;
  if most questions fell through, the bundle would be the problem.
- The Search API stays fast and unauthenticated.
- Analysis arrives over hosted MCP ([spec 006](../006-mcp-hosting/spec.md)), not a
  spawned subprocess. In the event it was built against the Analysis Service and
  ReactomeGSA directly (`src/analysis/`, `src/gsa/`).

## Out of scope for the proposal

- Rebuilding the bundle. That was the planned Release 98 work (spec 005, D1), which
  would reduce how often the cascade goes deep, but it was not this.
- Which MCP tools beyond analysis to expose.
- Presenting analysis results in the UI.
