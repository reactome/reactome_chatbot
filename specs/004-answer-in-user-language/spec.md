# Feature Specification: Answer in the User's Language

**Feature Branch**: `spec/answer-in-user-language`

**Created**: 2026-09-10

**Status**: Draft. Two contributed PRs to harvest; one decision (D1) for the team.

**Input**: Issue #104, "RAG only responds in English". Two competing pull requests, #125 and #140.

## The gap, precisely

Language detection already works. `detect_language` runs on every message, concurrently
with the safety check, and `detected_language` is on `BaseState` for every profile.

It reaches some places and not others:

| where | uses the detected language? |
|---|---|
| refusal messages (React-to-Me, Plant Reactome) | **yes** |
| Cross-Database final summary | **yes** |
| **React-to-Me's answer** | **no** |
| **Plant Reactome's answer** | **no** |

So a question in French is detected as French, rephrased into English for retrieval,
answered in English — and if it had instead been *refused*, the refusal would have
come back in French. The two profiles that serve real users are the two that answer
in the wrong language; the one that gets it right is the prototype.

That asymmetry is the whole feature. Nothing needs detecting, routing or storing that
is not already detected, routed and stored.

Reproduced against the Release95 bundle, running exactly what `generate_answer` does:

| step | result |
|---|---|
| question | *Quel role joue TP53 dans l'apoptose ?* |
| detected language | **French** |
| rephrased for retrieval | *What role does TP53 play in apoptosis?* |
| retrieval | good — the English rephrasing does its job |
| **answer** | **English** |

Four steps of five work. The detector is right, the translation is right, retrieval
is right, and then the answer comes back in the wrong language.

One detail makes it unambiguous: `chat_history` still holds the original French
question, and the model answered in English anyway. It is not that the model cannot
tell what language was used — nothing asks it to reply in that language.

## The constraint that shapes the design

**Retrieval must stay in English.** The Reactome corpus is English: the CSV rows BM25
tokenises, and the vectors built from them.

Measured on the `summations` collection rather than assumed, because the obvious
version of this claim is wrong. BM25 does **not** collapse on a non-English query —
it still returns a full ten documents, because named entities survive translation:

| query | documents shared with the English form |
|---|---|
| *Quel role joue TP53 dans l'apoptose ?* | 3 of 10 |
| *Wie wird die Glykolyse reguliert?* | 3 of 10 |

So the cost of retrieving in French is that **seven of ten documents change**, not
that retrieval fails. That is still a large enough change to be worth avoiding, and
the rephrase step already translates to English for exactly this reason — but the
reason is "materially different results", not "no results".

So the language instruction has to reach the **answer** without reaching the
**retrieval**. That is the entire design question, and it is where the two
contributed PRs diverge.

## Two contributed pull requests

Both from GSoC applicants who will not update them. Two people arriving at the same
issue independently is strong evidence it is worth fixing.

### #140 — @bleedblack1

Targets **React-to-Me**, which is the right profile: it is the one users use, and
the one with the gap. The intent is exactly right.

The mechanism is not. It appends the language instruction to `input`:

```python
query = f"{query}\n\n[CRITICAL INSTRUCTION: You MUST write your entire response in {detected_language}...]"
result = await self.reactome_rag.ainvoke({"input": query, ...})
```

`create_retrieval_chain` passes `input` **straight to the retriever**. Verified in
`langchain_classic`: *"it is expected that an `input` key be passed in — this is what
will be used to pass into the retriever."* So that 61-word block of English prose
about response languages becomes part of the BM25 query and the embedded vector, for
every non-English question.

Measured through the **whole retriever**, appending it to an English question:

| query | fused documents surviving the appended instruction |
|---|---|
| *What role does TP53 play in apoptosis?* | 20 of 40 |
| *Which complexes contain EGFR?* | 21 of 40 |

**About half the context changes.** That is the number to argue from.

It is worth saying what the first version of this section got wrong, because the
mistake is instructive. Measuring BM25 *directly* on the polluted string gives 0 of
10, 1 of 10, 0 of 10 — sixty-one words of instruction outweigh a six-word question,
so lexical ranking collapses entirely. But BM25 never sees that string in production:
`HybridRetriever` expands the query into four LLM-generated alternates first and
appends the original last, so four of the five queries are clean rewrites and the
fusion recovers most of the damage.

The component number was dramatic and irrelevant; the pipeline number is half, and
real. Constitution Article I, arrived at the hard way — twice in one week, after
`evaluator.py` measured a retriever the product no longer used.

Half the retrieved context silently differing for non-English users is still reason
enough to reject the mechanism. The instruction belongs in the answer prompt, where
it changes nothing about what is retrieved.

A second, smaller problem: #140 edits a method called `call_model`, which `main`
renamed to `generate_answer`. The patch does not apply as written.

It also carries the answer to a real sub-problem, and gets it right: gene symbols,
protein names, `R-HSA-*` identifiers and URLs must survive untranslated. That
requirement is kept.

### #125 — @bhavyakeerthi3

Strengthens the **Cross-Database summarizer** prompt, where `{detected_language}` is
already a template variable — the clean mechanism, applied to the profile that
already worked. It also rewrites the rephrase prompt while **keeping** the
translate-to-English step, which is correct and worth noting because removing it
would have been the obvious mistake.

Its problem is scope. The same PR adds a hallucination grader and web-search wiring
to Cross-Database, overlapping #123, so the language change cannot be taken without
taking a second feature that needs its own judgement.

### What each contributes

| | #140 | #125 |
|---|---|---|
| right profile | **yes** | no (prototype only) |
| right mechanism | no — pollutes retrieval | **yes** — prompt variable |
| protects nomenclature | **yes** | partly |
| self-contained | **yes** | no — bundles #123's feature |

Neither is mergeable as it stands. Between them they contain the whole answer:
**#140's target and its nomenclature rule, #125's mechanism.**

## User Scenarios & Testing *(mandatory)*

### User Story 1 — A question in French is answered in French (Priority: P1)

A researcher asks React-to-Me a question in French. Retrieval happens in English, and
the answer comes back in French, with gene names and Reactome links untouched.

**Why this priority**: It is the issue. Everything else here is a refinement of it.

**Independent Test**: Ask the same question in English and in French. The retrieved
documents should be substantially the same; the answers should differ only in
language.

**Acceptance Scenarios**:

1. **Given** a question in French, **When** it is answered, **Then** the answer is in
   French and retrieval used the English rephrasing, so the documents are the set the
   English rendering retrieves.
2. **Given** a question in English, **When** it is answered, **Then** nothing about
   the behaviour differs from today — no extra instruction, no extra call.
3. **Given** an answer in any language, **When** it names a gene, protein, pathway or
   `R-HSA-*` identifier, **Then** that term appears in its original English form.
4. **Given** an answer in any language, **When** it cites Reactome, **Then** the URL
   is unchanged and still resolves.

---

### User Story 2 — Retrieval quality does not move (Priority: P1)

A developer changes the language handling and can show that what reaches the model is
unchanged for English questions.

**Why this priority**: Equal to Story 1, because the plausible way to implement Story
1 is the way #140 does it, and that quietly degrades retrieval for exactly the users
the feature is meant to help. A feature that answers in French by retrieving worse
has not helped anyone.

**Independent Test**: `bin/retrieval_baseline` before and after. English questions
must be unchanged beyond the known ANN noise floor.

**Acceptance Scenarios**:

1. **Given** the change, **When** the baseline is compared, **Then** English
   retrieval differs only within the documented run-to-run variance.
2. **Given** a non-English question, **When** retrieval runs, **Then** the text sent
   to BM25 and the vector store contains no instruction prose — only the rephrased
   question.

---

### User Story 3 — Plant Reactome behaves the same way (Priority: P2)

The Plant Reactome deployment answers in the user's language too.

**Why this priority**: The same gap, the same fix, a second profile. P2 only because
the Reactome deployment is the larger audience.

**Acceptance Scenarios**:

1. **Given** a non-English question to Plant Reactome, **Then** the answer is in that
   language, with nomenclature preserved.

### Edge Cases

- **The detector is wrong.** It returns a single label from one LLM call on the raw
  input; a short question, or one mixing languages, can be misread. The cost is an
  answer in the wrong language, which the user can see and rephrase — acceptable, but
  it argues against building anything expensive on top of the label.
- **A language the model answers poorly.** The instruction is best-effort; there is
  no verification step and this specification does not add one.
- **Nomenclature that is also an ordinary word.** Gene symbols like `SET`, `MAX` or
  `CAT` are English words. A translator may render them as words rather than symbols.
- **A question already in English.** Must take exactly today's path, with no added
  instruction — otherwise every existing user pays for a feature they do not use.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: React-to-Me MUST answer in the language its question was asked in.
- **FR-002**: Plant Reactome MUST do the same.
- **FR-003**: The text used for retrieval MUST contain only the rephrased question —
  no language instruction, no prompt scaffolding.
- **FR-004**: Retrieval MUST continue to happen in English.
- **FR-005**: Scientific nomenclature — gene symbols, protein names, pathway names,
  `R-HSA-*` identifiers — MUST NOT be translated.
- **FR-006**: URLs and citation links MUST NOT be translated or altered.
- **FR-007**: An English question MUST follow exactly the current path, with no
  additional model call and no additional prompt content.
- **FR-008**: The language MUST reach the answer prompt as its own input, not
  concatenated into another field.

### Key Entities

- **Detected language**: one label per message, already on `BaseState`, produced
  concurrently with the safety check.
- **Rephrased question**: the English rendering, already produced, already used for
  retrieval.
- **Answer prompt**: per profile; the place the language belongs.

## Success Criteria *(mandatory)*

- **SC-001**: A question asked in French, German, Spanish or Japanese is answered in
  that language by React-to-Me.
- **SC-002**: `bin/retrieval_baseline` shows English retrieval unchanged beyond the
  documented noise floor.
- **SC-003**: For a non-English question, the retrieval query is byte-identical to
  what the same question produces today.
- **SC-004**: In a non-English answer, every gene symbol, pathway name and `R-HSA-*`
  identifier appears in English, and every Reactome URL resolves.
- **SC-005**: An English question costs the same number of model calls as today.

## Decisions for the team

### D1 — Instruct the answer prompt, or translate afterwards?

| option | what it means |
|---|---|
| **A. A language variable in the answer prompt** (recommended) | The answer is generated in the target language directly. No extra call, no extra latency. It is the mechanism Cross-Database already uses successfully, so the pattern is proven in this codebase. Quality depends on the model answering well in that language. |
| B. Answer in English, then translate | A separate translation step over the finished answer. Easier to protect nomenclature — a translator can be told to leave marked spans alone — but adds a model call and its latency to every non-English message, and translation of a scientific answer can introduce its own errors. |
| C. Both, chosen per language | Honest about models being better at some languages than others, and a table nobody will maintain. |

**Recommendation: A.** It is what already works for Cross-Database, it costs nothing
extra, and B's advantage on nomenclature is reachable from A with prompt instructions —
which is what both contributors did, independently.

## Assumptions

- The existing detector is good enough. It is already trusted for refusals, which are
  the more sensitive case; nothing here makes it more load-bearing.
- Users would rather have an answer in their own language than a marginally better
  English one. If that is wrong, the feature is wrong, not the implementation.
- The corpus stays English. Translating Reactome content is a different project.

## Out of Scope

- Translating the corpus, or multilingual embeddings.
- Verifying that the answer really is in the requested language.
- The web-search and hallucination-grading work bundled into #125 — that is #123's
  decision to make.
- The user interface: language is detected per message, not chosen in a setting.
