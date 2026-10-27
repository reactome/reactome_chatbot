# Feature Specification: Answer in the User's Language

**Feature Branch**: `spec/answer-in-user-language`

**Created**: 2026-09-10

**Status**: Historical record of the answer-language work, 2026-09-10. Implemented
2026-09-10 (#204 spec, #205 implementation) for React-to-Me and Plant Reactome, with
D1 implemented as recommended: the answer prompt is instructed and there is no
translation pass. The requirements below were set for that work, not for later
changes. Current behaviour is in `src/agent/tasks/language_instruction.py`, the answer
prompts that include it (`src/retrievers/{reactome,plantreactome,userguide}/prompt.py`)
and the call sites in `src/agent/profiles/`. The outcome is at the end of this file.

**Input**: Issue #104, "RAG only responds in English". Two competing pull requests, #125 and #140.

## The gap before this work (2026-09-10)

Language detection already worked. `detect_language` ran on every message,
concurrently with the safety check, and `detected_language` was on `BaseState` for
every profile. (React-to-Me has since moved detection to run alongside the rephrase
instead, 9677ecb, 2026-09-18: `src/agent/profiles/react_to_me.py:155-164`. The shared
`preprocess` used by Plant Reactome and Cross-Database still pairs it with the safety
check: `src/agent/profiles/base.py:69-72`.)

It reached some places and not others:

| where | uses the detected language? |
|---|---|
| refusal messages (React-to-Me, Plant Reactome) | **yes** |
| Cross-Database final summary | **yes** |
| **React-to-Me's answer** | **no** |
| **Plant Reactome's answer** | **no** |

So a question in French was detected as French, rephrased into English for retrieval,
answered in English — and if it had instead been *refused*, the refusal would have
come back in French. The two profiles that serve real users were the two that answered
in the wrong language; the one that got it right was the prototype. Since this work
they answer in the detected language (see the requirements table).

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

Measured on the `summations` collection (2026-09-10, Release95) rather than assumed, because the obvious
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

Measured through the **whole retriever** (2026-09-10, Release95, four collections), appending it to an English question:

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

## What the work set out to do (2026-09-10)

- **User Story 1 (P1)** — a question to React-to-Me in French is answered in French;
  retrieval still uses the English rephrasing; gene names, `R-HSA-*` identifiers and
  Reactome links are left untouched.
- **User Story 2 (P1)** — retrieval does not move: the text sent to BM25 and the
  vector store holds only the rephrased question. The planned test was a
  `bin/retrieval_baseline` diff. It was replaced by a test that the retrieval query is
  byte-identical across languages (`tests/agent/test_answer_language.py:69-79`), and
  no baseline was run.
- **User Story 3 (P2)** — Plant Reactome does the same.

### Edge cases noted at the time

- **The detector is wrong.** One label from one LLM call on the raw input; a short or
  mixed-language question can be misread, and the answer comes back in the wrong
  language, which the user can see and rephrase. Unknown or ambiguous input is
  labelled English (`src/agent/tasks/detect_language.py:12`).
- **A language the model answers poorly.** The instruction is best-effort; nothing
  verifies the answer's language.
- **Nomenclature that is also an ordinary word** (`SET`, `MAX`, `CAT`). The
  instruction tells the model not to translate symbols even when they are ordinary
  words (`src/agent/tasks/language_instruction.py:24-26`).
- **A question already in English.** A first draft required English questions to take
  exactly the old path, with no added instruction. That was dropped before
  implementation: every question, English included, gets the instruction. An English
  question costs no extra model call and reaches the retriever unchanged (FR-007).

## Requirements of this work, and where they stand (checked 2026-10-21)

Tests cite these IDs.

| ID | Required by this work | Today |
|---|---|---|
| FR-001 | React-to-Me answers in the question's language | Holds for all three React-to-Me sources. The reactome and userguide answer prompts include `LANGUAGE_INSTRUCTION` (`src/retrievers/reactome/prompt.py:49`; `src/retrievers/userguide/prompt.py:50`, since f0e8581, 2026-10-04). Live answers carry their own "Answer in {language}" (`src/reactome_mcp/answer.py:81`, passed from `src/agent/profiles/react_to_me.py:252`) |
| FR-002 | Plant Reactome does the same | Holds (`src/retrievers/plantreactome/prompt.py:46`; `src/agent/profiles/plantreactome.py:87`) |
| FR-003 | Retrieval text holds only the rephrased question | Holds: the language travels in a separate `detected_language` key (`react_to_me.py:307-314`, `plantreactome.py:81-95`); `tests/agent/test_answer_language.py:60-66` |
| FR-004 | Retrieval stays in English | Holds: the rephrase step translates to English (`src/agent/tasks/rephrase.py:8`, `:16`) |
| FR-005 | Nomenclature not translated | An instruction to the model (`language_instruction.py:24-26`), not enforced. Checked on one French answer, 2026-09-10 |
| FR-006 | URLs and citation links not altered | Same: instructed, not enforced |
| FR-007 | An English question costs no extra model call and reaches the retriever with a byte-identical query | Holds. Byte identity is tested (`test_answer_language.py:69-79`); no extra call because there is no translation step. English questions do get the instruction ("Answer in English", about 60 words, `language_instruction.py:19-26`), stated rather than hidden. Branching on "is this English" was rejected because the common path would then go untested by anyone who only asks in English |
| FR-008 | Language reaches the answer prompt as its own input | Holds (`test_answer_language.py:82-89`) |
| SC-001 | French, German, Spanish or Japanese answered in that language | Verified for French only (2026-09-10). German, Spanish and Japanese are not recorded as checked |
| SC-002 | `bin/retrieval_baseline` shows English retrieval unchanged | Not run; replaced by the byte-identity test (User Story 2) |
| SC-003 | Non-English retrieval query byte-identical to before | Holds by the same test |
| SC-004 | Nomenclature in English, Reactome URLs resolve | Checked on one French answer, 2026-09-10 |
| SC-005 | English costs the same model calls and stays English | Calls: holds (FR-007). The English answer was checked once, 2026-09-10 |

### Key terms

- **Detected language**: one label per message, on `BaseState`.
- **Rephrased question**: the English rendering, used for retrieval.
- **Answer prompt**: per source; where the language instruction goes.

## Decision D1 (put to the team 2026-09-10)

Implemented as option A in #205. This repository does not record whether the team
chose it explicitly beyond "taken as recommended".

### D1 — Instruct the answer prompt, or translate afterwards?

| option | what it means |
|---|---|
| **A. A language variable in the answer prompt** (recommended) | The answer is generated in the target language directly. No extra call, no extra latency. It is the mechanism Cross-Database already uses successfully, so the pattern is proven in this codebase. Quality depends on the model answering well in that language. |
| B. Answer in English, then translate | A separate translation step over the finished answer. Easier to protect nomenclature — a translator can be told to leave marked spans alone — but adds a model call and its latency to every non-English message, and translation of a scientific answer can introduce its own errors. |
| C. Both, chosen per language | Honest about models being better at some languages than others, and a table nobody will maintain. |

**Recommendation: A.** It is what already works for Cross-Database, it costs nothing
extra, and B's advantage on nomenclature is reachable from A with prompt instructions —
which is what both contributors did, independently.

## Assumptions at the time (2026-09-10)

- The existing detector is good enough. It is already trusted for refusals, which are
  the more sensitive case; nothing here makes it more load-bearing.
- Users would rather have an answer in their own language than a marginally better
  English one. If that is wrong, the feature is wrong, not the implementation.
- The corpus stays English. Translating Reactome content is a different project.

## Out of scope for this work

- Translating the corpus, or multilingual embeddings.
- Verifying that the answer really is in the requested language.
- The web-search and hallucination-grading work bundled into #125 — that is #123's
  decision to make.
- The user interface: language is detected per message, not chosen in a setting.


---

## Outcome (2026-09-10)

Implemented in #205. `detected_language` reaches the answer prompt in React-to-Me and
Plant Reactome as its own variable; `input` is untouched, so the retrieval query and
the query expansion in front of it are byte-identical across languages. A test
asserts that directly, which is exact where a retrieval baseline would only show
noise.

Verified against the Release95 bundle: a French question is answered in French with
`R-HSA` identifiers intact, an English question is unchanged, both retrieve 40
documents.

### What the reviews changed

The specification's own claims were wrong twice, both caught before implementation.

**"BM25 retrieves close to nothing for a French query"** — false. It returns a full
ten documents; named entities survive translation. The real cost is that seven of ten
differ.

**"#140's appended instruction leaves 0 of 10 documents"** — measured on BM25
directly, which the pipeline never does: the query expander rewrites the question
first, so four of five queries reach BM25 clean. Through the whole retriever it is
**20 of 40**. Half the context, not all of it. A review questioned the number as
looking too low, and was right.

The second is the more useful lesson. The first review checked whether each claim was
*true* without checking whether the *test measured the product* — the same failure
`evaluator.py` had four days earlier. An adversarial review has to attack the
measurement as well as the claim.

**And the plan (removed; in git history at 75ca531) contradicted the spec.** FR-007 promised English questions "no
additional prompt content" while the plan always passes the language, which adds a
sentence to every English prompt. FR-007 was the wrong half and was corrected to
promise what is actually true: no extra model call, byte-identical retrieval query,
and one added sentence stated rather than hidden.

### Worth watching

The French answer carried 2 `R-HSA` citations against the English answer's 9. One
sample, so not a finding — but if non-English answers systematically cite less, this
feature would be creating a quality gap while closing a language one. Not followed up
in this repository. The evaluator (`./bin/evaluate`, measuring the shipping chain since
2026-09-09; see [spec 002](../002-default-llm-choice/spec.md)) could look at it, but its
default question set is English only (`tests/golden/questions.txt`).

### Not resolved

#125 also carried a hallucination grader and web-search wiring for Cross-Database.
Neither is touched here, and closing that PR is not a decision about them; they
belong to #123.

### Alternatives rejected during this work

| Decision | Simpler alternative rejected | Why |
|---|---|---|
| A prompt variable | Append to `input`, as #140 did | Measured 2026-09-10: about half the fused documents change. The instruction would reach the retriever and the query expander. |
| Instruct the generator | Translate the finished answer | D1. An extra model call and its latency on every non-English message, plus translation errors over a scientific answer, for a nomenclature protection the prompt already gives. |
| Always pass the language | Branch on "is it English" | Two paths, one rarely exercised. Passing "English" is the same code doing the same thing. |
| No baseline run | `bin/retrieval_baseline` before and after | The retrieval query is unchanged by construction; a test asserting that is exact, where a baseline diff would only show noise. |
