# Feature Specification: Choosing the Default Answering Model

**Feature Branch**: `spec/default-llm-choice`

**Created**: 2026-09-09

**Status**: Evidence gathered, decision open. Three questions for the team below.

**Input**: `gpt-4o-mini` is today's default. `gpt-5.6-luna` is now usable (#186). Does it become the default, on which surfaces, and what evidence settles it?

## Why this is a specification and not a dependency bump

The constitution reserves Spec Kit for "design work with real, unmade decisions" and
excludes "dependency bumps, where the ceremony costs more than the fix". Changing the
answering model looks like a bump and is not one:

- it changes **what every user reads**, on every surface, immediately
- it trades **determinism** for capability — an unavoidable consequence, not a setting
- it changes **latency** by roughly a factor of two on a chat interface
- it changes **cost**, in a direction nobody has yet measured
- and the same choice will be made again for chat-alongside-search and analysis
  summarisation, which have different tolerances

None of that is recoverable from the diff. That is what this file is for.

## What has already been settled

`gpt-5.6-luna` **works end to end**. #186 removed the two things that stopped it:

| blocker | resolution |
|---|---|
| `temperature=0.0` hardcoded; the gpt-5.5/5.6/6 families accept only their default of 1 | `resolve_temperature()` per model family, `LLM_TEMPERATURE` to override |
| the three graders used `function_calling`, which gpt-5.6 refuses on `/v1/chat/completions` | `method="json_schema"`, verified on **both** models so the path cannot rot |

Switching is now one environment variable: `LLM_MODEL=gpt-5.6-luna`. **This
specification is about whether to set it, not how.**

## Evidence gathered

Measured through `AgentGraph.ainvoke` on the React-to-Me profile — the entry point
`bin/chat-chainlit.py` uses — against the Release95 reactome bundle.

### Latency and shape (3 questions, averaged)

| | seconds/question | LLM calls/question | input tokens |
|---|---|---|---|
| gpt-4o-mini | 22.5 | 6 | ~2685 |
| gpt-5.6-luna | 41.2 | 6 | ~3238 |

Output tokens are omitted deliberately: the OpenAI callback undercounts streamed
completions, and inconsistently between the two models, so the numbers it gave
would have looked authoritative and been wrong.

### Determinism (10 runs each)

`temperature=1` is not a preference; it is the only value these models accept. The
concern is the graders, which gate the whole conversation.

| input | gpt-4o-mini @ 0.0 | gpt-5.6-luna @ 1.0 |
|---|---|---|
| science question → intent | `reactome` ×10 | `reactome` ×10 |
| how-to question → intent | `userguide` ×10 | `userguide` ×10 |
| benign text → safety | `true` ×10 | `true` ×10 |
| prompt injection → safety | `false` ×10 | `false` ×10 |

Stable on four inputs. That is evidence, not a guarantee — four inputs is a smoke
test, and the failure mode being ruled out is a rare flip, which is precisely what
a small sample cannot rule out.

### Answer character (qualitative, n=3, no rubric)

On *"Which complexes contain EGFR?"* luna named four specific complexes with
Reactome links; gpt-4o-mini gave a general description of EGFR signalling naming
none. Luna's answers cite the retrieved records ("In the supplied Reactome
records...", "Reactome describes..."); gpt-4o-mini's read as recalled background.

This is the observation that makes the decision interesting, and it is the one with
the weakest evidence behind it. **A read of three answers is an anecdote.**

### What could not be measured

- **Price per token.** No endpoint reports it. The premise "just as cheap" is
  unverified here and has to come from the pricing page or from a billing period.
- **Answer quality at any scale.** See below — the harness that would do it does
  not currently measure the pipeline that ships.

## The blocker behind the blocker

`src/evaluation/evaluator.py` runs ragas with `faithfulness`, `answer_relevancy`,
`context_recall` and `context_utilization` over a golden set — exactly the metrics
that would settle the answer-quality question, and `faithfulness` is exactly the
axis on which luna appeared to differ.

It cannot be used as it stands. It builds its own retriever — `SelfQueryRetriever`
+ `EnsembleRetriever` + `MergerRetriever` — rather than calling
`create_bm25_chroma_ensemble_retriever`. Stage 2 removed `SelfQueryRetriever` from
the shipping pipeline, so the evaluator now measures a configuration that no longer
exists. Running it today would produce numbers that look like an answer and are not.

This is the same drift `bin/retrieval_baseline` had and had fixed: a measurement
tool that has quietly stopped measuring the thing.

It also blocks work already owed: the four retrieval changes from the week of
2026-09-04 (#169, #170, over-fetch, D1) went in **unevaluated for answer quality**.
Pointing the evaluator at the real pipeline pays for both.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — The evaluator measures what ships (Priority: P1)

A developer changes the model, or the retriever, and runs one command that reports
faithfulness, answer relevancy, context recall and context utilization for the
pipeline a user actually talks to.

**Why this priority**: Nothing else here can be decided without it, and it is owed
regardless of which model wins. It is the only item on this page whose value does
not depend on the outcome of the decision.

**Independent Test**: Run the evaluator against `main` twice with no change in
between; the metrics agree within noise. Then swap the retriever's document budget
and watch context utilization move.

**Acceptance Scenarios**:

1. **Given** the evaluator, **When** it constructs the chain, **Then** it calls the
   same factory `bin/chat-chainlit.py` reaches, so a change to retrieval cannot
   alter the product without altering the measurement.
2. **Given** a golden question set, **When** the evaluator runs on two models,
   **Then** it emits a per-metric comparison rather than two unrelated reports.
3. **Given** the evaluator, **When** `SelfQueryRetriever` no longer exists in the
   pipeline, **Then** the evaluator does not construct one.

---

### User Story 2 — The default model is chosen on evidence (Priority: P2)

The team picks the default from a table of measured differences, not from a
recollection of three answers.

**Why this priority**: This is the actual question. It is P2 only because it is
gated on P1.

**Independent Test**: Both models are run over the golden set; the report shows
faithfulness, relevancy, recall, utilization, latency and token counts side by
side.

**Acceptance Scenarios**:

1. **Given** both models evaluated, **When** faithfulness differs by less than the
   run-to-run noise, **Then** the difference is reported as "not measurable", not
   as a win.
2. **Given** a chosen default, **When** it is committed, **Then** the reason and
   the numbers are recorded here, so the next person does not re-derive them.

---

### User Story 3 — Different surfaces may choose differently (Priority: P3)

*Superseded: this is now [spec 003](../003-model-configuration/spec.md), which
covers how a model is chosen. This specification keeps only the question of which
model wins.*

Chat, chat-alongside-search-results, and analysis summarisation each pick a model
suited to their tolerance for latency.

**Why this priority**: Only chat exists today. Recording it now costs a paragraph;
discovering later that the model is a global constant costs a refactor — which is
the same mistake the context budget made, fixed in Stage 3.

**Acceptance Scenarios**:

1. **Given** a surface that must answer inside a search-results page, **When** it
   builds its chain, **Then** it can request a faster model without changing what
   the chat surface uses.

### Edge Cases

- **A model is added to a fixed-temperature family that the prefix table has not
  met.** It returns a 400 on the first user question, not at startup. `LLM_TEMPERATURE`
  is the escape hatch; the table is matched on prefix so dated snapshots are covered.
- **The evaluator's own judge model.** ragas uses an LLM to score. If the judge is
  the same model being evaluated, the comparison is biased. The judge must be pinned
  and stated, and must not change between the two runs being compared.
- **Latency on a streamed interface.** 41s to a complete answer is not 41s of
  silence if tokens stream; the perceived cost depends on time-to-first-token, which
  has not been measured and is the number a user actually experiences.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The evaluator MUST build its chain from the same factory the
  application uses, so that retrieval changes cannot alter the product without
  altering the measurement.
- **FR-002**: The evaluator MUST accept the model under test as an argument and
  report metrics per model, so two models can be compared in one run.
- **FR-003**: The evaluator MUST pin and report the judge model separately from the
  model under test.
- **FR-004**: The evaluator MUST report run-to-run variance, so a difference smaller
  than noise is not read as a result.
- **FR-005**: The default model MUST remain selectable per deployment without a code
  change.
- **FR-006**: A model whose temperature requirement is unknown MUST fail with a
  message naming the environment variable that fixes it.
- **FR-007**: The chosen default and the numbers behind it MUST be recorded in this
  specification when the decision is made.

### Key Entities

- **Model under test**: the model that answers, and that the graders use.
- **Judge model**: the model ragas uses to score. Independent of the above, pinned.
- **Golden set**: `tests/golden/questions.txt`, 28 questions, already committed and
  already used by `bin/retrieval_baseline`.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: One command reports answer quality for the pipeline a user talks to,
  for a named model, against the golden set.
- **SC-002**: Two consecutive runs of that command on unchanged code differ by less
  than the threshold the command itself reports as noise.
- **SC-003**: The default-model decision is recorded with a per-metric comparison
  covering both candidates.
- **SC-004**: Switching the default requires changing one environment variable, and
  reverting it requires changing it back — no rebuild, no code edit.
- **SC-005**: The four retrieval changes from 2026-09-04 have an answer-quality
  measurement attached, closing the gap left open by spec 001.

## Assumptions

- The golden set is representative enough to compare two models. It was assembled
  for retrieval measurement, not answer grading; if it proves too narrow that is a
  finding, not a reason to skip the measurement.
- `gpt-4o-mini` remains available. Nothing here is a migration forced by deprecation.
- Beta is the place to observe a model change before production. Both pin an image
  tag, so the model can differ between them by environment alone.
- Cost is a real constraint but not the binding one at current volume; latency and
  answer quality are what the team will notice first.

## Decisions for the team

Not gaps in the specification — the P1 work proceeds however these are answered.
They are recorded so they are answered once, in the open.

### D1 — Does the default flip before or after the quality evaluation?

| option | what it means |
|---|---|
| **A. Evaluate first** (recommended) | Fix the evaluator (P1), run both models over the golden set, then decide. Slower; the decision is defensible afterwards. |
| B. Flip beta now, evaluate alongside | Beta users see luna immediately. Real questions are better than golden ones, but nothing is being recorded, so "it seems better" stays an impression. |
| C. Flip both now | Fastest. Trades a measurable 2x latency increase on production for an answer-quality improvement supported by three answers. |

**Recommendation: A**, and it costs less than it sounds — the evaluator fix is owed
anyway for the four unevaluated retrieval changes (SC-005). B is defensible if the
grounding difference matters more than the latency; C is not, on this evidence.

### D2 — Is ~2x latency acceptable on the chat surface?

22.5s → 41.2s per question, measured to the complete answer. Chainlit streams, so
what a user feels is time-to-first-token, which has not been measured. If the answer
is "no", that alone settles the default for chat and D1 becomes moot for that
surface — though luna may still suit analysis summarisation, where nobody is
watching a cursor.

### D3 — What is the actual price per token?

The premise for trying luna was "just as cheap". No endpoint reports pricing, so
this could not be checked here. If luna is materially more expensive per token, the
larger answers it produces multiply that, and cost becomes a first-order input
rather than the third one.

## Out of Scope

- The LangChain upgrade. Unblocked by spec 001, unrelated to this.
- Fine-tuning, or any change to the prompts. Changing prompt and model together
  would make the comparison unattributable — the same reason spec 001 was staged.
- Streaming behaviour and time-to-first-token, beyond noting above that it is the
  number a user feels. Worth its own measurement.
