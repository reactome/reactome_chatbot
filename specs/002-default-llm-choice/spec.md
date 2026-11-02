# Feature Specification: Choosing the Default Answering Model

**Feature Branch**: `spec/default-llm-choice`

**Created**: 2026-09-09

**Status**: Historical record of the default-model evaluation, 2026-09-09: the measurements taken then, and three questions (D1-D3) put to the team. No answer to D1-D3 and no choice of default is recorded in this repository; the default is still `gpt-4o-mini` (`src/agent/graph.py:172`). The evaluator work this spec asked for was done the same day (cf9ad6a). The requirements below were set for that work, not for later changes. How the model is chosen today is in [spec 003](../003-model-configuration/spec.md) and `resolve_llm_model` (`src/agent/graph.py:175-197`).

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

## What was settled before this spec (2026-09-09)

`gpt-5.6-luna` worked end to end after #186, which removed the two things that stopped it:

| blocker | resolution in #186 | today |
|---|---|---|
| `temperature=0.0` hardcoded; the gpt-5.5/5.6/6 families accept only their default of 1 | `resolve_temperature()` matching those name prefixes, `LLM_TEMPERATURE` to override | The prefix rule was wrong for eleven models and was replaced (#189) by an exact-match set of measured models, with dated snapshot suffixes stripped (`src/agent/graph.py:103-138`) |
| the three graders used `function_calling`, which gpt-5.6 refuses on `/v1/chat/completions` | `method="json_schema"`, verified on both models | Unchanged (`src/agent/tasks/safety_checker.py:97-99`, `intent_classifier.py:206`, `completeness_grader.py:38-40`) |

Switching was one environment variable, `LLM_MODEL=gpt-5.6-luna`; since spec 003 it can
also be `llm.model` in `config.yml`. This spec was about whether to switch, not how.

## Evidence gathered (2026-09-09)

Measured on 2026-09-09 through `AgentGraph.ainvoke` on the React-to-Me profile — the
entry point `bin/chat-chainlit.py` used — against the Release95 reactome bundle. The
pipeline has changed since (a fifth collection, preprocessing in two rounds, collection
routing), so these figures describe that date, not current behaviour.

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
- **Answer quality at any scale.** At the time, the harness that would do it did
  not measure the pipeline that ships (next section).
- **Time to first token.** Not measured for either model here. Spec 010 later
  measured it for `gpt-4o-mini` only (p50 9.6s, 2026-09-18;
  `specs/010-search-page-answers/contracts/answer_endpoint.md`, "Budget").

## The evaluator gap, and how it was closed

`src/evaluation/evaluator.py` runs ragas with `faithfulness`, `answer_relevancy`,
`context_recall` and `context_utilization` over a question set — the metrics that
would settle the answer-quality question, and `faithfulness` is the axis on which
luna appeared to differ.

When this spec was written it could not be used: it built its own retriever
(`SelfQueryRetriever` + `EnsembleRetriever` + `MergerRetriever`) rather than the
shipping one, and the retriever rewrite (spec 001) had removed `SelfQueryRetriever`
from the product, so it measured a configuration that no longer existed. The same
day it was changed to call `create_reactome_rag` (cf9ad6a), with a repeatable
`--model`, `--repeat`, and a pinned judge; see `src/evaluation/README.md`.

The four retrieval changes from the week of 2026-09-04 (#169, #170, over-fetch, and
spec 001's D1) went in unevaluated for answer quality. This repository records no
evaluator run since, for them or for the model comparison.

## What this work set out to do (2026-09-09)

In priority order: make the evaluator measure the pipeline a user talks to (P1, User
Story 1), then choose the default model from a table of measured differences rather
than from three answers (P2, User Story 2). A third story, letting each surface choose
its own model, moved to [spec 003](../003-model-configuration/spec.md) the same day and
is unbuilt there.

User Story 2 set two acceptance conditions: a faithfulness difference smaller than
run-to-run noise is reported as "not measurable" rather than as a win, and the chosen
default is recorded here with its numbers. Neither has been met, because no comparison
run is recorded.

### Edge cases noted at the time

- **A fixed-temperature model the table has not met.** It returns a 400 on the first
  user question, not at startup; `LLM_TEMPERATURE` is the escape hatch. The table is
  now an exact-match set, and dated snapshots (`<model>-YYYY-MM-DD`) are covered by
  stripping the suffix, not by prefix matching (`src/agent/graph.py:131-138`).
- **The evaluator's own judge model.** If the judge is the model being evaluated, the
  comparison is biased. The evaluator pins the judge (`--judge-model`, default
  `gpt-4o`) and refuses one that is also under test (`src/evaluation/evaluator.py:67`,
  `:433-437`). Keeping it the same between two compared runs is left to whoever runs
  it (`src/evaluation/README.md`, "The judge").
- **Latency on a streamed interface.** 41s to a complete answer is not 41s of silence
  if tokens stream; what a user feels is time to first token, which was not measured
  for luna (see "What could not be measured").

## Requirements of this work, and where they stand (checked 2026-10-21)

| ID | Required by this work | Today |
|---|---|---|
| FR-001 | Evaluator builds its chain from the application's factory | Holds with a gap: `build_chain` calls `create_reactome_rag` (`src/evaluation/evaluator.py:260-275`, cf9ad6a), but it does not apply collection routing (added 2026-09-19, spec 009), so it searches every collection. It also skips the safety check and intent classification (`src/evaluation/README.md:96-106`) |
| FR-002 | Model under test as an argument; per-model metrics, several models in one run | Holds: `--model` is repeatable, and the report puts models side by side (`evaluator.py:382-386`, `:578-610`) |
| FR-003 | Judge pinned and reported separately | Holds: `--judge-model`, default `gpt-4o`, named in the report; a judge that is also under test is refused (`evaluator.py:67`, `:433-437`, `:582-585`) |
| FR-004 | Report run-to-run variance | Holds as a spread: `--repeat N` reports max−min per metric (`evaluator.py:399-405`, `:592-605`) |
| FR-005 | Default selectable per deployment without a code change | Holds: `LLM_MODEL`, or `llm.model` in `config.yml` since spec 003 (`src/agent/graph.py:175-197`) |
| FR-006 | A model with an unknown temperature requirement fails, naming the variable | Not adopted. Spec 003's FR-007 (2026-09-10) reversed it: a model the table has not met is not validated and does not block startup (`graph.py:150-151`; `tests/agent/test_model_temperature.py:186-190`) |
| FR-007 | Record the chosen default and its numbers here | Not met: no decision or comparison is recorded |
| SC-001 | One command reports answer quality for the shipping pipeline, for a named model | `./bin/evaluate --model <name>`, with FR-001's gap |
| SC-002 | Two runs on unchanged code differ by less than the reported noise | No recorded run |
| SC-003 | Default-model decision recorded with a per-metric comparison | Not met |
| SC-004 | Switching the default is one environment variable, no rebuild | Holds (`LLM_MODEL`) |
| SC-005 | Answer-quality measurement for the four retrieval changes of 2026-09-04 | No recorded measurement |

### Key terms

- **Model under test**: the model that answers, and that the graders use.
- **Judge model**: the model ragas uses to score. Independent of the above, pinned.
- **Golden set**: `tests/golden/questions.txt`, 20 questions (the file's other lines are comments),
  used by `bin/retrieval_baseline` and, by default, by the evaluator.

## Assumptions at the time (2026-09-09)

- The golden set is representative enough to compare two models. It was assembled
  for retrieval measurement, not answer grading; if it proves too narrow that is a
  finding, not a reason to skip the measurement.
- `gpt-4o-mini` remains available. Nothing here is a migration forced by deprecation.
- Beta is the place to observe a model change before production. Both pin an image
  tag, so the model can differ between them by environment alone.
- Cost is a real constraint but not the binding one at current volume; latency and
  answer quality are what the team will notice first.

## Open questions put to the team (2026-09-09)

These were proposals for a decision, not decisions. No answer to any of them is
recorded in this repository; that does not show whether they were discussed
elsewhere. Each "Recommendation" is the one made at the time. The evaluator work,
which none of them blocked, was done on 2026-09-09.

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
what a user feels is time-to-first-token, which was not measured for luna. If the answer
is "no", that alone settles the default for chat and D1 becomes moot for that
surface — though luna may still suit analysis summarisation, where nobody is
watching a cursor.

### D3 — What is the actual price per token?

The premise for trying luna was "just as cheap". No endpoint reports pricing, so
this could not be checked here. If luna is materially more expensive per token, the
larger answers it produces multiply that, and cost becomes a first-order input
rather than the third one.

## Out of scope for this work

- The LangChain upgrade (done separately the same day, 4e607c8).
- Fine-tuning, or any change to the prompts. Changing prompt and model together
  would make the comparison unattributable — the same reason spec 001 was staged.
- Streaming behaviour and time-to-first-token, beyond noting above that it is the
  number a user feels. Worth its own measurement.
