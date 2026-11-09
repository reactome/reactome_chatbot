# Feature Specification: OpenAI reasoning effort

**Feature Branch**: `llm-migration`

**Created**: 2026-11-03

**Status**: Built on `llm-migration`, 2026-11-03 to 2026-11-05, and not yet merged. The
spec was revised on 2026-11-05 from one reasoning effort for every call to two: `low`,
and `medium` for the answer. The startup refusals were checked against the API before
that revision. Nothing else was: the sweep, the completeness grader, the streaming path,
the command-line tools and the live-data loop were all left unchecked. The cost
measurement (Story 5) was not kept. The requirements below were set for that work, not
for later changes. Current behaviour is in `resolve_model_parameters`, `check_model` and
`build_llm` (`src/agent/graph.py:133-330`) and in `src/util/config_yml/models.py:41-55`.

**Input**: Request, 2026-11-03: "OpenAI models no longer accept temperature; send
reasoning effort instead, default medium; retire the temperature table and its probe."
Revised 2026-11-05: "I want the default llm reasoning effort to be low and only use
medium for the final answer generation step."

## Context (2026-11-03)

Spec 003 made temperature a measured fact about each model. A table, filled in by a
probe script, recorded which models accept `0.0` and which accept only `1.0`, and
every call sent a temperature from it.

That stopped being true. Current OpenAI models accept only their own default
temperature, so there is nothing to choose, and sending none gets the same default.
They take a reasoning effort instead. gpt-4o, gpt-4o-mini, gpt-4.1 and gpt-4.1-mini
are still being served while they are retired, and they reject any reasoning effort
(measured, [research](./research.md) R1). The code sent `0.0` to every model the table
did not list, gpt-6-luna included, so gpt-6-luna failed on a user's first question
rather than at startup.

Which levels a model accepts varies by model. Only `medium` is accepted by all of them.
Rather than keep a second hand-made table, the API is asked once at startup (R4).

The built-in default model changes to gpt-6-luna. That was decided before this spec;
the spec covers making it work.

A question makes several model calls, all on the answering model: rephrase, safety,
language, intent, query expansion, the answer, and, with post-processing on, the
completeness grader. Since 2026-11-05 the answer runs at `medium` and every other call
at `low`. The other calls return short or structured results, and they all run before
the first answer token, so reasoning spent on them is time the user waits through.

## User stories (set 2026-11-03, revised 2026-11-05)

### User Story 1 — Answer with an OpenAI reasoning model (P1; built)

With no `llm:` section, a question is answered by gpt-6-luna. No call sends a
temperature. The answer is sent `medium`, and every other call `low`.

**Check used**: offline, on the request each model would send (no `temperature` key,
the right `reasoning_effort`). The answer sweep, the completeness grader and the
streaming path were not run on the API.

### User Story 2 — A setting that cannot be honoured stops startup (P1; built)

Each of these makes the server exit before it accepts traffic, with a message naming
the setting, where it came from, and the fix: a temperature with OpenAI, from
`config.yml` or `LLM_TEMPERATURE`; a reasoning effort with Ollama; a level that does
not exist; an older model that takes no reasoning effort; a level the chosen model
rejects.

**Check used**: the six refused configurations were each started against the API
before the 2026-11-05 revision (T028). After the revision, the refusals for
`answer_reasoning_effort` were checked offline only.

### User Story 3 — Every model-building path behaves the same (P1; built)

These build a model: the chat and search graph, analysis summaries, the evaluator's
model under test, the evaluator's judge, the routing probe, and the retrieval
baseline's self-query. All of them go through `build_llm`, so the same settings send
the same parameters and are refused the same way. The routing probe and the answer
sweep used to ignore `config.yml`. The judge and the retrieval baseline deliberately
read no `llm:` section, so their results don't move when a deployment's config does.

**Check used**: offline, at each call site. None of the command-line tools was run on
the API.

### User Story 4 — The temperature table and its probe are gone (P2; built)

There is one rule per provider and no per-model table. Adding an OpenAI model means
naming it in `config.yml`, and nothing else.

### User Story 5 — The cost of the change is measured (P2; not kept)

The aim was a committed record of first-token and complete latency, and cost per
question, for gpt-6-luna on the sweep's questions on the served path. The tool was
written (R7), then left out of the repository on 2026-11-09 without being run. No
record exists.

## Requirements of this work, and where they stand (checked 2026-11-09)

| ID | Required by the work | Today |
|---|---|---|
| FR-001 | OpenAI calls send no temperature | Holds (`resolve_model_parameters`, `src/agent/graph.py:139`) |
| FR-002 | OpenAI calls send a reasoning effort: answer calls send `llm.answer_reasoning_effort`, otherwise `medium`; every other call sends `llm.reasoning_effort`, otherwise `low` (revised 2026-11-05; was one level, default `medium`) | Holds (`graph.py:125-129`, `build_llm(..., answer=True)`) |
| FR-002a | Answer calls are the calls whose text reaches the user as the answer: the RAG answer, every round of the live-data tool loop, the reply to a refused question, the analysis summary, and the parked Cross-Database final summary. No other call is an answer call | Holds. `AgentGraph` builds `llm` and `answer_llm` (`graph.py:434-487`). Uses: `react_to_me.py:68-85`, `plantreactome.py`, `cross_database.py:73`, the `answer_llm=` keyword on the Reactome and Plant Reactome RAG factories, `_summary_llm` (`src/api/analysis_summary.py:350`, from `AgentGraph.answer_model`) |
| FR-003 | `llm.reasoning_effort` and `llm.answer_reasoning_effort`, each optional, each one of `none`, `minimal`, `low`, `medium`, `high`, `xhigh`. Whether the model accepts the level is the API's answer (FR-013), not a list kept here | Holds (`src/util/config_yml/models.py:45-55`) |
| FR-004 | Ollama keeps its temperature behaviour and is sent no reasoning effort | Holds |
| FR-005 | Startup refuses: (a) `llm.temperature` with OpenAI, at any value, `1` included, because accepting the one value that does nothing would let an operator think they had chosen it; (b) `LLM_TEMPERATURE` with OpenAI; (c) either effort key with Ollama; (d) a level not listed in FR-003 | Holds. (a) to (c) in `resolve_model_parameters`, the same for both kinds of call; (d) when the config is parsed |
| FR-006 | One shared decision for every model-building path (Story 3). The judge and the retrieval baseline read no `llm:` section | Holds (`resolve_model_parameters`, passed to `build_llm`, `graph.py:300`). Judge: `evaluator.py:290`; baseline: `bin/retrieval_baseline:170` |
| FR-007 | The routing probe uses the deployment's `llm:` config | Holds (`src/evaluation/routing_probe.py:146`) |
| FR-008 | The safety checker, intent classifier and completeness grader return valid structured output while a reasoning effort is sent | Verified at `medium` for one question each (R1). They now run at `low`, which was not verified |
| FR-009 | The built-in default model is gpt-6-luna. A model named in `config.yml` or `LLM_MODEL` is kept | Holds (`graph.py:91`) |
| FR-010 | Startup logs the provider, the model and both sets of parameters | Holds (`graph.py:424`) |
| FR-011 | The temperature table, its logic and its probe are removed | Holds. The table's tests went with it |
| FR-012 | Comments, templates, the config schema and docs describe the new behaviour | Holds: `config_default.yml`, `.config.schema.yaml`, `env_template`, `deploy/beta/env.beta.template`, `src/evaluation/README.md` |
| FR-013 | Before other work, send one request for each distinct set of parameters. Stop if the API rejects it, quoting the provider's error with a fix. If the API gives no answer (unreachable, 429, 5xx), log a warning and start (revised 2026-11-09; was "stop on any failure"). Ollama is not checked | Holds (`check_model`, `graph.py:223`). Two requests at default settings, one if both levels are equal |
| FR-014 | The evaluator's default judge is gpt-6-astra, not the answering model. Its default model under test is the built-in default | Holds (`evaluator.py:71, 461`) |

## Out of scope (set 2026-11-03)

- A level for each task, such as a separate one for the classifier. There are exactly
  two levels. Different models for different surfaces (spec 003 Story 3) are also out.
- Prompt changes, the embedding model, and an environment variable for reasoning
  effort.
- Keeping gpt-4o, gpt-4o-mini, gpt-4.1 and gpt-4.1-mini working while they are
  retired. A deployment that pins one of them refuses to start until it changes model.

## Open items (recorded 2026-11-09)

- **Live-data answers are unverified.** The live loop uses function tools, and gpt-5.6
  refused function tools whenever a reasoning effort was sent. Nothing has tried the
  combination on gpt-6-luna. If it fails, live-data questions fail for users.
- **The graders at `low`** (FR-008), and the sweep, the streaming path and the
  command-line tools on the API, were not run.
- **chat-latest** accepts only `medium` (R1), so with the default `low` it refuses to
  start until `llm.reasoning_effort: medium` is set.
- **The judge now runs at `low`**, because it is not an answer call. That changes its
  scores. Judging at `medium` would be a deliberate choice.
- **Scores are not comparable across the judge change.** Scores judged by gpt-4o
  cannot be compared with scores judged by gpt-6-astra.
- **The chat's cost counter shows $0 for these models.** `OpenAICallbackHandler.total_cost`
  (`src/util/chainlit_helpers.py:95`) records $0 for any model missing from
  langchain-community's price table, gpt-5.6-luna and gpt-6-luna included. That is
  outside this spec.

## What was built, 2026-11-03 to 2026-11-05

- **One rule per provider, decided once.** `resolve_model_parameters` returns what to
  send (`ModelParameters` in `src/agent/models.py`). `get_llm` takes it as a required
  argument, so a caller that skips the decision does not type-check. The routing
  probe was one such caller, silently sending `0.0`.
- **A startup request.** `check_model` sends one request with exactly the parameters
  later calls will send, before the roughly 85 seconds of BM25 indexing.
  A refusal is a `ModelConfigError`, which startup turns into an exit. If the API
  gives no answer (an outage), startup logs a warning and goes on. `_summary_llm`
  runs during a request, so it resolves nothing: it builds from
  `AgentGraph.answer_model`, which startup resolved and checked, and reads no
  config.yml that could fail the request.
- **Two copies of one model.** Every profile is given `(llm, answer_llm, embedding)`.
- **Deleted**: the temperature table, `resolve_temperature`, `bin/probe_model_temperature`
  and their tests.
- **Also fixed** (principle VI): the routing probe and the answer sweep now read
  `config.yml`, and the retrieval baseline builds its self-query model once, only when
  self-query is on.
- **Added for measurement**: `astream_answer` takes `callbacks`, as `ainvoke` already
  does. The measurement tool that used it was not kept.
