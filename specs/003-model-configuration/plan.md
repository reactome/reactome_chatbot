# Implementation Plan: Model Configuration

**Branch**: `feat/model-configuration` | **Date**: 2026-09-10 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/003-model-configuration/spec.md`

## Summary

Move the answering model into `config.yml`, where a deployment already declares its
profiles, quotas, features and messages. Keep the model's *call requirements* —
temperature, structured-output method — derived in code, and refuse at startup a
configuration that contradicts them.

Harvests the LLM half of #112 (@AaryanCode69) and #151 (@bhavyakeerthi3), and
rejects the embedding half of both.

## Technical Context

**Language/Version**: Python 3.12

**Primary Dependencies**: pydantic v2 (`Config` is already a `BaseModel`),
`langchain-openai` 1.6 via `agent.models.get_llm`.

**Storage**: `config.yml` at the repo root, bind-mounted read-only into the
container; `config_default.yml` shipped in the image as the fallback.

**Testing**: pytest. `tests/util/test_config.py` already pins the loader's failure
behaviour, including that an invalid file is fatal rather than silently defaulted.
`tests/agent/test_model_temperature.py` pins the derived table.

**Target Platform**: Linux container, and developer machines.

**Project Type**: Library within a single application repository.

**Performance Goals**: None. This is a startup-time concern; it runs once.

**Constraints**: A deployment with no model configured must behave exactly as it
does today (FR-002). `resolve_embedding_model()` must remain the only source of the
embedding model (FR-004).

**Scale/Scope**: One new config section, one validation, four call sites at most.

## Constitution Check

*GATE: must pass before implementation, re-checked after.*

| Article | How this plan satisfies it |
|---|---|
| I — verify the user path | The exit criterion is the server starting with a model set in `config.yml` and answering a question with it — not a unit test asserting the field parses. |
| II — measure, don't argue | Nothing here changes retrieval, so no baseline is owed. The model *choice* is spec 002's measurement, and this plan deliberately does not make it. |
| III — characterization tests pin behaviour | `test_config.py`'s existing tests must pass untouched: adding a section must not change what an invalid config does. |
| IV — fail loudly | The whole point of Stage 2. A contradictory model/temperature pair stops startup; it does not get silently corrected. |
| V — derive from the source of truth | FR-004 and FR-005 are this article restated: the embedding model comes from the bundle, the temperature from the measured table. Neither becomes configurable. |
| VI — bias to doing over filing | Two contributed PRs are harvested here rather than left open another six months. |

**No violations.** The one judgement call is D1 (validate against the table, not the
API), taken as recommended in the spec: no network call at startup.

## Project Structure

### Documentation (this feature)

```text
specs/003-model-configuration/
├── spec.md          # what and why, with D1 recommended
├── plan.md          # this file
└── checklists/
    └── requirements.md
```

No `research.md`: there are no unresolved unknowns. The two contributed PRs are the
research, the temperature behaviour was measured empirically in #189, and D1 is
decided. No `data-model.md` or `contracts/` — the "model" is three optional fields
on an existing pydantic class, described below in full, and the only external
contract is `config.yml`, whose schema file is edited in Stage 1.

### Source Code

```text
src/util/config_yml/
├── __init__.py       # Config gains an optional `llm` section
└── models.py         # new: LLMConfig, after #112's shape

src/agent/
├── graph.py          # AgentGraph reads the config; resolve_temperature validates
└── models.py         # unchanged -- get_llm already takes model and temperature

.config.schema.yaml   # the editor-facing schema gains the same section
config_default.yml    # documents the section without setting it
```

## Implementation Stages

### Stage 1 — The model becomes configurable, with today's behaviour as the default

Add `LLMConfig` (`provider`, `model`, `base_url`) and an optional `llm` field on
`Config`. `AgentGraph` prefers it, falling back to `LLM_MODEL`, then to the current
default.

Precedence is **environment over file** (FR-003), the opposite of Docker secrets in
`util/secrets.py`, and the difference is worth stating: a secret is mounted *by* the
deployment and should beat a checked-in file, whereas `LLM_MODEL` is how an operator
overrides a committed `config.yml` for one container. Both rules are "the more
specific thing wins"; they only look contradictory.

Nothing is required. A `config.yml` with no `llm:` section behaves exactly as today,
which is what keeps every existing deployment working (FR-002).

**Exit criteria**: `test_config.py` passes untouched; a `config.yml` naming
`gpt-5.6-luna` starts the server and answers a question with it; the effective model
appears in the startup log (FR-008).

### Stage 2 — A contradictory configuration stops startup

`resolve_temperature` already knows which models refuse `0.0`. If `config.yml` (or
`LLM_TEMPERATURE`) sets a temperature a model will reject, startup fails naming the
model, the value, and the fix.

This is the reason the feature is a specification rather than a bump. Today that
mistake is a 400 on a user's first question — visible to a user, attributed to the
chatbot, and diagnosable only from logs.

Per D1 the check is against the table alone. No API call, so it is instant, works
offline, and cannot make startup depend on OpenAI being reachable. A model the table
has not met does not block startup (FR-007).

**Exit criteria**: `gpt-5.6-luna` with `temperature: 0` refuses to start and names
all three of model, value and fix; an unknown model starts normally; the existing
`LLM_TEMPERATURE` escape hatch still wins.

### Stage 3 — Close #112 and #151

Both are superseded by Stages 1–2. Close them courteously, crediting each
contributor in the commit that lands their idea, and say plainly why the embedding
half was rejected — it bypasses `resolve_embedding_model()` and would silently break
Plant Reactome retrieval.

**Exit criteria**: both PRs closed with credit; `.config.schema.yaml` and
`config_default.yml` document the new section; no configuration file can name an
embedding model (SC-004).

## Complexity Tracking

| Decision | Simpler alternative rejected | Why |
|---|---|---|
| A nested `llm:` section | A flat `llm: "openai/gpt-4o-mini"` string, as #151 does | Re-parsing a string to recover provider and model is work the loader can do once. `base_url` has nowhere to live in a flat string, and Plant Reactome needs it. |
| Validate at startup | Validate on first use | First use is a user's first question. The whole point is that the operator learns before a user does. |
| Table only, no probe | One API call at startup | D1. It catches the mistake actually being made, for free and offline. A mistyped model name fails immediately and unmistakably anyway. |
| Leave the embedding model alone | Make it configurable "for symmetry", as both PRs do | It is derived from the bundle that built the vectors. Configuring it is how you get silent nonsense instead of an error. |

## Out of Scope

- **Which model becomes the default.** [Spec 002](../002-default-llm-choice/spec.md),
  gated on an answer-quality run. This plan makes the choice expressible, not made.
- Per-surface model selection (spec 003 User Story 3). Only chat exists today; the
  nesting introduced in Stage 1 is what makes it cheap later.

  Confirmed nestable without a schema break (T020): a second surface adds a key
  beside `llm:` holding the same `LLMConfig` shape --

  ```yaml
  llm:                 # the deployment default
    model: gpt-4o-mini
  surfaces:
    analysis_summary:  # slower is fine; nobody is watching a cursor
      model: gpt-5.6-luna
  ```

  `LLMConfig` needs no change for that, and `resolve_llm_model` takes the config
  object rather than reading globals, so a caller can pass a different one.

  Why surfaces will want to differ (T021): spec 002 measured 22.5s per question
  for gpt-4o-mini against 41.2s for gpt-5.6-luna. A search-results panel and a
  background summarisation have opposite tolerances for that, so forcing them to
  agree is a choice with a real cost.
- The embedding model, permanently.
