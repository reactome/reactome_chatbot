# Feature Specification: Model Configuration

**Feature Branch**: `spec/model-configuration`

**Created**: 2026-09-09

**Status**: Historical record of the model-configuration work, 2026-09-09 to
2026-09-10. User Stories 1 and 2 were implemented 2026-09-10 (#200 plan, #201
implementation). User Story 3, a model per surface, was not built. D1 was implemented
as the spec recommended (table only, no API call). The requirements below were set for
that work, not for later changes. Current behaviour is in `resolve_temperature` and
`resolve_llm_model` (`src/agent/graph.py:141-197`) and `src/util/config_yml/`. The
outcome is recorded at the end of this file.

**Input**: Make it possible to switch between models — `gpt-4o-mini` and `gpt-5.6-luna` in particular — when per-model settings such as temperature differ. Does this need something to configure it with?

## The question, and the answer this specification gives

> "It would be nice to be able to switch between different models... being that
> things like the temperature need to be set up differently we probably need
> something to configure it with."

Half of that is right, and the half that is wrong is the expensive half.

**Which model to use is a choice.** It differs per deployment and, soon, per
surface. It belongs in configuration.

**How a model must be called is a fact.** `gpt-5.6-luna` accepts only
`temperature=1` and refuses function tools on `/v1/chat/completions`.
`gpt-4o-mini` wants `0.0`. These are not preferences an operator holds; they are
properties of the model, discoverable only by asking the API and getting a 400.

Making a fact configurable does not add flexibility. It adds a way to be wrong,
and the failure lands on a user's first question after deploy rather than at the
moment the mistake is made. The evidence is close to hand: the temperature table
was written by someone working on nothing else, and it was **wrong for eleven
models** (#189) — `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `o3`, `o4-mini` and
`chat-latest` all refuse `0.0`, while `gpt-5.1`, `gpt-5.2` and `gpt-5.4` accept
it. An operator editing YAML has no better information and less context.

So: **configure the choice, derive the consequences, and validate the pair before
the server accepts traffic.**

## What worked before this change (2026-09-09)

`LLM_MODEL` switched the answering model, and `resolve_temperature` and the
`json_schema` grader method (#186, #189) made `gpt-5.6-luna` work through it. The gap
was not "can we switch" but:

1. the model was **one global value**, so every surface had to use the same one —
   still true (see Outcome)
2. it lived in an environment variable rather than beside the other settings that
   shape a deployment — closed by the `llm:` section of `config.yml`
3. nothing checked the model was usable **until a user asked a question** — closed
   for a configured temperature the table knows the model refuses; still open, by
   D1, for an unknown or mistyped model name

## Two contributed pull requests (reviewed 2026-09-09)

Both came from GSoC applicants who were not expected to update them, and both were
closed when this work landed (11c3116). Two independent people hitting the same gap
was strong evidence it was real.

| | #112 — @AaryanCode69 | #151 — @bhavyakeerthi3 |
|---|---|---|
| shape | nested `ModelsConfig` → `LLMConfig` / `EmbeddingConfig`, with `provider`, `model`, `base_url`, `device` | flat strings, `llm: "openai/gpt-4o-mini"` |
| size | +50 / −5 across 5 files | +126 / −3 across 10 files |
| takes from | mirrors `get_llm`'s signature | uses `get_llm`'s existing `"provider/model"` split |

**#112's structure is the better starting point** — it names the fields rather than
re-parsing a string, and `base_url` matters for the Plant Reactome deployment,
which serves its embedding model from a self-hosted OpenAI-compatible endpoint.

### Both of them contain the same serious defect

Both make the **embedding model** configurable, and both pass it to
`get_embedding` directly, bypassing `resolve_embedding_model()`.

That function exists because a query embedded with a different model than built the
stored vectors returns nonsense rather than an error. It reads the model from the
bundle path — `openai/text-embedding-3-large/reactome/Release95` — which is the
only durable source of truth. It was written because a hardcoded default of
`bge-m3` broke every fresh Reactome deployment (a 404 on the first query) while
being correct for Plant Reactome; hardcoding either breaks the other.

`config_default.yml` in #112 sets `embedding.model: text-embedding-3-large`. Applied
to the Plant Reactome deployment, that is the `bge-m3` bug again, from the other
direction — and silent, because the wrong embedding model does not error, it just
retrieves the wrong documents.

**The embedding model must not be configurable at all.** `EMBEDDING_MODEL` already
exists as an override for the one case that needs it, and it is documented as
"leave unset". This specification harvests the LLM half of both PRs and rejects the
embedding half, with credit to both contributors.

## What the work set out to do (2026-09-09)

- **User Story 1 (P1)** — a deployment names its answering model in `config.yml`,
  beside profiles, quotas and features. With no model named, nothing changes, and the
  precedence against `LLM_MODEL` is defined rather than accidental.
- **User Story 2 (P1)** — an unusable model stops the server, not the conversation.
  A configured temperature that contradicts a known model requirement refuses to
  start, naming the model, the value and the fix (independent test: `gpt-5.6-luna`
  with `temperature: 0`). A model the table has not met still starts, and no API call
  is made at startup (D1).
- **User Story 3 (P2)** — chat, chat-alongside-search-results and analysis
  summarisation each name their own model, falling back to the deployment default.
  The reason given: `gpt-4o-mini` answered in 22.5s and `gpt-5.6-luna` in 41.2s
  (spec 002, 2026-09-09). Not built; see Outcome.

US2 also said that for an unknown model "the first failure names `LLM_TEMPERATURE`".
That was not built. The first failure is the provider's own error on the first
request; only the startup refusal for a known model names `LLM_TEMPERATURE`
(`src/agent/graph.py:160-166`).

### Edge cases considered at the time, and how they stand

- **`config.yml` names a model the API does not have.** Still a 404 on the first
  question, not at startup: D1 chose not to spend an API call to check.
- **`LLM_MODEL` and `config.yml` disagree.** The environment wins. The spec asked for
  the losing value to be reported; it is not. `resolve_llm_model` returns `LLM_MODEL`
  without logging the `config.yml` value it overrides (`src/agent/graph.py:197`), and
  only the effective model is logged (`:315-317`).
- **A legal temperature the repository does not want** (`0.7` on `gpt-4o-mini`
  makes the graders non-deterministic). Allowed; the effective temperature is in the
  same startup log line.
- **Provider `ollama` with a local model.** The table matches exact model names only
  (`src/agent/graph.py:114-138`), so a local model gets 0.0 unless its name is in the
  set or a temperature is configured.

## Requirements of this work, and where they stand (checked 2026-10-21)

Code and tests cite these IDs.

| ID | Required by this work | Today |
|---|---|---|
| FR-001 | Answering model settable in `config.yml` | Holds: optional `llm:` section, `LLMConfig` (`src/util/config_yml/models.py:18-45`, `src/util/config_yml/__init__.py:37`) |
| FR-002 | No model configured behaves as before | Holds: `llm` defaults to `None` and the built-in default is `gpt-4o-mini` (`src/agent/graph.py:172`; `tests/agent/test_model_configuration.py:27-38`) |
| FR-003 | Precedence between `config.yml` and `LLM_MODEL` defined, documented, tested | Holds: environment, then `config.yml`, then the default; `LLM_BASE_URL` works the same way (`graph.py:175-197`; `test_model_configuration.py:46-58`) |
| FR-004 | Embedding model not settable in `config.yml` | Holds: there is no such field, and `LLMConfig` and `Config` reject unknown keys (`models.py:27`, `__init__.py:30`), which `Config.from_yaml` turns into a refusal to start. The model comes from `resolve_embedding_model()`; `EMBEDDING_MODEL` remains an environment override |
| FR-005 | Per-model call requirements derived in code | Holds: temperature from `FIXED_TEMPERATURE_MODELS` (`graph.py:114-128`); the graders fix `json_schema` |
| FR-006 | A configured temperature that contradicts a known requirement stops startup, naming model, value and fix | Holds for `llm.temperature` (`graph.py:160-166`; `tests/agent/test_model_temperature.py:162-174`). `LLM_TEMPERATURE` is deliberately not checked: it is the escape hatch and wins over the table (`graph.py:153-158`; `test_model_temperature.py:119-124`) |
| FR-007 | A model unknown to the table does not block startup | Holds (`graph.py:150-151`; `test_model_temperature.py:186-190`) |
| FR-008 | Effective model logged at startup | Holds, with the temperature (`graph.py:315-317`) |
| FR-009 | Each surface can name its own model | Not built. Every surface uses the one deployment model (see Outcome) |
| FR-010 | Configuration that cannot be honoured stops the process | Holds for an invalid `config.yml`, including unknown keys (`__init__.py:113-128`), a refused configured temperature, and an unparseable `LLM_TEMPERATURE` (`graph.py:155-158`) |
| SC-001 | Switching model needs one file edit and a restart | Holds (`config.yml`, or `LLM_MODEL`) |
| SC-002 | Existing `config.yml` files keep working | Checked when shipped: neither the deployment `config.yml` nor `config_default.yml` carried an unknown key (`__init__.py:24-29`); `tests/util/test_config.py:154-156` covers the shipped default |
| SC-003 | A contradictory model/temperature pair is refused at startup | Holds for `config.yml`; not for `LLM_TEMPERATURE` (see FR-006) |
| SC-004 | No configuration file can name an embedding model | Holds (`tests/agent/test_model_configuration.py:93-109`) |
| SC-005 | Two surfaces run different models in one process | Not built (FR-009) |
| SC-006 | #112 and #151 credited and closed | Done 2026-09-10 (11c3116) |

### Key terms

- **Deployment default model**: what answers when a surface names nothing.
- **Surface**: chat (Chainlit), search-page answers (spec 010) and analysis
  summarisation (spec 011). Only chat existed when this was written. All three now
  use the same model: the answer endpoint shares the chat's graph
  (`src/agent/registry.py:1-16`), and analysis summaries resolve the model the same
  way (`src/api/analysis_summary.py:333-353`).
- **Model requirements**: temperature and structured-output method, derived in code;
  see `FIXED_TEMPERATURE_MODELS` and `bin/probe_model_temperature`. Temperature can
  be overridden (`llm.temperature`, `LLM_TEMPERATURE`) for a model the table has not met.

## Decision D1 (put to the team 2026-09-09)

Implemented as option A in #201. This repository does not record whether the team
chose it explicitly; the commit and the Outcome say it was "taken as recommended".

### D1 — Does startup validate against the API, or only against the table?

| option | what it means |
|---|---|
| **A. Table only** (recommended) | Startup checks the configured temperature against `FIXED_TEMPERATURE_MODELS`. No network call, instant, offline-friendly. Catches the mistake actually being made; misses a mistyped model name, which still 404s on the first question. |
| B. One probe request at startup | Catches everything — bad name, no access, wrong temperature — by asking the API. Costs one request and a second of boot, and makes startup fail when OpenAI is unreachable, which is a new way to be down. |
| C. Both, B behind a flag | Honest, and one more thing to configure. |

**Recommendation: A.** The failure mode this specification exists to prevent is a
plausible-but-wrong pairing, which the table catches for free. A mistyped model
name is a different and more obvious failure — it fails immediately, for everyone,
on the first request, and no one is confused about the cause.

## Assumptions at the time (2026-09-09)

- `config.yml` is the right home. It already carries profiles, quotas, features and
  messages; it has a schema, and a loader that refuses to start on an invalid file.
  A second configuration mechanism would be the thing worth avoiding.
- Operators are trusted but not omniscient. They may legitimately want a model this
  repository has never run; they should not have to know its temperature rules.
- Provider stays OpenAI-compatible. `base_url` covers self-hosting, as Plant
  Reactome already does.

## Out of scope for this work

- **Which model becomes the default.** Left to [spec 002](../002-default-llm-choice/spec.md),
  which made it wait on an answer-quality measurement. No decision is recorded there.
  This work made the choice expressible; it did not make it.
- Per-request model selection by an end user.
- The embedding model, permanently. FR-004.


---

## Outcome (2026-09-10)

Implemented in #201, planned in #200. The answering model is an optional `llm:`
section in `config.yml`; how it must be *called* stays derived in code.

D1 was taken as recommended: startup validates a configured temperature against
the measured table, with no API call. Instant, works offline, and startup cannot
come to depend on OpenAI being reachable.

### What writing it found

**Pydantic ignores unknown keys by default.** `embedding_model:` inside an `llm:`
section would have been accepted, silently discarded, and left an operator
believing they had set it — the exact failure FR-004 exists to prevent, arriving
through the door FR-004 was meant to lock. `LLMConfig` now forbids extras.

**And the guard was one level too low.** An adversarial review found that a typo in
the *section name* — `llmm:` for `llm:` — still loaded cleanly and did nothing.
`Config` forbids extras too now. Both were checked against the real `config.yml`
and `config_default.yml` first: neither carries an unknown key, so nothing that
works today is refused.

The pattern in both: the guard was placed on the thing being built rather than on
the seam beside it.

### Not built: a model per surface (User Story 3)

When this shipped only chat existed. Search-page answers (spec 010) and analysis
summaries (spec 011) have been added since, and all three use the one deployment
model. There is no per-surface setting. Because `Config` rejects unknown keys
(`src/util/config_yml/__init__.py:30`), adding one needs a code change. Whether it is
still wanted is not recorded.

The plan for this work checked (2026-09-10) that a per-surface section could reuse
`LLMConfig` unchanged, for example:

```yaml
llm:                 # the deployment default
  model: gpt-4o-mini
surfaces:            # proposed, not implemented
  analysis_summary:
    model: gpt-5.6-luna
```

It also noted that `resolve_llm_model` takes the config object rather than reading
globals, so a caller can pass a different one.

Spec 002 records no decision on whether luna becomes the default; the default is
still `gpt-4o-mini` (`src/agent/graph.py:172`).

### Why the environment beats the file

`LLM_MODEL` overrides `config.yml`, the reverse of `util/secrets.py`, where a mounted
Docker secret beats the environment. Both follow "the more specific source wins": a
secret is mounted by a deployment and should beat a committed file, while `LLM_MODEL`
is how an operator overrides a committed `config.yml` for one container
(`src/agent/graph.py:180-186`).

### Alternatives rejected during this work

| Choice | Simpler alternative rejected | Why |
|---|---|---|
| A nested `llm:` section | A flat `llm: "openai/gpt-4o-mini"` string, as #151 did | `base_url` has nowhere to live in a flat string, and Plant Reactome needs it. |
| Validate at startup | Validate on first use | First use is a user's first question. |
| Table only, no probe | One API call at startup | D1. |
| Leave the embedding model alone | Make it configurable, as both PRs did | It is derived from the bundle that built the vectors; configuring it gives silent nonsense instead of an error. |
