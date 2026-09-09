# Feature Specification: Model Configuration

**Feature Branch**: `spec/model-configuration`

**Created**: 2026-09-09

**Status**: Draft. Two contributed PRs to harvest; one decision (D1) for the team.

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

## What already works

`LLM_MODEL` switches the answering model today, and `resolve_temperature` and the
`json_schema` grader method (#186, #189) make `gpt-5.6-luna` work through it. The
gap is not "can we switch" — it is:

1. the model is **one global value**, so every surface must use the same one
2. it lives in an environment variable rather than beside the other settings that
   shape a deployment
3. nothing checks the model is usable **until a user asks a question**

## Two contributed pull requests to harvest

Both from GSoC applicants who will not update them. Two independent people hitting
the same gap is strong evidence it is real.

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

## User Scenarios & Testing *(mandatory)*

### User Story 1 — A deployment names its model beside its other settings (Priority: P1)

An operator sets the answering model in `config.yml`, next to profiles, quotas and
features, rather than in a separate environment variable.

**Why this priority**: It is the smallest thing that closes the gap, and both
contributed PRs already do most of it.

**Independent Test**: Set the model in `config.yml`, start the server, ask a
question, and confirm from the response which model answered.

**Acceptance Scenarios**:

1. **Given** a `config.yml` naming a model, **When** the server starts, **Then**
   that model answers.
2. **Given** no model in `config.yml`, **When** the server starts, **Then** the
   current default answers and nothing breaks — every existing deployment keeps
   working untouched.
3. **Given** a model in `config.yml` and `LLM_MODEL` set, **Then** the precedence
   between them is defined and documented, not accidental.

---

### User Story 2 — An unusable model stops the server, not the conversation (Priority: P1)

An operator names a model that cannot work as configured. The server refuses to
start and says why.

**Why this priority**: Equal to Story 1 and the reason this is a specification. The
failure today is a 400 on a user's first question — visible to a user, attributed
to the chatbot, and traceable only by reading logs. Both contributed PRs make this
*more* likely by adding two more ways to get the pair wrong.

**Independent Test**: Configure `gpt-5.6-luna` with `temperature: 0`. The server
must refuse to start and name the conflict.

**Acceptance Scenarios**:

1. **Given** a model whose temperature requirement is known, **When** configuration
   contradicts it, **Then** startup fails naming the model, the value and the fix.
2. **Given** a model the table has not met, **When** the server starts, **Then** it
   starts — an unknown model is not an error — and the first failure names
   `LLM_TEMPERATURE`.
3. **Given** a valid configuration, **When** the server starts, **Then** no
   additional API call is made on the happy path unless D1 says otherwise.

---

### User Story 3 — Surfaces choose their own model (Priority: P2)

Chat, chat-alongside-search-results and analysis summarisation each name a model
suited to their tolerance for latency. Measured today: `gpt-4o-mini` answers in
22.5s and `gpt-5.6-luna` in 41.2s, so a search-results panel and a background
summarisation should not be forced to agree.

**Why this priority**: Only chat exists today, so this is not yet blocking. It is
P2 rather than P3 because the context budget made exactly this mistake — a global
constant that had to be unpicked in spec 001 Stage 3 — and the cost of designing
for it now is one nesting level.

**Acceptance Scenarios**:

1. **Given** two surfaces configured with different models, **When** both answer,
   **Then** each uses its own, in one process.
2. **Given** a surface with no model of its own, **Then** it uses the deployment
   default rather than failing.

### Edge Cases

- **`config.yml` names a model the API does not have.** A 404 on the first
  question. Same class as the temperature mismatch and wanted at startup for the
  same reason — but only D1 decides whether that costs a network call.
- **`LLM_MODEL` and `config.yml` disagree.** One must win, and the loser must be
  reported, not silently discarded. Rate limiting was broken once by a config that
  quietly failed open; the constitution's Article IV is that scar.
- **An operator sets a temperature the model accepts but the repository does not
  want.** `0.7` on `gpt-4o-mini` is legal and makes the graders non-deterministic.
  That is the operator's call, and it must be visible in the logs at startup.
- **Provider is `ollama` with a local model.** The temperature table is
  OpenAI-derived and says nothing about it; local models must not be forced to 1.0.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The answering model MUST be settable in `config.yml`.
- **FR-002**: A deployment with no model configured MUST behave exactly as it does
  today.
- **FR-003**: Precedence between `config.yml` and `LLM_MODEL` MUST be defined,
  documented and tested.
- **FR-004**: The embedding model MUST NOT be settable in `config.yml`. It is
  derived from the installed bundle by `resolve_embedding_model()`.
- **FR-005**: Per-model call requirements — temperature, structured-output method —
  MUST remain derived in code, not configured.
- **FR-006**: A configured temperature that contradicts a known model requirement
  MUST stop startup, naming the model, the value and the fix.
- **FR-007**: A model unknown to the temperature table MUST NOT block startup.
- **FR-008**: The effective model MUST be logged at startup, so which one answered
  is answerable from the logs alone.
- **FR-009**: Each surface MUST be able to name its own model, falling back to the
  deployment default.
- **FR-010**: Configuration that cannot be honoured MUST stop the process rather
  than fall back to a default (Article IV, and `Config.from_yaml`'s existing
  behaviour).

### Key Entities

- **Deployment default model**: what answers when a surface names nothing.
- **Surface**: chat, chat-alongside-search, analysis summarisation. Only the first
  exists today.
- **Model requirements**: temperature and structured-output method. Derived, never
  configured; see `FIXED_TEMPERATURE_MODELS` and `bin/probe_model_temperature`.

## Success Criteria *(mandatory)*

- **SC-001**: Switching the answering model requires editing one file and
  restarting — no code change, no rebuild.
- **SC-002**: Every existing deployment keeps working with its `config.yml`
  unchanged.
- **SC-003**: A contradictory model/temperature pair is refused at startup, and no
  user ever sees the resulting error.
- **SC-004**: No configuration file can name an embedding model.
- **SC-005**: Two surfaces run different models in one process.
- **SC-006**: The credit for #112 and #151 appears in the commit that harvests
  them, and both PRs are closed courteously.

## Decisions for the team

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

## Assumptions

- `config.yml` is the right home. It already carries profiles, quotas, features and
  messages; it has a schema, and a loader that refuses to start on an invalid file.
  A second configuration mechanism would be the thing worth avoiding.
- Operators are trusted but not omniscient. They may legitimately want a model this
  repository has never run; they should not have to know its temperature rules.
- Provider stays OpenAI-compatible. `base_url` covers self-hosting, as Plant
  Reactome already does.

## Out of Scope

- **Which model becomes the default.** That is [spec 002](../002-default-llm-choice/spec.md),
  and it is gated on an answer-quality measurement that does not run yet. This
  specification makes the choice expressible; it does not make it.
- Per-request model selection by an end user.
- The embedding model, permanently. FR-004.
