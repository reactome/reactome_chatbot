---
description: "Task list for model configuration"
---

# Tasks: Model Configuration

**Input**: Design documents from `/specs/003-model-configuration/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [quickstart.md](./quickstart.md)

**Tests**: Included. This repository's constitution (Article III) makes tests the
tripwire for behaviour, and Article IV's "fail loudly" requirement is only real if a
test asserts the failure. They are written with the code they cover, not after.

**Branch**: `feat/model-configuration`

## Phase 1: Setup

- [ ] T001 Create branch `feat/model-configuration` from `origin/main`
- [ ] T002 Re-read #112 and #151 with `gh pr diff`, to credit them accurately in the commits that land their idea

## Phase 2: Foundational

**Blocking: every user story below depends on the config field existing.**

- [ ] T003 Create `LLMConfig` (`provider: str = "openai"`, `model: str | None = None`, `base_url: str | None = None`, `temperature: float | None = None`) in `src/util/config_yml/models.py`, after the shape in #112 and crediting @AaryanCode69
- [ ] T004 Add `llm: LLMConfig | None = None` to `Config` in `src/util/config_yml/__init__.py` — optional, so a config without it is unchanged (FR-002)
- [ ] T005 [P] Add the matching `llm` object to `.config.schema.yaml`, with **no** embedding field (FR-004)
- [ ] T006 [P] Document the section, commented out, in `config_default.yml`

## Phase 3: User Story 1 — A deployment names its model beside its other settings (P1)

**Goal**: the model comes from `config.yml`.

**Independent test**: set a model in `config.yml`, start the server, ask a question,
confirm from the log which model answered. Quickstart steps 1–3.

- [ ] T007 [US1] Add `resolve_llm_model(config)` to `src/agent/graph.py`: `LLM_MODEL` beats `config.llm.model` beats the current default, and document why the precedence is the reverse of `util/secrets.py` (both are "the more specific wins")
- [ ] T008 [US1] Wire `AgentGraph.__init__` to it, passing `base_url` and `provider` from the config when present, in `src/agent/graph.py`
- [ ] T009 [US1] Log the effective model at startup in `src/agent/graph.py` (FR-008) — the name only, never a key
- [ ] T010 [P] [US1] Test in `tests/agent/test_model_configuration.py`: no `llm` section behaves exactly as today (FR-002)
- [ ] T011 [P] [US1] Test in `tests/agent/test_model_configuration.py`: a configured model is the one selected
- [ ] T012 [P] [US1] Test in `tests/agent/test_model_configuration.py`: `LLM_MODEL` overrides `config.yml` (FR-003)
- [ ] T013 [US1] Run quickstart steps 1–3 against a real bundle and confirm the log names the expected model each time (constitution Article I)

## Phase 4: User Story 2 — An unusable model stops the server, not the conversation (P1)

**Goal**: a contradictory model/temperature pair is refused at startup.

**Independent test**: `gpt-5.6-luna` with `temperature: 0` must refuse to start.
Quickstart steps 4–5.

- [ ] T014 [US2] Extend `resolve_temperature` in `src/agent/graph.py` to accept a configured temperature and raise `SystemExit` naming model, value and fix when the model refuses it (FR-006)
- [ ] T015 [P] [US2] Test in `tests/agent/test_model_temperature.py`: luna + `temperature: 0` exits, and the message contains all three of model, value and remedy
- [ ] T016 [P] [US2] Test in `tests/agent/test_model_temperature.py`: a model absent from the table starts normally (FR-007)
- [ ] T017 [P] [US2] Test in `tests/agent/test_model_temperature.py`: `LLM_TEMPERATURE` still wins over the configured value
- [ ] T018 [US2] Perturbation check: delete the guard and confirm T015 fails — a test that cannot fail is not a tripwire (Article III)
- [ ] T019 [US2] Run quickstart steps 4–5 and confirm the server refuses to start rather than failing on the first question

## Phase 5: User Story 3 — Surfaces choose their own model (P2)

**Goal**: not built now; make sure Stage 1's shape does not preclude it.

**Independent test**: none — this phase ships no behaviour.

- [ ] T020 [US3] Confirm `LLMConfig` is nestable per surface without a schema break, and record in `specs/003-model-configuration/plan.md` what a second surface would add
- [ ] T021 [US3] Cross-reference spec 002's latency table (22.5s vs 41.2s per question) in the spec as the reason surfaces will want to differ

## Phase 6: Polish & Cross-Cutting

- [ ] T022 [P] Verify `grep -rn embedding .config.schema.yaml config_default.yml` finds no embedding model field (SC-004), and add a test asserting it
- [ ] T023 [P] Confirm `tests/util/test_config.py` passes **untouched** — adding a section must not change what an invalid config does (Article III)
- [ ] T024 Run `ruff check`, `ruff format --check`, `mypy`, `pytest`
- [ ] T025 Close #112 with credit to @AaryanCode69, stating plainly that the LLM half is harvested and the embedding half rejected because it bypasses `resolve_embedding_model()` and would silently break Plant Reactome
- [ ] T026 Close #151 with credit to @bhavyakeerthi3, noting the flat-string shape was reasonable but `base_url` has nowhere to live in it
- [ ] T027 Update `specs/003-model-configuration/spec.md` with the outcome, and record D1 as taken-as-recommended

## Dependencies

```text
Phase 1 (T001-T002)
      |
Phase 2 (T003-T006)   <- blocking; nothing below works without the field
      |
      +-- Phase 3 US1 (T007-T013)   P1
      |         |
      +-- Phase 4 US2 (T014-T019)   P1, needs US1's resolution to validate
      |
      +-- Phase 5 US3 (T020-T021)   P2, documentation only
                |
Phase 6 (T022-T027)
```

US2 depends on US1: there is nothing to validate until the model is configurable.
US3 depends on neither and ships no code.

## Parallel Opportunities

- **T005, T006** — different files, no shared state
- **T010, T011, T012** — three tests in one new file; write together, they share fixtures
- **T015, T016, T017** — same, in the existing temperature test file
- **T022, T023** — independent checks

`T025` and `T026` are deliberately **not** parallel with the rest: close a
contributor's PR only once the thing that replaces it has actually landed and its
gates are green.

## Implementation Strategy

**MVP is Phase 2 + Phase 3 (US1).** That alone delivers FR-001 through FR-003 and
FR-008 — a deployment can name its model — and is independently shippable.

**Phase 4 (US2) is the reason this is a specification.** It is P1 alongside US1
rather than after it, because shipping US1 alone adds two new ways to configure a
model wrongly without adding any way to find out before a user does.

Phase 5 ships nothing and could be dropped without loss; it exists so the nesting
decision in T003 is made deliberately rather than discovered later, which is the
mistake the context budget made in spec 001.
