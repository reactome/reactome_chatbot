---
description: "Task list for answering in the user's language"
---

# Tasks: Answer in the User's Language

**Input**: [plan.md](./plan.md), [spec.md](./spec.md)

**Tests**: Included. FR-003 and FR-007 are both "nothing changed" claims, and an
unasserted claim of that kind is worth nothing.

## Phase 1: Foundational

- [x] T001 Add a shared language instruction constant carrying the nomenclature rule (gene symbols, protein names, pathway names, `R-HSA-*`, URLs stay English) in `src/agent/tasks/language_instruction.py`, crediting @bleedblack1 for the wording

## Phase 2: User Story 1 — React-to-Me answers in the detected language (P1)

**Independent test**: ask the assembled chain a French question; read the answer.

- [x] T002 [US1] Add a `{detected_language}` variable and the instruction to `src/retrievers/reactome/prompt.py`
- [x] T003 [US1] Pass `state["detected_language"]` in `generate_answer` in `src/agent/profiles/react_to_me.py`, leaving `input` untouched
- [x] T004 [P] [US1] Test in `tests/agent/test_answer_language.py`: the retrieval query is byte-identical with and without a language (FR-003)
- [x] T005 [P] [US1] Test in `tests/agent/test_answer_language.py`: the prompt receives the language as its own variable, never concatenated into `input` (FR-008)
- [x] T006 [P] [US1] Test in `tests/agent/test_answer_language.py`: an English question adds no extra model call and reaches the retriever with a byte-identical query (FR-007)
- [x] T006a [US1] Ask the real chain an English question and confirm the answer is still English and well-formed — the instruction is new prompt content for every existing user (SC-005)
- [x] T007 [US1] Ask the real chain a French question against the Release95 bundle; confirm the answer is French and gene symbols, `R-HSA-*` IDs and URLs are unchanged (Article I)

## Phase 3: User Story 3 — Plant Reactome, identically (P2)

- [x] T008 [US3] Same change to `src/retrievers/plantreactome/prompt.py` and its call site in `src/agent/profiles/plantreactome.py`
- [x] T009 [P] [US3] Test that both profiles use the same shared instruction, so they cannot drift

## Phase 4: Polish & Cross-Cutting

- [x] T010 Perturbation check: revert the call site and confirm the French test fails
- [x] T011 Run `ruff check`, `ruff format --check`, `mypy`, `pytest`
- [ ] T012 Close #140 with credit to @bleedblack1 — the target and the nomenclature rule are kept; the mechanism is not, with the measured number
- [ ] T013 Close #125 with credit to @bhavyakeerthi3 for the mechanism, noting its hallucination-grading work belongs to #123 and is untouched
- [ ] T014 Record the outcome in `specs/004-answer-in-user-language/spec.md`

## Dependencies

```text
T001 -> Phase 2 (T002-T007) -> Phase 3 (T008-T009) -> Phase 4
```

Phase 3 depends on Phase 2 only for the shared constant's final shape.

## Implementation Strategy

**MVP is Phase 2.** React-to-Me is the deployment with users. Phase 3 is the same
change to a second profile and could ship separately.
