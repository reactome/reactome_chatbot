# Tasks: Searching Only the Collections a Question Needs

**Feature**: 009-collection-routing | **Plan**: [plan.md](./plan.md) | **Spec**: [spec.md](./spec.md)

Test tasks are included. Principle II requires a before-and-after measurement on any
change to what reaches the LLM, and Principle III requires current behaviour pinned
before it is changed.

## Phase 1: Setup

- [ ] T001 Create branch `009-collection-routing` from main and set `.specify/feature.json` to `specs/009-collection-routing`
- [x] T002 Capture the retrieval baseline before any code change: `./bin/retrieval_baseline capture --out specs/009-collection-routing/before.json` against the installed Release97 bundle
- [ ] T003 [P] Record the current cost in `specs/009-collection-routing/quickstart.md`: documents, context tokens and retrieval seconds for one question needing no variant data

## Phase 2: Foundational (blocks every user story)

**The gate has a hole.** Four of the five collections have no question that fails if
routing stops searching them. Routing must not land before this closes, or the
acceptance criterion cannot detect the failure the feature can cause.

- [x] T004 [P] Add a `summations`-dependent question to `EXPECTATIONS` in src/evaluation/answer_sweep.py (Selective autophagy / lysosome; verified by removal)
- [x] T005 [P] ~~Find a `complexes` candidate that fails without the collection~~ **Attempted and abandoned with a reason, 2026-09-19.** Nine questions over six complexes, in two shapes; every one answered just as well without the collection, because complex names appear throughout `reactions` (as input and output names) and `summations` prose. The one thing unique to `complexes` -- the component list -- has answers too variable to assert on: the same configuration returned 1 to 6 of 7 components. See research.md
- [x] T005a [P] Assert at retrieval level that `complexes` was searched, in tests/retrievers/test_sync_async_equivalence.py -- parametrised over all five real collection names: each is reachable, and selecting it searches nothing else. Verified by sabotaging `resolve_collections` to ignore the selection, which fails all five. **Catches a plumbing failure, not a routing one**: a classifier that never chooses a collection is still undetectable, and needs a production probe rather than a test (research.md)
- [ ] T005b **The risk T005 was written for is still open.** T005a catches a plumbing failure; a classifier that never *chooses* a collection is undetectable by any deterministic test, and produces confident plausible answers rather than an error. Needs the routing distribution observed over real traffic, or a periodic probe asking known-collection questions and checking what was selected. **Do not treat Phase 2 as closing this** -- a register that reads "handled" is why nobody looks again
- [x] T006 [P] Add an `ewas`-dependent question to `EXPECTATIONS` in src/evaluation/answer_sweep.py (TP53 UniProt P04637; verified by removal)
- [x] T007 Assert at retrieval level that `reactions` was searched -- closed by the same parametrised test as T005a, in tests/retrievers/test_sync_async_equivalence.py. No answer-level question can guard it, because every reaction name also appears in `summations`
- [x] T008 Verify each new question FAILS when its collection is removed from the bundle copy, and passes with it present; record the evidence in the PR (method established; two of four candidates survived it)
- [x] T009 Pin current behaviour: a characterization test in tests/retrievers/test_collection_selection.py asserting that with no selection every collection in the bundle is searched
- [ ] T010 Run `./bin/answer-sweep` against Release97 and confirm green before any behaviour change

## Phase 3: User Story 1 — a question searches only the collections it needs (P1)

**Goal**: cut context tokens and retrieval latency on questions that do not need every
collection, with no extra LLM call.

**Independent test**: `./bin/answer-sweep` stays green while the measured context
tokens for a question needing one collection fall relative to `before.json`.

**The second half of that test was the wrong measurement, and it failed.**
Context does not fall: the chain caps the fused list at ten documents however
many collections fed it, so narrowing changes which documents arrive, not how
many (research.md, 2026-09-19). What falls is retrieval time, 1.82s to 1.43s.
The sweep half stands: 13/13 with the classifier choosing.

- [x] T011 [US1] Add `collections: list[str] = []` to `QueryIntent` in src/agent/tasks/intent_classifier.py, defaulting to empty so an omitted field means "all"
- [x] T012 [US1] Extend the classifier prompt in src/agent/tasks/intent_classifier.py to name selectable collections, sourced from `reactome_descriptions_info` rather than a literal list
- [x] T013 [P] [US1] Add `resolve_collections(selected, available)` to src/retrievers/csv_chroma.py implementing data-model.md: empty means all, unknown names log WARNING and return all
- [x] T014 [P] [US1] Unit-test `resolve_collections` in tests/retrievers/test_collection_selection.py for empty, all-valid, some-unknown and all-unknown, asserting every failure widens rather than narrows
- [x] T015 [US1] Filter `self.collection_retrievers` by the selection in `retrieve_documents` in src/retrievers/csv_chroma.py, reading it from `RunnableConfig["configurable"]["collections"]`
- [x] T016 [US1] Apply the identical filter in `aretrieve_documents` in src/retrievers/csv_chroma.py — this is the served path
- [x] T017 [US1] Extend tests/retrievers/test_sync_async_equivalence.py to assert both paths honour the same selection, and confirm it fails when only one is filtered
- [x] T018 [US1] Carry `collections` on `ReactToMeState` in src/agent/profiles/react_to_me.py, set in `preprocess` beside `active_sources`
- [x] T019 [US1] Pass the selection into retrieval in `generate_answer` in src/agent/profiles/react_to_me.py — **via the `selected_collections` ContextVar, not `config["configurable"]`**: `create_retrieval_chain` gives the retriever no path for extra arguments. LangGraph copies the context into the tasks it spawns, so a value set around the call reaches the retriever inside them. Reset in `finally`, because the graph reuses one task across turns
- [x] T020 [US1] Verify through the agent, not the retriever: a test that a variant question routes to `disease_variants` and a userguide question does not, per Principle I

## Phase 4: Measurement and acceptance

- [ ] T021 Capture `./bin/retrieval_baseline capture --out specs/009-collection-routing/after.json` and `compare` it with before.json; report the diff in the PR as information, not as a gate
- [ ] T022 Re-measure context tokens and retrieval seconds for the same question as T003 and state the change as a number
- [ ] T023 Run `./bin/answer-sweep` against Release97 with live MCP; it must be green including all nine collection-dependent questions
- [ ] T024 Record in the PR how often the classifier selected a subset, and how often it named an unknown collection

## Phase 5: Polish

- [ ] T025 [P] Update the Status line and add a Results section to specs/009-collection-routing/spec.md with the measured before/after
- [ ] T026 [P] Note in src/retrievers/reactome/metadata_info.py that `reactome_descriptions_info` is now a serving input, so an inaccurate description degrades routing
- [ ] T027 Run `/speckit-analyze` across spec, plan and tasks and resolve anything CRITICAL or HIGH

## Dependencies

```
Phase 1 (T001-T003)
   ↓
Phase 2 (T004-T010)   ← blocking: the gate must detect routing mistakes first
   ↓
Phase 3 (T011-T020)   ← the feature
   ↓
Phase 4 (T021-T024)   ← acceptance
   ↓
Phase 5 (T025-T027)
```

Within Phase 3: T011 → T012; T013 → T014, T015, T016; T015 and T016 → T017; T018 → T019 → T020.

## Parallel opportunities

- T004-T007: four sweep questions, four independent edits to the same list — write together, run T008 once
- T013 and T014 alongside T011 and T012: `resolve_collections` is pure and does not depend on the prompt
- T025 and T026: different files

## Implementation strategy

**MVP is Phase 2 plus Phase 3.** Phase 2 alone is worth landing on its own: it closes
a hole in the gate that exists today, independently of whether routing is ever built.

Ship Phase 2 as its own PR. If routing then turns out to cost more recall than it
saves, that PR still stands on its own merit.
