# Collection Routing: Task-ID Index

Historical. The step-by-step task list for this work (T001–T027) is in git history
up to 8387647. Kept here are the IDs that code comments and this folder refer to.

| ID | Task | Outcome |
|---|---|---|
| T003 | Record the pre-routing cost | Done 2026-09-17; figures in spec.md |
| T004, T006 | Sweep questions guarding `summations` and `ewas` | In `src/evaluation/answer_sweep.py`; each verified by removing its collection |
| T005 | Sweep question guarding `complexes` | Abandoned 2026-09-19: complex names also appear in `reactions` and `summations` (research.md) |
| T005a | Retrieval-level test that every collection is reachable and selecting one searches nothing else | `tests/retrievers/test_sync_async_equivalence.py`. Catches plumbing failures, not a classifier that never chooses a collection |
| T005b | Probe what the classifier selects | `bin/routing-probe`; first run 2026-09-19, 10/10 |
| T005c | Run the probe on every deploy | Wired into the beta deploy script 2026-09-19 (`deploy/beta/README.md`, section 5); the script is not in this repository |
| T007 | Guard `reactions` | No question can; covered by the T005a test |
| T008 | Check each guard question fails without its collection | Done; two of four candidates held |
| T017 | Both retrieval paths honour the same selection | `tests/retrievers/test_sync_async_equivalence.py` |

Not done when the list was retired, and no record of whether they are still wanted:
T021 (capture `after.json` and compare it with `before.json`) and T026 (note in
`src/retrievers/reactome/metadata_info.py` that `reactome_descriptions_info` is a
serving input).
