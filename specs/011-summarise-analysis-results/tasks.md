# Summarise Analysis Results: Task-ID Index

Historical. The step-by-step task list for this work (T001–T037, with T031a–c)
is in git history up to b0ff6ec. Every task was marked done there; T031b was
only partly done (below). Kept here are the IDs that code comments and this
folder refer to, and one repository rule found along the way.

| ID | Task | Outcome |
|---|---|---|
| T002 | Create `tests/analysis/` | Created without an `__init__.py`. No test directory in this repository has one, and adding one made `tests/analysis` shadow `src/analysis`. Test basenames must be unique for the same reason, which is why the client tests are `tests/analysis/test_analysis_client.py` (the task named `test_client.py`) |
| T019 | Test that the `identifiers` tier is never reached without an explicit request | `tests/api/test_analysis_summary.py:307-318`, by recording the outbound not-found call |
| T031 | Agree how human presence is asserted | Agreed with the website 2026-09-19 (research D6). The task text named the third claim `subject`; the agreed and implemented claim is `human_sub` |
| T031b | Rate limit per person, and per analysis token | Partly done: the limiter keys on `human_sub`, falling back to the caller identity (`src/api/analysis_summary.py:157-164`). The per-analysis-token limit was not built, though the task was marked done |
| T034 | Bound the summary so a stuck upstream cannot hold a connection | 120 s bound (`src/api/analysis_summary.py:47`); `tests/api/test_analysis_summary.py:636-667` also pins that a timed-out summary is not stored |
| T035 | Log an abandoned summary stream | `src/api/analysis_summary.py:311-315`; `tests/api/test_analysis_summary.py:670` |
| T036 | Run the quickstart scenarios against beta | Done 2026-09-21 on image `cc4315c`; results in [quickstart.md](./quickstart.md). Scenarios 2 and 8 could not be constructed |
| T037 | Tell the website once live | Done 2026-09-21: the terminal states, the scenarios verified only by unit tests, and that summaries are lost on deploy |
