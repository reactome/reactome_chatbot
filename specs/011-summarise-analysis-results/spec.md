# Feature Specification: Summarise analysis results

**Feature Branch**: `011-summarise-analysis-results`
**Created**: 2026-09-18
**Status**: Historical record of the analysis-summary work, 2026-09-18 to
2026-09-21. The endpoint was built 2026-09-19 to 2026-09-20 (4e2d8ab, 9fe1461)
and was live on beta on 2026-09-21; review fixes followed on 2026-10-03
(d153fbf). Two pieces were not built: a durable summary store (research D4)
and a rate limit per analysis token (research D6). The spec's status field was
never moved past "Draft". FR-011 to FR-015 record the answers to two
clarification questions (privacy and stability) given on 2026-09-18. The
requirements below were set for that work, not for later changes. Current
behaviour is in `src/api/analysis_summary.py`, `src/analysis/` and
[contracts/summary_endpoint.md](./contracts/summary_endpoint.md).
**Input**: Summarise Reactome analysis results in natural language, for users who have run an analysis and want to know what it means.

## Why this exists

A Reactome analysis returns a table. A user who has just uploaded a gene list gets
back hundreds of pathway rows with p-values, FDRs, and found/total ratios, sorted
by significance, and has to work out for themselves which rows matter, whether
they are trustworthy, and what the biology has in common.

The gap is not the numbers. It is that reading them correctly requires knowing
what FDR means, that a pathway with 2 of 3 entities found is not strong evidence,
and that unmatched identifiers usually indicate the wrong identifier type rather
than a biological absence. That knowledge is exactly what the curators put in the
user guide and exactly what a reader does not have to hand at the moment they get
their result.

## User stories (as specified 2026-09-18)

The stories and acceptance scenarios were the targets for the work. Notes dated
2026-10-21 say where the built feature differs.

### User Story 1 - What does my result say? (Priority: P1)

A researcher has run a pathway analysis and is looking at the result. They want a
few sentences telling them which pathways came out on top, whether those are
significant once multiple testing is accounted for, and what the top hits have in
common biologically — instead of reading the table themselves.

**Why this priority**: It is the question every user of every analysis type has,
and it is answerable from the result alone. On its own it is a usable feature:
someone with a result gets a readable account of it.

**Acceptance Scenarios**:

1. **Given** a result whose top pathways pass FDR correction, **When** a summary is
   requested, **Then** the summary names those pathways, says they remain
   significant after correction, and cites each by stable id.
2. **Given** a result where no pathway passes FDR correction, **When** a summary is
   requested, **Then** the summary says so plainly rather than describing the
   lowest p-values as though they were findings.
3. **Given** a result with no pathways at all, **When** a summary is requested,
   **Then** the summary reports that nothing was found and does not speculate.

---

### User Story 2 - Why were my identifiers not found? (Priority: P2)

A researcher sees that a large share of their submitted identifiers were not
matched. They want to know why, and whether the result can be trusted.

**Why this priority**: The single most common source of confusion with Reactome
analysis, and it has a small number of well-understood causes — an identifier type
Reactome does not index, the wrong species, or identifiers that are genuinely
absent. It is independently valuable: a user can ask only this and be helped.

**Acceptance Scenarios**:

1. **Given** a result where most identifiers were not found and the matched ones
   resolved through a single resource, **When** a summary is requested, **Then** the
   summary reports the proportion and identifies the probable cause as an
   identifier-type or species mismatch.
2. **Given** a result where every identifier was found, **When** a summary is
   requested, **Then** the summary says so without inventing a problem.
3. **Given** a result whose unmatched proportion is high enough to undermine the
   findings, **Then** the summary says the result should be treated with caution
   and why.

*Built differently (found 2026-09-20).* The aggregate result carries how many
identifiers were not found and nothing about how many were submitted, so no
proportion can be derived without the reader's identifiers. The summary
reports the count and is told never to state a proportion
(`src/analysis/summarise.py:155-171`). Scenarios 1 and 3 therefore hold only as
far as the count and `resourceSummary`/`speciesSummary` allow; nothing computes
a "too many unmatched" threshold.

---

### User Story 3 - What do these numbers mean? (Priority: P3)

A researcher wants the statistics explained in the context of their own result:
what separates the p-value from the FDR, what the found/total ratio implies, and
whether a hit resting on very few entities is worth pursuing.

**Why this priority**: Genuinely useful and the most reusable across analysis
types, but a user can get value from stories 1 and 2 without it, and it is closest
to material the user guide already covers.

*As built (2026-09-20).* There is no request for one pathway. Every pathway
sent to the model carries its own counts and a flag, computed in code, for a
hit resting on fewer than five matched entities (`src/analysis/summarise.py:18-48`).
The model is told to explain those in the reader's terms.

**Acceptance Scenarios**:

1. **Given** a pathway with a small number of found entities, **When** its numbers
   are explained, **Then** the explanation states that few entities make the result
   fragile, using that pathway's actual counts.
2. **Given** a pathway significant by p-value but not after FDR correction, **When**
   its numbers are explained, **Then** the explanation distinguishes the two.

---

### User Story 4 - Readings specific to the analysis type (Priority: P4)

An expression analysis carries values across one or more columns; a species
comparison carries inferred events in another organism. Each supports a reading the
others do not, and a summary that ignores the type either says nothing useful or
says something wrong.

**Why this priority**: It multiplies the value of story 1 for two of the analysis
types, but story 1 must be right first. Deferring it is safe because the type is
visible in the result, so a summary can decline to make type-specific claims until
this is built.

*Built 2026-09-20* as per-type instructions to the model
(`TYPE_INSTRUCTION`, `src/analysis/summarise.py:178-209`), including one for
over-representation results.

**Acceptance Scenarios**:

1. **Given** an expression result with several columns, **When** a summary is
   requested, **Then** it describes how the highlighted pathways behave across those
   columns rather than treating the result as a single enrichment.
2. **Given** a species-comparison result, **When** a summary is requested, **Then** it
   states that findings are inferred by orthology and what that does not establish.

---

### Edge cases identified, and how they are handled (checked 2026-10-21)

| Edge case | Today |
|---|---|
| A token that does not exist, has expired, or belongs to a discarded analysis | `not_found` for 404, 500 or a malformed token; `gone` for 410 (FR-009) |
| A result too large to summarise in full | Only the first 12 pathways are sent, and the model is told when the significant count is a lower bound (research D9). Expression values are sent for every column; the number of columns is not bounded |
| Findings that are entirely disease pathways, often an artefact of the list | Not handled. `inDisease` is allow-listed but not given to the model (`src/analysis/summarise.py:99-125`) |
| A species the summary's knowledge does not cover well | Not handled specifically. `speciesSummary` is given to the model; `summary.species` is not |
| A result already summarised | The stored summary is reused (FR-014) |
| An identifier list that is the user's unpublished data | The `aggregate` tier never sends identifiers; `identifiers` sends up to 50 unmatched ones (FR-012) |

## Requirements of this work, and where they stand (checked 2026-10-21)

"Instructed" means the behaviour is asked of the model in its prompt, not
checked in code.

| ID | Required by this work (2026-09-18) | Today |
|---|---|---|
| FR-001 | Summarise a result the user has already run, by token; never run, re-run or re-implement an analysis | Holds for this endpoint, which takes only a token. (`src/analysis/client.py` gained `submit_identifiers` on 2026-09-25 for the chat's gene-list feature, 3b2c792; this endpoint does not call it.) |
| FR-002 | Derive every quantitative claim from the result; state no statistic it does not contain | Instructed (`DATA_RULES` rule 1, `src/analysis/summarise.py:266-276`). The input says when the significant count is only a lower bound (research D9). Not checked mechanically |
| FR-003 | Distinguish significance before and after multiple-testing correction | Instructed (`STATISTICS_INSTRUCTION`, `src/analysis/summarise.py:213-226`; `DATA_RULES` rule 3) |
| FR-004 | Report a weak, empty or heavily unmatched result rather than presenting the top row as a finding | Empty and nothing-significant are decided in code (`verdict`, `src/analysis/summarise.py:58-66`) with a matching instruction. Hits on fewer than five entities are flagged in code (`:38-48`). "Heavily unmatched" cannot be judged as a proportion (US2 note) |
| FR-005 | Every pathway discussed attributable by stable id | Holds: a `citation` event for every pathway sent to the model (`src/api/analysis_summary.py:263-278`) |
| FR-006 | Identify the analysis type; no claims specific to another type | `analysis_type` on `start`; per-type instructions (`src/analysis/summarise.py:178-209`); ReactomeGSA types declined (`src/analysis/client.py:84-109`) |
| FR-007 | Refuse without a verified caller, before any model call, on the answer endpoint's terms | Holds: the same `verify` (`src/api/analysis_summary.py:133-140`); zero model calls tested (`tests/api/test_analysis_summary.py:215-222`) |
| FR-008 | Fail invisibly: a terminal state, never a broken panel or an HTTP error | Holds for refusals, lookup outcomes, failures and timeouts (HTTP 200 with `done`). A malformed request body gets HTTP 422 (`tests/api/test_analysis_summary.py:285-292`) |
| FR-009 | An unknown, expired or malformed token is a normal negative outcome | Holds: 404, 500 and a malformed token give `not_found`; 410 gives `gone` (`src/analysis/client.py:127-148`) |
| FR-010 | Read from beta's Analysis Service, never production | Beta is the default; `ANALYSIS_BASE_URL` overrides it (`src/analysis/client.py:22-23`, `:63-64`) |
| FR-011 | Opt-in: nothing sent to a model provider until the user asks | This endpoint generates only when called and keeps no consent state. When it is called is the website's decision |
| FR-012 | A meaningful choice of what is shared, with one useful option sending no identifiers, explained before choosing | `disclosure` is required with no default (`src/api/analysis_summary.py:88-89`); `aggregate` sends no identifiers. How the choice is presented is the website's (contract, question 2) |
| FR-013 | Evidence that a person is present, not only a known service | Holds: `human` and a 30-minute `human_iat` (`src/util/caller_token.py:228`, `:280-309`); research D6 |
| FR-014 | The same token yields the same summary on every request | Holds within one process while the summary stays in the 512-entry store and the release is unchanged; lost on deploy. A durable store was not built (research D4) |
| FR-015 | Transparent that it is generated, which analysis it describes, and that regenerating may differ; stability by reuse | `cached`, `release`, `analysis_type` and `disclosure` on `start` (`src/api/analysis_summary.py:241-261`). Telling the reader is the website's |

### Key Entities

- **Analysis token**: the identifier under which a completed analysis result is
  retrievable. The input to every summary. Not secret, but it addresses data the
  user uploaded.
- **Analysis result**: the completed analysis — its type, the species analysed, the
  pathways with their entity and reaction statistics, the resources identifiers
  resolved through, unmatched identifier counts, and any warnings.
- **Pathway hit**: one pathway in the result, with the counts and probabilities that
  determine whether it is worth a reader's attention.
- **Summary**: the natural-language account produced for a result, with the pathway
  citations that let a reader check it.

## Success criteria of this work, and what was checked

| ID | Criterion (2026-09-18) | Checked |
|---|---|---|
| SC-001 | For a result whose top pathways pass FDR, the summary names the pathways the result ranks highest, across a tracked analysis set | Once live, on one token, 2026-09-21 ([quickstart.md](./quickstart.md) scenario 1). This repository records no tracked analysis set or evaluation over one |
| SC-002 | When nothing passes correction, the summary says so and presents no p-value as a finding | Unit-tested on the model input (`tests/analysis/test_summarise.py`). Not run live: no such result could be produced on 2026-09-21 |
| SC-003 | Every pathway named resolves to a stable id present in the result, checked mechanically | Holds by construction: citations come from the result (`tests/api/test_analysis_summary.py:225-233`) |
| SC-004 | Zero model calls without a verified caller | `tests/api/test_analysis_summary.py:187-222`, counted on a patched model |
| SC-005 | An unknown or expired token gives a terminal "no summary" outcome | `not_found` / `gone` in an HTTP 200 stream (`tests/api/test_analysis_summary.py:256-282`) |
| SC-006 | A reader can tell from the summary whether to trust the result, by curator review | This repository records no curator review |

## Assumptions at the time (2026-09-18)

- The caller has already run the analysis and holds its token; this feature never
  accepts a raw identifier list, which keeps it clear of the Analysis Service's job.
- The caller is the Reactome website, reached through its server-side proxy,
  presenting the caller token the answer endpoint verifies. That held, but the
  token gained human-presence claims for this endpoint (FR-013, research D6;
  `src/util/caller_token.py:203-309`), so the assumption that no new
  authentication would be needed did not.
- Delivery follows the existing answer endpoint's shape — progressive output,
  citations as structured events, failure as a terminal state — because a summary
  takes comparable time to produce and the website already renders that shape.
  The state names differ (see the contract).
- Results are read from beta's Analysis Service, consistent with the standing
  instruction to keep off production (FR-010).
- ReactomeGSA results are recognised from the result, but summarising them was
  left for a later increment: GSA is a separate service with its own result
  shape. It has not been built.
- The measured non-reproducibility of generated answers applies here too, which is
  why FR-014 and FR-015 make stability a matter of reuse rather than assuming it.

## Scope (as set 2026-09-18)

**In scope**: summarising a completed analysis result; explaining its statistics;
explaining unmatched identifiers; recognising the analysis type.

**Out of scope**: running analyses; ranking or re-ranking pathways by any measure
the result does not contain; comparing two analyses; storing results; the analysis
UI itself.

## Dependencies (as of 2026-09-18)

- The Analysis Service on beta, for retrieving results by token.
- The existing caller-token verification and streaming answer surface, reused
  rather than rebuilt.
- ReactomeGSA, only to the extent of recognising that a result came from it.
