# Feature Specification: Run a gene set analysis from the chat

**Feature Branch**: `012-run-gene-analysis`

**Created**: 2026-09-21

**Status**: Draft

**Input**: User description: "Run a gene set analysis from an uploaded expression matrix in the chat, and return the pathway results as a downloadable file"

## Context

Reactome runs gene set analysis at `gsa.reactome.org` — PADOG, Camera
("similar to the classical GSEA algorithm"), ssGSEA, terapadog. The chatbot
cannot start one. The gap was found the hard way: asked to "run a GSEA with
my list of genes", the chatbot replied that Reactome could not, and offered
`fgsea` and a YouTube tutorial.

Five catalogue tools were added to reactome-mcp in September (methods, data
types, dataset search, examples, sources). They let the model *describe* the
service. None of them runs anything.

**Everything below was measured against the live service on 2026-09-21**, not
read off the specification.

### What the service actually requires

`POST /analysis` takes `datasets[].data`: the whole expression matrix, inline,
as a tab-delimited string. It is a required field and there is no
by-reference variant.

| | measured |
|---|---|
| matrix, 16-sample example dataset | **1.2 MB** |
| submitted payload | **1.5 MB** |
| result | **2.0 MB** — 2,679 pathways × 9 columns |
| run time | minutes (PADOG, 1,000 permutations) |

A full round trip was completed: load → summary → submit → poll → result.
Reactome release 97, columns `Pathway, Name, Direction, FDR, PValue, NGenes,
MeanAbsT0, MeanWeightT0, av_foldchange`, plus a Pathway Browser link carrying
the analysis token.

### Two routes in, and only one needs an upload

`POST /data/load/{resourceId}` loads a **public** dataset by identifier —
Expression Atlas, Single Cell Expression Atlas, GREIN, GEO, and the bundled
examples. `GET /data/summary/{id}` then returns the sample IDs and factors
(measured: 16 samples; `condition` = MCM/MOCK, `cell.type` = PBMCB/TIBC,
`patient.id` = P1–P4), which is exactly what a user needs to choose a
comparison, and is small enough to show them.

The user's own data has no such route. It must be uploaded.

### Where the data goes, and where the decision is

Uploading is **not** the boundary that matters. Reactome is not OICR: the
service runs on AWS operated by the Reactome project, and reactome.org's own
analysis page already accepts exactly these files. A user uploading their
matrix here is doing what they can already do on the website. Users know what
they are uploading.

**The boundary this feature moves is the results reaching OpenAI**, and only
if the user asks for a summary. That already has a mechanism and a warning,
and this feature reuses both rather than inventing either.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Analyse a public dataset (Priority: P1)

A user says "run a gene set analysis on GSE12345 comparing treated with
control". The chatbot loads the dataset by identifier, shows the sample
groups it found, runs the analysis, and returns the significant pathways plus
a downloadable table.

**Why this priority**: it is the whole feature minus the upload, it needs no
file handling and no disk, and it is the case a user with no data of their own
can still reach. If only this ships, the chatbot can run real analyses.

**Acceptance**: given a public dataset identifier and a named comparison, the
chat reports the pathway count, shows the top pathways with FDR, offers the
full table as a file, and links the Pathway Browser view.

### User Story 2 — Analyse an uploaded matrix (Priority: P2)

A user uploads their own expression matrix through the chat UI, names the
comparison, and gets the same output.

**Why this priority**: it is what was asked for, and it is P2 only because it
depends on the submit-and-poll machinery that Story 1 already builds.

**Acceptance**: a matrix uploaded through the UI reaches the service and
produces results, and the file is deleted from the container once submitted.

### User Story 3 — Results without a model summary (Priority: P1)

A user who declines the summary still gets the pathway table and the Pathway
Browser link.

**Why this priority**: same priority as Story 1 deliberately. The analysis is
the product; the summary is a convenience. A user must never have to send
results to a third party in order to see their own results.

**Acceptance**: with the summary declined, the file and the link are present
and nothing has been sent to the model.

## Requirements *(mandatory)*

- **FR-001** The matrix MUST NOT pass through the model's context or an MCP
  tool call. At 1.2 MB it would bury the answer and cost more than the
  analysis. The chatbot submits it server-side.
- **FR-002** The result table MUST NOT be sent to the model whole. 2,679 rows
  is the measured size of one small run; the model receives a bounded top-N.
- **FR-003** Results sent to the model MUST go through the existing
  `src/analysis/disclosure.py` allow-list, **extended for ReactomeGSA's own
  free-text fields**. The current `NEVER_SENT` names `fileName`, `sampleName`
  and `columnNames`, which are the Analysis Service's. GSA carries the same
  hazard under different names — `datasets[].name` is chosen by the user, and
  `design.samples` is their column headers, which in the measured example were
  `patient.id` values. An allow-list is wrong only by omission; these must be
  omitted deliberately, not by luck.
- **FR-004** The existing warning MUST be shown before any result reaches the
  model, and declining MUST still yield the file (Story 3).
- **FR-005** A successful `POST /analysis` MUST be treated as *accepted*, never
  as *succeeded*. Measured: a submission returned HTTP 200 and then failed with
  `CONNECTION_FORCED - broker forced connection closure` while the service was
  mid-upgrade. The failure was visible only through `GET /status`.
- **FR-006** Uploads MUST be capped well below Chainlit's default. The config
  ships `max_size_mb = 500`; the host has **4.8 GB free of 88 GB (95% used)**,
  and `~/update-beta-chat.sh` already needs 6 GB to deploy. A handful of
  default-sized uploads takes down the chat and blocks the next deploy. ~20 MB
  covers a real matrix.
- **FR-007** An uploaded file MUST be deleted once submitted, whether the
  analysis succeeds or fails.
- **FR-008** The run takes minutes, so the flow MUST NOT be shaped like a
  streamed answer. The user is told it started and told again when it finishes.

## Out of scope

- Hosting any of this in reactome-mcp. The MCP is public, stateless and was
  deliberately narrowed this month; file handling and long-running jobs belong
  in the chat application. The five catalogue tools stay where they are.
- Single-cell clustering parameters (`k` for Single Cell Expression Atlas).
- Report generation: `GET /report_status/{id}` returned 404 for a completed
  analysis, so whatever produces reports is not reachable this way and needs
  its own investigation.

## Open questions

- Which method to default to. PADOG is the service's own default and is what
  was measured; Camera is faster. Needs a decision informed by run time on a
  realistic dataset, not by preference.
- Whether an analysis should survive a chat session ending. Results live
  behind an analysis ID at the service, so resuming is possible; whether it is
  wanted is a product question.
