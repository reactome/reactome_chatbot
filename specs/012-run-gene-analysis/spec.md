# Feature Specification: Run a gene set analysis from the chat

**Feature Branch**: `012-run-gene-analysis`

**Created**: 2026-09-21

**Status**: Historical record of running ReactomeGSA from the chat, 2026-09-21 to 2026-09-22 (1678026, 0a9c5c0, 980ea94, f51283a), with fixes through 2026-10-03 (6dee089, b31669d, 48d2a7d). The upload path (Story 2) is built. The public-dataset path (Story 1) and a model-written summary (Story 3's premise) are not. The requirements below were set for that work, not for later changes. Current behaviour is in `src/gsa/`, entered from `bin/chat-chainlit.py:693-713`.

**Input**: User description: "Run a gene set analysis from an uploaded expression matrix in the chat, and return the pathway results as a downloadable file"

## Context (2026-09-21)

Reactome runs gene set analysis at `gsa.reactome.org` — PADOG, Camera
("similar to the classical GSEA algorithm"), ssGSEA, terapadog. When this spec
was written the chatbot could not start one. The gap was found the hard way:
asked to "run a GSEA with my list of genes", the chatbot replied that Reactome
could not, and offered `fgsea` and a YouTube tutorial.

Since then the chat runs a ReactomeGSA analysis on an attached matrix (this
work), and a gene list typed into the chat is offered an over-representation
analysis instead (`src/analysis/gene_list.py`; 3b2c792, 2026-09-25), which this
spec does not cover.

Five catalogue tools were added to reactome-mcp in September 2026 (methods,
data types, dataset search, examples, sources). They let the model *describe*
the service; none of them ran anything.

**The figures in this section were measured against the live service on
2026-09-21**, not read off the specification. This repository records no
re-measurement.

### What the service required (measured 2026-09-21)

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

**The boundary the spec identified is the results reaching OpenAI**, which
would happen only if the user asked for a summary. The spec planned to reuse
the existing allow-list mechanism and the opt-in set for analysis summaries
(spec 011) rather than invent new ones. No summary was built, so no GSA
result reaches a model; the GSA allow-list was written separately (FR-003).

## User stories as proposed (2026-09-21)

Priorities are as the spec set them; the build order differed (see "What
shipped"). Story numbers are kept because code cites them
(`src/gsa/chainlit_flow.py:283`).

### User Story 1 — Analyse a public dataset (proposed P1; not built)

Proposed: a user says "run a gene set analysis on GSE12345 comparing treated
with control"; the chatbot loads the dataset by identifier, shows the sample
groups it found, runs the analysis, and returns the significant pathways plus
a downloadable table. The spec ranked it first because it needs no file
handling and no disk, and a user with no data of their own can reach it.
Proposed acceptance: given a public dataset identifier and a named comparison,
the chat reports the pathway count, shows the top pathways with FDR, offers
the full table as a file, and links the Pathway Browser view.

**Status**: not built as a chat feature. `submit_public_dataset`
(`src/gsa/job.py:86-140`) loads, checks and submits a public dataset and is
tested (`tests/gsa/test_gsa_job.py:103-135`), but nothing outside the tests
calls it. The blocker recorded on 2026-09-22: reaching it needs the model to
choose a dataset and a factor from the conversation, which is conversation
design, not plumbing. Whether it is still wanted is not recorded.

### User Story 2 — Analyse an uploaded matrix (proposed P2; built 2026-09-22)

A user uploads their own expression matrix through the chat UI, says which
group each sample is in, and gets the same output. It was what had been asked
for; the spec ranked it P2 only because it would reuse Story 1's
submit-and-poll machinery, and it was built first. Proposed acceptance: a
matrix uploaded through the UI reaches the service and produces results, and
the file is deleted once submitted.

**How it works now**: the attachment is checked and its sample names echoed
(`src/gsa/chainlit_flow.py:144-164`, `src/gsa/upload.py:113`). The matrix then
waits, up to 30 minutes and one per session, for the user's next message that
reads as one label per sample (`bin/chat-chainlit.py:704-713`,
`src/gsa/pending.py:22`); other messages are answered as usual. Exactly two
groups, one comparison; the first label given is group 1
(`src/gsa/chat.py:73-122`). The method is PADOG (`src/gsa/job.py:59`) and the
dataset type is always `rnaseq_counts` (`src/gsa/chainlit_flow.py:230`).

### User Story 3 — Results without a model summary (proposed P1)

Proposed: a user who declines the summary still gets the pathway table and the
Pathway Browser link, and nothing is sent to the model. The spec's reason,
which still describes the design: the analysis is the product and the summary
a convenience, and a user must never have to send results to a third party in
order to see their own results.

**Status**: moot as written. No model summary was built, so every user gets
the table and the link and nothing is sent to a model
(`src/gsa/chainlit_flow.py:278-297`). An opt-in GSA summary is proposed and not
built; what blocks it and whether it is still wanted are not recorded.

## Requirements of this work, and where they stand (checked 2026-10-21)

| ID | Required by the work (2026-09-21) | Today |
|---|---|---|
| FR-001 | The matrix not to pass through the model's context or an MCP tool call (at 1.2 MB it would bury the answer and cost more than the analysis); the chatbot submits it server-side | Holds. An attached `.tsv`, `.csv` or `.txt` routes to the analysis flow instead of the graph (`bin/chat-chainlit.py:693-702`, `src/gsa/chainlit_flow.py:51-53, 94-107`), and the chat server posts the matrix inline to ReactomeGSA (`src/gsa/job.py:174-183`, `src/gsa/client.py:292-310`) |
| FR-002 | The result table not sent to the model whole (2,679 rows in one small measured run); the model to receive a bounded top-N | The bounded view exists: the 20 most significant rows plus exact counts (`src/gsa/results.py:50-52, 156-198`). No GSA result is sent to a model at all; the view is used only to write the chat's own description of the result (`src/gsa/chat.py:155-196`) |
| FR-003 | Results sent to the model to go through `src/analysis/disclosure.py`, extended for ReactomeGSA's own free-text fields: `datasets[].name` (chosen by the user) and `design.samples` (their column headers, which in the measured example were `patient.id` values) | Built differently. GSA has its own allow-list in `src/gsa/results.py`: `MODEL_COLUMNS`, and `NEVER_SENT` = `fold_changes`, `design`, `samples`, `mappings` (`results.py:22-48`), tested by name (`tests/gsa/test_gsa_results.py:85-108`). `disclosure.py` was not extended; its `NEVER_SENT` is still the Analysis Service's three fields (`disclosure.py:80`). The submitted dataset is named `uploaded`, never the user's filename (`src/gsa/job.py:176`), and the Pathway Browser link, which carries the analysis token, is kept out of the model view (`results.py:201-219`; 21acf75) |
| FR-004 | The existing warning shown before any result reaches the model, and declining still to yield the file (Story 3) | Nothing to apply to: no result reaches a model, and the GSA flow has no summary or warning step |
| FR-005 | A successful `POST /analysis` treated as accepted, never as succeeded. Measured 2026-09-21: a submission returned HTTP 200 and then failed with `CONNECTION_FORCED - broker forced connection closure` while the service was mid-upgrade, visible only through `GET /status` | Holds. `submit_*` returns an ID only; success or failure is decided in `await_result` (`src/gsa/job.py:9-12, 291-355`), and a failure after acceptance is reported to the user (`src/gsa/chainlit_flow.py:264-276`) |
| FR-006 | Uploads capped well below Chainlit's default (`max_size_mb = 500`), at about 20 MB. Reason given 2026-09-21: the host had 4.8 GB free of 88 GB (95% used), `~/update-beta-chat.sh` needed 6 GB to deploy, and a handful of default-sized uploads would take down the chat and block the next deploy | Holds, in two places: `.chainlit/config.toml:65` (`max_size_mb = 20`, enforced in the browser) and `src/gsa/upload.py:23-24, 118-126` (20 MiB by default, `GSA_MAX_UPLOAD_BYTES`, enforced on the server) |
| FR-007 | An uploaded file deleted once submitted, whether the analysis succeeds or fails | Holds, and on more paths than submission: in `submit_uploaded_matrix`'s `finally` (`src/gsa/job.py:166-185`), on a refused or unreadable upload (`src/gsa/chainlit_flow.py:152-161`), after a labelled run (`chainlit_flow.py:183-186`), when a newer upload replaces it or the 500-session bound evicts it (`src/gsa/pending.py:34-46`), when it is found expired after 30 minutes (`pending.py:48-57`), and at session cleanup (`bin/chat-chainlit.py:207`). Result tables are pruned before each write, at 24 hours or 200 MB in total (`src/gsa/job.py:200-259`) |
| FR-008 | The run takes minutes, so the flow not shaped like a streamed answer: the user told it started, and told again when it finishes | Holds: a "Started" message (`src/gsa/chainlit_flow.py:252-255`), a progress line edited in place (`bin/chat-chainlit.py:216-245`), then the result and the file (`chainlit_flow.py:294-297`) |

## Out of scope for this work (set 2026-09-21)

- Hosting any of this in reactome-mcp. The MCP was public and stateless and had
  been deliberately narrowed in September 2026, so the spec put file handling
  and long-running jobs in the chat application and left the five catalogue
  tools where they were.
- Single-cell clustering parameters (`k` for Single Cell Expression Atlas). Not
  supported: the submission carries no parameters (`src/gsa/client.py:292-306`).
- Report generation: on 2026-09-21 `GET /report_status/{id}` returned 404 for a
  completed analysis, so reports were not reachable this way and were left for
  a separate investigation. `src/gsa/` has no report support.

## Open questions (recorded 2026-09-21; no decision recorded since)

- Which method to default to. PADOG is the service's own default, was what was
  measured, and is what the chat uses (`src/gsa/job.py:56-59`); Camera is
  faster. The spec wanted the choice informed by run time on a realistic
  dataset rather than by preference.
- Whether an analysis should survive a chat session ending. Results live
  behind an analysis ID at the service, so resuming is possible; whether it is
  wanted is a product question. Nothing resumes today; if waiting fails, the
  chat gives the user the analysis ID (`src/gsa/chainlit_flow.py:268-276`).

---

## What shipped, 2026-09-22

**User Story 2 (upload) shipped, and Story 3's outcome holds because no summary
was built.** Story 1 (public datasets by identifier) was built underneath --
`submit_public_dataset` works and is tested -- but has no chat route,
because reaching it needs the model to choose a dataset and a factor, which
is a conversation design rather than a plumbing job. Still the case on
2026-10-21.

**As deployed on 2026-09-22 the upload path could not complete.** Driving it
in a browser on 2026-09-25 found four bugs the tests had missed -- among them,
every submission failed because the service answers with a bare text ID, and
the chat UI crashed on delivering the TSV. They were fixed in 6dee089.

Shipping Story 2 before Story 1 inverts the spec's priorities. The reason is
that the upload path turned out to be the *simpler* one: a file arrives with
its own sample names, so there is nothing to search for and nothing to
disambiguate.

**Nothing reaches OpenAI.** Story 3 was written as "a user who declines the
summary still gets their results". What shipped is that with no summary at
all: the result is described from the table and attached as a file, both of
which are the user's own data returning to the user. `Finished.for_model` is
used only to write that description (`src/gsa/chat.py:155-196`) and is never
sent to a model. The position recorded at the time: a summary, if one is
added, should be given the bounded allow-listed view behind an opt-in, not
the result. None has been added.

**FR-006 landed in two places on purpose.** `.chainlit/config.toml` caps
uploads at 20 MB, and `src/gsa/upload.py` caps them again. The first stops
a browser sending the file; the second stops the server accepting it. Only
the second is a guarantee.

## Later changes (2026-09-25 to 2026-10-03)

- **Asking in words** (f615bdd). A request to run a GSA, typed rather than
  attached, gets instructions to attach a matrix, instead of a model answer
  saying the chat cannot do it (`src/gsa/chat.py:199-253`,
  `bin/chat-chainlit.py:753-760`).
- **Labels across messages** (48d2a7d). The first version asked for labels with
  Chainlit's `AskUserMessage`, which a reconnect broke. The matrix now waits in
  `src/gsa/pending.py`, and the next message that reads as labels runs it.
  `run_analysis` (`src/gsa/chainlit_flow.py:110`) is the old single-step flow,
  still exercised by tests but no longer called by the chat.
- **Correctness fixes from review** (b31669d): the first label given is group 1
  (it was the alphabetically first); R `write.table` headers keep their first
  sample; CSV uploads are converted to tabs before submission; files are
  decoded the same way at validation and at submit; polling retries transient
  failures up to four times, and a failure gives the user the analysis ID.
