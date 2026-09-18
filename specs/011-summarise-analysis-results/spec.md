# Feature Specification: Summarise analysis results

**Feature Branch**: `011-summarise-analysis-results`
**Created**: 2026-09-18
**Status**: Draft
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

## User Scenarios & Testing *(mandatory)*

### User Story 1 - What does my result say? (Priority: P1)

A researcher has run a pathway analysis and is looking at the result. They want a
few sentences telling them which pathways came out on top, whether those are
significant once multiple testing is accounted for, and what the top hits have in
common biologically — instead of reading the table themselves.

**Why this priority**: It is the question every user of every analysis type has,
and it is answerable from the result alone. On its own it is a usable feature:
someone with a result gets a readable account of it.

**Independent Test**: Submit a known analysis token, and check the summary names
the pathways the result actually ranks highest, describes their significance in
terms the result supports, and cites each pathway it discusses by stable id.

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

**Independent Test**: Submit a token from an analysis with a deliberate identifier
mismatch and check the summary reports the proportion unmatched and names the
likely cause from the evidence in the result.

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

---

### User Story 3 - What do these numbers mean? (Priority: P3)

A researcher wants the statistics explained in the context of their own result:
what separates the p-value from the FDR, what the found/total ratio implies, and
whether a hit resting on very few entities is worth pursuing.

**Why this priority**: Genuinely useful and the most reusable across analysis
types, but a user can get value from stories 1 and 2 without it, and it is closest
to material the user guide already covers.

**Independent Test**: Request an explanation for a specific pathway in a result and
check the explanation uses that pathway's own numbers rather than generic
definitions.

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

**Independent Test**: Submit an expression result and a species-comparison result,
and check each summary addresses what is specific to that type and neither
describes the other's.

**Acceptance Scenarios**:

1. **Given** an expression result with several columns, **When** a summary is
   requested, **Then** it describes how the highlighted pathways behave across those
   columns rather than treating the result as a single enrichment.
2. **Given** a species-comparison result, **When** a summary is requested, **Then** it
   states that findings are inferred by orthology and what that does not establish.

---

### Edge Cases

- A token that does not exist, or has expired, or belongs to an analysis that has
  been discarded.
- A result too large to summarise in full: hundreds of significant pathways, or an
  expression matrix with many columns.
- A result whose findings are entirely disease pathways, which is often an artefact
  of the submitted list rather than a finding.
- An analysis run against a species the summary's knowledge does not cover well.
- A result already summarised: a reader who reloads should not be told something
  materially different about a fixed artefact (see Assumptions).
- An identifier list that is itself the user's unpublished research data.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST summarise an analysis result the user has already run,
  identified by its analysis token, and MUST NOT run, re-run or re-implement any
  analysis.
- **FR-002**: The system MUST derive every quantitative claim from the analysis
  result itself, and MUST NOT state statistics the result does not contain.
- **FR-003**: The system MUST distinguish significance before and after multiple
  testing correction whenever it describes a pathway as significant.
- **FR-004**: The system MUST report when a result is weak, empty, or heavily
  unmatched, rather than describing the strongest available row as a finding.
- **FR-005**: Every pathway the summary discusses MUST be attributable by Reactome
  stable id, so a reader can open it.
- **FR-006**: The system MUST identify which of the analysis types it is
  summarising, and MUST NOT make claims specific to a type the result is not.
- **FR-007**: The system MUST refuse a request that does not carry a verified
  caller, before any model call, on the same terms as the existing answer endpoint.
  That is the floor, not the bar — see FR-013.
- **FR-008**: The system MUST fail invisibly to the caller: any error, timeout or
  refusal yields a terminal state the caller can render as "no summary", never a
  broken panel or an HTTP error.
- **FR-009**: The system MUST treat an unknown, expired or malformed token as a
  normal negative outcome, not an error condition.
- **FR-010**: The system MUST read analysis results from the beta Analysis Service
  for now, never from production.
- **FR-011**: Summarising MUST be opt-in. The system MUST NOT send any part of an
  analysis result to a model provider until the user has actively asked for a
  summary. A result being viewed is not consent; nothing is summarised in the
  background or in anticipation.
- **FR-012**: The user MUST be offered a choice of what is shared, and the choice
  MUST be meaningful — at least one option MUST produce a useful summary without
  transmitting their submitted identifiers. The user is choosing between summaries
  of different quality at different disclosure, and MUST be told which is which
  before choosing, not after.
- **FR-013**: The system MUST require evidence that a person is present, not merely
  that a known service is calling. This is a stricter bar than the answer
  endpoint's, which verifies caller identity and deliberately asserts nothing about
  humanity, and it exists because this feature discloses a user's own uploaded data
  rather than public pathway text.
- **FR-014**: A summary MUST be stable for a given analysis token: the same token
  MUST yield the same summary on request after request, so that a reader who
  reloads, or who cites it, sees what they saw before. An analysis result is a
  fixed artefact and its summary must behave like one.
- **FR-015**: The system MUST be transparent about what a summary is: that it was
  generated rather than curated, which analysis it describes, and that regenerating
  it may produce different wording. Stability under FR-014 is achieved by reuse,
  not by the generator being deterministic, and the interface MUST NOT imply
  otherwise.

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

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For a result whose top pathways pass FDR correction, the summary names
  the same pathways the result ranks highest, verified against the result across the
  tracked analysis set.
- **SC-002**: For a result where nothing passes correction, the summary says so; it
  never presents the lowest p-value as a finding. Verified with deliberately null
  results.
- **SC-003**: Every pathway named in a summary resolves to a real Reactome stable id
  present in that result — no invented or mismatched identifiers, checked
  mechanically rather than by reading.
- **SC-004**: Zero model calls for requests without a verified caller, measured by
  counting calls under unauthenticated load.
- **SC-005**: An unknown or expired token produces a terminal "no summary" outcome,
  and never an error the caller must special-case.
- **SC-006**: A reader can tell, from the summary alone, whether the result is
  trustworthy enough to act on — assessed by curator review of summaries for a set
  of results chosen to span strong, weak and empty outcomes.

## Assumptions

- The caller has already run the analysis and holds its token; this feature never
  accepts a raw identifier list, which keeps it clear of the Analysis Service's job.
- The caller is the Reactome website, reached through its server-side proxy, and
  presents the same caller token the answer endpoint verifies. No new
  authentication mechanism is introduced.
- Delivery follows the existing answer endpoint's shape — progressive output,
  citations as structured events, failure as a terminal state — because a summary
  takes comparable time to produce and the website already renders that shape.
- Results are read from beta's Analysis Service, consistent with the standing
  instruction to keep off production.
- ReactomeGSA results are recognised from the result, but summarising them is a
  later increment: GSA is a separate service with its own result shape, and
  including it in the first increment would double the surface.
- The measured non-reproducibility of generated answers applies here too, which is
  why FR-012 asks the question rather than assuming stability.

## Scope

**In scope**: summarising a completed analysis result; explaining its statistics;
explaining unmatched identifiers; recognising the analysis type.

**Out of scope**: running analyses; ranking or re-ranking pathways by any measure
the result does not contain; comparing two analyses; storing results; the analysis
UI itself.

## Dependencies

- The Analysis Service on beta, for retrieving results by token.
- The existing caller-token verification and streaming answer surface, reused
  rather than rebuilt.
- ReactomeGSA, only to the extent of recognising that a result came from it.
