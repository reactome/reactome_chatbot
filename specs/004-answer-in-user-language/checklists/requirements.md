# Specification Quality Checklist: Answer in the User's Language

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-09-10
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Success criteria are technology-agnostic
- [x] All acceptance scenarios are defined
- [x] Edge cases are identified
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria
- [x] User scenarios cover primary flows
- [x] Feature meets measurable outcomes defined in Success Criteria
- [x] No implementation details leak into specification

## Adversarial review of this specification

Run before trusting it. Two load-bearing claims were checked against the corpus
rather than reasoned about, and **one of them was wrong**.

| claim as first written | verdict |
|---|---|
| "BM25, which is lexical, retrieves close to nothing for a French query" | **wrong.** It returns a full ten documents; named entities survive translation. The real cost is that 7 of 10 differ. Corrected, with the measurement. |
| "#140's appended instruction pollutes retrieval" | **right, and understated.** Measured: 0/10, 1/10 and 0/10 documents survive. It replaces lexical retrieval rather than degrading it. |
| "#140 targets React-to-Me" | right, but it edits `call_model`, which main renamed to `generate_answer`. Added. |
| "Plant Reactome's answer does not receive the language" | verified in `plantreactome.py`. |
| "Cross-Database's summary does" | verified in `cross_database.py`. |
| "#125 keeps the translate-to-English step" | verified in its diff. The obvious mistake, not made. |

The wrong claim mattered: it would have justified FR-004 with a reason a reader could
disprove in five minutes, which is worse than justifying it with the smaller, true
one.

## Notes

Deviations from the template, deliberate:

1. **Two contributed PRs are reviewed in the spec body.** They are the starting
   material and the reason to reject each is inseparable from the reason to read it.
2. **Measurements appear in a specification.** Constitution Article II says retrieval
   claims come with numbers. The numbers are what make the choice between the two
   approaches obvious rather than aesthetic.
