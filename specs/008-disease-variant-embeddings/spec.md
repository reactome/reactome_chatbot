# Feature Specification: Disease and Variant Embeddings

**Feature Branch**: `spec/disease-variant-embeddings`

**Created**: 2026-09-16

**Status**: Historical record of the disease-variant collection work, 2026-09-16 to
2026-09-17. Built 2026-09-16 (2855de4, review fixes 7b733ea) and merged in #225 on
2026-09-17. The two decisions were delegated to the implementer, who recorded them
below; one original recommendation was withdrawn as impossible. They were made for
that work, not for later changes. Current behaviour is in
`src/data_generation/disease_variant/__init__.py`, with the retrieval-side handling in
`src/retrievers/csv_chroma.py:35-39`.

**Input**: Add disease and variant data to the embeddings, from
`disease_variant_ewas_mapping.tsv` in the release download directory. Do it as a new
collection alongside regenerating the existing ones for Release 97.

## Why (2026-09-16): the chatbot answered about diseases, but could not name a variant

Asked on 2026-09-16 against beta running `a4de878`, with live MCP:

> **List the ABCA1 variants in Reactome and the disease each one causes.**
>
> "The Reactome database lists the following disease associated with ABCA1 variants:
> **Defective ABCA1 causes Tangier Disease (TGD)** ... This is the primary disease
> explicitly mentioned in the context of ABCA1 variants. Other diseases may be
> associated with ABCA1, bu[t] ..."

Zero variants named. Reactome curates **six** ABCA1 variants, and the file has them:
`ABCA1 W590S`, `N935S`, `C1417R` and three more, each with its stable ID and the
residue change in prose ("L-tryptophan 590 replaced with L-serine").

> **Which diseases involve variants of the PTEN gene in Reactome?**
>
> "... **PTEN Loss of Function in Cancer**: This pathway highlights ..."

Answered at pathway level. Reactome curates **108 PTEN variants across 86 diseases**.

The pattern is consistent: the four existing collections (`complexes`, `ewas`,
`reactions`, `summations`) hold pathway- and reaction-level prose about disease, so
the chatbot says *"defective ABCA1 causes Tangier disease"* fluently. What it has no
document for is the individual variant entity. It therefore answers a level up from
the question and hedges, which is the failure mode this project keeps finding: a
fluent answer that does not contain the fact asked for.

## Why it could be done first (2026-09-16)

It was the only part of the embeddings work that was **not blocked**.

| | source | needs |
|---|---|---|
| `complexes`, `ewas`, `reactions`, `summations` | Neo4j at `bolt://localhost:7687` | credentials not available at the time |
| **disease/variant** | **a flat TSV in the download directory** | **nothing** |

`generate_reactome_embeddings` reads Neo4j. `generate_alliance_embeddings` does not --
it reads flat files through `MetaDataCSVLoader` and `build_embeddings`. That was the
precedent followed, so this collection could be built, installed and tested while the
Release 97 regeneration of the other four waited on access. This spec does not record
whether that regeneration has since been done.

## The data (Release 97 file, examined 2026-09-16)

`.../static/download/97/disease_variant_ewas_mapping.tsv`, 5.3 MB, **6,294 rows**,
26 columns, one row per variant entity (6,294 distinct `stable_id`, so no duplication).

- **400** genes, **462** disease pathways
- **443** distinct diseases -- *not* the 968 distinct `disease` strings, see D2
- Best covered: cancer (1,700 variants), Kabuki syndrome (564), acute myeloid
  leukaemia (126), ornithine carbamoyltransferase deficiency (104)

Each row already carries the chain a user actually asks about: gene, variant display
name, the residue change in prose, the disease with Mondo/DOID identifiers, the
reaction the variant takes part in, its functional status, **and the normal reaction
and pathway it is the defective counterpart of**.

Fill rates are high: 20 of 26 columns are ≥95% populated. Two are effectively empty
(`normal_reaction_like_event_go_biological_process_*`, 6.3%), and
`entityWithAccessionedSequence_literatureReference_pubMedIdentifier` is 30%. All three
are dropped (see D1).

## Decisions

Both were delegated: *"I think you should decide on the columns ... make your best
design and go ahead."*

### D1 -- what is embedded, and what is metadata (decided by the implementer, 2026-09-16)

The loader renders each field as `name: value`, so the **column names are embedded
too**. The release names are the query paths that produced them, up to 104 characters
of `entityWithAccessionedSequence_reactionLikeEvent_entityFunctionalStatus_...`. Left
alone they would contribute more tokens than the values, identically in every
document. They are renamed to what a person would call them.

Embedded: `gene`, `variant`, `protein`, `residue_change`, `mutation_type`, `disease`,
`reaction`, `functional_status`, `disease_pathway`, `normal_reaction`,
`normal_pathway`, `normal_process`. A document reads:

```
gene: ABCA1
variant: ABCA1 W590S [plasma membrane]
residue_change: L-tryptophan 590 replaced with L-serine
mutation_type: ReplacedResidue
disease: Tangier disease
reaction: Defective ABCA1 does not transport CHOL from transport vesicle membrane...
functional_status: loss_of_function
normal_reaction: 4xPALM-C-p-2S-ABCA1 tetramer transports CHOL from transport vesicle...
```

Metadata, not embedded: every identifier -- `st_id`, `uniprot_id`, `disease_id`,
`cross_references`, the four Reactome stable IDs and `reaction_pubmed_ids`. The
metadata also repeats `gene`, `protein`, `disease` and `mutation_type`, which are
embedded too (`METADATA_COLUMNS`, `src/data_generation/disease_variant/__init__.py:86-100`).
`cross_references` was first named `disease_cross_reference`. It was renamed in
7b733ea because it also carries COSMIC, ClinVar, ClinGen and LOVD *variant* ids
(`:51-55`). Nobody types
`R-HSA-5682201` at a chatbot, and embedding it costs tokens in every document. The
variant's own identifier is named **`st_id`** because that is the key `csv_chroma`
de-duplicates on; a different name would silently disable de-duplication here.

Dropped: five of the 26 columns -- the two `normal_reaction_like_event_go_biological_process_*`
columns at 6.3% fill; `entityWithAccessionedSequence_pathway_normalPathway_goBiologicalProcess_accession`
(the process name is kept as `normal_process`); the entity-level
`entityWithAccessionedSequence_literatureReference_pubMedIdentifier` (30% fill); and
`first_entitySet` (`DROPPED`, `__init__.py:102-110`; the CSV keeps only the 21 columns
named in `COLUMNS`, `:36-65`). The recorded reason covers the sparse ones: a column
empty in 94% of documents earns nothing and costs a line of `name:` in every one. No
reason is recorded for the other three.

Median document: 556 characters, about 140 tokens (measured 2026-09-16).

The keyword (BM25) index has used the same 12 content columns since f0e8581
(2026-10-04). Before that it indexed all 21 CSV columns, so BM25 and vector results
for the same variant never had the same text, and RRF, which keys on the text, could
not fuse them (`src/retrievers/csv_chroma.py:35-39`, `:447-463`).

### D2 -- the pipe-delimited `disease` field (decided by the implementer, 2026-09-16; the earlier recommendation withdrawn)

This spec first recommended splitting the field into a list for metadata. **That is
not possible**: Chroma accepts only `str`, `int`, `float` or `bool` as a metadata
value and rejects a list outright.

Fanning out to one document per variant-disease pair was measured rather than
estimated: 6,294 documents become **10,500** (+67%), and `p16INK4A R80*` would be
repeated **45 times**. Forty-five near-identical documents can fill an entire result
set, which is a worse failure than a long disease string.

Decided: **one document per variant**, with `|` rewritten to `, ` so the field reads
as a list rather than a path. All disease names stay searchable in the content, and
the metadata value stays a string Chroma will accept.

## Scope of this work

In: one new collection built from this one file; wiring it into the retriever
alongside the existing four; answer-sweep expectations that fail if it regresses.

Out: `HumanDiseasePathways.txt` and `Reactome2OMIM.txt` -- not needed for this
(confirmed 2026-09-16); regenerating the four Neo4j-backed collections for Release 97;
publishing to S3, which was blocked separately at the time -- the build host's instance
profile was `EC2CloudwatchAgentRole` and `head_bucket` on `download.reactome.org`
returned 403.

## Result, measured (2026-09-16, before the BM25 column fix noted under D1)

Built into a scratch bundle and asked through the real retriever:

| | before | after |
|---|---|---|
| ABCA1 | *"Defective ABCA1 causes Tangier Disease ... Other diseases may be associated with ABCA1, bu[t]"* -- no variant named | **C1417R, Q537R, N935S, R587W, S1446L**, each with its disease and loss-of-function status |
| PTEN | *"PTEN Loss of Function in Cancer"* -- a pathway | **Q17\*, Q97\*, Q171\***, named against endometrial cancer |

Two answer-sweep expectations pin this. They match a *pattern* for a named variant
rather than a specific one, because which of the six come back depends on retrieval
order and pinning one would fail a good answer -- the mistake made in `007` and fixed
there. The pattern was first `\bABCA1 [A-Z]\d{2,4}[A-Z*]`, checked against the
recorded answers on 2026-09-16. On 2026-09-19 (7598b17) it failed a correct answer that
listed the variants as `**C1417R**` without the gene beside them. Both expectations
now use `\b[A-Z]\d{2,4}[A-Z*]\b`, anchored to the topic by `must=("Tangier",)` and
`must=("PTEN",)` (`src/evaluation/answer_sweep.py:125-148`; pinned by
`tests/evaluation/test_answer_sweep.py:244`).

They carry `needs_collection="disease_variants"` and are skipped, with a reason, when
the installed bundle has no such collection (`answer_sweep.py:266-285`), rather than
turning the deploy gate red for a reason nobody can act on. A test covers the
direction that matters: that they run once it is installed
(`tests/evaluation/test_answer_sweep.py:209`).

Cost was cents. The collection is a fraction of the 3.4 GB the existing four take.

## Installing it (beta host, as found 2026-09-16)

The embeddings tree on that host was root-owned, so installing needed sudo -- and
using sudo kept it root-owned. It was not the container's doing: the image has run as
`appuser` (uid 3001) since 2025-04-17 and the bundle was created 2026-09-02, so
something had been run under sudo on the host. A host-side script,
`~/fix-embeddings-ownership.sh` (not in this repository), set the tree to the
operator's own user, group `reactome`, with world-read. That suits both: the owner can
manage bundles without sudo, and the container, whose uid is not a host user and which
only reads at runtime, still can. The host's current state is not recorded here.
