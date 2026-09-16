# Feature Specification: Disease and Variant Embeddings

**Feature Branch**: `spec/disease-variant-embeddings`

**Created**: 2026-09-16

**Status**: Draft. Two decisions (D1, D2) for the team.

**Input**: Add disease and variant data to the embeddings, from
`disease_variant_ewas_mapping.tsv` in the release download directory. Do it as a new
collection alongside regenerating the existing ones for Release 97.

## Why: the chatbot answers about diseases, but cannot name a variant

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

## Why this one is worth doing now

It is the only part of the embeddings work that is **not blocked**.

| | source | needs |
|---|---|---|
| `complexes`, `ewas`, `reactions`, `summations` | Neo4j at `bolt://localhost:7687` | credentials we do not have |
| **disease/variant** | **a flat TSV in the download directory** | **nothing** |

`generate_reactome_embeddings` reads Neo4j. `generate_alliance_embeddings` does not --
it reads flat files through `MetaDataCSVLoader` and `build_embeddings`. That is the
precedent to follow, and it means this collection can be built, installed and tested
today while the Release 97 regeneration waits on access.

## The data

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
and should be dropped (`normal_reaction_like_event_go_biological_process_*`, 6.3%);
`entityWithAccessionedSequence_literatureReference_pubMedIdentifier` is 30%.

## Decisions

### D1 -- what goes in the embedded text, and what stays metadata

`MetaDataCSVLoader` takes `content_columns` and `metadata_columns`. Getting this wrong
is how a collection retrieves badly: stable IDs embedded as text add tokens and match
nothing a person types.

Recommendation -- content: `Genename`, `displayName`, `hasModifiedResidue_displayName`,
`disease`, the reaction and pathway `displayName`s, and the *normal* reaction and
pathway `displayName`s. Metadata (filterable, not embedded): every `stable_id`,
`referenceEntity_id`, `disease_identifier`, `cross_reference`, `modifiedResidue_class`,
PubMed identifiers.

The normal-counterpart names belong in content because "what is the healthy version of
this reaction" is a question the chain uniquely answers.

### D2 -- the pipe-delimited `disease` field

33% of rows (2,105) carry more than one disease in one field, up to nineteen:

```
Barrett's esophagus|esophagus squamous cell carcinoma|brain meningioma|...|astrocytoma
```

Left as-is, "968 diseases" is really 443, a search for `melanoma` competes with a
wall of unrelated text in the same document, and the metadata value is unfilterable.

Options: (a) leave as-is, simplest, retrieval suffers on the long ones; (b) split into
a list for metadata, keep the joined string in content; (c) one document per
variant-disease pair, which fixes retrieval but inflates 6,294 rows to ~10,000
documents and repeats the variant text.

Recommendation: **(b)**. It costs nothing at generation time and makes the identifiers
usable, without multiplying near-duplicate documents.

## Scope

In: one new collection built from this one file; wiring it into the retriever
alongside the existing four; answer-sweep expectations that fail if it regresses.

Out: `HumanDiseasePathways.txt` and `Reactome2OMIM.txt` -- not needed for this
(confirmed 2026-09-16); regenerating the four Neo4j-backed collections for Release 97;
publishing to S3, which is blocked separately -- this host's instance profile is
`EC2CloudwatchAgentRole` and `head_bucket` on `download.reactome.org` returns 403.

## How we will know it worked

`src/evaluation/answer_sweep.py` gains expectations that fail today and pass after:

| question | must contain | why |
|---|---|---|
| List the ABCA1 variants in Reactome | `W590S` | today it names none |
| Which diseases involve PTEN variants in Reactome? | `Cowden` | today it answers at pathway level |

Cost is not a blocker: 6,294 short documents on `text-embedding-3-large` is cents.
Disk is not either -- the collection is a fraction of the 3.4 GB the existing four take.
