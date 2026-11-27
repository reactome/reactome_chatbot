# Feature Specification: Three Reactome collections, normal and disease together

**Feature Branch**: `update-reactome-tables`

**Created**: 2026-11-13

**Status**: Built on `update-reactome-tables`, 2026-11-13 to 2026-11-27 (0b5f952 to
b06f046), in an open pull request, not merged. A Release 98 bundle was built from the
new queries and served by a local chatbot on 2026-11-27. The answer sweep and a
retrieval-baseline comparison against the current bundle have not been run, and no new
bundle has been deployed. The new modules were tested with fake Neo4j answers and fake
embeddings, but those tests are not committed (Open items). The requirements below were
set for that work, not for later changes. Current behaviour is in
`src/data_generation/reactome/` (`neo4j_connector.py`, `collection_definitions.py`,
`bundle_generator.py`), `src/util/bundle_manifest.py`, `src/retrievers/csv_chroma.py`,
`src/retrievers/reactome/rag.py` and `src/agent/graph.py:352-399`.

**Input**: Request, 2026-11-13: "right now we are pulling in infomation about
complexes, ewas, summations, events and diseases context - all of which are stored in
seperate chromaDB instances. that's not a good way of providing info to the LLM. what i
want instead is: 1- pathways: Normal + Diseased - both stored in 1 chroma DB 2-
reactions: Normal + Diseased - both stored in 1 chroma DB 3- Complexes and Ewas: Normal +
Diseased - both stored in 1 chroma DB". Revised 2026-11-17: "the disease_variant
collection can be retired. we dont need to route between disease and normal." Revised
2026-11-26: "I want the chatbot to search all three bundles everytime", "instead of it
coming up with 4 different rephrased versions, i want to do 2", and "i only want the top
10 passed to the LLM but i want each of the lists to return 20 docs instead of 10."

## Context (2026-11-13)

The Reactome bundle held five Chroma collections: `reactions`, `summations`,
`complexes` and `ewas` from four Neo4j queries, and `disease_variants` from a release
TSV (spec 008). One record was spread across collections. A reaction's prose sat in
`summations`, apart from its inputs and outputs. Disease reactions and mutant complexes
were in the same collections as normal ones, with nothing labelling them. Variants were
only in `disease_variants`.

The old queries returned one row per reaction and pathway pair, and per complex and
component, so one entity filled several result slots. Vector search over-fetched 4x
and kept the top distinct records to make up for it (`VECTOR_OVERFETCH`,
`src/retrievers/csv_chroma.py:154`). The shipped bundle also stored every reaction
twice, which spent half that over-fetch on copies
(`src/data_generation/reactome/__init__.py:72-75`). Every column was embedded,
identifiers and links included, and lists were written in Python list syntax. A router
narrowed each question to some of the five collections (spec 009).

## User stories (set 2026-11-13, revised 2026-11-17 and 2026-11-26)

### User Story 1 — Three collections, normal and disease together (P1; built)

`pathways`, `reactions` and `entities` replace the five collections. Each is fed by
its own queries, eight in all, and every row carries a `record_type` label such as
"disease reaction". `entities` holds complexes and sets, one row per UniProt protein
listing its disease forms, and Disease Ontology terms. `disease_variants` is retired.

**Check used**: the CSVs were written from Release 98 on 2026-11-27, then built into a
bundle and served locally.

| Collection | Records written from Release 98 |
|---|---|
| pathways | 2,100 normal, 789 disease |
| reactions | 14,365 normal, 2,059 disease |
| entities | 19,383 normal and 2,144 disease complexes or sets, 18,249 proteins, 771 disease terms |

### User Story 2 — Each record is self-contained (P1; built)

One row per entity. A pathway's or reaction's own summations sit in its row. Lists are
plain text. Only names and prose are embedded; identifiers and links are metadata.
Empty fields are left out, and each document reads identity first, prose last.

**Check used**: tests with fake answers, and the Release 98 CSVs read by eye.

### User Story 3 — Proteins and disease terms are cited by their own link (P1; built)

A record with no stable ID is cited by its URL, to UniProt or the Disease Ontology. A
record with a stable ID is still cited by it, as the website contract expects.

**Check used**: tests on the citation builder, not committed. The search page has not
been tried with a mix of both kinds.

### User Story 4 — Every question searches every collection (P1; built)

Collection routing is removed. The router only picks the source: Reactome, the user
guide or live lookups.

**Check used**: CI, and questions asked of the local chatbot.

### User Story 5 — Older bundles are read as before (P1; built)

A bundle's `manifest.json` turns on the new reading: keyword text built exactly as
embedded, no vector over-fetch, and each record's link and accession shown to the
model. A bundle without one, the deployed Reactome bundle and Plant Reactome's, is
read as before.

**Check used**: the existing test suite, on old-layout fixtures. Not run against the
deployed bundle.

### User Story 6 — Deeper search, fewer rephrasings (P2; built, not measured)

Query expansion writes 2 rephrasings instead of 4. Each keyword and vector search keeps
20 results, and 10 per collection still reach the model. This applies to every bundle,
Plant Reactome included.

**Check used**: none beyond tests. On the old bundle, 2 rephrasings cut retrieval from
1.22s to 0.65s but kept 84% of the documents 4 did (2026-09-20,
`specs/010-search-page-answers/research.md:176-191`). Not re-measured with the deeper
search.

## Requirements of this work, and where they stand (checked 2026-11-27)

| ID | Required by the work | Today |
|---|---|---|
| FR-001 | Eight source queries: normal and disease pathways, reactions and complexes; proteins; disease terms | Holds (`SOURCE_QUERIES`, `src/data_generation/reactome/neo4j_connector.py:614`) |
| FR-002 | Every row has its identity and a web link, as single non-blank values. A row with a stable ID links to its Reactome page | Holds (`fetch_source`, `neo4j_connector.py:639`) |
| FR-003 | Proteins are identified by `protein_id` and `url`, disease terms by `display_name` and `url` | Holds (`REQUIRED_COLUMNS_BY_SOURCE`, `neo4j_connector.py:633`) |
| FR-004 | Each collection names its embedded, metadata and model-visible columns. Identifiers and links are never embedded | Holds (`src/data_generation/reactome/collection_definitions.py:65-179`) |
| FR-005 | One CSV per collection. Every source is fetched before any CSV is written. A repeated record, or a column no definition places, stops the build | Holds (`collection_rows`, `write_collection_csvs`, `src/data_generation/reactome/bundle_generator.py:99, 192`) |
| FR-006 | Lists are joined with "; " and prose with a space. Line breaks are removed | Holds (`serialise`, `bundle_generator.py:89`) |
| FR-007 | The release is read from the graph's `DBInfo.releaseNumber`, saved beside the CSVs, and must match the bundle's name | Holds (`fetch_release`, `neo4j_connector.py:734`; `check_release`, `bundle_generator.py:227`) |
| FR-008 | Every CSV is checked against the definitions before anything is embedded | Holds (`check_csv`, `bundle_generator.py:260`) |
| FR-009 | A rebuild replaces each store. The manifest is written last, and a bundle stays marked incomplete until then. An incomplete bundle is refused | Holds (`build_bundle`, `bundle_generator.py:342`; `read_manifest`, `src/util/bundle_manifest.py:47`) |
| FR-010 | `make` for Reactome builds this bundle and reuses existing CSVs unless `--force`. `--disease-variant-tsv` is refused. Plant Reactome keeps the old path | Holds (`bin/embeddings_manager:102-104`; `generate_reactome_bundle`, `bundle_generator.py:382`) |
| FR-011 | The retriever loads a manifest bundle's collections as built. A listed collection with no store stops startup; an unlisted store is ignored | Holds (`collection_layouts`, `load_keyword_documents`, `src/retrievers/csv_chroma.py:265, 299`) |
| FR-012 | Vector search over-fetches only for bundles without a manifest | Holds (`vector_k`, `csv_chroma.py:294`) |
| FR-013 | For a manifest bundle, the model sees each record's `url` and `protein_id` after its text. Neither is embedded | Holds (`show_fields`, `src/retrievers/reactome/rag.py:15`; `create_rag_chain`, `src/retrievers/rag_chain.py:22`) |
| FR-014 | Citations use `st_id`, then the record's `url`, then `source` | Holds (`_citation_for`, `src/agent/graph.py:352`) |
| FR-015 | The router returns only a source. Every Reactome question searches every collection | Holds (`QueryIntent`, `src/agent/tasks/intent_classifier.py:85`; `retrieve_documents`, `csv_chroma.py:501`) |
| FR-016 | 2 rephrasings, and the prompt asks for exactly the configured number | Holds (`DEFAULT_ALTERNATES`, `multi_query_prompt`, `csv_chroma.py:86, 113`) |
| FR-017 | 20 results per search; 10 per collection reach the model | Holds (`csv_chroma.py:147, 158`) |
| FR-018 | The answer sweep's variant checks run on either layout | Holds (`src/evaluation/answer_sweep.py:65, 138, 148`) |
| FR-019 | `retrieval_baseline` loads and searches as the chatbot does, and refuses to compare captures from different layouts or depths | Holds (`bin/retrieval_baseline:92, 143, 327-334`) |
| FR-020 | The CSV loader keeps empty fields unless asked not to | Holds (`skip_empty_fields`, `src/data_generation/metadata_csv_loader.py:40`) |

## Decisions (made by the requester)

- **D1, 2026-11-13.** A summation sits inside its pathway or reaction row, so each
  record is self-contained.
- **D2, 2026-11-17.** `disease_variants` is retired. Variants come from the protein
  rows.
- **D3, 2026-11-17.** Disease terms join `entities`. They have no stable ID, so they are
  identified and cited by their link.
- **D4, 2026-11-17.** No routing between normal and disease records. Ranking is left to
  handle it.
- **D5, 2026-11-25.** One protein query, one row per UniProt record, its disease forms
  listed on the row. Pairing each variant with its disease, and isoforms, are left as
  they are.
- **D6, 2026-11-25.** Every query returns its own link, and the fetch checks it. One
  name per idea across queries: `failure_reason`, `disrupted_in_diseases`, `summation`.
- **D7, 2026-11-25.** Identifiers and links are metadata. `url` and `protein_id` are
  shown to the model, not embedded.
- **D8, 2026-11-26.** No guide to record types in the answer prompt: "the data itself
  is clear enough".
- **D9, 2026-11-26.** Collection routing is removed.
- **D10, 2026-11-26.** The retrieval tuning applies to every bundle, Plant Reactome
  included.

## Out of scope (set 2026-11-17 to 2026-11-26)

- Plant Reactome's generation. It keeps the old path.
- Records of their own for small molecules, drugs and pathogen proteins. No old
  collection had them either.
- Telling complexes from sets, capping long lists, stripping HTML from summations,
  marking inferred reactions, and splitting regulators by type.

## Open items (recorded 2026-11-27)

- **Not measured.** Neither the answer sweep nor a retrieval-baseline capture has been
  run on Release 98, so there is no before-and-after for these retrieval changes.
  Stories 4 and 6 change answers on deploy, even before a new bundle ships.
- **Variant answers are unchecked.** A protein row lists variant names and disease
  names separately. The sweep's ABCA1 and PTEN checks have not been run on it.
- **The beta deploy script** still runs `bin/routing-probe`, now a stub that always
  passes. Remove that step from `~/update-beta-chat.sh`, then the stub.
- **Tests are not committed** for the queries' checks, the bundle build, the manifest,
  citation by link or the loader option. CI covers the retrieval changes only.
- **The website contract** section on link citations, in
  `specs/010-search-page-answers/contracts/answer_endpoint.md`, is written but not
  committed.
- **The first local message is lost.** Under `chainlit run`, the graph is built inside
  the first message: 58.7s for Release 98, against about 85s for the old bundle
  (`src/agent/registry.py:55`). That freezes the server, the browser reconnects, and
  the answer goes to the old session. The deployed app builds at startup
  (`bin/chat-fastapi.py:60`), so production is unaffected.
- **Query values not yet checked:** what `failure_reason` holds, what regulation names
  hold, the five-child pathway preview with no total, sets without a species being
  dropped, and participant ID and name lists that can drift apart.
- **Cleanup:** the old generation path and `csv_generator.py`, still imported for Plant
  Reactome; the `disease_variant` module, once the new bundle is live; and the unused
  link branch in `csv_generator.py`.

## What was built (committed 2026-11-26)

- **Queries and fetch checks** (`neo4j_connector.py`). The old four queries stay, for
  Plant Reactome's import chain.
- **Collection definitions** (`collection_definitions.py`): sources, labels, and where
  each column goes.
- **The bundle generator** (`bundle_generator.py`): CSVs, checks, stores and manifest,
  in that order. `python -m data_generation.reactome.bundle_generator <bundle-dir>`
  writes the CSVs alone, for reading before anything is embedded.
- **The manifest** (`src/util/bundle_manifest.py`): its format and reader, shared by the
  build, the retriever and the answer chain.
- **The retriever** (`csv_chroma.py`): manifest-aware loading, shared with
  `retrieval_baseline`; deeper search; fewer rephrasings, read once per question.
- **The answer chain** (`rag_chain.py`, `reactome/rag.py`): an optional function that
  writes each document into the prompt, used to show the link and accession.
- **Citations** (`graph.py`): the link step.
- **Deleted:** collection routing (the selection variable, its resolver, the router's
  collection rules and output field), the routing probe and its tests, and the
  collection descriptions only the router read.
- **Also changed:** one release-name parser (`release_from_name`,
  `src/util/embedding_environment.py:8`); `csv_generator.py` marked obsolete for
  Reactome; the CSV loader's opt-in `skip_empty_fields`.
