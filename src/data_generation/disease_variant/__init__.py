"""Disease variants: the collection built from a file rather than from Neo4j.

Asked to list the ABCA1 variants in Reactome, the chatbot answered "Defective
ABCA1 causes Tangier Disease" and named none of the six. The four Neo4j-backed
collections hold pathway- and reaction-level prose about disease, so it talks
about disease fluently and has no document for the variant itself.

`disease_variant_ewas_mapping.tsv` in the release download directory has all of
them -- 6,294 variants over 400 genes -- each with the residue change already in
prose, the disease with its Mondo and DOID identifiers, and the normal reaction
the defective one replaces. It needs no database, which is also why this
collection can be built when the others cannot.
"""

import csv
import os
from pathlib import Path

from langchain_chroma import Chroma

from data_generation.embeddings import build_embeddings
from data_generation.metadata_csv_loader import MetaDataCSVLoader

COLLECTION = "disease_variants"

# The source column names are the query paths that produced them, up to 104
# characters of `entityWithAccessionedSequence_reactionLikeEvent_...`. That
# matters because the loader embeds each field as "name: value", so the column
# names are themselves embedded: left alone they would contribute more tokens
# than the values, identically in every document. These are the names a person
# would use.
COLUMNS: dict[str, str] = {
    "Genename": "gene",
    "displayName": "variant",
    "referenceEntity_name": "protein",
    "hasModifiedResidue_displayName": "residue_change",
    "modifiedResidue_class": "mutation_type",
    "disease": "disease",
    "entityWithAccessionedSequence_reactionLikeEvent_displayName": "reaction",
    "entityWithAccessionedSequence_reactionLikeEvent_entityFunctionalStatus_functionalStatus_functionalStatusType_displayName": "functional_status",
    "entityWithAccessionedSequence_pathway_displayName": "disease_pathway",
    "entityWithAccessionedSequence_reactionLikeEvent_normalReaction_displayName": "normal_reaction",
    "entityWithAccessionedSequence_pathway_normalPathway_displayName": "normal_pathway",
    "entityWithAccessionedSequence_pathway_normalPathway_goBiologicalProcess_displayName": "normal_process",
    # Identifiers. Kept out of the embedded text -- nobody types R-HSA-5682201
    # at a chatbot, and embedding it costs tokens in every document.
    "stable_id": "st_id",
    "referenceEntity_id": "uniprot_id",
    "cross_reference": "disease_cross_reference",
    "disease_identifier": "disease_id",
    "entityWithAccessionedSequence_reactionLikeEvent_stable_id": "reaction_id",
    "entityWithAccessionedSequence_pathway_stable_id": "disease_pathway_id",
    "entityWithAccessionedSequence_reactionLikeEvent_normalReaction_stable_id": "normal_reaction_id",
    "entityWithAccessionedSequence_pathway_normalPathway_stable_id": "normal_pathway_id",
    "reactionLikeEvent_literatureReference_pubMedIdentifier": "reaction_pubmed_ids",
}

# What a question is actually about: names, the change in prose, the disease,
# and what the variant breaks.
CONTENT_COLUMNS: list[str] = [
    "gene",
    "variant",
    "protein",
    "residue_change",
    "mutation_type",
    "disease",
    "reaction",
    "functional_status",
    "disease_pathway",
    "normal_reaction",
    "normal_pathway",
    "normal_process",
]

# Filterable, and carried into the answer so a citation can be made. `st_id` is
# named for the key `csv_chroma` de-duplicates on.
METADATA_COLUMNS: list[str] = [
    "st_id",
    "gene",
    "protein",
    "uniprot_id",
    "disease",
    "disease_id",
    "disease_cross_reference",
    "mutation_type",
    "reaction_id",
    "disease_pathway_id",
    "normal_reaction_id",
    "normal_pathway_id",
    "reaction_pubmed_ids",
]

# 6.3% populated. A column that is empty in 94% of documents earns nothing and
# costs a line of "name: " in every one of them.
DROPPED = (
    "normal_reaction_like_event_go_biological_process_accession",
    "normal_reaction_like_event_go_biological_process_displayName",
    "entityWithAccessionedSequence_pathway_normalPathway_goBiologicalProcess_accession",
    "entityWithAccessionedSequence_literatureReference_pubMedIdentifier",
    "first_entitySet",
)


def _tidy(value: str) -> str:
    """One row's field, made readable.

    A third of rows pack several diseases into one pipe-delimited string, up to
    nineteen of them. They stay in one document -- fanning out to one document
    per variant-disease pair would repeat `p16INK4A R80*` forty-five times, and
    forty-five near-identical documents can fill a whole result set. The
    separator becomes a comma so it reads as a list rather than a path.
    """
    return ", ".join(part.strip() for part in value.split("|") if part.strip())


def write_csv(tsv_path: Path, csv_path: Path) -> int:
    """Rewrite the release TSV as the CSV both retrievers read. Returns rows."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tsv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    missing = set(COLUMNS) - set(rows[0]) if rows else set(COLUMNS)
    if missing:
        raise ValueError(
            f"{tsv_path} is missing expected columns: {sorted(missing)}. "
            "The release file's shape has changed; update COLUMNS."
        )

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS.values()))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {new: _tidy(row[old] or "") for old, new in COLUMNS.items()}
            )
    return len(rows)


def generate_disease_variant_embeddings(
    embeddings_dir: str,
    tsv_path: str | Path,
    hf_model: str | None = None,
    device: str | None = None,
) -> Chroma:
    """Build the disease_variants collection inside an existing bundle."""
    bundle = Path(embeddings_dir)
    csv_path = bundle / "csv_files" / f"{COLLECTION}.csv"
    count = write_csv(Path(tsv_path), csv_path)
    print(f"  wrote {csv_path} ({count} variants)")

    loader = MetaDataCSVLoader(
        file_path=str(csv_path),
        content_columns=CONTENT_COLUMNS,
        metadata_columns=METADATA_COLUMNS,
        encoding="utf-8",
    )
    docs = loader.load()
    print(f"  loaded {len(docs)} documents")

    return Chroma.from_documents(
        documents=docs,
        embedding=build_embeddings(hf_model, device, chunk_size=400),
        persist_directory=os.path.join(embeddings_dir, COLLECTION),
    )
