"""The disease_variants collection, built from a release file rather than Neo4j."""

import csv
from pathlib import Path

import pytest

from data_generation.disease_variant import (
    COLUMNS,
    CONTENT_COLUMNS,
    METADATA_COLUMNS,
    _tidy,
    write_csv,
)
from data_generation.metadata_csv_loader import MetaDataCSVLoader

ROW = {
    "Genename": "ABCA1",
    "displayName": "ABCA1 W590S [plasma membrane]",
    "stable_id": "R-HSA-5682201",
    "referenceEntity_name": "ABCA1",
    "referenceEntity_id": "UniProt:O95477",
    "hasModifiedResidue_displayName": "L-tryptophan 590 replaced with L-serine",
    "modifiedResidue_class": "ReplacedResidue",
    "cross_reference": "Mondo:0008783",
    "disease": "Tangier disease",
    "disease_identifier": "DOID:1388",
    "entityWithAccessionedSequence_literatureReference_pubMedIdentifier": "",
    "first_entitySet": "",
    "entityWithAccessionedSequence_reactionLikeEvent_stable_id": "R-HSA-5682111",
    "entityWithAccessionedSequence_reactionLikeEvent_displayName": "Defective ABCA1 does not transport CHOL",
    "entityWithAccessionedSequence_reactionLikeEvent_entityFunctionalStatus_functionalStatus_functionalStatusType_displayName": "loss_of_function",
    "reactionLikeEvent_literatureReference_pubMedIdentifier": "pubmed:12509412",
    "entityWithAccessionedSequence_pathway_stable_id": "R-HSA-5682113",
    "entityWithAccessionedSequence_pathway_displayName": "Defective ABCA1 causes TGD",
    "entityWithAccessionedSequence_reactionLikeEvent_normalReaction_stable_id": "R-HSA-216723",
    "entityWithAccessionedSequence_reactionLikeEvent_normalReaction_displayName": "ABCA1 tetramer transports CHOL",
    "entityWithAccessionedSequence_pathway_normalPathway_displayName": "Plasma lipoprotein assembly",
    "entityWithAccessionedSequence_pathway_normalPathway_stable_id": "R-HSA-174824",
    "normal_reaction_like_event_go_biological_process_accession": "",
    "normal_reaction_like_event_go_biological_process_displayName": "",
    "entityWithAccessionedSequence_pathway_normalPathway_goBiologicalProcess_accession": "GO:0071827",
    "entityWithAccessionedSequence_pathway_normalPathway_goBiologicalProcess_displayName": "plasma lipoprotein particle organization",
}


def _tsv(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    path = tmp_path / "mapping.tsv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ROW), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_the_release_columns_are_renamed_to_readable_ones(tmp_path: Path) -> None:
    # The loader embeds each field as "name: value", so the column names are
    # themselves embedded. The release names are query paths up to 104 chars.
    out = tmp_path / "csv_files" / "disease_variants.csv"
    assert write_csv(_tsv(tmp_path, [ROW]), out) == 1
    header = out.read_text().splitlines()[0]
    assert "entityWithAccessionedSequence" not in header
    assert "residue_change" in header
    assert "normal_reaction" in header


def test_identifiers_are_metadata_and_never_embedded(tmp_path: Path) -> None:
    # Nobody types R-HSA-5682201 at a chatbot, and embedding it costs tokens in
    # every document.
    out = tmp_path / "csv_files" / "disease_variants.csv"
    write_csv(_tsv(tmp_path, [ROW]), out)
    (doc,) = MetaDataCSVLoader(
        file_path=str(out),
        content_columns=CONTENT_COLUMNS,
        metadata_columns=METADATA_COLUMNS,
        encoding="utf-8",
    ).load()

    assert "R-HSA-5682201" not in doc.page_content
    assert "UniProt:O95477" not in doc.page_content
    assert doc.metadata["st_id"] == "R-HSA-5682201"
    # csv_chroma de-duplicates on metadata["st_id"]; a different key there
    # would silently stop that working for this collection.
    assert "st_id" in METADATA_COLUMNS

    assert "ABCA1 W590S [plasma membrane]" in doc.page_content
    assert "L-tryptophan 590 replaced with L-serine" in doc.page_content
    assert "Tangier disease" in doc.page_content
    # The normal counterpart is content, because "what is the healthy version
    # of this" is a question only this chain answers.
    assert "ABCA1 tetramer transports CHOL" in doc.page_content


def test_multiple_diseases_stay_in_one_document(tmp_path: Path) -> None:
    # Fanning out to one document per variant-disease pair would repeat
    # p16INK4A R80* forty-five times, and that many near-identical documents
    # can fill a whole result set.
    row = dict(ROW, disease="Cowden syndrome|breast cancer|melanoma")
    out = tmp_path / "csv_files" / "disease_variants.csv"
    assert write_csv(_tsv(tmp_path, [row]), out) == 1
    (doc,) = MetaDataCSVLoader(
        file_path=str(out),
        content_columns=CONTENT_COLUMNS,
        metadata_columns=METADATA_COLUMNS,
        encoding="utf-8",
    ).load()
    assert "Cowden syndrome, breast cancer, melanoma" in doc.page_content


def test_tidy_normalises_the_pipe_separator() -> None:
    assert _tidy("a|b|c") == "a, b, c"
    assert _tidy("") == ""
    assert _tidy("only") == "only"
    assert _tidy("a||b") == "a, b"


def test_a_changed_release_file_fails_loudly(tmp_path: Path) -> None:
    # Silently writing empty columns would produce a bundle that embeds
    # nothing useful and reports no error.
    path = tmp_path / "mapping.tsv"
    path.write_text("Genename\tdisease\nABCA1\tTangier disease\n")
    with pytest.raises(ValueError, match="missing expected columns"):
        write_csv(path, tmp_path / "out.csv")


def test_every_renamed_column_is_used_somewhere() -> None:
    # A column renamed but left out of both lists is silently dropped.
    used = set(CONTENT_COLUMNS) | set(METADATA_COLUMNS)
    assert set(COLUMNS.values()) - used == set()
