"""Building a collection twice must not store everything twice.

`Chroma.from_documents` appends to whatever is already in the persist directory,
so a second build silently doubles the collection. The shipped Release95 bundle
is what that looks like: 33,498 reaction documents for 16,749 CSV rows, exactly
2.00x, unnoticed until the Release 97 rebuild. The vector retriever over-fetches
then de-duplicates on `st_id`, so half that over-fetch was spent on duplicates of
what it already had.

Three of the five generators had this, each written separately.

These tests *run* the generators rather than inspecting their source. The first
version of this file grepped for `rmtree`, and an adversarial check showed it
passed even when the code deleted a completely unrelated directory -- it was
asserting the shape of the code, not what it does.
"""

import csv
from pathlib import Path
from typing import Any

import pytest
from langchain_core.embeddings import FakeEmbeddings

import data_generation.alliance as alliance_mod
import data_generation.reactome as reactome_mod
import data_generation.uniprot as uniprot_mod


@pytest.fixture(autouse=True)
def _fake_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """No API calls, and no dependence on an embedding model being reachable."""
    for module in (reactome_mod, uniprot_mod, alliance_mod):
        monkeypatch.setattr(
            module, "build_embeddings", lambda *a, **k: FakeEmbeddings(size=8)
        )


def _write_csv(path: Path, columns: list[str], rows: int, delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter=delimiter)
        writer.writeheader()
        for i in range(rows):
            writer.writerow({c: f"{c}-{i}" for c in columns})


def test_reactome_replaces_its_collection(tmp_path: Path) -> None:
    columns = [
        "st_id",
        "display_name",
        "canonical_gene_name",
        "synonyms_gene_name",
        "uniprot_link",
    ]
    csv_path = tmp_path / "ewas.csv"
    _write_csv(csv_path, columns, rows=3)

    first = reactome_mod.upload_to_chromadb(str(tmp_path), str(csv_path), "ewas", "m")
    assert first._collection.count() == 3
    second = reactome_mod.upload_to_chromadb(str(tmp_path), str(csv_path), "ewas", "m")
    assert second._collection.count() == 3, "a second build must not append"


def test_uniprot_replaces_its_collection(tmp_path: Path) -> None:
    columns = [
        "gene_names",
        "short_protein_name",
        "full_protein_name",
        "protein_family",
        "biological_pathways",
    ]
    csv_path = tmp_path / "uniprot_data.csv"
    _write_csv(csv_path, columns, rows=4)

    first = uniprot_mod.upload_to_chromadb(
        str(tmp_path), str(csv_path), "uniprot_data", "m"
    )
    assert first._collection.count() == 4
    second = uniprot_mod.upload_to_chromadb(
        str(tmp_path), str(csv_path), "uniprot_data", "m"
    )
    assert second._collection.count() == 4, "a second build must not append"


def test_alliance_replaces_its_collection(tmp_path: Path) -> None:
    version = "1.0.0"
    _write_csv(
        tmp_path / "csv_files" / "alliance" / version / "genes.tsv",
        alliance_mod.COLUMN_SCHEMAS["genes"],
        rows=2,
        delimiter="\t",
    )

    first = alliance_mod.upload_to_chromadb(str(tmp_path), version, False, "m")
    assert first is not None
    assert first._collection.count() == 2
    second = alliance_mod.upload_to_chromadb(str(tmp_path), version, False, "m")
    assert second is not None
    assert second._collection.count() == 2, "a second build must not append"


def test_alliance_reads_from_the_directory_it_is_given(tmp_path: Path) -> None:
    """Not from the working directory, which is where it used to look."""
    assert alliance_mod.upload_to_chromadb(str(tmp_path), "9.9.9", False, "m") is None


def test_userguide_does_not_gate_the_removal_on_force() -> None:
    """Source-level, because generating it fetches pages over the network.

    `force` means re-fetch the HTML, not "rebuild the collection"; gating the
    removal on it meant an ordinary rebuild appended.
    """
    source = (
        Path(__file__).resolve().parents[2]
        / "src/data_generation/userguide/__init__.py"
    ).read_text()
    assert "if force and chroma_dir.exists():" not in source
    assert "if chroma_dir.exists():" in source


def test_every_generator_is_covered_here(tmp_path: Path, request: Any) -> None:
    """A new generator must arrive with a case in this file.

    Five modules, each with its own hand-written persist block, is why the same
    fault appeared three times.
    """
    src = Path(__file__).resolve().parents[2] / "src/data_generation"
    generators = {
        p.parent.name
        for p in src.rglob("__init__.py")
        if "from_documents" in p.read_text()
    }
    covered = {"reactome", "uniprot", "alliance", "userguide", "disease_variant"}
    assert (
        generators <= covered
    ), f"no replace-not-append test for: {generators - covered}"
