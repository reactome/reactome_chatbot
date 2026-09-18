"""BM25 documents must carry the same metadata the embedded ones do.

The ensemble retriever loads each CSV twice over: Chroma holds documents embedded
with `MetaDataCSVLoader`, and BM25 is built by loading the same CSV at startup.
When the second used a plain `CSVLoader`, its documents had metadata of only
{row, source} -- so a document retrieved by keyword could not be cited, even
though its stable id was sitting in the page content where nothing could reach
it. Measured on one question: 50 of 62 documents the citation path walked were
unattributable.
"""

from pathlib import Path

import pytest

from data_generation.metadata_csv_loader import MetaDataCSVLoader
from retrievers.csv_chroma import _csv_column_names

HEADER = "st_id,display_name,pathway_name,url"
ROW = "R-HSA-8863013,CDK5 binds p25,Neuronal System,https://reactome.org/x"


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    path = tmp_path / "reactions.csv"
    path.write_text(f"{HEADER}\n{ROW}\n")
    return path


def test_every_column_is_read_from_the_header(csv_file: Path) -> None:
    assert _csv_column_names(csv_file) == [
        "st_id",
        "display_name",
        "pathway_name",
        "url",
    ]


def test_a_missing_or_empty_file_asks_for_no_metadata(tmp_path: Path) -> None:
    """An empty list is falsy, so the loader behaves as it did before."""
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    assert _csv_column_names(empty) == []


def test_loaded_documents_carry_the_stable_id(csv_file: Path) -> None:
    """The property the citation path depends on."""
    documents = MetaDataCSVLoader(
        file_path=str(csv_file), metadata_columns=_csv_column_names(csv_file)
    ).load()

    assert len(documents) == 1
    assert documents[0].metadata["st_id"] == "R-HSA-8863013"
    assert documents[0].metadata["display_name"] == "CDK5 binds p25"


def test_the_content_is_not_trimmed_by_promoting_columns(csv_file: Path) -> None:
    """BM25 scores page_content, so changing it would change retrieval.

    `MetaDataCSVLoader` only narrows content when `content_columns` is given, and
    the retriever does not give it. This pins that: every column stays in the
    text, which is why retrieval measured identical before and after the change.
    """
    documents = MetaDataCSVLoader(
        file_path=str(csv_file), metadata_columns=_csv_column_names(csv_file)
    ).load()

    content = documents[0].page_content
    for column in ("st_id", "display_name", "pathway_name", "url"):
        assert f"{column}:" in content, f"{column} vanished from the indexed text"
