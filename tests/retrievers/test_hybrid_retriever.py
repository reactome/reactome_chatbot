"""Characterization of the hybrid retrieval fusion.

`HybridRetriever` fuses BM25 and vector hits per Chroma subdirectory using
LangChain's Reciprocal Rank Fusion. RRF's constant and its de-duplication key are
LangChain *implementation details* that this repo depends on, so an upgrade across
the 0.3 -> 1.x boundary can silently change result ordering. These tests exist to
make that change loud.

Skipped unless the retrieval stack is installed; run them on both sides of the
upgrade and diff the output.
"""

from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("langchain", reason="retrieval stack not installed")
pytest.importorskip("chromadb", reason="retrieval stack not installed")

from langchain_core.documents import Document  # noqa: E402

from retrievers.csv_chroma import (  # noqa: E402
    HybridRetriever,
    list_chroma_subdirectories,
)

pytestmark = pytest.mark.requires_retrieval_stack


def _doc(text: str) -> Document:
    return Document(page_content=text)


def _fuse(doc_lists: list[list[Document]]) -> list[Document]:
    """`weighted_reciprocal_rank` never touches `self`; call it without an instance.

    Building a real HybridRetriever requires an LLM and an on-disk Chroma store,
    which would make this an integration test rather than a characterization of
    the ranking maths.
    """
    return HybridRetriever.weighted_reciprocal_rank(cast(Any, None), doc_lists)


def test_subdirectories_are_discovered_by_chroma_sqlite_marker(tmp_path: Path) -> None:
    for name in ("reactions", "summations", "complexes"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "chroma.sqlite3").touch()
    (tmp_path / "csv_files").mkdir()  # sibling data dir, not a collection
    (tmp_path / "empty").mkdir()  # no marker file

    assert sorted(list_chroma_subdirectories(tmp_path)) == [
        "complexes",
        "reactions",
        "summations",
    ]


def test_missing_directory_yields_no_subdirectories(tmp_path: Path) -> None:
    assert list_chroma_subdirectories(tmp_path / "nope") == []


def test_documents_in_both_lists_outrank_documents_in_one() -> None:
    """The core reason for hybrid retrieval: BM25/vector agreement should win."""
    bm25 = [_doc("agreed"), _doc("bm25 only")]
    vector = [_doc("vector only"), _doc("agreed")]

    ranked = [d.page_content for d in _fuse([bm25, vector])]

    assert ranked[0] == "agreed"
    assert set(ranked) == {"agreed", "bm25 only", "vector only"}


def test_fusion_deduplicates_on_page_content() -> None:
    """De-dup key is page_content, not metadata -- identical text collapses."""
    duplicated = [[_doc("same")], [_doc("same")], [_doc("same")]]
    assert len(_fuse(duplicated)) == 1


def test_rank_order_within_a_single_list_is_preserved() -> None:
    single = [[_doc("first"), _doc("second"), _doc("third")]]
    assert [d.page_content for d in _fuse(single)] == ["first", "second", "third"]


def test_empty_input_is_not_an_error() -> None:
    assert _fuse([[]]) == []


def test_weights_are_uniform_across_lists() -> None:
    """Each subdirectory's lists are weighted 1/len(doc_lists) -- no list dominates.

    Swapping the order of equally-ranked lists must not change the outcome.
    """
    a, b = [_doc("x"), _doc("y")], [_doc("y"), _doc("x")]
    assert [d.page_content for d in _fuse([a, b])] == [
        d.page_content for d in _fuse([b, a])
    ]


@pytest.mark.requires_embeddings
def test_installed_bundle_exposes_the_expected_collections() -> None:
    """Guards the bundle layout the retriever assumes: <db>/<collection>/chroma.sqlite3
    plus a sibling csv_files/<collection>.csv for BM25."""
    from util.embedding_environment import EmbeddingEnvironment

    directory = EmbeddingEnvironment.get_dir("reactome")
    if directory is None or not directory.is_dir():
        pytest.skip("no reactome embeddings installed")

    collections = list_chroma_subdirectories(directory)
    assert collections, "bundle contains no chroma collections"
    for collection in collections:
        assert (
            directory / "csv_files" / f"{collection}.csv"
        ).is_file(), f"BM25 source CSV missing for collection '{collection}'"
