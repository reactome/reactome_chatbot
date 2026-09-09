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
from typing import Any, cast, get_type_hints

import pytest

pytest.importorskip("langchain", reason="retrieval stack not installed")
pytest.importorskip("chromadb", reason="retrieval stack not installed")

from langchain_core.documents import Document  # noqa: E402
from langchain_core.retrievers import BaseRetriever  # noqa: E402

from retrievers import csv_chroma  # noqa: E402
from retrievers.csv_chroma import (  # noqa: E402
    MAX_DOCUMENTS_PER_COLLECTION,
    HybridRetriever,
    RetrieverDict,
    dedupe_by_entity,
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


def test_ties_are_broken_by_position_not_by_score() -> None:
    """Equal RRF scores are resolved by which list came first.

    x is rank 1 in the first list and rank 2 in the second; y is the mirror
    image, so both score w/61 + w/62 exactly. sorted() is stable, so the order
    that survives is the order of chain.from_iterable(doc_lists) -- meaning the
    caller's list order decides.

    In HybridRetriever the lists are the query variants, and within each list
    BM25's results precede the vector retriever's. So on a tie, earlier query
    variants win, and BM25 wins over the vector store. Deterministic, but a
    consequence of iteration order rather than of relevance.
    """
    a, b = [_doc("x"), _doc("y")], [_doc("y"), _doc("x")]
    assert [d.page_content for d in _fuse([a, b])] == ["x", "y"]
    assert [d.page_content for d in _fuse([b, a])] == ["y", "x"]


def test_weights_are_uniform_so_they_cannot_change_the_ordering() -> None:
    """create_bm25_chroma_ensemble_retriever passes [1/n] * n.

    A constant multiplier across every list scales all scores equally, so the
    weighting is currently inert. It is the obvious place to put a BM25-vs-vector
    balance, but nothing uses it today.
    """
    lists = [[_doc("p"), _doc("q")], [_doc("q"), _doc("r")]]
    uniform_half = HybridRetriever.weighted_reciprocal_rank(cast(Any, None), lists)
    assert [d.page_content for d in uniform_half] == ["q", "p", "r"]


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


def _doc_with_id(st_id: str, text: str) -> Document:
    return Document(page_content=text, metadata={"st_id": st_id})


def test_dedupe_keeps_the_highest_ranked_row_per_entity() -> None:
    """One reaction occupies several CSV rows -- one per pathway/input/output
    combination -- and those rows have different page_content, so nothing
    upstream collapses them. Before this, vector search on `reactions` returned
    ten results containing about five distinct reactions (issue #169)."""
    docs = [
        _doc_with_id("R-HSA-1", "in pathway A"),
        _doc_with_id("R-HSA-1", "in pathway B"),
        _doc_with_id("R-HSA-2", "second reaction"),
        _doc_with_id("R-HSA-1", "in pathway C"),
        _doc_with_id("R-HSA-3", "third reaction"),
    ]
    kept = dedupe_by_entity(docs, limit=10)
    assert [d.metadata["st_id"] for d in kept] == ["R-HSA-1", "R-HSA-2", "R-HSA-3"]
    assert kept[0].page_content == "in pathway A", "keeps the highest-ranked row"


def test_dedupe_respects_the_limit() -> None:
    docs = [_doc_with_id(f"R-HSA-{i}", f"doc {i}") for i in range(20)]
    assert len(dedupe_by_entity(docs, limit=10)) == 10


def test_dedupe_falls_back_to_page_content_without_st_id() -> None:
    """A collection whose metadata lacks st_id degrades to the old behaviour
    rather than raising."""
    docs = [_doc("same"), _doc("same"), _doc("different")]
    kept = dedupe_by_entity(docs, limit=10)
    assert [d.page_content for d in kept] == ["same", "different"]


def test_bm25_and_vector_are_fused_as_separate_lists() -> None:
    """Both retrievers' top hits must score equally.

    Concatenating them into one list put every vector result at rank 11+, so the
    best vector hit scored 1/71 against BM25's 1/61 (issue #170). As separate
    lists both are rank 1, and a document only one of them found cannot outrank
    a document they agree on.
    """
    bm25 = [_doc("agreed"), _doc("bm25 only")]
    vector = [_doc("agreed"), _doc("vector only")]

    ranked = [d.page_content for d in _fuse([bm25, vector])]
    assert ranked[0] == "agreed", "agreement between the two retrievers wins"
    # the two single-source documents are tied, so only membership is asserted
    assert set(ranked[1:]) == {"bm25 only", "vector only"}


def test_fused_results_are_capped_per_collection() -> None:
    """RRF returns every unique document it is given, not a top-N.

    Uncapped, the retriever ranked ~222 documents and sent all of them --
    roughly 32k tokens per message, which made the ranking decorative since
    nothing acted on it. The cap is applied per collection so that one
    collection cannot crowd out the others.
    """
    many = [[_doc(f"doc {i}") for i in range(50)]]
    assert len(_fuse(many)) == 50, "the fusion itself still returns everything"
    assert (
        MAX_DOCUMENTS_PER_COLLECTION < 50
    ), "the cap, applied by the caller, is what bounds the prompt"


def test_the_vector_side_makes_no_llm_call() -> None:
    """D1: SelfQueryRetriever is gone, so retrieval costs one LLM call, not 21.

    SelfQueryRetriever translated each question into a Chroma metadata filter with
    an LLM call, once per collection per query variant -- 4 x 5 = 20 per message,
    plus the expansion. The vector side is now plain similarity search.

    Asserted structurally rather than by counting calls, because a call counter
    would pass just as well against a cached or mocked LLM.
    """
    source = Path(csv_chroma.__file__).read_text()
    # Checks the import, not the word: the name still appears in a comment
    # explaining what was replaced, and that comment is worth keeping.
    assert (
        "from langchain.retrievers.self_query" not in source
    ), "the vector side must not reintroduce an LLM-backed retriever"
    assert "as_retriever(" in source, "plain similarity search is expected"


def test_retriever_dict_accepts_any_base_retriever() -> None:
    """The vector slot is typed to the contract, not to one implementation.

    It was `SelfQueryRetriever`, which meant swapping the implementation was a
    type change as well as a behaviour change.
    """
    hints = get_type_hints(RetrieverDict)
    assert hints["vector"] is BaseRetriever
