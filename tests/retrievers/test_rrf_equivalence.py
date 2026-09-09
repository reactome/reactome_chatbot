"""The vendored RRF must behave exactly like the LangChain method it replaces.

`reciprocal_rank_fusion` was copied out of `EnsembleRetriever` so that ranking --
the core of retrieval quality -- is owned here and cannot be reordered by a
library upgrade without anyone noticing. That is only safe if the copy is
faithful, so this compares the two implementations directly rather than trusting
that the maths was transcribed correctly.

This test is the reason Stage 1 of the retriever rewrite can claim "no behaviour
change". An end-to-end diff cannot show it: SelfQueryRetriever makes an LLM call,
so the same code run twice already differs on roughly a quarter of questions.
"""

import random

import pytest
from langchain_core.documents import Document

from retrievers.csv_chroma import RRF_K, reciprocal_rank_fusion

pytest.importorskip("langchain", reason="retrieval stack not installed")

from langchain.retrievers import EnsembleRetriever  # noqa: E402

pytestmark = pytest.mark.requires_retrieval_stack


def _langchain_rrf(doc_lists: list[list[Document]]) -> list[str]:
    n = len(doc_lists)
    ensemble = EnsembleRetriever(retrievers=[], weights=[1 / n] * n)
    return [d.page_content for d in ensemble.weighted_reciprocal_rank(doc_lists)]


def _ours(doc_lists: list[list[Document]]) -> list[str]:
    return [d.page_content for d in reciprocal_rank_fusion(doc_lists)]


def test_the_constant_matches_the_source_it_was_copied_from() -> None:
    assert EnsembleRetriever(retrievers=[], weights=[1.0]).c == RRF_K


@pytest.mark.parametrize("seed", range(25))
def test_equivalent_on_random_inputs(seed: int) -> None:
    """Random shapes: varying list counts, overlaps, orderings and empties."""
    # S311: generating test data, not keys.
    rng = random.Random(seed)  # noqa: S311
    for _ in range(80):
        n_lists = rng.randint(1, 5)
        pool = [f"doc-{i}" for i in range(rng.randint(1, 12))]
        lists = [
            [
                Document(page_content=c)
                for c in rng.sample(pool, rng.randint(0, len(pool)))
            ]
            for _ in range(n_lists)
        ]
        assert _ours(lists) == _langchain_rrf(lists), f"diverged on {lists}"


def test_equivalent_when_every_list_is_empty() -> None:
    assert _ours([[], []]) == _langchain_rrf([[], []]) == []


def test_equivalent_when_one_retriever_returns_nothing() -> None:
    """A retriever finding nothing must not drop the other's results."""
    found = [Document(page_content="a"), Document(page_content="b")]
    assert _ours([found, []]) == _langchain_rrf([found, []]) == ["a", "b"]


def test_mismatched_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="one to one"):
        reciprocal_rank_fusion([[Document(page_content="a")]], weights=[0.5, 0.5])
