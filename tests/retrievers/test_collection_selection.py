"""Collection routing: which collections a question searches.

The rule that matters is the asymmetry. Every failure path widens the search and
none narrows it, because searching everything when unsure costs latency while
searching a subset chosen on a misunderstanding costs the answer (Principle IV).

The characterization test at the bottom pins today's behaviour -- every
collection searched, always -- so that turning routing on is a visible change
rather than a silent one.
"""

from typing import Any

import pytest

from retrievers.csv_chroma import resolve_collections

ALL = ["complexes", "disease_variants", "ewas", "reactions", "summations"]


class TestResolveCollections:
    def test_empty_searches_everything(self) -> None:
        """The default, and what every failure degrades to."""
        assert resolve_collections([], ALL) == ALL

    def test_absent_searches_everything(self) -> None:
        """An omitted field, an older prompt and a parse failure all arrive here."""
        assert resolve_collections(None, ALL) == ALL

    def test_a_valid_selection_narrows(self) -> None:
        assert resolve_collections(["reactions", "ewas"], ALL) == ["ewas", "reactions"]

    def test_the_bundle_order_is_kept_not_the_selection_order(self) -> None:
        """Retrieval order must not depend on how a model happened to list them."""
        assert resolve_collections(["summations", "complexes"], ALL) == [
            "complexes",
            "summations",
        ]

    def test_one_unknown_name_widens_to_everything(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The case that must not narrow.

        An unrecognised name means the classifier's prompt and the installed
        bundle disagree about what exists. Narrowing on that disagreement would
        drop a collection the question needed.
        """
        with caplog.at_level("WARNING"):
            got = resolve_collections(["reactions", "not_a_collection"], ALL)
        assert got == ALL, "narrowed on a selection it did not understand"
        assert "not_a_collection" in caplog.text

    def test_all_unknown_names_widen_to_everything(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            assert resolve_collections(["nope", "also_nope"], ALL) == ALL
        assert "also_nope" in caplog.text

    def test_the_warning_names_what_is_actually_available(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """So an operator can see the disagreement rather than guess at it."""
        with caplog.at_level("WARNING"):
            resolve_collections(["typo_reactions"], ALL)
        for name in ALL:
            assert name in caplog.text

    def test_an_empty_bundle_yields_nothing_rather_than_raising(self) -> None:
        """Degenerate, but it must not explode on the retrieval path."""
        assert resolve_collections(["reactions"], []) == []
        assert resolve_collections([], []) == []


class _StubRetriever:
    def __init__(self, name: str, seen: list[str]) -> None:
        self._name = name
        self._seen = seen

    def invoke(self, _query: str, **_kw: Any) -> list[Any]:
        self._seen.append(self._name)
        return []


def test_characterization_no_selection_searches_every_collection() -> None:
    """Today's behaviour, pinned before routing changes it.

    Recorded so that narrowing becomes a visible change. If this starts failing
    without `collections` being set, something narrowed retrieval by accident --
    which is the failure that costs answers rather than time.
    """
    seen: list[str] = []
    collection_retrievers = {name: _StubRetriever(name, seen) for name in ALL}

    for name in resolve_collections([], list(collection_retrievers)):
        collection_retrievers[name].invoke("any query")

    assert sorted(seen) == sorted(ALL)


class _FakeHybrid:
    """Enough of HybridRetriever to exercise the filtering, without a bundle."""

    def __init__(self, seen: list[str]) -> None:
        self.collection_retrievers = dict.fromkeys(ALL, None)
        self._seen = seen

    def search(self) -> None:
        from retrievers.csv_chroma import resolve_collections, selected_collections

        for name in resolve_collections(
            selected_collections.get(), self.collection_retrievers
        ):
            self._seen.append(name)


def test_the_selection_narrows_retrieval() -> None:
    from retrievers.csv_chroma import selected_collections

    seen: list[str] = []
    token = selected_collections.set(["reactions", "ewas"])
    try:
        _FakeHybrid(seen).search()
    finally:
        selected_collections.reset(token)
    assert seen == ["ewas", "reactions"]


def test_no_selection_still_searches_everything() -> None:
    seen: list[str] = []
    _FakeHybrid(seen).search()
    assert sorted(seen) == sorted(ALL)


def test_one_request_cannot_see_anothers_selection() -> None:
    """The reason this is a ContextVar and not an attribute.

    The retriever is built once at startup and shared by every request. An
    attribute would race; a ContextVar is copied per asyncio task, so a narrow
    selection in one request cannot narrow another's.
    """
    import asyncio

    from retrievers.csv_chroma import selected_collections

    async def one(selection: list[str] | None, out: list[str]) -> None:
        if selection is not None:
            selected_collections.set(selection)
        await asyncio.sleep(0)  # force interleaving
        _FakeHybrid(out).search()

    narrow: list[str] = []
    wide: list[str] = []

    async def both() -> None:
        await asyncio.gather(one(["reactions"], narrow), one(None, wide))

    asyncio.run(both())
    assert narrow == ["reactions"], "the narrow request did not get its selection"
    assert sorted(wide) == sorted(ALL), "a concurrent request leaked its narrowing"
