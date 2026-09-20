"""Stability by reuse, and the two keys that stop it being wrong."""

from analysis.store import SummaryStore

CITES = (("R-HSA-109581", "Apoptosis"),)


def test_the_same_request_returns_byte_identical_text() -> None:
    # FR-014. Generation cannot provide this -- the same question through the
    # same surface scores 0.33 similarity twice -- so stability is reuse.
    store = SummaryStore()
    store.put("tok", "97", "aggregate", "Four pathways pass correction.", CITES)
    first = store.get("tok", "97", "aggregate")
    second = store.get("tok", "97", "aggregate")
    assert first is not None
    assert second is not None
    assert first.text == second.text == "Four pathways pass correction."
    assert first.citations == CITES


def test_a_release_change_discards_the_summary() -> None:
    # The Analysis Service deletes results on a new release, so without the
    # release in the key a stored summary outlives the result it describes
    # and we serve a confident account of an analysis that no longer exists.
    store = SummaryStore()
    store.put("tok", "97", "aggregate", "from release 97", CITES)
    assert store.get("tok", "98", "aggregate") is None
    assert store.get("tok", "97", "aggregate") is not None


def test_the_two_tiers_are_different_artefacts() -> None:
    # Sharing a key would serve a reader who chose the default a summary
    # built from their identifiers, or the reverse. One of those is a
    # disclosure nobody asked for.
    store = SummaryStore()
    store.put("tok", "97", "aggregate", "no identifiers named", CITES)
    store.put("tok", "97", "identifiers", "ABCA1_TYPO was not matched", CITES)
    aggregate = store.get("tok", "97", "aggregate")
    identifiers = store.get("tok", "97", "identifiers")
    assert aggregate is not None
    assert identifiers is not None
    assert aggregate.text != identifiers.text
    assert "ABCA1_TYPO" not in aggregate.text


def test_an_empty_summary_is_never_stored() -> None:
    # A failed or abandoned generation leaves no text. Storing it would serve
    # the emptiness back forever as though it were the answer, and nothing
    # downstream could tell it from a result with nothing to say.
    store = SummaryStore()
    store.put("tok", "97", "aggregate", "", CITES)
    store.put("tok", "97", "identifiers", "   \n ", CITES)
    assert store.get("tok", "97", "aggregate") is None
    assert store.get("tok", "97", "identifiers") is None
    assert len(store) == 0


def test_the_oldest_is_dropped_rather_than_the_newest_refused() -> None:
    # Bounded because this lives for the life of the process. A reader whose
    # summary was evicted regenerates, which is the documented behaviour
    # anyway; refusing to store new ones would silently stop the feature
    # working for everyone after the first few hundred readers.
    store = SummaryStore(max_entries=2)
    for n in ("a", "b", "c"):
        store.put(n, "97", "aggregate", f"summary {n}", CITES)
    assert len(store) == 2
    assert store.get("a", "97", "aggregate") is None
    assert store.get("c", "97", "aggregate") is not None


def test_reading_a_summary_keeps_it_from_being_evicted() -> None:
    # The one people actually reload is the one worth keeping.
    store = SummaryStore(max_entries=2)
    store.put("a", "97", "aggregate", "summary a", CITES)
    store.put("b", "97", "aggregate", "summary b", CITES)
    store.get("a", "97", "aggregate")
    store.put("c", "97", "aggregate", "summary c", CITES)
    assert store.get("a", "97", "aggregate") is not None
    assert store.get("b", "97", "aggregate") is None
