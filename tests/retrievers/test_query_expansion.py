"""How many alternate questions retrieval asks for.

Measured, not argued (spec 010 T020): the expansion call costs about 1.27s
whatever it returns, so trimming the count saves only fan-out and the cost
disappears only at zero. The default is unchanged regardless -- thirteen
tracked questions show those answers do not need expansion, not that recall
is unaffected in general.
"""

import asyncio

import pytest

from retrievers.csv_chroma import (
    ALTERNATES_ENV,
    DEFAULT_ALTERNATES,
    HybridRetriever,
    expansion_alternates,
)


def test_the_default_is_unchanged_without_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ALTERNATES_ENV, raising=False)
    assert expansion_alternates() == DEFAULT_ALTERNATES == 4


@pytest.mark.parametrize(("raw", "expected"), [("0", 0), ("2", 2), ("7", 7)])
def test_a_configured_count_is_honoured(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: int
) -> None:
    monkeypatch.setenv(ALTERNATES_ENV, raw)
    assert expansion_alternates() == expected


@pytest.mark.parametrize("raw", ["", "  ", "lots", "-1", "3.5"])
def test_a_bad_value_falls_back_loudly_rather_than_disabling_recall(
    monkeypatch: pytest.MonkeyPatch, raw: str, caplog: pytest.LogCaptureFixture
) -> None:
    # The dangerous direction: a typo must not silently turn expansion off,
    # because nothing downstream would look any different.
    monkeypatch.setenv(ALTERNATES_ENV, raw)
    assert expansion_alternates() == DEFAULT_ALTERNATES


class _Expander:
    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, _inputs: object, config: object = None) -> list[str]:
        self.calls += 1
        return [f"alt {i}" for i in range(4)]


def _expand(
    count: str | None, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], int]:
    if count is None:
        monkeypatch.delenv(ALTERNATES_ENV, raising=False)
    else:
        monkeypatch.setenv(ALTERNATES_ENV, count)
    retriever = HybridRetriever.__new__(HybridRetriever)
    object.__setattr__(retriever, "include_original", True)
    expander = _Expander()
    queries = asyncio.run(
        retriever._expand("the question", lambda: expander.ainvoke(None))
    )
    return queries, expander.calls


def test_the_original_question_is_always_asked_and_always_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # RRF breaks ties by first appearance, so the position is ranking, not
    # cosmetics.
    for count in (None, "2", "0"):
        queries, _ = _expand(count, monkeypatch)
        assert queries[-1] == "the question"


def test_at_zero_the_expansion_call_is_skipped_not_made_and_discarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The call is the larger half of the cost. Making it and throwing the
    # result away would keep the expense and lose the benefit -- and would
    # look identical in every other measurement.
    queries, calls = _expand("0", monkeypatch)
    assert queries == ["the question"]
    assert calls == 0


def test_a_lower_count_truncates_rather_than_trusting_the_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Measured: asked for "exactly 1" and "exactly 0" alternates, the model
    # produced four anyway. Enforcing the count in code removes the model's
    # obedience from the question.
    queries, calls = _expand("2", monkeypatch)
    assert queries == ["alt 0", "alt 1", "the question"]
    assert calls == 1
