"""The safety check and language detection must overlap.

They used to run back to back, costing an extra LLM round trip on every message
on the path every profile shares. The safety check needs the rephrased text so it
has to follow the rephrase; language detection reads the raw input and does not.

Idea from @bleedblack1 in PR #111.
"""

import asyncio
import time
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.runnables import RunnableConfig, RunnableLambda

from agent.profiles.base import BaseGraphBuilder, BaseState
from agent.tasks.safety_checker import SafetyCheck

if TYPE_CHECKING:
    from agent.profiles.react_to_me import ReactToMeGraphBuilder

DELAY = 0.2


def _slow(
    result: Any, log: list[tuple[str, float, float]], name: str
) -> RunnableLambda:
    async def run(_: Any) -> Any:
        start = time.perf_counter()
        await asyncio.sleep(DELAY)
        log.append((name, start, time.perf_counter()))
        return result

    return RunnableLambda(run)


def _builder(log: list[tuple[str, float, float]]) -> BaseGraphBuilder:
    """Build without __init__, which would construct real LLM chains."""
    b = BaseGraphBuilder.__new__(BaseGraphBuilder)
    b.rephrase_chain = _slow("rephrased", log, "rephrase")
    b.safety_checker = _slow(
        SafetyCheck(safety="true", reason_unsafe=""), log, "safety"
    )
    b.language_detector = _slow("en", log, "language")
    return b


def test_safety_and_language_overlap() -> None:
    log: list[tuple[str, float, float]] = []
    state = BaseState(user_input="what is TP53?")

    result = asyncio.run(_builder(log).preprocess(state, RunnableConfig()))

    assert result["rephrased_input"] == "rephrased"
    assert result["safety"] == "true"
    assert result["detected_language"] == "en"

    spans = {name: (start, end) for name, start, end in log}
    safety, language = spans["safety"], spans["language"]
    # Two intervals overlap when each starts before the other ends.
    assert safety[0] < language[1], "safety started after language detection finished"
    assert language[0] < safety[1], "language detection started after safety finished"

    # How MUCH they overlap, not how long the whole call took.
    #
    # This used to assert `elapsed < DELAY * 2.8` against the wall clock, and it
    # failed on CI at 0.63s against a 0.56s budget. That budget covers the two
    # sleeps plus asyncio.run start-up, three coroutine hand-offs and whatever
    # else a shared runner is doing, so it measures the runner as much as the
    # code. The claim being made is that these two steps run concurrently, and
    # the overlap measures exactly that -- from the same perf_counter marks the
    # assertions above use. Two steps that ran back to back overlap by ~0; two
    # started together overlap by ~DELAY.
    overlap = min(safety[1], language[1]) - max(safety[0], language[0])
    assert overlap > DELAY / 2, (
        f"safety and language overlapped for only {overlap:.3f}s of {DELAY}s; "
        "they are running sequentially again"
    )


def test_rephrase_still_precedes_the_safety_check() -> None:
    """Ordering that must not be lost: the safety check reads the rephrased text."""
    log: list[tuple[str, float, float]] = []
    state = BaseState(user_input="what is TP53?")
    asyncio.run(_builder(log).preprocess(state, RunnableConfig()))

    spans = {name: (start, end) for name, start, end in log}
    assert (
        spans["rephrase"][1] <= spans["safety"][0]
    ), "the safety check started before the rephrase finished"


# --- the profile that is actually served -------------------------------------
#
# Everything above pins BaseGraphBuilder. ReactToMeGraphBuilder overrides
# `preprocess` outright, and its override ran all four calls back to back -- so
# the profile behind both the chat UI and the answer endpoint discarded the
# overlap these tests exist to protect, and no test noticed. These pin the
# override on its own terms.


def _react_builder(log: list[tuple[str, float, float]]) -> "ReactToMeGraphBuilder":
    from agent.profiles.react_to_me import ReactToMeGraphBuilder
    from agent.tasks.intent_classifier import QueryIntent

    builder = ReactToMeGraphBuilder.__new__(ReactToMeGraphBuilder)
    builder.rephrase_chain = _slow("rephrased", log, "rephrase")
    builder.safety_checker = _slow(
        SafetyCheck(safety="true", reason_unsafe=""), log, "safety"
    )
    builder.language_detector = _slow("en", log, "language")
    builder.intent_classifier = _slow(QueryIntent(source="reactome"), log, "intent")
    builder._available_sources = frozenset({"reactome"})
    return builder


def _spans(log: list[tuple[str, float, float]]) -> dict[str, tuple[float, float]]:
    return {name: (start, end) for name, start, end in log}


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return min(a[1], b[1]) - max(a[0], b[0])


def test_react_to_me_runs_preprocessing_in_two_rounds() -> None:
    """Four calls, two rounds: (rephrase | language) then (safety | intent).

    Language detection reads the raw user input, so it need not wait for the
    rephraser; safety and intent both read the rephrased text, so they must
    follow it but not each other.
    """
    from agent.profiles.react_to_me import ReactToMeState

    log: list[tuple[str, float, float]] = []
    state = ReactToMeState(user_input="what is TP53?", active_sources=[])

    asyncio.run(_react_builder(log).preprocess(state, RunnableConfig()))

    spans = _spans(log)
    assert set(spans) == {"rephrase", "safety", "language", "intent"}

    assert (
        _overlap(spans["rephrase"], spans["language"]) > DELAY / 2
    ), "language detection is waiting for the rephraser it does not depend on"
    assert (
        _overlap(spans["safety"], spans["intent"]) > DELAY / 2
    ), "safety and intent are running one after the other again"


def test_react_to_me_keeps_the_ordering_the_dependencies_require() -> None:
    """Overlap must not be bought by running something before its input exists."""
    from agent.profiles.react_to_me import ReactToMeState

    log: list[tuple[str, float, float]] = []
    asyncio.run(
        _react_builder(log).preprocess(
            ReactToMeState(user_input="what is TP53?", active_sources=[]),
            RunnableConfig(),
        )
    )

    spans = _spans(log)
    for dependent in ("safety", "intent"):
        assert (
            spans["rephrase"][1] <= spans[dependent][0]
        ), f"{dependent} started before the rephrased text it reads existed"


def _failing(message: str) -> RunnableLambda:
    async def run(_: Any) -> Any:
        raise RuntimeError(message)

    return RunnableLambda(run)


def test_a_failed_rephrase_still_surfaces_as_an_error() -> None:
    """`gather` changes how a failure travels, so pin that it still travels.

    Sequentially, a rephrase failure short-circuited: nothing after it ran. Under
    `gather` the sibling call is already in flight and keeps going, and only the
    first exception propagates. What must not change is that preprocess still
    raises rather than returning a half-built state -- the endpoint turns an
    exception into `state: failed`, and a silently empty `rephrased_input` would
    instead retrieve against nothing and answer from it.
    """
    from agent.profiles.react_to_me import ReactToMeState

    log: list[tuple[str, float, float]] = []
    builder = _react_builder(log)
    builder.rephrase_chain = _failing("rephrase upstream is down")

    with pytest.raises(RuntimeError, match="rephrase upstream is down"):
        asyncio.run(
            builder.preprocess(
                ReactToMeState(user_input="what is TP53?", active_sources=[]),
                RunnableConfig(),
            )
        )


def test_a_failed_intent_classification_still_surfaces_as_an_error() -> None:
    """The second round has the same property, and it is the round that pairs
    two calls neither of which the other needs."""
    from agent.profiles.react_to_me import ReactToMeState

    log: list[tuple[str, float, float]] = []
    builder = _react_builder(log)
    builder.intent_classifier = _failing("intent classifier is down")

    with pytest.raises(RuntimeError, match="intent classifier is down"):
        asyncio.run(
            builder.preprocess(
                ReactToMeState(user_input="what is TP53?", active_sources=[]),
                RunnableConfig(),
            )
        )
