"""The safety check and language detection must overlap.

They used to run back to back, costing an extra LLM round trip on every message
on the path every profile shares. The safety check needs the rephrased text so it
has to follow the rephrase; language detection reads the raw input and does not.

Idea from @bleedblack1 in PR #111.
"""

import asyncio
import time
from typing import Any

from langchain_core.runnables import RunnableConfig, RunnableLambda

from agent.profiles.base import BaseGraphBuilder, BaseState
from agent.tasks.safety_checker import SafetyCheck

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
    b.rephrase_chain = _slow("rephrased", log, "rephrase")  # type: ignore[assignment]
    b.safety_checker = _slow(  # type: ignore[assignment]
        SafetyCheck(safety="true", reason_unsafe=""), log, "safety"
    )
    b.language_detector = _slow("en", log, "language")  # type: ignore[assignment]
    return b


def test_safety_and_language_overlap() -> None:
    log: list[tuple[str, float, float]] = []
    state = BaseState(user_input="what is TP53?")  # type: ignore[typeddict-item]

    elapsed = time.perf_counter()
    result = asyncio.run(_builder(log).preprocess(state, RunnableConfig()))
    elapsed = time.perf_counter() - elapsed

    assert result["rephrased_input"] == "rephrased"
    assert result["safety"] == "true"
    assert result["detected_language"] == "en"

    spans = {name: (start, end) for name, start, end in log}
    safety, language = spans["safety"], spans["language"]
    # Two intervals overlap when each starts before the other ends.
    assert safety[0] < language[1], "safety started after language detection finished"
    assert language[0] < safety[1], "language detection started after safety finished"

    # Three sequential calls would take 3 * DELAY; overlapping two takes about 2.
    assert (
        elapsed < DELAY * 2.8
    ), f"took {elapsed:.2f}s, expected roughly {DELAY * 2:.2f}s"


def test_rephrase_still_precedes_the_safety_check() -> None:
    """Ordering that must not be lost: the safety check reads the rephrased text."""
    log: list[tuple[str, float, float]] = []
    state = BaseState(user_input="what is TP53?")  # type: ignore[typeddict-item]
    asyncio.run(_builder(log).preprocess(state, RunnableConfig()))

    spans = {name: (start, end) for name, start, end in log}
    assert (
        spans["rephrase"][1] <= spans["safety"][0]
    ), "the safety check started before the rephrase finished"
