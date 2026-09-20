"""The harness exists to catch three mistakes, so the tests are that it does.

Each test recreates a mistake actually made in this repository rather than a
hypothetical one.
"""

import asyncio

import pytest

from evaluation.ab import ADVISORY_MIN_SAMPLES, Arm, run


def _arm(name: str, holds: bool = True) -> tuple[Arm[float], list[str]]:
    applied: list[str] = []
    return (
        Arm(
            name=name,
            apply=lambda: applied.append(name),
            precondition=lambda: holds,
        ),
        applied,
    )


def _measure_order(order: list[str]):  # type: ignore[no-untyped-def]
    async def measure(_case: object, arm: str) -> float:
        order.append(arm)
        return 1.0

    return measure


def test_the_arms_alternate_rather_than_running_in_sequence() -> None:
    # The mistake: measuring arm A fully, then arm B, so B inherits a warm
    # process and whatever the API is doing. It has twice produced a result
    # in the direction the author wanted.
    a, _ = _arm("a")
    b, _ = _arm("b")
    order: list[str] = []
    asyncio.run(run([a, b], ["q1", "q2"], _measure_order(order), repeats=2))
    first_of_each_pair = order[::2]
    assert set(first_of_each_pair) == {"a", "b"}, "one arm always went first"


def test_a_failed_precondition_refuses_the_comparison() -> None:
    # The mistake: a prompt asked for "exactly 1" alternate and got four; a
    # ContextVar was overwritten by the node it was set around. Both produced
    # ordinary-looking numbers that measured the baseline twice.
    good, _ = _arm("good")
    broken, _ = _arm("broken", holds=False)
    summary = asyncio.run(run([good, broken], ["q"], _measure_order([]), repeats=3))
    assert not summary.trustworthy
    assert "PRECONDITION FAILED" in summary.report()
    assert "means nothing" in summary.report()
    assert "Refusing to draw a conclusion" in summary.report()


def test_a_held_precondition_reports_normally() -> None:
    a, _ = _arm("a")
    b, _ = _arm("b")
    summary = asyncio.run(run([a, b], ["q"], _measure_order([]), repeats=6))
    assert summary.trustworthy
    assert "PRECONDITION FAILED" not in summary.report()
    assert "Refusing" not in summary.report()


def test_the_configuration_is_applied_per_sample_not_once_per_arm() -> None:
    # A setting applied once and mutated in between is the same class of
    # failure as never applying it: the numbers look fine either way.
    a, applied_a = _arm("a")
    b, _ = _arm("b")
    asyncio.run(run([a, b], ["q1", "q2"], _measure_order([]), repeats=3))
    assert len(applied_a) == len(a.samples) == 6


def test_a_small_sample_is_flagged_rather_than_quoted_confidently() -> None:
    # p90 from fifteen samples is about the second-highest value, and one was
    # quoted from fifteen. The harness reports min and max instead, and says
    # when the median is thin.
    a, _ = _arm("a")
    b, _ = _arm("b")
    summary = asyncio.run(run([a, b], ["q"], _measure_order([]), repeats=2))
    assert all(len(arm.samples) < ADVISORY_MIN_SAMPLES for arm in summary.arms)
    assert "small; treat the median loosely" in summary.report()
    assert "p90" not in summary.report()


def test_one_arm_is_not_a_comparison() -> None:
    a, _ = _arm("a")
    with pytest.raises(ValueError, match="at least two arms"):
        asyncio.run(run([a], ["q"], _measure_order([]), repeats=1))
