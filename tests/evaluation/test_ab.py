"""The harness exists to catch three mistakes, so the tests are that it does.

Each test recreates a mistake actually made in this repository rather than a
hypothetical one.
"""

import asyncio

import pytest

from evaluation.ab import ADVISORY_MIN_SAMPLES, Arm, run

#: Which arm's configuration is currently applied. Shared, because a
#: precondition that does not read something the arms actually change cannot
#: tell them apart -- which the harness now refuses, and which this helper
#: used to do by returning a constant.
ACTIVE: list[str] = []


def _arm(name: str, holds: bool = True) -> tuple[Arm[float], list[str]]:
    applied: list[str] = []

    def apply() -> None:
        applied.append(name)
        ACTIVE.clear()
        ACTIVE.append(name)

    return (
        Arm(
            name=name,
            apply=apply,
            precondition=(lambda: [name] == ACTIVE) if holds else (lambda: False),
        ),
        applied,
    )


def _always_true_arm(name: str) -> Arm[float]:
    """A guard that cannot fail, which is the thing the harness must catch."""
    return Arm(name=name, apply=lambda: None, precondition=lambda: True)


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


def test_a_precondition_that_cannot_fail_is_caught() -> None:
    # The hole this harness had: `lambda: True` reads like a guard, passes
    # every sample, and proves nothing. Same shape as an absence test that
    # never showed the thing can be present -- which is the mistake this file
    # was written to stop, present in the file itself.
    summary = asyncio.run(
        run(
            [_always_true_arm("always-a"), _always_true_arm("always-b")],
            ["q"],
            _measure_order([]),
            repeats=3,
        )
    )
    assert not summary.trustworthy
    assert "DOES NOT DISCRIMINATE" in summary.report()
    assert "cannot tell the arms apart" in summary.report()


def test_discriminating_preconditions_are_accepted() -> None:
    # The honest case: a shared flag the arms actually set, so each arm's
    # precondition is false under the other's configuration.
    active: list[str] = []

    def arm(name: str) -> Arm[float]:
        return Arm(
            name=name,
            apply=lambda: active.clear() or active.append(name),  # type: ignore[func-returns-value]
            precondition=lambda: active == [name],
        )

    summary = asyncio.run(
        run([arm("a"), arm("b")], ["q"], _measure_order([]), repeats=6)
    )
    assert summary.trustworthy
    assert "DISCRIMINATE" not in summary.report()


def test_reusing_arms_across_runs_is_refused() -> None:
    # Two runs into one distribution looks exactly like one noisy run.
    a, _ = _arm("a")
    b, _ = _arm("b")
    asyncio.run(run([a, b], ["q"], _measure_order([]), repeats=2))
    with pytest.raises(ValueError, match="already hold samples"):
        asyncio.run(run([a, b], ["q"], _measure_order([]), repeats=2))


def test_three_arms_each_take_every_position() -> None:
    # Reversing gives two orderings, so with three arms the middle one is
    # always in the middle and carries a systematic warming bias.
    order: list[str] = []
    arms = [_arm(name)[0] for name in ("a", "b", "c")]
    asyncio.run(run(arms, ["q"], _measure_order(order), repeats=3))
    positions: dict[str, set[int]] = {name: set() for name in ("a", "b", "c")}
    for start in range(0, len(order), 3):
        for position, name in enumerate(order[start : start + 3]):
            positions[name].add(position)
    assert all(len(seen) == 3 for seen in positions.values()), positions
