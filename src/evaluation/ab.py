"""Compare two configurations without the three mistakes that keep recurring.

Every latency and retrieval comparison in this repository has gone wrong in
one of three ways, more than once each, and knowing the lesson has not
prevented repeating it within hours. So the checks live here rather than in
whoever is writing the script.

**1. Arms measured in sequence.** The second arm benefits from a warm process,
a warm connection and whatever the API is doing that minute. Twice this has
produced a result in the direction the author was hoping for -- and once, in
an earlier measurement, showed a real improvement as a 2.5s regression. `run`
alternates the arms and never offers a mode that does not.

**2. The manipulation never verified.** A prompt asking for "exactly 1"
alternate produced four; a ContextVar set around a call was overwritten by the
node inside it. Both looked like ordinary results. Each arm must declare a
`precondition`, and `run` refuses to report a comparison whose precondition
did not hold rather than printing numbers that mean nothing.

**3. A precondition that cannot fail.** The check above is only worth the
discrimination in it: `lambda: True` passes every time and reads like a guard.
So after each sample the *other* arms' preconditions are evaluated against the
configuration that is actually active, and if one of them also holds, the two
preconditions do not tell the arms apart and the comparison is refused. This
is the same rule the rest of the repository learned the hard way -- an
assertion that something is absent proves nothing until the same check has
shown it can be present.

**4. A percentile the sample cannot support.** p90 from fifteen samples is
about the second-highest value. `Summary` reports p50 with min and max, and
says so, instead of implying a tail estimate that is not there.
"""

import statistics
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")

#: Below this, a median is a gesture. Not enforced -- a small run is often the
#: right thing -- but reported, so nobody quotes six samples as a p50.
ADVISORY_MIN_SAMPLES = 10


@dataclass
class Arm(Generic[T]):
    """One configuration under test.

    `apply` makes the configuration active and is called immediately before
    each sample, never once per arm -- a setting applied once and mutated by
    something else in between is exactly the failure this guards.

    `precondition` is checked after each sample and must return True for the
    manipulation to be believed. Returning True unconditionally defeats the
    purpose, so write it against something the configuration actually changes.
    """

    name: str
    apply: Callable[[], None]
    precondition: Callable[[], bool]
    samples: list[float] = field(default_factory=list)
    precondition_failures: int = 0
    #: Times another arm's precondition also held under this arm's
    #: configuration, meaning the two do not discriminate.
    indiscriminate: int = 0

    @property
    def held(self) -> bool:
        return (
            self.precondition_failures == 0
            and self.indiscriminate == 0
            and bool(self.samples)
        )


@dataclass
class Summary:
    arms: list[Arm[float]]

    @property
    def trustworthy(self) -> bool:
        return all(arm.held for arm in self.arms)

    def report(self) -> str:
        lines = []
        for arm in self.arms:
            if not arm.samples:
                lines.append(f"  {arm.name:24s} no samples")
                continue
            values = sorted(arm.samples)
            note = ""
            if arm.precondition_failures:
                note = (
                    f"   PRECONDITION FAILED on {arm.precondition_failures} "
                    f"of {len(values)} samples -- this comparison means nothing"
                )
            elif arm.indiscriminate:
                note = (
                    f"   PRECONDITION DOES NOT DISCRIMINATE: another arm's held "
                    f"under this one's configuration on {arm.indiscriminate} "
                    f"samples, so it cannot tell the arms apart"
                )
            elif len(values) < ADVISORY_MIN_SAMPLES:
                note = f"   (n={len(values)}, small; treat the median loosely)"
            lines.append(
                f"  {arm.name:24s} p50 {statistics.median(values):6.2f}  "
                f"min {values[0]:6.2f}  max {values[-1]:6.2f}  n={len(values)}{note}"
            )
        if not self.trustworthy:
            lines.append(
                "  Refusing to draw a conclusion: an arm's precondition did not hold."
            )
        return "\n".join(lines)


async def run(
    arms: list[Arm[float]],
    cases: Iterable[object],
    measure: Callable[[object, str], Awaitable[float]],
    *,
    repeats: int = 5,
) -> Summary:
    """Measure every case under every arm, alternating which arm goes first.

    The alternation is not optional and the order is derived from the case and
    repeat indices, so a run is deterministic in structure while still
    splitting any warming effect evenly between the arms.
    """
    if len(arms) < 2:
        raise ValueError("an A/B comparison needs at least two arms")
    if any(arm.samples for arm in arms):
        # Re-using arms silently mixes two runs into one distribution, and the
        # result looks like an ordinary noisy measurement.
        raise ValueError("these arms already hold samples; build fresh ones")
    cases = list(cases)
    for repeat in range(repeats):
        for index, case in enumerate(cases):
            # Rotate rather than reverse: with more than two arms, reversing
            # leaves the middle one always in the middle.
            offset = (repeat + index) % len(arms)
            ordered = arms[offset:] + arms[:offset]
            for arm in ordered:
                arm.apply()
                arm.samples.append(await measure(case, arm.name))
                if not arm.precondition():
                    arm.precondition_failures += 1
                    continue
                # The configuration for `arm` is still active, so any other
                # arm whose precondition also holds is not distinguishing.
                if any(other is not arm and other.precondition() for other in arms):
                    arm.indiscriminate += 1
    return Summary(arms)
