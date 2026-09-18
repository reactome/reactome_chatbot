"""The sweep is a regression gate, so the thing worth testing is that it fails.

A gate that only ever passes is worse than no gate: it is a green tick that
means nothing, and the deploy script treats it as evidence. These tests drive
`run()` against a stub graph, so they check the harness itself rather than any
answer the real chatbot gives.
"""

import asyncio
from collections.abc import Callable

import pytest

from evaluation.answer_sweep import EXPECTATIONS, Expectation, _contains, run


class StubGraph:
    """Stands in for AgentGraph, answering from a scripted list."""

    def __init__(self, answers: list[str | Exception]) -> None:
        self.answers = answers
        self.asked: list[str] = []

    async def ainvoke(
        self, question: str, *_args: object, **_kwargs: object
    ) -> dict[str, object]:
        self.asked.append(question)
        answer = self.answers[min(len(self.asked) - 1, len(self.answers) - 1)]
        if isinstance(answer, Exception):
            raise answer
        return {"answer": answer}

    async def close_pool(self) -> None:
        pass


Install = Callable[[list[str | Exception]], StubGraph]


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> Install:
    def install(answers: list[str | Exception]) -> StubGraph:
        graph = StubGraph(answers)
        monkeypatch.setattr(
            "evaluation.answer_sweep.AgentGraph", lambda *_a, **_k: graph
        )
        return graph

    return install


ONE = (
    Expectation(
        question="Does Reactome do GSEA?",
        why="The safety checker used to refuse this.",
        must=("ReactomeGSA",),
        must_not=("does not provide",),
    ),
)


def test_good_answer_passes(stub: Install) -> None:
    stub(["Yes -- ReactomeGSA runs gene set analysis in the browser."])
    (result,) = asyncio.run(run(ONE))
    assert result.ok


def test_missing_term_fails(stub: Install) -> None:
    stub(["Reactome offers several analysis options."])
    (result,) = asyncio.run(run(ONE))
    assert not result.ok
    assert result.missing == ["ReactomeGSA"]


def test_forbidden_term_fails_even_with_the_required_one(stub: Install) -> None:
    # The real regression looked exactly like this: confident, plausible, and
    # wrong in the middle of an otherwise on-topic answer.
    stub(["Reactome does not provide a GSEA tool, though ReactomeGSA exists."])
    (result,) = asyncio.run(run(ONE))
    assert not result.ok
    assert result.forbidden == ["does not provide"]


def test_an_exception_is_a_failure_not_a_crash(stub: Install) -> None:
    stub([RuntimeError("upstream is down")] * 2)
    (result,) = asyncio.run(run(ONE))
    assert not result.ok
    assert "upstream is down" in result.error


def test_a_transient_blip_is_retried_and_the_retry_is_reported(stub: Install) -> None:
    graph = stub(["A service error occurred.", "Use ReactomeGSA."])
    (result,) = asyncio.run(run(ONE))
    assert result.ok
    assert len(graph.asked) == 2
    # The report prints "(retried once)" from this flag. It used to be set on
    # the result that the retry threw away, so a retried question was reported
    # as a clean pass.
    assert result.retried


def test_a_real_failure_is_not_retried_away(stub: Install) -> None:
    graph = stub(["Reactome offers several analysis options."])
    (result,) = asyncio.run(run(ONE))
    assert not result.ok
    assert len(graph.asked) == 1, "a wrong answer must not be retried"


def test_contains_is_bounded_at_word_edges() -> None:
    # "96" matching "1996" made the release check assert almost nothing.
    assert not _contains("released in 1996", "96")
    assert _contains("a total of 96 species", "96")
    assert _contains("Cannot determine", "cannot")
    assert _contains("see R-HSA-1234", "R-HSA-")


def test_every_expectation_asserts_something() -> None:
    for expectation in EXPECTATIONS:
        assert expectation.must or expectation.must_not or expectation.must_match
        assert expectation.why, f"{expectation.question} does not say why"


LIVE = (
    Expectation(
        question="Which release of Reactome is this?",
        why="The bundle is a snapshot and cannot know.",
        must=("release",),
        needs_live=True,
    ),
)


def test_a_live_question_is_skipped_when_there_is_no_mcp(
    stub: Install, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = stub(["I have no idea."])
    monkeypatch.setattr("evaluation.answer_sweep.is_configured", lambda: False)
    (result,) = asyncio.run(run(LIVE))
    assert result.skipped
    assert result.ok, "a skip is not a failure"
    assert graph.asked == [], "it should not have been asked at all"


def test_a_live_question_still_runs_when_mcp_is_configured(
    stub: Install, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The dangerous direction: skipping these inside the container would
    # quietly stop checking the questions the live service exists to answer.
    graph = stub(["I have no idea."])
    monkeypatch.setattr("evaluation.answer_sweep.is_configured", lambda: True)
    (result,) = asyncio.run(run(LIVE))
    assert not result.skipped
    assert not result.ok
    assert len(graph.asked) == 1


def test_a_must_not_guard_still_catches_inflections() -> None:
    # "consult" guards against medical advice. Closing the pattern at both
    # ends let "consulting your physician" through, which is the whole thing
    # it is there to catch.
    assert _contains("Please consult your physician.", "consult")
    assert _contains("consulting your physician is best", "consult")
    assert _contains("reports of muscle pains", "muscle pain")
    # A number stays closed at both ends: a longer one is a different number.
    assert not _contains("released in 1996", "96")
    assert not _contains("there are 965 of them", "96")


COLLECTION = (
    Expectation(
        question="List the ABCA1 variants in Reactome.",
        why="Pathway-level prose instead of the variant.",
        must=("W590S",),
        needs_collection="disease_variants",
    ),
)


def test_a_question_is_skipped_when_its_collection_is_not_installed(
    stub: Install, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = stub(["Defective ABCA1 causes Tangier Disease."])
    monkeypatch.setattr("evaluation.answer_sweep._has_collection", lambda _n: False)
    (result,) = asyncio.run(run(COLLECTION))
    assert result.skipped
    assert result.ok
    assert graph.asked == []


def test_it_runs_once_the_collection_is_installed(
    stub: Install, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The direction that matters: once the bundle ships, this must actually be
    # checked rather than skipped forever.
    graph = stub(["Defective ABCA1 causes Tangier Disease."])
    monkeypatch.setattr("evaluation.answer_sweep._has_collection", lambda _n: True)
    (result,) = asyncio.run(run(COLLECTION))
    assert not result.skipped
    assert not result.ok
    assert len(graph.asked) == 1


def test_the_sweep_does_not_pay_for_a_web_search() -> None:
    """The sweep reads `answer` and nothing else.

    With postprocess left at its default it ran a Tavily search per question,
    discarded the result, and slowed each question by the length of that search
    -- fifteen of them, after every beta deploy. Pinned here rather than trusted,
    because the cost is invisible: nothing fails when it happens.
    """
    import inspect

    from evaluation import answer_sweep

    source = inspect.getsource(answer_sweep)
    assert (
        "enable_postprocess=False" in source
    ), "the sweep is running the postprocess web search again"
    assert "additional_content" not in source, (
        "the sweep now reads additional_content, so the assertion above is no "
        "longer the right guard"
    )
