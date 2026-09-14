"""The parts of the evaluator that can be checked without spending an API call.

The scoring itself needs the network and a bundle; what is pinned here is the
input handling and the guards -- including the one that stops a model from
grading its own answers, which would silently produce numbers rather than fail.
"""

from pathlib import Path

import pytest

pytest.importorskip("ragas", reason="evaluation stack not installed")

from evaluation.evaluator import (  # noqa: E402
    DEFAULT_JUDGE_MODEL,
    DEFAULT_QUESTIONS,
    read_questions,
    read_references,
    resolve_references,
)


def test_the_default_question_set_is_the_committed_one() -> None:
    """The evaluator and bin/retrieval_baseline must ask the same questions.

    Two measurement tools disagreeing about the question set would make their
    results incomparable for no reason.
    """
    assert DEFAULT_QUESTIONS.exists(), DEFAULT_QUESTIONS
    assert DEFAULT_QUESTIONS.name == "questions.txt"
    assert len(read_questions(DEFAULT_QUESTIONS)) >= 20


def test_comments_and_blank_lines_are_not_questions(tmp_path: Path) -> None:
    path = tmp_path / "q.txt"
    path.write_text("# a heading\n\nWhat is TP53?\n\n#another\n  Which complexes?  \n")
    assert read_questions(path) == ["What is TP53?", "Which complexes?"]


def test_references_are_read_as_a_question_to_answer_map(tmp_path: Path) -> None:
    path = tmp_path / "refs.json"
    path.write_text('{"What is TP53?": "A tumour suppressor."}')
    assert read_references(path) == {"What is TP53?": "A tumour suppressor."}


def test_the_default_judge_is_not_a_model_this_repo_answers_with() -> None:
    """The judge must be pinned and distinct; see the guard in main().

    gpt-4o-mini is the default answering model, so the default judge must not be
    it -- otherwise the out-of-the-box invocation grades its own homework.
    """
    assert DEFAULT_JUDGE_MODEL != "gpt-4o-mini"


def test_references_are_looked_up_by_the_original_wording() -> None:
    """A reference file is written against the questions a person typed.

    The evaluator retrieves and scores on the *rephrased* question, because that
    is what production does. References must still be found by the original --
    an intermediate version keyed them by the rephrased text, which matched
    nothing and would have dropped context_recall from every run while looking
    exactly like "no references were supplied".
    """
    questions = ["What is TP53?", "How does PTEN work?"]
    references = {"What is TP53?": "A tumour suppressor."}

    assert resolve_references(questions, references) == [
        "A tumour suppressor.",
        None,
    ], "positional, aligned with the questions asked"


def test_a_missing_reference_is_none_and_not_empty_text() -> None:
    """score() drops context_recall when any reference is falsy.

    The distinction matters because ragas would score an answer against an empty
    reference and report a number for it. `""` is falsy too, so the guard would
    still hold -- but a reference file containing an empty string for a question
    is a different mistake from omitting it, and only None says which happened.
    """
    assert resolve_references(["unmatched"], {"other": "x"}) == [None]
    assert resolve_references(["q"], {"q": ""}) == [
        ""
    ], "an explicitly empty reference is preserved, not turned into None"


class _FakeRephrase:
    """Stands in for the rephrase chain; returns the question unchanged."""

    def invoke(self, payload: dict) -> str:
        return payload["user_input"]


class _FakeChain:
    """A chain that answers, unless the question is one it is told to fail on."""

    def __init__(self, fail_on: set[str] | None = None) -> None:
        self.fail_on = fail_on or set()

    def invoke(self, payload: dict) -> dict:
        question = payload["input"]
        if question in self.fail_on:
            raise RuntimeError(f"boom: {question}")

        class _Doc:
            def __init__(self, text: str) -> None:
                self.page_content = text

        return {"answer": f"answer to {question}", "context": [_Doc(f"ctx {question}")]}


def test_answer_questions_keeps_what_succeeded() -> None:
    """One bad question must not discard the answers already paid for."""
    from evaluation.evaluator import answer_questions

    questions = ["q1", "q2", "q3", "q4"]
    rephrased, answers, contexts, _elapsed, failures = answer_questions(
        _FakeChain(fail_on={"q3"}), _FakeRephrase(), questions
    )

    assert [f.question for f in failures] == ["q3"]
    assert answers == ["answer to q1", "answer to q2", "answer to q4"]
    assert rephrased == ["q1", "q2", "q4"]
    assert len(contexts) == 3


def test_answer_questions_preserves_order_under_concurrency() -> None:
    """Results are placed by index, never appended as they arrive.

    Appending would order answers by completion time, so a slow question would
    push every later answer onto the wrong reference. That does not crash -- it
    produces a plausible score for the wrong pairing.
    """
    from evaluation.evaluator import answer_questions

    questions = [f"q{i}" for i in range(12)]
    rephrased, answers, _contexts, _elapsed, failures = answer_questions(
        _FakeChain(), _FakeRephrase(), questions, concurrency=6
    )

    assert not failures
    assert rephrased == questions
    assert answers == [f"answer to {q}" for q in questions]


def test_answer_questions_order_holds_when_a_middle_question_fails() -> None:
    """The alignment that matters: survivors keep their original order."""
    from evaluation.evaluator import answer_questions

    questions = [f"q{i}" for i in range(10)]
    _rephrased, answers, _contexts, _elapsed, failures = answer_questions(
        _FakeChain(fail_on={"q2", "q7"}), _FakeRephrase(), questions, concurrency=4
    )

    assert sorted(f.question for f in failures) == ["q2", "q7"]
    expected = [f"answer to q{i}" for i in range(10) if i not in (2, 7)]
    assert answers == expected


def test_surviving_questions_line_up_with_their_references() -> None:
    """A dropped question must not shift every later reference by one.

    This is the failure that reads as a result: scores come out, they are just
    computed against the wrong pairing.
    """
    from evaluation.evaluator import _kept, answer_questions

    questions = ["q0", "q1", "q2", "q3"]
    references = {q: f"ref {q}" for q in questions}

    _rephrased, answers, _contexts, _elapsed, failures = answer_questions(
        _FakeChain(fail_on={"q1"}), _FakeRephrase(), questions
    )
    answered = [q for i, q in enumerate(questions) if _kept(i, failures)]
    resolved = resolve_references(answered, references)

    assert answered == ["q0", "q2", "q3"]
    assert resolved == ["ref q0", "ref q2", "ref q3"]
    assert len(resolved) == len(answers)


def test_transcript_log_survives_the_run_that_wrote_it(tmp_path: Path) -> None:
    """Every answer is on disk as it is produced, not held until the end."""
    import json

    from evaluation.evaluator import answer_questions, make_transcript_writer

    log = tmp_path / "nested" / "transcript.jsonl"
    questions = ["q0", "q1", "q2"]
    answer_questions(
        _FakeChain(fail_on={"q1"}),
        _FakeRephrase(),
        questions,
        concurrency=3,
        on_answered=make_transcript_writer(log, "test-model", 1, questions),
    )

    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert {r["question"] for r in records} == {"q0", "q2"}
    assert all(r["model"] == "test-model" for r in records)
    # Written per answer, so a crash keeps what was already bought.
    assert len(records) == 2


def test_no_transcript_writer_when_no_path_given() -> None:
    from evaluation.evaluator import make_transcript_writer

    assert make_transcript_writer(None, "m", 1, []) is None
