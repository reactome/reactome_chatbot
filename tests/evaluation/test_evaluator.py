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
