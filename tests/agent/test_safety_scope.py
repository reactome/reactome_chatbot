"""The safety checker must not refuse people for asking Reactome to do its job.

Measured on beta 2026-09-16: "can you run gsea for me" was refused 4 times out
of 4, with

    Requests a specific analysis task without providing context or data,
    which is outside the scope

and "run a pathway analysis on my genes" was refused 1 time in 4 -- the same
user, the same intent, a different answer depending on the roll.

Nothing downstream ever saw those questions. The user guide bundle that answers
them and the ReactomeGSA tools that describe them are both two nodes further on
in the graph.

These assert the prompt's *wording*, not the model's behaviour -- a behavioural
test would cost an API call per case and drift with the model. The wording is
what was wrong, and it is what a future edit could silently undo.
"""

from agent.tasks.safety_checker import safety_check_message


def test_asking_reactome_to_do_something_is_relevant() -> None:
    """The failure this file exists for."""
    assert "run an analysis on my genes" in safety_check_message
    assert "can you run GSEA for me" in safety_check_message.lower() or (
        "Can you run GSEA for me?" in safety_check_message
    )


def test_a_task_request_is_distinguished_from_personal_advice() -> None:
    """ "Outside the scope of scientific knowledge" was being read as "asks the
    assistant to do a task". It means medical, legal and personal advice."""
    assert "It does not mean" in safety_check_message
    assert "asks the assistant to do a task" in safety_check_message


def test_the_cost_of_refusing_is_stated() -> None:
    """So a later editor tightening this knows what it buys and what it costs."""
    assert (
        "off-topic" in safety_check_message or "inappropriate" in safety_check_message
    )


def test_the_genuine_refusals_are_still_described() -> None:
    """Loosening this must not loosen the part that matters. Medical advice,
    dual-use and irrelevance all keep their examples."""
    for kept in ("medical", "dual-use", "harmful", "statins", "gene drives"):
        assert (
            kept in safety_check_message
        ), f"{kept!r} disappeared from the safety prompt"


def test_capability_is_answered_downstream_not_refused_here() -> None:
    """Whether the assistant can run an analysis is a different question from
    whether it is allowed to be asked."""
    assert "decided later" in safety_check_message
