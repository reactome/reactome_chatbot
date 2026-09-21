"""The two SSE endpoints, held against each other.

Each endpoint's contract was correct about itself and they diverged anyway:
different success-state names, and `release` an int on one and a string on
the other. Nobody was wrong; nothing compared them. A consumer did, and
rendered a correct summary as a truncated failure.

Prose in both contracts is what came out of that. Prose does not fail when
the next divergence appears, so these do.
"""

from pathlib import Path

from agent.graph import AnswerEvent
from api.analysis_summary import _as_number

ANSWER_CONTRACT = Path("specs/010-search-page-answers/contracts/answer_endpoint.md")
SUMMARY_CONTRACT = Path(
    "specs/011-summarise-analysis-results/contracts/summary_endpoint.md"
)

#: What each endpoint's `done` can carry. The answer endpoint's come from the
#: graph's own Literal; the summary's are written here because they are
#: string literals in the handler with nothing to import.
ANSWER_STATES = {"answered", "nothing_found", "refused", "failed"}
SUMMARY_STATES = {
    "summarised",
    "not_found",
    "gone",
    "unsupported",
    "refused",
    "failed",
}


def test_the_answer_states_are_still_what_the_contract_says() -> None:
    # Taken from the graph rather than restated, so a new state there fails
    # here rather than silently escaping both contracts.
    declared = set(AnswerEvent.__annotations__["state"].__args__[0].__args__)
    assert declared == ANSWER_STATES


def test_the_two_endpoints_disagree_only_where_both_contracts_say_so() -> None:
    # `refused` and `failed` mean the same thing on both. Everything else is
    # deliberately different, and each contract has to warn about the other
    # -- because the shapes are near-identical and a consumer will reuse one
    # state list for both. One did.
    shared = ANSWER_STATES & SUMMARY_STATES
    assert shared == {"refused", "failed"}

    answer_doc = ANSWER_CONTRACT.read_text()
    summary_doc = SUMMARY_CONTRACT.read_text()
    for state in SUMMARY_STATES - shared:
        assert state in summary_doc, f"{state} undocumented"
    for state in ANSWER_STATES - shared:
        assert state in answer_doc, f"{state} undocumented"

    # And each must name the other's divergent success state, which is the
    # one that caused actual harm.
    assert "summarised" in answer_doc, "the answer contract does not warn"
    assert "answered" in summary_doc, "the summary contract does not warn"


def test_release_is_the_same_type_on_both_endpoints() -> None:
    # It was an int on one and a string on the other, in the same field of
    # the same event. A consumer requiring a number got null from one of
    # them and did not notice, because null is legitimate on both.
    import typing

    from util.embedding_environment import EmbeddingEnvironment

    # The answer endpoint puts `get_release`'s value straight on the wire,
    # so its declared return type is the contract. A declaration check, not
    # a behavioural one -- reading the real value needs an installed bundle,
    # and it is checked against the deployed build in the quickstart.
    declared = typing.get_type_hints(EmbeddingEnvironment.get_release)["return"]
    assert declared == (int | None)

    # The summary endpoint's arrives as text from the Analysis Service, so
    # the normalisation is where its type is decided. That half is checked
    # by behaviour.
    assert _as_number("97") == 97
    assert _as_number(None) is None
    assert _as_number("97-beta") is None
