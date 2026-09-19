"""The trailing source list must go, and nothing else with it."""

import pytest

from util.sources_section import SourcesSectionStripper

ANSWER = (
    "CDK5 phosphorylates tau in Alzheimer disease, and the reaction is "
    "curated in Reactome.\n\n"
)

# Every heading below was actually observed by the website session before the
# prompts pinned one. They are the reason this exists.
OBSERVED = (
    "## Sources",
    "### Sources",
    "## Most relevant sources",
    "**Most relevant sources**",
    "## Key sources",
    "## Top citations",
    "## Relevant references",
    "**Sources:**",
)


@pytest.mark.parametrize("heading", OBSERVED)
def test_an_observed_heading_and_everything_after_it_goes(heading: str) -> None:
    stripper = SourcesSectionStripper()
    out = stripper.feed(f"{ANSWER}{heading}\n- Apoptosis\n- Cell Cycle\n")
    assert stripper.flush() == ""
    assert out == ANSWER


@pytest.mark.parametrize("heading", OBSERVED)
def test_it_survives_being_split_at_every_character(heading: str) -> None:
    # The heading arrives in fragments like any other text: measured on the
    # live endpoint, one anchor came as twenty-odd pieces.
    whole = f"{ANSWER}{heading}\n- Apoptosis\n"
    stripper = SourcesSectionStripper()
    out = "".join(stripper.feed(c) for c in whole) + stripper.flush()
    assert out == ANSWER


def test_prose_mentioning_sources_is_not_a_heading() -> None:
    # The failure that would matter: eating the answer. A sentence about
    # sources is not a source list, and neither is a bolded phrase inside one.
    prose = (
        "Reactome draws on several **sources** of evidence, and the primary "
        "sources for this reaction are listed in the literature references. "
        "Citations of this pathway are numerous.\n"
    )
    stripper = SourcesSectionStripper()
    assert "".join(stripper.feed(c) for c in prose) + stripper.flush() == prose


def test_a_long_bold_line_is_prose_not_a_heading() -> None:
    # Bounded at five words: a bold sentence that happens to end in "sources"
    # is prose, and dropping the rest of the answer would be the worst
    # possible failure here.
    text = (
        "**The following mechanism is supported by several independent "
        "curated sources**\n\nIt proceeds in three steps.\n"
    )
    stripper = SourcesSectionStripper()
    assert "".join(stripper.feed(c) for c in text) + stripper.flush() == text


def test_an_answer_with_no_source_list_is_unchanged() -> None:
    stripper = SourcesSectionStripper()
    assert stripper.feed(ANSWER) + stripper.flush() == ANSWER


def test_nothing_is_emitted_after_the_heading_even_in_later_feeds() -> None:
    stripper = SourcesSectionStripper()
    stripper.feed(f"{ANSWER}## Sources\n")
    assert stripper.feed("- Apoptosis\n") == ""
    assert stripper.flush() == ""


def test_prose_is_not_held_back_waiting_for_a_newline() -> None:
    # The first version held every unterminated line, so an answer with no
    # newline until the end arrived as one blob at flush -- which defeats the
    # endpoint. Two endpoint tests caught it; this states it directly.
    stripper = SourcesSectionStripper()
    assert stripper.feed("CDK5 ") == "CDK5 "
    assert stripper.feed("phosphorylates ") == "phosphorylates "
    assert stripper.feed("tau.") == "tau."
    assert stripper.flush() == ""


def test_a_bullet_is_not_mistaken_for_the_start_of_a_heading() -> None:
    # "* " is a bullet; only "**" can open the bold form of the heading.
    stripper = SourcesSectionStripper()
    assert stripper.feed("- one\n* two") == "- one\n* two"
