"""The anchor stripper, driven the way the model actually streams.

The bug this exists for is not "an anchor is present" but "an anchor is split".
A fragment-at-a-time regex passes a test that feeds whole anchors and fails on
the real stream, so every case here is also run at every possible split point.
"""

import random

import pytest

from util.anchor_strip import AnchorStripper

# Verbatim from the live endpoint, 2026-09-17: how one anchor actually arrived.
LIVE_FRAGMENTS = [
    " <",
    "a",
    " href",
    '="',
    "https",
    "://",
    "react",
    "ome",
    ".org",
    "/content",
    "/detail",
    "/R",
    "-H",
    "SA",
    "-",
    "960",
    "1234",
    '">',
    "ABCA",
    "1",
    " transports",
    " cholesterol",
    "</a",
    ">",
    " out",
    " of",
    " the",
    " cell",
    ".",
]


def _through(stripper: AnchorStripper, pieces: list[str]) -> str:
    return "".join(stripper.feed(piece) for piece in pieces) + stripper.flush()


def _every_split(text: str) -> list[list[str]]:
    return [[text[:i], text[i:]] for i in range(len(text) + 1)]


def test_the_live_fragment_sequence_loses_its_anchor_and_keeps_its_words() -> None:
    assert _through(AnchorStripper(), LIVE_FRAGMENTS) == (
        " ABCA1 transports cholesterol out of the cell."
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('<a href="https://x">text</a>', "text"),
        ("before <a href='x'>link</a> after", "before link after"),
        ("<A HREF='x'>upper</A>", "upper"),
        ("no anchors at all", "no anchors at all"),
        ("", ""),
        # Prose that merely contains '<' must not be held back or eaten.
        ("if x < y then", "if x < y then"),
        ("a < b and c > d", "a < b and c > d"),
        # Non-anchor markup is prose here; only anchors are the contract's problem.
        ("<b>bold</b>", "<b>bold</b>"),
        ("</a>orphan close", "orphan close"),
        ("two <a href='1'>one</a> and <a href='2'>two</a>", "two one and two"),
    ],
)
def test_cases_hold_at_every_split_point(raw: str, expected: str) -> None:
    assert _through(AnchorStripper(), [raw]) == expected
    for pieces in _every_split(raw):
        assert _through(AnchorStripper(), pieces) == expected, f"split: {pieces!r}"


def test_character_at_a_time_is_the_same_as_all_at_once() -> None:
    """The worst split there is."""
    raw = 'CDK5 <a href="https://reactome.org/content/detail/R-HSA-1">p25</a> binds.'
    assert _through(AnchorStripper(), list(raw)) == "CDK5 p25 binds."


def test_random_splits_agree_with_whole_input() -> None:
    raw = "x <a href='u'>a</a> y <a href='v'>b</a> z < w"
    whole = _through(AnchorStripper(), [raw])
    rng = random.Random(0)
    for _ in range(200):
        pieces, rest = [], raw
        while rest:
            cut = rng.randint(1, min(4, len(rest)))
            pieces.append(rest[:cut])
            rest = rest[cut:]
        assert _through(AnchorStripper(), pieces) == whole


def test_an_unclosed_tag_is_not_swallowed_at_the_end() -> None:
    """A stream that dies mid-tag must not silently lose the text before it."""
    stripper = AnchorStripper()
    out = stripper.feed('kept text <a href="https://truncated')
    assert out.startswith("kept text ")
    assert "kept text " + stripper.flush() == 'kept text <a href="https://truncated'
