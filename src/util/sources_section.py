"""Drop the trailing source list from a token stream split at arbitrary points.

The search page renders citations as its own chips, built from the `citation`
events, so the prose list at the end of an answer is a strictly worse copy of
data the caller already has -- worse still after `AnchorStripper`, which leaves
it as bare display names with no links.

The website was matching the heading itself and could not win. Measured
2026-09-19, our prompts never specified one: they asked for "a bullet-point
list of each unique citation anchor" and said nothing about a heading, so the
model invented one per answer -- `## Sources`, `## Most relevant sources`,
`relevant references`, `Key sources`, `Top citations`. That is not five
phrasings of a contract, it is five samples from an unconstrained generator.

The prompts now pin the heading to `## Sources`. This is the second half: strip
it here so no caller has to pattern-match model output at all.

**One risk is known, measured, and deliberately not defended against.** This
stripper is a state machine: once it commits to a heading it drops everything
after it, and unlike a function that recomputes over accumulated text it cannot
change its mind. So a *complete* `## Sources` line in the middle of an answer,
with real prose after it, would cost the reader that prose.

Measured 2026-09-19 over six real answers spanning the reactome and userguide
prompts: the heading was `## Sources` every time, last every time, with nothing
but list items after it. Zero mid-answer occurrences. Guarding against it would
mean holding the text after a heading until a list item confirms the verdict --
buildable, and more state than an unobserved failure justifies. If a mid-answer
heading is ever seen, that is the fix; it is recorded here rather than built. The pattern below
is still tolerant, because the prompt is an instruction and not a guarantee, but
it is bounded -- a heading or bold-only line, at most five words, naming sources
-- rather than any line mentioning the word.
"""

import re

# A whole line that is a markdown heading or a bold-only line naming nothing
# but the source list. Two rules keep it off real content, and the first was
# learned the hard way: an earlier version allowed any short heading that
# mentioned sources, and "## Sources of reactive oxygen species" -- an entirely
# plausible Reactome heading -- truncated the answer there.
#
# 1. The noun must be the LAST word. "Sources of oxidative stress" is about
#    biology; "Most relevant sources" is a source list.
# 2. Only a closed set of qualifiers may precede it, so "Cellular sources" is
#    left alone.
#
# The asymmetry justifies the strictness. A heading this misses costs the
# reader a duplicate list at the end -- cosmetic, and exactly what the website
# lives with today. A heading this matches wrongly costs them the rest of the
# answer. When in doubt, do not match.
_QUALIFIER = (
    r"(?:most|more|relevant|key|top|main|primary|all|further|additional"
    r"|related|supporting|complete|full|cited|used)"
)
_HEADING = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]+|\*\*[ \t]*)"
    rf"(?:{_QUALIFIER}[ \t]+){{0,3}}"
    r"(?:sources?|references?|citations?)"
    r"[ \t]*:?[ \t]*(?:\*\*)?[ \t]*:?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

# Held back while a partial line could still turn out to be that heading. A
# heading is one short line; beyond this the text is prose and is released.
_MAX_HELD = 120


def _could_become_heading(partial: str) -> bool:
    """True while an unterminated line might still turn into the heading.

    A heading starts a line with `#` or `**`. A bullet (`* `) cannot, and
    neither can prose, so both stream straight through.
    """
    stripped = partial.lstrip(" \t")
    if len(stripped) > _MAX_HELD:
        return False  # Too long to be a heading; it is prose.
    if not stripped:
        return True
    if stripped[0] == "#":
        return True
    return stripped == "*" or stripped.startswith("**")


class SourcesSectionStripper:
    """Feed fragments in, get the answer without its trailing source list.

    Everything from the heading to the end of the stream is dropped: the
    prompts place the list last, so there is nothing after it to keep.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._done = False
        # Whether the buffer currently begins at a real start of line. Once
        # text has been released, position 0 is mid-line -- and `^` under
        # re.MULTILINE matches there anyway, which turned a bolded word in
        # mid-sentence into a heading and ate the rest of the answer.
        self._at_line_start = True

    def feed(self, text: str) -> str:
        if self._done:
            return ""
        self._buffer += text
        # Terminated only: mid-stream, "## Sources" matches before
        # " of reactive oxygen species" has arrived, and deciding then drops
        # the rest of a perfectly good answer. Only the whole-string probe
        # caught this; the character-by-character one is what found it.
        match = self._find_heading(terminated_only=True)
        if match:
            out = self._buffer[: match.start()]
            self._buffer = ""
            self._done = True
            return out
        # Hold back only a partial line that could still become that heading.
        # Holding every unterminated line instead would defeat the point of
        # the endpoint: an answer often has no newline until it ends, so the
        # whole thing would arrive in one blob at `flush`. Two existing tests
        # caught exactly that.
        newline = self._buffer.rfind("\n")
        if newline == -1 and not self._at_line_start:
            cut = len(self._buffer)  # No line start in here at all.
        else:
            cut = newline + 1
            if not _could_become_heading(self._buffer[cut:]):
                cut = len(self._buffer)
        out, self._buffer = self._buffer[:cut], self._buffer[cut:]
        if out:
            self._at_line_start = out.endswith("\n")
        return out

    def flush(self) -> str:
        """Whatever is still held, once the stream has ended."""
        if self._done:
            return ""
        match = self._find_heading(terminated_only=False)
        held = self._buffer[: match.start()] if match else self._buffer
        self._buffer = ""
        self._done = True
        return held

    def _find_heading(self, *, terminated_only: bool) -> re.Match[str] | None:
        """The heading, ignoring a match at offset 0 when that is mid-line.

        `terminated_only` rejects a match that runs to the end of the buffer,
        because more of that line may still arrive. At `flush` the stream has
        ended, so there is nothing more to wait for.
        """
        position = 0
        while True:
            match = _HEADING.search(self._buffer, position)
            if match is None:
                return None
            if terminated_only and match.end() >= len(self._buffer):
                return None  # The line has not ended; it may yet grow.
            if match.start() or self._at_line_start:
                return match
            newline = self._buffer.find("\n")
            if newline == -1:
                return None
            position = newline + 1
