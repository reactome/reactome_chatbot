"""Remove HTML anchors from a token stream that is split at arbitrary points.

contracts/answer_endpoint.md promises the search page prose without embedded
anchors -- citations arrive as their own events so the website can style links
itself. The `react-to-me` profile is the chat UI's, and its prompt asks for
inline `<a href=...>` links, so the endpoint inherited them.

They cannot be stripped a fragment at a time. Measured on the live endpoint, one
anchor arrived as twenty-odd fragments: ' <', 'a', ' href', '="', 'https', '://',
'react', 'ome', '.org', '/content', '/detail', '/R', '-H', 'SA', ... A caller
rendering incrementally would show that verbatim before it became a link.

So this holds back only the part of the buffer that could still turn into an
anchor, and passes everything else straight through.
"""

import re

# A complete opening or closing anchor tag.
_TAG = re.compile(r"<a\b[^>]*>|</a\s*>", re.IGNORECASE)

# An unclosed '<...' is held only while it could still become an anchor. Beyond
# this it is treated as prose: real tags here are about seventy characters, and
# holding forever would strand text that never arrives.
_MAX_HELD = 200


class AnchorStripper:
    """Feed fragments in, get anchor-free text out. Call `flush` at the end."""

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, text: str) -> str:
        self._buffer += text
        out: list[str] = []
        while self._buffer:
            start = self._buffer.find("<")
            if start == -1:
                out.append(self._buffer)
                self._buffer = ""
                break
            out.append(self._buffer[:start])
            self._buffer = self._buffer[start:]

            match = _TAG.match(self._buffer)
            if match:
                self._buffer = self._buffer[match.end() :]
                continue
            if not self._could_become_anchor():
                out.append("<")
                self._buffer = self._buffer[1:]
                continue
            break  # Incomplete: wait for more.
        return "".join(out)

    def _could_become_anchor(self) -> bool:
        """True while the held '<...' might still complete into an anchor tag.

        Checked before length, so prose like "x < y" is released immediately
        rather than waiting: the character after '<' settles it.
        """
        held = self._buffer
        if len(held) == 1:
            return True  # Just '<'; the next character decides.
        if ">" in held:
            return False  # A complete tag that _TAG already declined.
        if len(held) > _MAX_HELD:
            return False
        after = held[1]
        if after == "/":
            return len(held) == 2 or held[2] in "aA"
        return after in "aA"

    def flush(self) -> str:
        """Whatever is still held, emitted as prose. A truncated tag is not one."""
        remaining, self._buffer = self._buffer, ""
        return remaining


# The longest link worth holding back for: display names are short, and URLs
# here are under a hundred characters. Past this, a held '[' is prose.
_MAX_LINK = 400


class MarkdownLinkStripper:
    """`[text](url)` becomes `text`, across arbitrary fragment boundaries.

    Since citations moved from HTML anchors to markdown links (so the chat can
    render without HTML -- review, area 1b), the search page's prose needs
    these stripped the way `AnchorStripper` strips anchors. Holds back only
    from a '[' that could still become a link; anything else streams straight
    through. Call `flush` at the end.
    """

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, text: str) -> str:
        self._buffer += text
        out: list[str] = []
        while self._buffer:
            start = self._buffer.find("[")
            if start == -1:
                out.append(self._buffer)
                self._buffer = ""
                break
            out.append(self._buffer[:start])
            self._buffer = self._buffer[start:]
            state, label, end = self._parse()
            if state == "partial":
                break  # Could still become a link: wait for more.
            if state == "prose":
                out.append("[")
                self._buffer = self._buffer[1:]
                continue
            out.append(label)
            self._buffer = self._buffer[end:]
        return "".join(out)

    def _parse(self) -> tuple[str, str, int]:
        """("partial" | "prose" | "link", label, end of the link)."""
        held = self._buffer
        close = held.find("]")
        if close == -1:
            state = (
                "partial" if len(held) <= _MAX_LINK and "\n" not in held else "prose"
            )
            return state, "", 0
        if close + 1 == len(held):
            return "partial", "", 0  # '[label]' -- the next character decides.
        if held[close + 1] != "(":
            return "prose", "", 0
        end = held.find(")", close + 2)
        if end == -1:
            tail = held[close + 2 :]
            if len(held) > _MAX_LINK or any(c.isspace() for c in tail):
                return "prose", "", 0
            return "partial", "", 0
        url = held[close + 2 : end]
        if not url or any(c.isspace() for c in url):
            return "prose", "", 0
        return "link", held[1:close], end + 1

    def flush(self) -> str:
        """Whatever is still held, as prose: a truncated link is not one."""
        remaining, self._buffer = self._buffer, ""
        return remaining
