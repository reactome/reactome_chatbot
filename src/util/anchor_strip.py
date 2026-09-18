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
