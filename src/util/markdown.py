"""Escaping for text placed into markdown the chat renders.

Reactome pathway names contain markdown syntax -- measured over a real
2,679-pathway result: `NOTCH1:M1580_K2555`, `H139Hfs13* PPM1K ...`. Two of
`_` or `*` in one table cell become emphasis, and a variant identifier
renders with characters silently missing; a `|` splits the cell.
"""

SPECIAL = "\\`*_[]<>|~"


def escape(text: str) -> str:
    """Literal text, on one line: a newline would end a table row."""
    flat = " ".join(text.splitlines())
    return "".join(f"\\{ch}" if ch in SPECIAL else ch for ch in flat)
