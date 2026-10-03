"""Escaping for text placed into markdown the chat renders.

Reactome pathway names contain markdown syntax -- measured over a real
2,679-pathway result: `NOTCH1:M1580_K2555`, `H139Hfs13* PPM1K ...`. Two of
`_` or `*` in one table cell become emphasis, and a variant identifier
renders with characters silently missing; a `|` splits the cell.
"""

import re

SPECIAL = "\\`*_[]<>|~"


def escape(text: str) -> str:
    """Literal text, on one line: a newline would end a table row."""
    flat = " ".join(text.splitlines())
    escaped = "".join(f"\\{ch}" if ch in SPECIAL else ch for ch in flat)
    return escape_directives(escaped)


def inert_html(text: str) -> str:
    """Markdown kept, HTML made literal: every `<` is escaped.

    For text a model wrote that reaches a reader other than the one who
    prompted it -- a handed-off summary. The chat renders HTML
    (`unsafe_allow_html`), so markup a visitor steered into an answer ran in
    the browser of whoever opened their handoff link (review, area 1b). `escape`
    would also flatten the summary's bold and lists; this only stops tags.
    """
    return escape_directives(text.replace("<", "\\<"))


#: A colon that the chat's markdown would read as a directive: `remark-directive`
#: is in Chainlit's renderer, so the ":p25" in "CDK5:p25" was parsed as markup
#: and dropped -- shown as "CDK5", a break, then the rest. Reactome names are
#: full of these (complexes are written A:B). Never "://" in a URL.
_DIRECTIVE_COLON = re.compile(r"(?<=[A-Za-z0-9]):(?=[A-Za-z])(?!//)")


def escape_directives(text: str) -> str:
    """Keep "A:B" literal in rendered markdown; nothing else changes."""
    return _DIRECTIVE_COLON.sub(r"\\:", text)
