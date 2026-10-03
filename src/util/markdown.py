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
#: and dropped. Any colon followed by a letter starts one, whatever precedes
#: it -- the first version only escaped colons after a letter or digit, and
#: 1,206 of Release 97's 25,286 colon names still broke ("(ACTA2,ACTG2):ATP",
#: "TNF-alpha:TNFR1"). Measured with the same parser chain: 0 now, in prose, link
#: labels and table cells. URLs ("://"), times and "Note: x" are untouched.
_DIRECTIVE_COLON = re.compile(r"(?<!\\):(?=[^\W\d_])")
#: Code is shown verbatim, so an escape there would show as a backslash.
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.S)


def escape_directives(text: str) -> str:
    """Keep "A:B" literal in rendered markdown; code is left alone."""
    parts = _CODE.split(text)
    return "".join(
        part if index % 2 else _DIRECTIVE_COLON.sub(r"\\:", part)
        for index, part in enumerate(parts)
    )
