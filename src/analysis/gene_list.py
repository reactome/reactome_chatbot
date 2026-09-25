"""A gene list typed into the chat, run as an over-representation analysis.

Asked in the chat, verbatim: "can we do a gsa analysis in the chat. I want
to do it with genes TP53, ERBB2 and RUNX2". It got upload instructions for an
expression matrix -- true, and no use: three genes are not a matrix. What
Reactome runs on a list of genes is over-representation, and the Analysis
Service will run it in a second, so the chat now does.

Pure functions, like `gsa.chat`: what counts as a gene list, and what the
reader is told, are tested here; the handler is wiring.

**Recognising a request is a heuristic, so it is narrow on purpose.** It
needs all three of: an analysis term, a request to do something, and at
least `MIN_IDENTIFIERS` identifier-shaped tokens. A question *about* genes
("can you explain how TP53 and MDM2 interact?") has no analysis term and
still goes to the model. A single gene is not a list: an enrichment of one
gene is every pathway that gene is in, which the model answers better.
"""

import re
from dataclasses import dataclass
from typing import Any

from analysis.client import MAX_SUBMITTED_IDENTIFIERS
from util.markdown import escape

MIN_IDENTIFIERS = 2
TOP_PATHWAYS = 10
#: Below this many matched identifiers, say the FDRs rest on very few hits.
SMALL_LIST = 10
MAX_UNMATCHED_LISTED = 20

_ANALYSIS_TERMS = re.compile(
    r"\b(gsea|gsa|reactome\s*gsa|ora|enrichment|enriched"
    r"|over[\s-]?representation|over[\s-]?represented"
    r"|(pathway|enrichment|gene[\s-]*set)\s+analy[sz]\w*"
    r"|analy[sz](e|is|ing)\s+(on\s+|of\s+|for\s+|with\s+)?"
    r"(my|our|these|this|the\s+following|a|the)?\s*"
    r"(gene|genes|list|proteins?|identifiers?|ids)\b)",
    re.IGNORECASE,
)
_WANTS_TO_DO_IT = re.compile(
    r"\b(run|do|perform|start|carry\s+out|execute|submit|analy[sz]e"
    r"|can\s+(we|i|you)|could\s+(we|i|you)|please|want\s+to|would\s+like\s+to"
    r"|help\s+me|with\s+(the\s+)?(genes?|proteins?|list)|for\s+(the\s+)?(genes?|proteins?))\b",
    re.IGNORECASE,
)

#: Split on anything an identifier cannot contain. Hyphens stay (isoforms
#: like P04637-2, symbols like HLA-A); dots split (Ensembl versions).
_SEPARATORS = re.compile(r"[^A-Za-z0-9_-]+")
_UNIPROT = re.compile(
    r"\A([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})(-\d+)?\Z"
)
_ENSEMBL = re.compile(r"\AENS[A-Z]*[GTP]\d{11}\Z")
#: A symbol in capitals: EGFR, KRAS, HLA-A. Needs no digit.
_CAPS_SYMBOL = re.compile(r"\A[A-Z][A-Z0-9-]{2,14}\Z")
#: Any case, but with a digit: TP53, Trp53, p53, ERBB2.
_DIGIT_SYMBOL = re.compile(r"\A[A-Za-z][A-Za-z0-9-]{1,14}\Z")

#: Capitals that are words, not genes, in messages about analyses.
_NOT_GENES = frozenset(
    """GSA GSEA ORA GSVA PADOG CAMERA RNA DNA MRNA CDNA RNA-SEQ TSV CSV TXT XLS
    XLSX PDF FDR API URL IDS AND THE FOR YOU CAN PLEASE RUN WITH GENE GENES
    LIST HELP HOW WHAT WHY USA NIH OICR EBI NCBI HGNC UNIPROT ENSEMBL REACTOME
    KEGG OK HELLO THANKS""".split()
)


def _looks_like_identifier(token: str, *, shouting: bool) -> bool:
    if token.upper() in _NOT_GENES:
        return False
    if _UNIPROT.match(token) or _ENSEMBL.match(token):
        return True
    has_digit = any(c.isdigit() for c in token)
    if has_digit:
        return bool(_DIGIT_SYMBOL.match(token))
    # A message typed in capitals makes every word look like a symbol.
    return not shouting and bool(_CAPS_SYMBOL.match(token))


def _is_shouting(text: str) -> bool:
    words = re.findall(r"[A-Za-z]{2,}", text)
    upper = sum(w.isupper() for w in words)
    return len(words) >= 4 and upper / len(words) > 0.5


def identifiers_in(text: str) -> list[str]:
    """Identifier-shaped tokens, in order, each once (case-insensitively)."""
    shouting = _is_shouting(text)
    seen: set[str] = set()
    found: list[str] = []
    for token in _SEPARATORS.split(text):
        token = token.strip("-")
        if (
            token
            and token.upper() not in seen
            and _looks_like_identifier(token, shouting=shouting)
        ):
            seen.add(token.upper())
            found.append(token)
    return found


def gene_list_request(text: str) -> list[str] | None:
    """The identifiers to analyse, if this message asks for an analysis of them."""
    if not (_ANALYSIS_TERMS.search(text) and _WANTS_TO_DO_IT.search(text)):
        return None
    found = identifiers_in(text)
    return found if len(found) >= MIN_IDENTIFIERS else None


@dataclass(frozen=True)
class Overrepresentation:
    """What the reader is told, and whether there was anything to continue."""

    text: str
    #: False when nothing matched -- nothing for a follow-up to be about.
    has_pathways: bool


def _fdr(value: Any) -> str:
    return f"{value:.2g}" if isinstance(value, int | float) else "–"


def describe_overrepresentation(
    submitted: list[str],
    result: dict[str, Any],
    browser_url: str,
    unmatched: list[str] | None,
    *,
    truncated: bool = False,
) -> Overrepresentation:
    """The reply to a gene list: which matched, the top pathways, a link.

    Also becomes the model's previous turn, so a follow-up ("which of these
    involve TP53?") is answered from this result. That is why the table
    carries stable identifiers as well as names.
    """
    not_found = result.get("identifiersNotFound")
    matched = len(submitted) - not_found if isinstance(not_found, int) else None
    pathways = [p for p in result.get("pathways") or [] if isinstance(p, dict)]
    total = result.get("pathwaysFound")

    lines = [
        "With a list of genes, the Reactome analysis to run is "
        "**over-representation**: which pathways contain more of your genes "
        "than chance would put there. (A gene set analysis with ReactomeGSA "
        "needs expression measurements for each sample — attach a matrix "
        "with 📎 if you have one.)",
        "",
    ]
    if matched is not None:
        lines.append(f"Matched **{matched} of {len(submitted)}** identifiers.")
    if truncated:
        lines.append(
            f"Only the first {MAX_SUBMITTED_IDENTIFIERS:,} identifiers were submitted."
        )

    if not pathways:
        lines += ["", "No Reactome pathways contain any of them."]
    else:
        count = f"{total:,}" if isinstance(total, int) else str(len(pathways))
        lines += [
            "",
            f"Top {len(pathways[:TOP_PATHWAYS])} of {count} pathways, by FDR:",
            "",
            "| Pathway | Entities found | FDR |",
            "|---|---|---|",
        ]
        for p in pathways[:TOP_PATHWAYS]:
            entities = p.get("entities") or {}
            name = escape(str(p.get("name", "")))
            st_id = escape(str(p.get("stId", "")))
            lines.append(
                f"| {name} ({st_id}) | {entities.get('found', '–')} of "
                f"{entities.get('total', '–')} | {_fdr(entities.get('fdr'))} |"
            )
        lines += [
            "",
            # Measured: TP53, ERBB2, RUNX2 found "7 of 1574" -- entities count
            # each form of a protein, so a column called "genes" read as 7 of 3.
            "Entities are Reactome's molecules, so one gene can count more "
            "than once (its protein in several forms or complexes).",
            "",
            f"[Open the full result in the Pathway Browser]({browser_url})",
        ]
        if matched is not None and matched < SMALL_LIST:
            lines += [
                "",
                f"With {matched} matched genes, each pathway rests on very few "
                "of your genes, so treat these as pointers rather than findings.",
            ]

    if unmatched:
        shown = ", ".join(escape(u) for u in unmatched[:MAX_UNMATCHED_LISTED])
        more = len(unmatched) - MAX_UNMATCHED_LISTED
        lines += [
            "",
            f"Not found in Reactome: {shown}"
            + (f" and {more} more" if more > 0 else ""),
        ]
    return Overrepresentation(text="\n".join(lines), has_pathways=bool(pathways))


FAILED = (
    "I tried to run an over-representation analysis on those genes, but "
    "Reactome's Analysis Service didn't return a result. Please try again in "
    "a moment."
)
