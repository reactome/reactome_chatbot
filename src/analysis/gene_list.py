"""A gene list typed into the chat, run as an over-representation analysis.

Asked in the chat, verbatim: "can we do a gsa analysis in the chat. I want
to do it with genes TP53, ERBB2 and RUNX2". It got upload instructions for an
expression matrix -- true, and no use: three genes are not a matrix. What
Reactome runs on a list of genes is over-representation, and the Analysis
Service will run it in a second, so the chat now does.

Pure functions, like `gsa.chat`: what counts as a gene list, and what the
reader is told, are tested here; the handler is wiring.

**Recognising a request is a heuristic, and it only proposes.** The chat
shows the identifiers it read and asks before submitting anything, so a
misfire costs one click ("No, answer my question") and a dropped or extra
token is visible before it matters. The first version ran immediately; an
adversarial review found it answered 14 of 44 ordinary questions ("Could you
explain why IFNG and TNF are enriched...") with a results table.

A message is a request when it has an analysis term, a request to do one,
no sign of being a question *about* something ("explain", "why",
"compare"...), and at least `MIN_IDENTIFIERS` identifiers. A single gene is
not a list: an enrichment of one gene is every pathway it is in, which the
model answers better.
"""

import re
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from analysis.client import MAX_SUBMITTED_IDENTIFIERS
from util.markdown import escape

MIN_IDENTIFIERS = 2
TOP_PATHWAYS = 10
#: Below this many matched identifiers, say the FDRs rest on very few hits.
SMALL_LIST = 10
MAX_UNMATCHED_LISTED = 20
#: How many parsed identifiers the confirmation lists by name.
MAX_PROPOSED_LISTED = 40

_ANALYSIS_TERMS = re.compile(
    r"\b(gsea|gsa|ora|enrich\w*|over[\s-]?represent\w*|analy[sz]\w*"
    r"|through\s+reactome|which\s+(\w+\s+)?pathways|to\s+pathways)\b",
    re.IGNORECASE,
)
_REQUEST = re.compile(
    r"\b(run|perform|carry\s+out|execute|submit|analy[sz]e|map|find"
    r"|do\s+(an?\s+|the\s+|my\s+|some\s+)?([\w-]+\s+){0,3}(analysis|enrichment|ora|gsa|gsea)"
    r"|which\s+(\w+\s+)?pathways|over[\s-]?represented\s+in"
    r"|enrichment\s+(for|on))\b",
    re.IGNORECASE,
)
#: A question about something, not a request to compute it.
_ABOUT = re.compile(
    r"\b(explain\w*|why|how|describe|what\s+(does|is|are|would|do)|difference"
    r"|compar\w*|check\s+if|whether|interpret\w*|understand|discuss|talk\s+about"
    r"|tell\s+me|summar\w*|meaning|mean|literature|correct)\b",
    re.IGNORECASE,
)

#: A token: anything an identifier cannot contain splits. Hyphens stay
#: (isoforms like P04637-2, symbols like HLA-A); dots split (Ensembl versions).
_TOKEN = re.compile(r"[A-Za-z0-9_-]+")
_UNIPROT = re.compile(
    r"\A([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})(-\d+)?\Z"
)
_ENSEMBL = re.compile(r"\AENS[A-Z]*[GTP]\d{11}\Z")
#: A symbol in capitals: EGFR, KRAS, HLA-A. Needs no digit.
_CAPS_SYMBOL = re.compile(r"\A[A-Z][A-Z0-9-]{2,14}\Z")
#: Any case, with a digit: TP53, Trp53, p53, ERBB2.
_DIGIT_SYMBOL = re.compile(r"\A[A-Za-z][A-Za-z0-9-]{1,14}\Z")
#: Any case, no digit -- only inside a list (see `_list_runs`): egfr, kras.
_ANY_SYMBOL = re.compile(r"\A[A-Za-z][A-Za-z0-9-]{1,14}\Z")
#: Shaped like identifiers, but of something else.
_OTHER_ACCESSIONS = re.compile(
    r"\A(chr[0-9XYM]|rs\d|GS[EM]\d|hg\d|GRCh|v\d|pH\d|Q\d\Z|R-[A-Z]{3}-\d"
    r"|log\d|COVID|SARS|PMID|HEK\d|MCF\d|HCT\d)",
    re.IGNORECASE,
)

#: Words, not genes, in messages about analyses -- compared upper-cased.
_NOT_GENES = frozenset(
    """GSA GSEA ORA GSVA PADOG CAMERA DAVID GEO KEGG GO RNA DNA MRNA CDNA
    RNA-SEQ SCRNA-SEQ TSV CSV TXT XLS XLSX PDF FDR API URL ID IDS DE DEG DEGS
    CHEBI HGNC UNIPROT ENSEMBL REACTOME NCBI EBI NIH OICR USA UK OK HELLO
    THANKS HUMAN MOUSE RAT CELLS CELL NOT ALSO AND OR THE FOR YOU CAN PLEASE
    RUN WITH GENE GENES LIST HELP HOW WHAT WHY THESE THIS THAT MY OUR ME IT
    THEM ON OF IN TO AN IS ARE ALL SOME ANALYSIS ANALYSE ANALYZE ENRICHMENT
    PATHWAY PATHWAYS PROTEINS PROTEIN IDENTIFIERS FOLLOWING HERE THANK""".split()
)


def _is_identifier(token: str, *, in_list: bool, shouting: bool) -> bool:
    if token.upper() in _NOT_GENES or _OTHER_ACCESSIONS.match(token):
        return False
    if _UNIPROT.match(token) or _ENSEMBL.match(token):
        return True
    if any(c.isdigit() for c in token):
        return bool(_DIGIT_SYMBOL.match(token))
    if in_list:
        return bool(_ANY_SYMBOL.match(token))
    return not shouting and bool(_CAPS_SYMBOL.match(token))


#: Where a list starts: after a colon, a question mark, a newline, or a
#: preposition; and a list continues across commas, semicolons, whitespace
#: and "and"/"or". Anything else -- a full stop, a word that is not an
#: identifier -- ends it.
_LIST_START = re.compile(r"[:?\n]|\b(for|on|of|with|genes|proteins)\b", re.IGNORECASE)
_LIST_GAP = re.compile(
    r"\A(\s*[,;]?\s*|\s+(and|or)\s+|\s*,\s*(and|or)\s+)\Z", re.IGNORECASE
)


def _list_runs(text: str, shouting: bool) -> set[tuple[int, int]]:
    """Spans of tokens inside a list, where lower-case symbols are accepted.

    A list is two or more identifier-shaped tokens in a row, starting after
    a list opener. Free text is not a list, so "enrichment for egfr, kras"
    reads egfr and kras, and "which pathways are enriched" reads nothing.
    """
    spans: set[tuple[int, int]] = set()
    for opener in _LIST_START.finditer(text):
        run: list[tuple[int, int]] = []
        pos = opener.end()
        for match in _TOKEN.finditer(text, pos):
            if not _LIST_GAP.match(text[pos : match.start()]):
                break
            if not _is_identifier(match.group(), in_list=True, shouting=shouting):
                break
            run.append(match.span())
            pos = match.end()
        separators = {text[a:b] for (_, a), (b, _) in pairwise(run)}
        # Space-separated lower-case words are prose unless a colon, question
        # mark or newline opened the list.
        if len(run) >= MIN_IDENTIFIERS and (
            opener.group() in ":?\n"
            or any(s.strip() for s in separators)
            or "\n" in "".join(separators)
        ):
            spans.update(run)
    return spans


def identifiers_in(text: str, *, shouting: bool = False) -> list[str]:
    """Identifier-shaped tokens, in order, each once (case-insensitively)."""
    in_list = _list_runs(text, shouting)
    seen: set[str] = set()
    found: list[str] = []
    for match in _TOKEN.finditer(text):
        token = match.group().strip("-")
        if not token or token.upper() in seen:
            continue
        if _is_identifier(token, in_list=match.span() in in_list, shouting=shouting):
            seen.add(token.upper())
            found.append(token)
    return found


def gene_list_request(text: str) -> list[str] | None:
    """The identifiers to propose analysing, if this message asks for it."""
    request = _REQUEST.search(text)
    if not (request and _ANALYSIS_TERMS.search(text)) or _ABOUT.search(text):
        return None
    # Typed in capitals, every word looks like a symbol; then only tokens
    # with a digit, or in the accession formats, count.
    found = identifiers_in(text, shouting=request.group().isupper())
    return found if len(found) >= MIN_IDENTIFIERS else None


def describe_proposal(identifiers: list[str]) -> str:
    """What the chat is about to submit, shown before it does."""
    shown = ", ".join(f"`{i}`" for i in identifiers[:MAX_PROPOSED_LISTED])
    more = len(identifiers) - MAX_PROPOSED_LISTED
    if more > 0:
        shown += f" and {more:,} more"
    limit = (
        f" Only the first {MAX_SUBMITTED_IDENTIFIERS:,} will be submitted."
        if len(identifiers) > MAX_SUBMITTED_IDENTIFIERS
        else ""
    )
    return (
        "With a list of genes, the Reactome analysis to run is "
        "**over-representation**: which pathways contain more of your genes "
        "than chance would put there. (A gene set analysis with ReactomeGSA "
        "needs expression measurements for each sample — attach a matrix "
        "with 📎 if you have one.)\n\n"
        f"I read **{len(identifiers)} identifiers** in your message: {shown}.{limit}\n\n"
        "Run the analysis on these?"
    )


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
    # Clamped, and hidden if the service counts more misses than we sent:
    # a count that cannot be right is not shown.
    matched = (
        len(submitted) - not_found
        if isinstance(not_found, int) and 0 <= not_found <= len(submitted)
        else None
    )
    pathways = [p for p in result.get("pathways") or [] if isinstance(p, dict)]
    total = result.get("pathwaysFound")

    lines = ["**Over-representation analysis** of your gene list, in Reactome."]
    if matched is not None:
        lines += ["", f"Matched **{matched} of {len(submitted)}** identifiers."]
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
            entities = p.get("entities")
            if not isinstance(entities, dict):
                entities = {}
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
        # The lookup is paged; the service's own count is the real total.
        total_unmatched = not_found if isinstance(not_found, int) else len(unmatched)
        more = max(total_unmatched, len(unmatched)) - min(
            len(unmatched), MAX_UNMATCHED_LISTED
        )
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
