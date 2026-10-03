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
    r"|enrichment\s+(for|on)|(enrichment|ora|gsea|analysis)\s+please)\b",
    re.IGNORECASE,
)
#: Handing a list over: a request and an analysis term at once, because in
#: a chat that analyses lists, "here is my gene list TP53, ERBB3 and JAX9"
#: means "analyse these". Reported as missed on 2026-09-29, straight after
#: the chat had said "include the genes in your message".
_HANDED_OVER = re.compile(
    r"\b(here(\s+is|\s+are|\s*'s|s)\s+(my|the|a|our)\s+(gene\s+list|list\s+of\s+genes|genes|list)"
    r"|(my|our)\s+(gene\s+list|genes|list\s+of\s+genes)\s*(is|are|:)"
    r"|these\s+are\s+(my|our|the)\s+genes|gene\s+list\s*:)",
    re.IGNORECASE,
)
#: A question about something, not a request to compute it.
_ABOUT = re.compile(
    r"\b(explain\w*|why|how|describe|what\s+(does|is|are|would|do)|difference"
    r"|compar\w*|check\s+if|whether|interpret\w*|understand|discuss|talk\s+about"
    r"|tell\s+me|summar\w*|meaning|mean|literature|correct"
    # From a second review's fresh questions: prose about genes, not a list.
    r"|role|relationship|between|shared|(?<![\w-])regulat\w*|involv\w*|papers?|steps"
    r"|including|mutations?|variants?"
    # A matrix's columns are samples, for the GSA flow, not identifiers.
    r"|matrix|columns?)\b",
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
#: Any case, no digit -- only inside a list (see `_accepted`): egfr, kras.
_ANY_SYMBOL = re.compile(r"\A[A-Za-z][A-Za-z0-9-]{1,14}\Z")
#: Shaped like identifiers, but of something else.
_OTHER_ACCESSIONS = re.compile(
    r"\A(chr[0-9XYM]|rs\d|GS[EM]\d|hg\d|GRCh|v\d|pH\d|Q\d\Z|R-[A-Z]{3}-\d"
    r"|log\d|COVID|SARS|PMID|HEK\d|MCF\d|HCT\d|U2OS|HepG2|A549\Z|K562\Z"
    # Sample and group labels: Sample1, rep2, day3, t0, ctrl1.
    r"|(sample|rep|replicate|day|week|condition|group|batch|patient|donor|ctrl|t)\d+\Z"
    # Protein variants: G12D, V600E, L858R, T790M.
    r"|[A-Z]\d{2,4}[A-Z]\Z)",
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
    PATHWAY PATHWAYS PROTEINS PROTEIN IDENTIFIERS FOLLOWING HERE THANK
    PERFORM SUBMIT MAP FIND DO EXECUTE ETC
    CONTROL CONTROLS TREATED UNTREATED TREATMENT CTRL TRT WT KO KD OE MOCK
    VEHICLE DMSO SAMPLE SAMPLES GENEID GENESYMBOL SYMBOL HELA JURKAT
    YES NO NOPE SURE LATER GREAT""".split()
)


#: Longer than any list worth typing (3,000 identifiers of ~15 characters is
#: 48K). A bound on the work, not a policy: an attached file is the way in
#: for more.
MAX_MESSAGE_CHARS = 60_000

#: Words a list follows: "for", "on", "genes", and the request verbs
#: themselves ("analyze TP53, MDM2").
_OPENER_WORDS = frozenset(
    """FOR ON OF IN WITH GENES PROTEINS LIST THESE FOLLOWING RUN PERFORM SUBMIT
    ANALYSE ANALYZE MAP FIND""".split()
)
#: Joining words, read as part of the gap between two tokens.
_JOINERS = frozenset({"and", "or"})
_REQUEST_START = re.compile(
    r"\A\s*(please\s+)?(run|perform|do|analy[sz]e|submit|execute|map|find)\b",
    re.IGNORECASE,
)
#: List markers people paste: "1. TP53", "2) MDM2", "- CDKN1A", "• BAX".
_MARKERS = re.compile(r"(?m)^[ \t]*(?:\d{1,4}[.)]|[-*•])[ \t]+")
_ENSEMBL_VERSION = re.compile(r"\b(ENS[A-Z]*[GTP]\d{11})\.\d+\b")
# Possessive quantifiers throughout: a gap can be 40,000 newlines long, and
# `\s*,?\s*` over that backtracks quadratically (measured: 24 s).
_AND = re.compile(r"\A\s*+,?\s*+(and|or)\s++\Z", re.IGNORECASE)
_COMMA = re.compile(r"\A[ \t]*+[,;\t|/][ \t]*+\Z")
_SPACE = re.compile(r"\A[ \t]++\Z")
_NEWLINE = re.compile(r"\A[ \t]*+[,;]?[ \t]*+\n[ \t]*+\Z")
_BLANK_LINE = re.compile(r"\n[ \t]*+\n")
#: Pasted text arrives with Windows line ends, non-breaking spaces, quotes
#: and full-width commas; each would otherwise end a list.
_NORMALISE = str.maketrans(
    {
        "\r": "\n",
        "\u00a0": " ",
        "\uff0c": ",",
        "\u3001": ",",
        '"': " ",
        "'": " ",
        "\u201c": " ",
        "\u201d": " ",
        "\u2018": " ",
        "\u2019": " ",
        "`": " ",
    }
)


def _gap_kind(gap: str) -> str | None:
    """How two neighbouring tokens are joined; None ends a list.

    Plain regexes over a gap that is itself bounded by two tokens, so the
    whole message is scanned once. The first version walked from every
    opener to the end of the message: a pasted column of 4,000 genes took
    38 s, on the event loop every session shares.
    """
    if _COMMA.match(gap):
        return "comma"
    if _NEWLINE.match(gap):
        return "newline"
    if _AND.match(gap):
        return "and"
    if _SPACE.match(gap):
        return "space"
    return None


def _strength(token: str) -> str | None:
    """ "strong" (a digit or an accession format), "caps", "weak", or None."""
    if token.upper() in _NOT_GENES or _OTHER_ACCESSIONS.match(token):
        return None
    if _UNIPROT.match(token) or _ENSEMBL.match(token):
        return "strong"
    if any(c.isdigit() for c in token):
        return "strong" if _DIGIT_SYMBOL.match(token) else None
    if _CAPS_SYMBOL.match(token):
        return "caps"
    # "down-regulated" is a word; HLA-A, in capitals, was caught above.
    return "weak" if _ANY_SYMBOL.match(token) and "-" not in token else None


@dataclass
class _Run:
    opened_by_mark: bool
    tokens: list[tuple[str, str]]  # (token, strength)
    kind: str | None = None
    ended_by_and: bool = False


def _accepted(run: _Run, *, shouting: bool, pair_ok: bool, question: bool) -> list[str]:
    tokens = run.tokens
    # Space-separated plain words are prose unless a colon, question mark or
    # newline announced a list: "on TP53 MDM2 using default settings" reads
    # TP53 and MDM2. Capitals are words too when the whole message is.
    if not run.opened_by_mark:
        for index, (_, strength) in enumerate(tokens):
            if (strength == "weak" and run.kind in (None, "space", "and")) or (
                strength == "caps" and shouting
            ):
                tokens = tokens[:index]
                break
    # A list typed in lower case is read in lower case. In any other list a
    # lower-case word is the sentence carrying on: "TP53, MDM2, then show
    # me the top hits", "..., cheers".
    if any(token != token.lower() for token, _ in tokens):
        for index, (token, strength) in enumerate(tokens):
            if strength == "weak" and token == token.lower():
                tokens = tokens[:index]
                break
    if len(tokens) < MIN_IDENTIFIERS:
        return []
    # "X and Y" is how prose names two genes; a list says it with commas.
    # And two genes in a question are the question's subject -- "My
    # enrichment for HIF1A, VEGFA came back empty. What went wrong?" -- more
    # often than a list to run: a third review's four misfires were all this.
    if len(tokens) == 2 and (run.kind == "and" or question) and not pair_ok:
        return []
    return [token for token, _ in tokens]


def identifiers_in(
    text: str, *, shouting: bool = False, anywhere: bool = False
) -> list[str]:
    """The identifiers in the lists in a message, in order, each once.

    A list is two or more identifier-shaped tokens joined the same way
    throughout -- commas, newlines, tabs or spaces, with "and" before the
    last -- and it starts at the beginning of the message, after a colon,
    question mark or newline, or after "for", "on", "genes"... A change of
    separator, a blank line, or a word that is not an identifier ends it,
    so a trailing sentence is not read as genes.
    """
    text = text.replace("\r\n", "\n").translate(_NORMALISE)
    text = _ENSEMBL_VERSION.sub(r"\1", _MARKERS.sub("", text))
    question = "?" in text
    pair_ok = not question and bool(_REQUEST_START.match(text))
    found: list[str] = []
    run: _Run | None = None
    previous_end = 0
    previous_token = ""

    def close() -> None:
        if run is not None:
            found.extend(
                _accepted(run, shouting=shouting, pair_ok=pair_ok, question=question)
            )

    for match in _TOKEN.finditer(text):
        token = match.group().strip("-")
        if not token or token.lower() in _JOINERS:
            # Stays in the gap: "TP53, ERBB2 and RUNX2", "my genes - CTNNB1".
            continue
        gap = text[previous_end : match.start()]
        previous_end = match.end()
        if _BLANK_LINE.search(gap):
            close()
            run = None
            if found:
                # A blank line after a list: what follows is a new paragraph
                # -- "Best,\nJohn" -- not more genes.
                break
        strength = _strength(token) if token else None
        kind = _gap_kind(gap)
        # One separator throughout; "and" may join the last one, after which
        # the list is over.
        if (
            run is not None
            and not run.ended_by_and
            and strength is not None
            and kind is not None
            and (run.kind is None or kind in (run.kind, "and"))
        ):
            run.tokens.append((token, strength))
            run.kind = run.kind or kind
            run.ended_by_and = kind == "and"
        else:
            close()
            run = None
            marked = any(c in gap for c in ":?\n")
            opens = (
                anywhere
                or match.start() == 0
                or marked
                or previous_token.upper() in _OPENER_WORDS
            )
            if strength is not None and opens:
                run = _Run(opened_by_mark=marked, tokens=[(token, strength)])
        previous_token = token
    close()

    seen: set[str] = set()
    unique: list[str] = []
    for token in found:
        if token.upper() not in seen:
            seen.add(token.upper())
            unique.append(token)
    return unique


def _looks_like_genes(found: list[str]) -> bool:
    """Enough to stand as a list without a request around it: three or more,
    or at least one that is gene-shaped rather than a plain word. "yes,
    great" and "control, treated" are replies, not lists."""
    return len(found) >= 3 or any(_strength(t) in ("strong", "caps") for t in found)


def gene_list_request(text: str) -> list[str] | None:
    """The identifiers to propose analysing, if this message asks for it."""
    if len(text) > MAX_MESSAGE_CHARS:
        return None
    request = _REQUEST.search(text) or _HANDED_OVER.search(text)
    handed_over = _HANDED_OVER.search(text) is not None
    if not (request and (handed_over or _ANALYSIS_TERMS.search(text))) or _ABOUT.search(
        text
    ):
        return None
    # Typed in capitals, every word looks like a symbol; then only tokens
    # with a digit, or in the accession formats, count.
    found = identifiers_in(text, shouting=request.group().isupper())
    if len(found) < MIN_IDENTIFIERS and handed_over:
        # "my genes are TP53, MDM2, CDKN1A": the list follows the phrase,
        # which opens it as a colon would. ("are" does not open lists in
        # general: "...where the controls are WT, KO".)
        match = _HANDED_OVER.search(text)
        if match is not None:
            # Opened as "for" would, not as a colon: a comma list of any case
            # is read, but space-separated words are not -- "my genes are
            # highly expressed in muscle" is prose (held-out set 5).
            found = identifiers_in("for " + text[match.end() :])
    return found if len(found) >= MIN_IDENTIFIERS else None


#: Pointing back at a list from an earlier message: "can you analyze the
#: gene list that I gave you" (reported 2026-09-29, answered by the model).
#: It must name a gene list -- "analyse the pathways above" or "run it again"
#: point back at something else.
_REFERS_BACK = re.compile(
    r"\b((the|my|that|this|our)\s+(gene\s+list|list\s+of\s+genes|list|genes)"
    r"|(those|these|them|the)\s+genes|gene\s+list)\b",
    re.IGNORECASE,
)
#: Pointing back at something that is not a gene list: a matrix, a file, a
#: GSA -- which the GSA how-to answers -- or the results.
_REFERS_ELSEWHERE = re.compile(
    r"\b(matri(x|ces)|files?|upload\w*|attach\w*|expression|samples?|pathways"
    r"|results?|website|gsa|gsea|reactome\s*gsa|counts?)\b",
    re.IGNORECASE,
)


def refers_back(text: str) -> bool:
    """A request to analyse a gene list given in an earlier message."""
    if len(text) > MAX_MESSAGE_CHARS or _ABOUT.search(text):
        return False
    return bool(
        _REQUEST.search(text)
        and _ANALYSIS_TERMS.search(text)
        and _REFERS_BACK.search(text)
        and not _REFERS_ELSEWHERE.search(text)
        and len(identifiers_in(text)) < MIN_IDENTIFIERS
    )


def listed(text: str) -> list[str] | None:
    """A list the reader sent, remembered in case they ask about it later."""
    if len(text) > MAX_MESSAGE_CHARS:
        return None
    # Anywhere: "What do TP53, MDM2 and CDKN1A have in common?" has no word
    # that opens a list, but it is the list "analyse those" will mean.
    found = identifiers_in(text, anywhere=True)
    # Three or more, one of them gene-shaped: "hmm, interesting" and
    # "PD-1 PD-L1 checkpoint blockade" are not the list a reader means.
    if len(found) >= 3 and any(_strength(t) in ("strong", "caps") for t in found):
        return found[:MAX_SUBMITTED_IDENTIFIERS]
    return None


def answer_to_invitation(text: str) -> list[str] | None:
    """The identifiers in a reply to "send me your genes", however phrased.

    Straight after the chat has told the reader to include their genes in a
    message, a message that lists two or more is that message -- no verb or
    analysis term needed. Still only proposes, and a question about the
    genes ("how do TP53 and MDM2 interact?") is still a question.
    """
    if len(text) > MAX_MESSAGE_CHARS or _ABOUT.search(text):
        return None
    found = identifiers_in(text)
    if len(found) >= MIN_IDENTIFIERS and _looks_like_genes(found):
        return found
    return None


@dataclass(frozen=True)
class Reading:
    """What one message means for the gene-list flow, decided in one place."""

    #: Identifiers in this message to offer an analysis of.
    offer: list[str] | None = None
    #: It asks about a gene list from an earlier message.
    refers_back: bool = False
    #: A list it mentions, to remember for a later "analyse those".
    listed: list[str] | None = None


def read_message(text: str, *, invited: bool) -> Reading:
    """Every gene-list decision about a message.

    Pure, so the handler's choices can be tested, and run off the event
    loop by the handler: on a 60K message the separate calls took up to a
    second between them.
    """
    offer = gene_list_request(text)
    if offer is None and invited:
        # Just told "include the genes in your message": a list is the reply.
        offer = answer_to_invitation(text)
    if offer is not None:
        return Reading(offer=offer, listed=offer[:MAX_SUBMITTED_IDENTIFIERS])
    if refers_back(text):
        return Reading(refers_back=True)
    return Reading(listed=listed(text))


def describe_proposal(identifiers: list[str], *, earlier: bool = False) -> str:
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
        f"I read **{len(identifiers)} identifiers** in your "
        f"{'earlier ' if earlier else ''}message: {shown}.{limit}\n\n"
        "Run the analysis on these? (Or just type *yes*.)"
    )


_CONFIRMS = re.compile(
    r"\A\s*(yes|y|yep|yeah|ok|okay|sure|go|go\s+ahead|run|run\s+it|do\s+it"
    r"|please|please\s+do|yes,?\s+please|please\s+run\s+it)\s*+[.!]*+\s*+\Z",
    re.IGNORECASE,
)


#: No typed "yes" is longer than this. Checked first: the pattern ran on the
#: raw message, and "y" plus a million spaces backtracked for an hour and a
#: half on the event loop every session shares (review, area 2).
MAX_CONFIRM_CHARS = 40


def confirms(text: str) -> bool:
    """A typed yes to the proposal just made, instead of clicking Run."""
    return len(text) <= MAX_CONFIRM_CHARS and bool(_CONFIRMS.match(text))


@dataclass(frozen=True)
class Overrepresentation:
    """What the reader is told, and whether there was anything to continue."""

    text: str
    #: False when nothing matched -- nothing for a follow-up to be about.
    has_pathways: bool
    #: What the model may see: `text` without the Pathway Browser link. The
    #: link embeds the analysis token, and anyone holding the token can
    #: fetch the result; it goes to the reader, never to the model.
    for_model: str = ""


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
    text = "\n".join(lines)
    return Overrepresentation(
        text=text,
        has_pathways=bool(pathways),
        for_model="\n".join(line for line in lines if browser_url not in line),
    )


FAILED = (
    "I tried to run an over-representation analysis on those genes, but "
    "Reactome's Analysis Service didn't return a result. Please try again in "
    "a moment."
)


EXPIRED = (
    "That list is no longer waiting to be analysed. Send it again and I'll "
    "offer to run it."
)

EXPIRED_DECLINED = (
    "That offer has lapsed, so I no longer have the question that went with "
    "it. Please ask it again."
)

FAILED_TO_ANSWER = "Something went wrong answering that. Please try again."

NO_LIST_YET = (
    "I don't have a gene list from you in this conversation yet. Paste the "
    "genes in your next message — for example *TP53, ERBB2, RUNX2* — and "
    "I'll offer to run an over-representation analysis on them."
)
