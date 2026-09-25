"""Turning a chat exchange into an analysis, and back into words.

Kept apart from the Chainlit handler on purpose. Everything here is a pure
function over plain values, so the decisions -- what counts as a usable
reply, what the user is told, what the model is allowed to see -- can be
tested. The handler that remains is wiring, and wiring is what a browser is
for.

**The audience split is the point.** A finished analysis produces three
things and they go to three different places: a bounded allow-listed view
for the model, a Pathway Browser link for the person, and a full table as a
file. `describe_result` writes what the person reads; it never becomes a
prompt, and `Finished.for_model` never becomes a message.
"""

import re
from dataclasses import dataclass

from gsa.client import AnalysisStatus
from gsa.job import Finished
from gsa.upload import Matrix

#: Shown when a matrix arrives, before anything is submitted.
MAX_SAMPLES_LISTED = 24


class ReplyUnusableError(ValueError):
    """The reply cannot be turned into a grouping, with a reason to show."""


@dataclass(frozen=True)
class Grouping:
    """One label per sample, and the two labels to compare."""

    labels: list[str]
    group1: str
    group2: str


def describe_matrix(matrix: Matrix) -> str:
    """What the user is told about the file they just sent.

    Their sample names are echoed back deliberately -- they need to see
    what was read in order to label it, and it is their own data being
    shown to them. It is the *model* that never sees these.
    """
    shown = matrix.samples[:MAX_SAMPLES_LISTED]
    more = len(matrix.samples) - len(shown)
    genes = (
        "an unknown number of"
        if matrix.gene_count is None
        else f"{matrix.gene_count:,}"
    )

    lines = [
        f"Read **{len(matrix.samples)} samples** and {genes} genes "
        f"({matrix.size_bytes / 1e6:.1f} MB).",
        "",
        "Samples, in column order:",
        "",
        "  "
        + ", ".join(f"`{name}`" for name in shown)
        + (f" …and {more} more" if more else ""),
        "",
        "To run the analysis I need to know which group each sample belongs "
        "to. Reply with one label per sample, in that order, separated by "
        "commas — for example `control, control, treated, treated`.",
    ]
    return "\n".join(lines)


def parse_grouping(reply: str, sample_count: int) -> Grouping:
    """Turn a reply into a grouping, or say why it cannot be one.

    Deliberately forgiving about separators and case, and strict about the
    count: a label list that does not line up with the columns produces an
    analysis of the wrong thing, which is worse than a refusal because it
    returns a plausible answer.
    """
    labels = [
        part.strip() for part in reply.replace("\t", ",").replace(";", ",").split(",")
    ]
    labels = [label for label in labels if label]

    if not labels:
        raise ReplyUnusableError("I could not find any group labels in that.")

    if len(labels) != sample_count:
        raise ReplyUnusableError(
            f"That is {len(labels)} label{'s' if len(labels) != 1 else ''} for "
            f"{sample_count} samples. I need exactly one per sample, in the "
            f"order the columns appear."
        )

    # Case-insensitive grouping, but the user's own spelling is kept: the
    # labels go to the service and come back in its output, and silently
    # lower-casing someone's "Treated" makes the result harder to read
    # against their own notes.
    seen: dict[str, str] = {}
    for label in labels:
        seen.setdefault(label.casefold(), label)

    if len(seen) < 2:
        raise ReplyUnusableError(
            "All the samples have the same label, so there is nothing to "
            "compare. A gene set analysis needs two groups."
        )
    if len(seen) > 2:
        names = ", ".join(sorted(seen.values()))
        raise ReplyUnusableError(
            f"I found more than two groups ({names}). This runs one "
            f"comparison at a time, so please use exactly two labels."
        )

    canonical = [seen[label.casefold()] for label in labels]
    group1, group2 = sorted(seen.values())
    return Grouping(labels=canonical, group1=group1, group2=group2)


def describe_progress(status: AnalysisStatus) -> str:
    """One line, safe to send repeatedly as an edit.

    **No percentage.** The first version showed `completed` as a percent,
    and against the real service that read "60% · Permutation 1000 / 1000":
    ReactomeGSA holds `completed` at 0.6 for the whole permutation phase
    while its description counts through it. The description is the
    service's own account and never contradicts itself; a number that does
    is worse than none.
    """
    detail = " ".join(status.description.split()) or "working"
    return f"Running the analysis — {detail}"


#: Characters with meaning in markdown. Reactome pathway names contain some
#: of them -- measured over a real 2,679-pathway result: `H139Hfs13* PPM1K
#: causes a mild variant of MSUD`, `NOTCH1:M1580_K2555`. Two of either in a
#: cell become emphasis, and a variant identifier like `M1580_K2555` renders
#: as `M1580K2555` with nothing to show it changed.
_MARKDOWN_SPECIAL = "\\`*_[]<>|"


def _escape(text: str) -> str:
    return "".join(f"\\{ch}" if ch in _MARKDOWN_SPECIAL else ch for ch in text)


def describe_result(finished: Finished) -> str:
    """What the person reads. Never a prompt.

    The Pathway Browser link is included *here* and not in anything the
    model sees: its URL carries the analysis token, and whoever holds that
    can fetch the unredacted result back from the service.
    """
    view = finished.for_model
    total = view.get("pathway_count", 0)
    significant = view.get("significant_count", 0)
    top = view.get("top_pathways") or []

    lines = [
        f"**{significant:,} of {total:,} pathways** are significant at FDR < 0.05.",
        "",
    ]
    if top:
        lines += [
            "| Pathway | Direction | FDR |",
            "|---|---|---|",
        ]
        for pathway in top[:10]:
            lines.append(
                f"| {_escape(pathway['name'])} | {pathway['direction']} | {pathway['fdr']:.2g} |"
            )
        lines.append("")

    for name, url in finished.links:
        lines.append(f"[{name}]({url})")
    if finished.links:
        lines.append("")
    lines.append("The full table, with every column, is attached.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# "Can we run a GSA in this chat?"
#
# Asked in words, the chat used to say no. The answer path grounds itself in
# the user guide, which describes the *website's* ReactomeGSA page, and its
# prompt forbids claiming anything the guide does not say -- so "you cannot do
# this in the chat, go to the website" was the only answer it could give.
# Reported 2026-09-25 from "can we run gsa in this chat please", which got an
# answer starting "no".
#
# The capability cannot be added to that prompt: the same prompt answers the
# website's search page, which has no upload, and would start telling people
# to attach files. So the request is recognised here, on the chat side only.
#
# Deliberately narrow. It needs both an analysis term and a request to *do*
# something, so "what is GSEA?" still reaches the model and is explained
# rather than being answered with upload instructions.

_ANALYSIS_TERMS = re.compile(
    r"\b(gsea|gsa|reactome\s*gsa|gene[\s-]*set(\s+enrichment)?\s+analy[sz]\w*"
    r"|expression\s+(matrix|matrices|data|profiles?|values?)|rna[\s-]*seq|microarray"
    r"|proteomics?\s+data|count\s+matrix)\b",
    re.IGNORECASE,
)
_WANTS_TO_DO_IT = re.compile(
    r"\b(run|do|perform|start|carry\s+out|execute|submit|upload|attach"
    r"|analy[sz]e\s+(my|our|this|these|it|them)"
    r"|can\s+(we|i|you)|could\s+(we|i|you)|how\s+(do|can|would|should)\s+(i|we)"
    r"|is\s+it\s+possible|want\s+to|would\s+like\s+to|help\s+me)\b",
    re.IGNORECASE,
)


def asks_to_run_gsa(text: str) -> bool:
    """True for a request to run a gene set analysis, not a question about one."""
    return bool(_ANALYSIS_TERMS.search(text) and _WANTS_TO_DO_IT.search(text))


#: The gene-list line points at the website, not at this chat. An earlier
#: draft said "paste the gene list and ask me to analyse it"; checked in a
#: browser, the chat does not run that analysis -- it sends the reader to the
#: website -- so the promise would have been false.
HOW_TO_RUN_GSA = """Yes — you can run a gene set analysis right here in the chat, using ReactomeGSA.

1. **Attach your expression matrix** with the 📎 button below: a `.tsv` or `.csv` file with genes (or proteins) as rows, samples as columns, and a first row naming the samples. Up to 20 MB.
2. **Tell me which group each sample is in** when I ask — for example `control, control, treated, treated`.
3. I'll run the analysis and give you the most significant pathways, a link to view the result in Reactome's Pathway Browser, and the full results table to download. It usually takes a few minutes.

If you only have a **list of genes** rather than measurements for each sample, that's an over-representation analysis instead. You can run it on Reactome's [Analyse Data](https://reactome.org/PathwayBrowser/#TOOL=AT) page by pasting the list."""
