"""Turn a 2 MB ReactomeGSA result into a small summary and a file.

Two consumers with opposite needs. The **file** wants everything: 2,679
pathways in the measured run, every column, so a researcher can sort and
filter it themselves. The **model** wants almost nothing: a bounded handful
of the most significant rows, because 2,679 rows would bury the answer, cost
more than the analysis, and say nothing the top twenty do not.

So nothing here returns "the result". Each function names its audience.

What may reach the model is an **allow-list**, for the reason
`analysis/disclosure.py` gives: a denial list is wrong by default the moment
the service adds a field, an allow-list is wrong only by omission. The
fields excluded here are not hypothetical -- a GSA submission carries the
user's own dataset name and their column headers, and the measured example's
headers were patient identifiers (`P1`..`P4` under a `patient.id` factor).
"""

from dataclasses import dataclass, field
from typing import Any

#: Columns of the pathway table that may be shown to a model. Everything
#: else -- present or future -- is excluded. `Pathway` and `Name` identify a
#: Reactome pathway, the rest are statistics over it; none is user content.
MODEL_COLUMNS: tuple[str, ...] = (
    "Pathway",
    "Name",
    "Direction",
    "FDR",
    "PValue",
    "NGenes",
)

#: Never sent to a model, under any circumstances, and named so a test can
#: assert on them.
#:
#: These are ReactomeGSA's own free-text carriers, and they are *different
#: field names* from the Analysis Service's `fileName` / `sampleName` /
#: `columnNames`. Reusing that list without adding these would have looked
#: like protection and provided none:
#:
#:   datasets[].name     chosen by the user, e.g. `smith_lab_unpublished`
#:   design.samples      the user's column headers
#:   fold_changes        one row per gene, and the columns are the samples
#:   mappings            the user's own row identifiers, mapped to UniProt --
#:                       8,035 entries and 499 KB in the measured run, and
#:                       the field nobody would have thought to exclude
NEVER_SENT: tuple[str, ...] = ("fold_changes", "design", "samples", "mappings")

#: A result can hold thousands of pathways. This bounds what is described,
#: not what is saved.
DEFAULT_TOP = 20


@dataclass(frozen=True)
class Pathway:
    stable_id: str
    name: str
    direction: str
    fdr: float
    p_value: float
    gene_count: int


@dataclass(frozen=True)
class GsaResult:
    """A parsed result. `pathways` is the whole table; bound it before use."""

    method: str
    release: str
    dataset_name: str
    pathways: list[Pathway]
    #: The service's own Pathway Browser view. Safe to show: it is a URL the
    #: service minted, and it is how a user sees the result properly.
    browser_links: list[tuple[str, str]]
    #: The raw table, kept verbatim for the file so a researcher gets every
    #: column rather than the six a model is allowed.
    #:
    #: `repr=False` is load-bearing. A dataclass's generated `__repr__`
    #: includes every field, and this one is ~500 KB in a real run, so a
    #: `logger.debug("%s", result)`, an exception context, or a failing
    #: test's output would print the entire pathway table into a log. The
    #: care taken over what reaches a *model* is wasted if the same content
    #: reaches a *log file* by default.
    raw_table: str = field(repr=False)

    @property
    def significant(self) -> list[Pathway]:
        return [p for p in self.pathways if p.fdr < 0.05]


def _as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        # A non-numeric FDR must not sort as "most significant". The
        # service has been seen returning "NA".
        return 1.0


def parse(result: dict[str, Any]) -> GsaResult:
    """Parse the service's result. Does no bounding -- see `for_model`."""
    datasets = result.get("results") or []
    first = datasets[0] if datasets else {}
    table = first.get("pathways")
    table = table if isinstance(table, str) else ""

    pathways: list[Pathway] = []
    lines = [line for line in table.splitlines() if line.strip()]
    if lines:
        header = lines[0].split("\t")
        index = {name: i for i, name in enumerate(header)}

        def cell(row: list[str], column: str) -> str:
            position = index.get(column)
            return row[position] if position is not None and position < len(row) else ""

        for line in lines[1:]:
            row = line.split("\t")
            try:
                genes = int(float(cell(row, "NGenes") or 0))
            except ValueError:
                genes = 0
            pathways.append(
                Pathway(
                    stable_id=cell(row, "Pathway"),
                    name=cell(row, "Name"),
                    direction=cell(row, "Direction"),
                    fdr=_as_float(cell(row, "FDR")),
                    p_value=_as_float(cell(row, "PValue")),
                    gene_count=genes,
                )
            )

    links = [
        (str(link.get("name") or "Reactome"), str(link.get("url")))
        for link in result.get("reactome_links") or []
        if link.get("url")
    ]

    return GsaResult(
        # `method_name`, not `methodName`. The swagger's AnalysisResult
        # definition says the latter; the service returns the former, so
        # this field read None until a fixture built from a real response
        # showed it. Same defect as the ten tools reactome-mcp fixed in
        # September: a field path asserted rather than verified.
        method=str(result.get("method_name") or result.get("methodName") or ""),
        release=str(result.get("release") or ""),
        dataset_name=str(first.get("name") or ""),
        pathways=pathways,
        browser_links=links,
        raw_table=table,
    )


def for_model(result: GsaResult, *, top: int = DEFAULT_TOP) -> dict[str, Any]:
    """The bounded, allow-listed view that may go into a prompt.

    Counts are exact and stated as such, so a summary can say "2,679
    pathways, 412 significant" without having seen 2,679 of anything.
    """
    ranked = sorted(result.pathways, key=lambda p: (p.fdr, p.p_value))[:top]

    # A result with no pathways is not an analysis that found nothing -- it
    # is an analysis whose table did not arrive, which is what a failed or
    # half-written result looks like. Reporting `0` with
    # `counts_are_exact: True` invites a model to tell the user their data
    # contained no enriched pathways, which is a confident answer to a
    # question nobody managed to ask. Say "no result" instead.
    if not result.pathways:
        return {
            "release": result.release,
            "no_result": True,
            "counts_are_exact": False,
            "pathway_count": 0,
            "significant_count": 0,
            "showing": 0,
            "top_pathways": [],
        }

    return {
        "release": result.release,
        "no_result": False,
        "pathway_count": len(result.pathways),
        "significant_count": len(result.significant),
        "counts_are_exact": True,
        "showing": len(ranked),
        "top_pathways": [
            {
                "stId": p.stable_id,
                "name": p.name,
                "direction": p.direction,
                "fdr": p.fdr,
                "genes": p.gene_count,
            }
            for p in ranked
        ],
    }


def for_user(result: GsaResult) -> list[tuple[str, str]]:
    """Links for the chat to show the user. **Not for the model.**

    The Pathway Browser URL embeds the analysis token:

        https://reactome.org/PathwayBrowser/#/DTAB=AN&ANALYSIS=MjAyNj...

    Anyone holding that token can fetch the whole result back from the
    service -- including `mappings`, which is the user's own gene
    identifiers, and `fold_changes`, whose columns are their samples. So
    the link is a *capability*, not a citation.

    `for_model` used to include it. Stripping user content from the payload
    and then handing over a key that retrieves it is not protection; it is
    the same content by a longer route. The user should absolutely see this
    link -- it is how they view their own result properly -- so the chat
    renders it and the model never receives it.
    """
    return list(result.browser_links)


def as_tsv(result: GsaResult) -> str:
    """The whole table, for the file the user downloads.

    Verbatim rather than reassembled from `Pathway` objects: the parsed form
    keeps six columns and the service returned nine, and a researcher asked
    for their results should get their results.
    """
    return result.raw_table
