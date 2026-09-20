"""Turn an allow-listed result into what the model is asked to describe.

Everything here is derived from the result. The one job this file has beyond
reshaping is to decide, in code rather than in the prompt, whether the result
supports a finding at all -- because "nothing passed correction" is precisely
the case a model will otherwise narrate as though the lowest p-value were a
discovery (FR-004).
"""

from itertools import pairwise
from typing import Any

#: The conventional threshold, and it is stated rather than assumed so a
#: reader of the prompt input can see what "significant" meant here.
FDR_THRESHOLD = 0.05


def _significant(pathway: dict[str, Any]) -> bool:
    fdr = pathway.get("entities", {}).get("fdr")
    return isinstance(fdr, int | float) and fdr <= FDR_THRESHOLD


def prompt_input(payload: dict[str, Any]) -> dict[str, Any]:
    """What to tell the model, from the aggregate payload.

    `verdict` is computed here on purpose. Handing a model a list of pathways
    sorted by p-value and asking it to be careful produces a confident
    account of the top row; handing it "nothing passed correction" does not.
    """
    pathways = payload.get("pathways") or []
    significant = [p for p in pathways if _significant(p)]

    if not pathways:
        verdict = "empty"
    elif not significant:
        verdict = "nothing_significant"
    else:
        verdict = "has_findings"

    unmatched = payload.get("identifiersNotFound")
    shown = len(pathways)
    total = payload.get("pathways_total", shown)

    # How many pathways are significant *overall* is not in this payload, and
    # saying so is not pedantry: handed "12 significant" and "1280 total", a
    # model writes "12 significant out of 1280". Measured against a real
    # result on 2026-09-19, and it is false -- the top twelve were sent, all
    # twelve passed, so the true count is at least twelve and unknown above.
    #
    # It is exact only when a non-significant pathway appears among those
    # shown, and the results are ordered worst-p-value-last. The ordering is
    # checked rather than assumed, because relying on someone else's default
    # sort is how this kind of claim becomes wrong quietly.
    p_values = [q.get("entities", {}).get("pValue") for q in pathways]
    ordered = all(
        a is not None and b is not None and a <= b for a, b in pairwise(p_values)
    )
    significant_is_exact = shown >= total or (
        ordered and bool(pathways) and not _significant(pathways[-1])
    )

    out: dict[str, Any] = {
        "analysis_type": payload.get("summary", {}).get("type"),
        "verdict": verdict,
        "fdr_threshold": FDR_THRESHOLD,
        "pathways_total": total,
        "pathways_shown": shown,
        "significant_among_shown": len(significant),
        "significant_count_is_exact": significant_is_exact,
        "identifiers_not_found": unmatched,
        "pathways": [
            {
                "st_id": p.get("stId"),
                "name": p.get("name"),
                "found": p.get("entities", {}).get("found"),
                "total": p.get("entities", {}).get("total"),
                "p_value": p.get("entities", {}).get("pValue"),
                "fdr": p.get("entities", {}).get("fdr"),
                "significant": _significant(p),
            }
            for p in pathways
        ],
    }
    # Stated rather than left for the model to notice, because "nothing went
    # wrong" is the case a model most readily embellishes into a caveat.
    out["all_identifiers_matched"] = unmatched == 0
    for optional in ("resourceSummary", "speciesSummary", "warnings", "expression"):
        if optional in payload:
            out[optional] = payload[optional]
    return out


#: Said in the prompt input rather than left to the model, because each is a
#: claim the result either supports or does not.
VERDICT_INSTRUCTION = {
    "empty": "No pathways were returned. Report that nothing was found and do "
    "not speculate about why.",
    "nothing_significant": "No pathway passed multiple-testing correction. Say "
    "so plainly. Do not describe the lowest p-values as findings.",
    "has_findings": "Describe the pathways that pass correction. Distinguish "
    "significance before and after correction wherever you mention it.",
}

#: Always appended. The aggregate result carries how many identifiers were
#: *not* found and nothing at all about how many were submitted -- the
#: denominator lives only behind `/found/all`, which returns the reader's own
#: identifiers and is therefore the disclosing tier.
#:
#: So a proportion cannot be derived, and asking for one would produce the
#: same class of invention as D9's "12 significant out of 1280". The reader
#: knows how many they submitted; the count alone is useful to them.
UNMATCHED_INSTRUCTION = (
    "The data gives how many identifiers were NOT found and does not give how "
    "many were submitted. State the count. Never state a proportion, a "
    "percentage, or how many were found -- none of those are derivable. Use "
    "`resourceSummary` and `speciesSummary` to say what the likely cause is: "
    "identifiers resolving through a single resource suggests an identifier "
    "type Reactome does not index, and pathways concentrated in one species "
    "suggests the wrong species was analysed."
)

#: Appended only when the reader chose the disclosing tier and unmatched
#: identifiers were actually retrieved.
#:
#: Without it the tier is the worst of both: their identifiers are sent to a
#: model provider and the summary says exactly what the aggregate one said.
#: Measured 2026-09-20 -- the first version sent the names and never
#: mentioned them, because nothing asked it to. A disclosure has to buy the
#: reader something or it should not be offered.
NAMED_UNMATCHED_INSTRUCTION = (
    "`identifiers_not_found_names` lists identifiers the reader submitted "
    "that Reactome did not match, because they asked for them. Name them, and "
    "say what their form suggests -- a gene symbol Reactome does not carry, an "
    "identifier from a resource it does not index, an obsolete or misspelled "
    "symbol. Only comment on the ones listed; the list may be truncated."
)

#: Appended whenever the count is a lower bound. Separate from the verdict
#: because it is about what the *data* omits rather than what it shows.
INEXACT_COUNT_INSTRUCTION = (
    "Only the highest-ranked pathways are included here, and every one of them "
    "is significant, so the number significant overall is NOT known. Never "
    "state how many of the total were significant, and never imply that only "
    "the pathways listed here passed."
)
