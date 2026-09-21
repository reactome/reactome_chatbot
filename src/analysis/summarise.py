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


#: Below this many matched entities, a pathway's p-value rests on so little
#: that it should not be read as evidence however small it is -- a pathway
#: with 2 of 3 entities found looks like a perfect hit and is nearly
#: meaningless.
#:
#: **Five is chosen, not derived.** An earlier version of this comment said
#: Reactome's user guide makes the point; it may, but nobody checked before
#: writing that, so the claim is withdrawn rather than left cited. The
#: number comes from this feature's own spec, which uses 2 of 3 as its
#: example of what is not evidence, and from wanting a margin above it.
#: Setting it properly is a curator's judgement, not a programmer's.
#:
#: It does discriminate, which is the part that was measured. Against beta on
#: 2026-09-20: a four-identifier analysis flagged 12 of 12 shown pathways
#: (found counts of 2), and a hundred-gene analysis flagged 0 of 12 (found
#: counts 13 to 67). So it fires on the inputs where a hit really does rest
#: on nothing and stays quiet on ordinary ones.
#:
#: Computed here rather than left to the model. Handed a small p-value and a
#: small count and asked to be careful, a model describes the p-value.
FRAGILE_BELOW_FOUND = 5


def _significant(pathway: dict[str, Any]) -> bool:
    fdr = pathway.get("entities", {}).get("fdr")
    return isinstance(fdr, int | float) and fdr <= FDR_THRESHOLD


def _fragile(pathway: dict[str, Any]) -> bool:
    found = pathway.get("entities", {}).get("found")
    return isinstance(found, int) and found < FRAGILE_BELOW_FOUND


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
                # True when the hit rests on too few entities to be evidence,
                # whatever the p-value says.
                "fragile": _fragile(p),
                # The per-column values, in column order. Carried explicitly
                # because the allow-list keeping them is not the same as the
                # prompt receiving them -- they were allow-listed and dropped
                # here, so the expression reading asked the model to describe
                # behaviour across columns using data it had never been
                # given, and it invented both the trends and a fourth column
                # of a three-column analysis.
                **(
                    {"exp": p["entities"]["exp"]}
                    if isinstance(p.get("entities", {}).get("exp"), list)
                    else {}
                ),
            }
            for p in pathways
        ],
    }
    # Stated rather than left for the model to notice, because "nothing went
    # wrong" is the case a model most readily embellishes into a caveat.
    out["all_identifiers_matched"] = unmatched == 0

    # How many columns there are, stated rather than left to be counted off
    # an array. A model asked to describe behaviour across columns will name
    # one that does not exist -- measured: a three-column analysis was
    # summarised as rising "from column 1 to column 4".
    columns = {len(p["exp"]) for p in out["pathways"] if isinstance(p.get("exp"), list)}
    if len(columns) == 1:
        out["expression_columns"] = columns.pop()
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

#: What each analysis type supports saying, and what it does not. A summary
#: that ignores the type either says nothing useful or says something wrong,
#: and the wrong thing is the likelier: an expression result read as a plain
#: enrichment loses the whole point of it, and a species comparison read as
#: observation states as fact what was inferred.
TYPE_INSTRUCTION = {
    "EXPRESSION": (
        "This is an expression analysis. Each pathway carries `exp`, its "
        "values across the submitted columns in order. Describe how the "
        "highlighted pathways behave *across* those columns -- rising, "
        "falling, mixed -- rather than treating the result as a single "
        "enrichment. **The columns are unlabelled here and you must not "
        "guess what they are**: never a condition, timepoint or sample name. "
        "Each pathway's `exp` holds its value per column, in order, and "
        "`expression_columns` says how many there are. **Those are the only "
        "columns that exist** -- never mention a column number beyond it, "
        "and never describe a trend you cannot read off `exp`. "
        "Refer to them in exactly this form -- `column 1`, `column 2`, "
        "numbered from one -- and in no other form, because the interface "
        "holds the real labels and substitutes them by matching that exact "
        "wording. 'The first column' or 'the leftmost sample' will not be "
        "matched and will reach the reader as written."
    ),
    "SPECIES_COMPARISON": (
        "This is a species comparison. The findings are **inferred by "
        "orthology**, not observed in the compared species, and you must say "
        "so. An inferred event means Reactome projected a human event onto "
        "that species because the proteins correspond; it is not evidence "
        "the event has been measured there."
    ),
    "OVERREPRESENTATION": (
        "This is an over-representation analysis: which pathways contain "
        "more of the submitted identifiers than chance would give. It says "
        "nothing about direction, magnitude or regulation, so do not "
        "describe anything as up, down, increased or activated."
    ),
}

#: Always appended. Both halves are things a model will otherwise get wrong
#: in the same direction -- towards overstating a finding.
STATISTICS_INSTRUCTION = (
    "When you call a pathway significant, say whether that is before or after "
    "multiple-testing correction; `significant` in the data is after. A "
    "pathway with a small p-value that does not pass correction is not a "
    "finding, and saying so is more useful than omitting it. "
    "Where `fragile` is true the hit rests on fewer than "
    f"{FRAGILE_BELOW_FOUND} matched entities. Explain that in the reader's "
    "terms using that pathway's own found and total counts -- a small p-value "
    "on three entities is not evidence however small it is. **Never use the "
    "word 'fragile' or any other field name from the data**: these are "
    "internal labels, and a reader should be told what the counts mean, not "
    "what we called it. Explain what the numbers mean for this result, never "
    "in general."
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
