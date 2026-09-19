"""What may leave this service, defined by field rather than by intention.

**An allow-list, not a denial list**, and that is the whole design. The
obvious implementation -- "strip the identifiers" -- misses three fields in
the *aggregate* result that are user-supplied free text:

- `summary.fileName`   e.g. `smith_lab_unpublished_2026.txt`
- `summary.sampleName` e.g. a patient sample label
- `expression.columnNames` e.g. `Patient_001_tumour`

A denial list is wrong by default the moment the Analysis Service adds a
field. An allow-list is only ever wrong by omission, which costs a missing
sentence rather than a disclosure.

Two tiers. `aggregate` is the default and the only one that may be
pre-selected; `identifiers` is never a default and requires the user to have
asked for it explicitly.
"""

from typing import Any, Literal

Tier = Literal["aggregate", "identifiers"]

#: Fields of `summary` that carry no user content. Everything absent from
#: this tuple is excluded, including fields that do not exist yet.
SUMMARY_FIELDS: tuple[str, ...] = (
    "type",
    "projection",
    "interactors",
    "includeDisease",
    "species",
    "speciesName",
)

#: Statistics per pathway. Measured against beta 2026-09-19: a result without
#: interactors carries no `curatedFound`/`interactorsFound`, so every field
#: here is optional and absence is normal, not an error.
ENTITY_FIELDS: tuple[str, ...] = (
    "found",
    "total",
    "ratio",
    "pValue",
    "fdr",
    "curatedFound",
    "interactorsFound",
    "resource",
)

PATHWAY_FIELDS: tuple[str, ...] = ("stId", "name", "species", "inDisease")

#: Top-level fields that are counts and summaries, never content.
RESULT_FIELDS: tuple[str, ...] = (
    "identifiersNotFound",
    "pathwaysFound",
    "resourceSummary",
    "speciesSummary",
    "warnings",
)

#: Never sent under any tier. Listed only so the test can assert on them by
#: name and so the reason is written down next to the list that excludes them.
NEVER_SENT: tuple[str, ...] = ("fileName", "sampleName", "columnNames")


def _pick(source: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(source, dict):
        return {}
    return {key: source[key] for key in fields if key in source}


def aggregate(result: dict[str, Any], *, top_pathways: int = 12) -> dict[str, Any]:
    """The result with only allow-listed fields, ready for a prompt.

    `top_pathways` bounds what is sent at all: a result can hold over a
    thousand pathways (measured: 1,280 for an eight-gene list), and sending
    them all would be slow, expensive and no more informative.
    """
    pathways = result.get("pathways") or []
    kept = []
    for pathway in pathways[:top_pathways]:
        if not isinstance(pathway, dict):
            continue
        entry = _pick(pathway, PATHWAY_FIELDS)
        entry["entities"] = _pick(pathway.get("entities"), ENTITY_FIELDS)
        kept.append(entry)

    out = _pick(result, RESULT_FIELDS)
    out["summary"] = _pick(result.get("summary"), SUMMARY_FIELDS)
    out["pathways"] = kept
    out["pathways_total"] = len(pathways)

    expression = result.get("expression")
    if isinstance(expression, dict):
        # The range, never the column labels.
        out["expression"] = _pick(expression, ("min", "max"))
    return out


def for_tier(result: dict[str, Any], tier: Tier) -> dict[str, Any]:
    """What may be sent for this tier.

    `identifiers` is a strict superset and is assembled by the caller, which
    must fetch the unmatched identifiers separately -- deliberately a second,
    explicit step rather than a flag on this function, so nothing reaches the
    identifier tier by passing a default through.
    """
    if tier not in ("aggregate", "identifiers"):
        raise ValueError(f"unknown disclosure tier: {tier!r}")
    return aggregate(result)
