"""What the model is told about a result, before any model is involved."""

from typing import Any

from analysis.summarise import VERDICT_INSTRUCTION, prompt_input


def _payload(*fdrs: float) -> dict[str, Any]:
    return {
        "summary": {"type": "OVERREPRESENTATION"},
        "pathways_total": len(fdrs),
        "identifiersNotFound": 0,
        "pathways": [
            {
                "stId": f"R-HSA-{i}",
                "name": f"Pathway {i}",
                "entities": {"found": 3, "total": 40, "pValue": 0.001, "fdr": fdr},
            }
            for i, fdr in enumerate(fdrs)
        ],
    }


def test_a_null_result_is_labelled_before_the_model_sees_it() -> None:
    # FR-004. A model handed rows sorted by p-value and asked to be careful
    # produces a confident account of the top row; the verdict is computed in
    # code so "nothing passed correction" is a fact, not a hope.
    out = prompt_input(_payload(0.4, 0.6, 0.9))
    assert out["verdict"] == "nothing_significant"
    assert out["significant_among_shown"] == 0
    assert "do not describe the lowest p-values as findings" in (
        VERDICT_INSTRUCTION[out["verdict"]].lower()
    )


def test_a_result_with_no_pathways_at_all_is_distinct_from_a_null_one() -> None:
    # "Nothing was found" and "things were found, none survived correction"
    # are different statements about a result and must not collapse.
    out = prompt_input(_payload())
    assert out["verdict"] == "empty"


def test_findings_are_labelled_per_pathway_not_just_overall() -> None:
    out = prompt_input(_payload(0.001, 0.2))
    assert out["verdict"] == "has_findings"
    assert out["significant_among_shown"] == 1
    assert [p["significant"] for p in out["pathways"]] == [True, False]


def test_the_threshold_is_stated_not_implied() -> None:
    # A reader of the prompt input can see what "significant" meant.
    assert prompt_input(_payload(0.01))["fdr_threshold"] == 0.05


def test_a_pathway_exactly_on_the_threshold_counts_as_significant() -> None:
    # Inclusive, and pinned because an unstated boundary is one nobody agreed.
    assert prompt_input(_payload(0.05))["significant_among_shown"] == 1
    assert prompt_input(_payload(0.050001))["significant_among_shown"] == 0


def test_a_missing_fdr_is_not_significant() -> None:
    # Absence is normal in this API; it must never read as passing.
    payload = _payload(0.01)
    del payload["pathways"][0]["entities"]["fdr"]
    assert prompt_input(payload)["significant_among_shown"] == 0


def test_the_significant_count_is_marked_inexact_when_it_is_a_lower_bound() -> None:
    # The error a real run produced: given "12 significant" and "1280 total",
    # the model wrote "12 significant out of 1280". Only the top twelve were
    # sent and all twelve passed, so the true count is at least twelve and
    # unknown above. FR-002 -- never state a statistic the result lacks.
    payload = _payload(*([0.001] * 12))
    payload["pathways_total"] = 1280
    out = prompt_input(payload)
    assert out["significant_among_shown"] == 12
    assert out["pathways_shown"] == 12
    assert out["significant_count_is_exact"] is False


def test_the_count_is_exact_once_a_non_significant_pathway_is_shown() -> None:
    # Ordered by p-value, a non-significant pathway among those shown means
    # everything below it is non-significant too, so the count is complete.
    payload = _payload(0.001, 0.002, 0.9)
    payload["pathways_total"] = 1280
    assert prompt_input(payload)["significant_count_is_exact"] is True


def test_an_unsorted_result_is_never_claimed_as_exact() -> None:
    # The ordering is the Analysis Service's default, not a guarantee.
    # Relying on someone else's default sort is how a claim goes quietly
    # wrong, so it is checked rather than assumed.
    payload = _payload(0.9, 0.001)
    payload["pathways_total"] = 1280
    assert prompt_input(payload)["significant_count_is_exact"] is False


def test_a_result_with_everything_matched_says_so_rather_than_hedging() -> None:
    # US2 scenario 2. "Nothing went wrong" is the case a model most readily
    # embellishes into a caveat, so it is stated as a fact in the input
    # rather than left to be noticed.
    payload = _payload(0.001)
    payload["identifiersNotFound"] = 0
    out = prompt_input(payload)
    assert out["all_identifiers_matched"] is True
    assert out["identifiers_not_found"] == 0


def test_unmatched_identifiers_are_reported_as_a_count_not_a_proportion() -> None:
    # Measured against beta: the aggregate result carries how many were NOT
    # found and nothing about how many were submitted. The denominator lives
    # only behind `/found/all`, which returns the reader's own identifiers --
    # so a proportion is not derivable at this tier, and asking for one would
    # invent a statistic the way "12 significant out of 1280" did (D9).
    from analysis.summarise import UNMATCHED_INSTRUCTION

    payload = _payload(0.001)
    payload["identifiersNotFound"] = 7
    out = prompt_input(payload)
    assert out["identifiers_not_found"] == 7
    assert out["all_identifiers_matched"] is False
    assert "how many were submitted" in UNMATCHED_INSTRUCTION
    assert "Never state a proportion" in UNMATCHED_INSTRUCTION
    # Nothing in the payload lets a proportion be computed.
    assert not any("submitted" in k or "total_identifiers" in k for k in out)


def test_a_hit_resting_on_few_entities_is_flagged_fragile() -> None:
    # US3 scenario 1, and the thing the spec says readers most often get
    # wrong: "a pathway with 2 of 3 entities found is not strong evidence".
    # Computed here because a model handed a tiny p-value and a tiny count
    # describes the p-value.
    payload = _payload(1e-9)
    payload["pathways"][0]["entities"].update({"found": 2, "total": 3})
    out = prompt_input(payload)
    assert out["pathways"][0]["fragile"] is True
    assert out["pathways"][0]["significant"] is True


def test_a_well_supported_hit_is_not_flagged_fragile() -> None:
    # A flag that is always true would pass the test above and say nothing.
    payload = _payload(1e-9)
    payload["pathways"][0]["entities"].update({"found": 40, "total": 120})
    pathways: list[dict[str, Any]] = prompt_input(payload)["pathways"]
    assert pathways[0]["fragile"] is False


def test_the_fragility_threshold_is_on_the_count_not_the_ratio() -> None:
    # 2 of 3 looks like a perfect hit by ratio and is nearly meaningless;
    # 40 of 200 looks poor by ratio and is real evidence. The ratio is the
    # misleading number here, so the flag deliberately ignores it.
    ratio_perfect = _payload(1e-9)
    ratio_perfect["pathways"][0]["entities"].update({"found": 2, "total": 3})
    ratio_poor = _payload(1e-9)
    ratio_poor["pathways"][0]["entities"].update({"found": 40, "total": 200})
    pathways: list[dict[str, Any]] = prompt_input(ratio_perfect)["pathways"]
    assert pathways[0]["fragile"] is True
    poor: list[dict[str, Any]] = prompt_input(ratio_poor)["pathways"]
    assert poor[0]["fragile"] is False


def test_significant_before_correction_is_distinguishable_from_after() -> None:
    # US3 scenario 2. Both numbers are present per pathway and `significant`
    # is defined as after correction, so the two can be told apart rather
    # than the model choosing which it means.
    from analysis.summarise import STATISTICS_INSTRUCTION

    payload = _payload(0.2)
    payload["pathways"][0]["entities"]["pValue"] = 0.001
    out = prompt_input(payload)
    pathway = out["pathways"][0]
    assert pathway["p_value"] == 0.001
    assert pathway["fdr"] == 0.2
    assert pathway["significant"] is False
    assert "before or after" in STATISTICS_INSTRUCTION
    assert "is after" in STATISTICS_INSTRUCTION


def test_a_pathway_with_no_found_count_is_not_called_fragile() -> None:
    # Absence is normal in this API and must not read as a finding either way.
    payload = _payload(1e-9)
    del payload["pathways"][0]["entities"]["found"]
    pathways: list[dict[str, Any]] = prompt_input(payload)["pathways"]
    assert pathways[0]["fragile"] is False


def test_the_model_is_told_not_to_repeat_our_field_names() -> None:
    # Measured against a real result: the summary said "these pathways are
    # classified as fragile", handing the reader an internal label instead of
    # the reason. The flag is ours; the explanation is theirs.
    from analysis.summarise import STATISTICS_INSTRUCTION

    assert "Never use the word 'fragile'" in STATISTICS_INSTRUCTION
    assert "internal labels" in STATISTICS_INSTRUCTION


def test_the_fragility_threshold_discriminates_on_realistic_inputs() -> None:
    # A flag that fires on everything is as useless as one that never fires,
    # and this one fired on 12 of 12 pathways for a four-identifier analysis.
    # Measured against beta: a hundred-gene analysis flags 0 of 12, with
    # found counts of 13 to 67. Both ends pinned here with those real shapes.
    tiny = _payload(1e-9, 1e-8, 1e-7)
    for pathway in tiny["pathways"]:
        pathway["entities"].update({"found": 2, "total": 4})
    realistic = _payload(1e-9, 1e-8, 1e-7)
    for pathway, found in zip(realistic["pathways"], (31, 17, 13), strict=True):
        pathway["entities"].update({"found": found, "total": 164})

    tiny_flags = [p["fragile"] for p in prompt_input(tiny)["pathways"]]
    real_flags = [p["fragile"] for p in prompt_input(realistic)["pathways"]]
    assert all(tiny_flags), "a hit on two entities must be flagged"
    assert not any(real_flags), "ordinary hits must not all be flagged"
