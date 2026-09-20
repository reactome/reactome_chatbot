"""What the model is told about a result, before any model is involved."""

from analysis.summarise import VERDICT_INSTRUCTION, prompt_input


def _payload(*fdrs: float) -> dict[str, object]:
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
    del payload["pathways"][0]["entities"]["fdr"]  # type: ignore[index]
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
