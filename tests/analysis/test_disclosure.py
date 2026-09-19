"""What must never leave this service.

Asserted on the payload, not on a generated summary. Reading a summary and
seeing nothing alarming is not evidence: the model may simply not have
mentioned the filename this time.
"""

import json

import pytest

from analysis.disclosure import NEVER_SENT, aggregate, for_tier

# A result shaped like beta's, with every dangerous field populated with
# something a lab would mind seeing sent anywhere.
RESULT = {
    "summary": {
        "token": "MjAyNjA5MTkxODExNDJfMTE%3D",
        "type": "EXPRESSION",
        "projection": False,
        "interactors": False,
        "includeDisease": True,
        "fileName": "smith_lab_unpublished_2026.txt",
        "sampleName": "Patient 4 biopsy, pre-treatment",
    },
    "expression": {
        "columnNames": ["Patient_001_tumour", "Patient_001_normal"],
        "min": -3.1,
        "max": 4.2,
    },
    "identifiersNotFound": 7,
    "pathwaysFound": 1280,
    "resourceSummary": [{"resource": "TOTAL", "pathways": 1280}],
    "warnings": [],
    "pathways": [
        {
            "stId": "R-HSA-109581",
            "name": "Apoptosis",
            "dbId": 109581,
            "entities": {"found": 4, "total": 11, "pValue": 1e-7, "fdr": 4e-5},
        }
    ],
}


def _sent(payload: dict[str, object]) -> str:
    """Everything that would go over the wire, keys and values alike."""
    return json.dumps(payload, default=str)


@pytest.mark.parametrize("tier", ["aggregate", "identifiers"])
def test_the_three_free_text_fields_are_never_sent(tier: str) -> None:
    # These are the trap. A tier defined as "do not send the gene list" passes
    # all three straight through, which is why the tier is an allow-list.
    sent = _sent(for_tier(RESULT, tier))  # type: ignore[arg-type]
    for field in NEVER_SENT:
        assert field not in sent, f"{field} is in the outbound payload"
    assert "smith_lab_unpublished_2026" not in sent
    assert "Patient 4 biopsy" not in sent
    assert "Patient_001_tumour" not in sent


def test_a_field_the_service_adds_tomorrow_is_excluded_by_default() -> None:
    # The property that makes this an allow-list rather than a denial list.
    # A denial list is wrong the moment the Analysis Service adds a field;
    # this must be wrong only by omission.
    result = json.loads(json.dumps(RESULT))
    result["summary"]["patientNotes"] = "consented 2026-03, arm B"
    result["donorIdentifier"] = "DONOR-88213"
    result["pathways"][0]["entities"]["submitterComment"] = "our unpublished hit"
    sent = _sent(aggregate(result))
    assert "patientNotes" not in sent
    assert "consented 2026-03" not in sent
    assert "DONOR-88213" not in sent
    assert "our unpublished hit" not in sent


def test_what_is_needed_for_a_summary_does_survive() -> None:
    # A guarantee that excluded everything would pass every test above and be
    # useless, so the useful half is asserted too.
    payload = aggregate(RESULT)
    assert payload["summary"]["type"] == "EXPRESSION"
    assert payload["identifiersNotFound"] == 7
    assert payload["pathways"][0]["stId"] == "R-HSA-109581"
    assert payload["pathways"][0]["entities"]["fdr"] == 4e-5
    assert payload["expression"] == {"min": -3.1, "max": 4.2}


def test_the_expression_range_survives_without_the_column_labels() -> None:
    payload = aggregate(RESULT)
    assert "columnNames" not in payload["expression"]
    assert payload["expression"]["max"] == 4.2


def test_a_result_with_thousands_of_pathways_is_bounded() -> None:
    # Measured: an eight-gene list produced 1,280 pathways. Sending them all
    # is slow and no more informative, and the total is kept so the summary
    # can still say how many there were.
    result = json.loads(json.dumps(RESULT))
    result["pathways"] = [
        {"stId": f"R-HSA-{i}", "name": f"p{i}", "entities": {"fdr": 0.01}}
        for i in range(1280)
    ]
    payload = aggregate(result)
    assert len(payload["pathways"]) == 12
    assert payload["pathways_total"] == 1280


def test_missing_statistics_are_normal_not_an_error() -> None:
    # A result without interactors carries no curatedFound/interactorsFound.
    result = json.loads(json.dumps(RESULT))
    del result["expression"]
    payload = aggregate(result)
    assert "expression" not in payload
    assert payload["pathways"][0]["entities"]["found"] == 4


def test_an_unknown_tier_is_refused_rather_than_guessed() -> None:
    with pytest.raises(ValueError, match="unknown disclosure tier"):
        for_tier(RESULT, "everything")  # type: ignore[arg-type]


def test_warnings_are_bounded_because_their_content_is_not_guaranteed() -> None:
    # `warnings` is the one allow-listed field whose *content* the service
    # composes freely. Observed on beta it is service-level ("Missing header.
    # Using a default one."), and one observation is not a guarantee -- this
    # service never sees the user's identifiers, so it cannot filter them out
    # of a warning it is handed. Bounding the exposure is what is available.
    result = json.loads(json.dumps(RESULT))
    result["warnings"] = [f"warning {i} " + "x" * 500 for i in range(20)]
    payload = aggregate(result)
    assert len(payload["warnings"]) == 5
    assert all(len(w) <= 200 for w in payload["warnings"])


def test_no_warnings_key_when_the_service_sent_none() -> None:
    result = json.loads(json.dumps(RESULT))
    result["warnings"] = []
    assert "warnings" not in aggregate(result)
