"""What reaches the model, and what only reaches the file.

The fixture is a real ReactomeGSA response, trimmed: the same keys, the same
column header, real rows, and `mappings` cut from 8,035 entries to two that
keep its hazard. Trimmed rather than invented, because the two defects found
while writing this code were both invisible to an invented one -- the result
key is `method_name` where the swagger says `methodName`, and `mappings`
carries the user's own row identifiers.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from gsa import results as gsa_results

FIXTURE = Path(__file__).parent / "result_fixture.json"


@pytest.fixture
def raw() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(FIXTURE.read_text())
    return loaded


@pytest.fixture
def parsed(raw: dict[str, Any]) -> gsa_results.GsaResult:
    return gsa_results.parse(raw)


def test_parses_the_service_shape(parsed: gsa_results.GsaResult) -> None:
    assert parsed.release == "97"
    # The assertion that would have failed against the swagger's field name.
    assert parsed.method == "padog"
    assert len(parsed.pathways) == 6
    first = parsed.pathways[0]
    assert first.stable_id.startswith("R-HSA-")
    assert first.name
    assert first.direction in {"Up", "Down"}
    assert 0.0 <= first.fdr <= 1.0


def test_browser_link_survives(parsed: gsa_results.GsaResult) -> None:
    # The user's real way of seeing the result. Losing it silently would
    # leave them with a table and no picture.
    assert parsed.browser_links
    assert all(url.startswith("http") for _, url in parsed.browser_links)


def test_model_view_is_bounded(parsed: gsa_results.GsaResult) -> None:
    view = gsa_results.for_model(parsed, top=3)
    assert view["showing"] == 3
    assert view["pathway_count"] == 6
    assert view["counts_are_exact"] is True
    # Bounded on what is shown, exact on what is counted: a summary may say
    # how many there were without having seen them.
    assert len(view["top_pathways"]) == 3


def test_model_view_is_ranked_by_significance(parsed: gsa_results.GsaResult) -> None:
    view = gsa_results.for_model(parsed, top=6)
    fdrs = [p["fdr"] for p in view["top_pathways"]]
    assert fdrs == sorted(fdrs)


def test_a_non_numeric_fdr_does_not_rank_first(raw: dict[str, Any]) -> None:
    # Measured behaviour elsewhere in Reactome: "NA" appears in numeric
    # columns. float() raises, and a default of 0.0 would sort it to the top
    # as the most significant result in the analysis.
    lines = raw["results"][0]["pathways"].split("\n")
    header = lines[0].split("\t")
    row = lines[1].split("\t")
    row[header.index("FDR")] = "NA"
    lines[1] = "\t".join(row)
    raw["results"][0]["pathways"] = "\n".join(lines)

    parsed = gsa_results.parse(raw)
    ranked = gsa_results.for_model(parsed, top=6)["top_pathways"]
    assert ranked[0]["fdr"] < 1.0
    assert ranked[-1]["fdr"] == 1.0


@pytest.mark.parametrize("field", gsa_results.NEVER_SENT)
def test_never_sent_fields_are_absent_by_name(
    parsed: gsa_results.GsaResult, field: str
) -> None:
    assert field not in json.dumps(gsa_results.for_model(parsed))


def test_no_user_content_reaches_the_model(raw: dict[str, Any]) -> None:
    """The test that matters: a marker planted in every user-supplied place
    must not appear anywhere in the model's view.

    Asserting on field *names* is not enough -- a field could be copied into
    a differently named one. This asserts on the values.
    """
    marker = "SMITH-LAB-UNPUBLISHED-2026"
    raw["results"][0]["name"] = marker
    raw["results"][0]["fold_changes"] = f"GENE\t{marker}\nENSG1\t1.0\n"
    raw["mappings"] = [{"identifier": marker, "mapped_to": ["P00001"]}]

    view = json.dumps(gsa_results.for_model(gsa_results.parse(raw)))
    assert marker not in view


def test_the_marker_test_can_fail(raw: dict[str, Any]) -> None:
    """The control for the test above.

    An absence assertion proves nothing until the same construction has been
    shown capable of producing a presence. If the marker cannot reach the
    view even when deliberately placed in an allow-listed field, the test
    above is passing vacuously.
    """
    marker = "SMITH-LAB-UNPUBLISHED-2026"
    lines = raw["results"][0]["pathways"].split("\n")
    header = lines[0].split("\t")
    row = lines[1].split("\t")
    row[header.index("Name")] = marker
    lines[1] = "\t".join(row)
    raw["results"][0]["pathways"] = "\n".join(lines)

    assert marker in json.dumps(gsa_results.for_model(gsa_results.parse(raw)))


def test_the_file_keeps_every_column(parsed: gsa_results.GsaResult) -> None:
    # The model gets six columns; the researcher asked for their results and
    # the service returned nine.
    header = gsa_results.as_tsv(parsed).splitlines()[0].split("\t")
    assert set(gsa_results.MODEL_COLUMNS).issubset(header)
    assert {"MeanAbsT0", "MeanWeightT0", "av_foldchange"}.issubset(header)
