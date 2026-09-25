"""The client's handling of shapes the service actually produces.

No network here. Everything asserted was seen from the live service on
2026-09-21 and is reproduced as a stub, so the suite keeps testing it after
the service moves on.
"""

import pytest

from gsa import client as gsa_client


@pytest.mark.parametrize(
    "value",
    ["../etc/passwd", "a/b", "GSE1 2345", "", "x" * 200, "tok;drop"],
)
def test_hostile_identifiers_are_refused(value: str) -> None:
    # These are interpolated into a URL path. A value containing `/`
    # addresses a different endpoint, which is a bypass, not a 404.
    with pytest.raises(gsa_client.GsaError):
        gsa_client._checked("dataset", value)


@pytest.mark.parametrize("value", ["EXAMPLE_MEL_RNA", "GSE12345", "E-MTAB-2770"])
def test_real_identifiers_are_accepted(value: str) -> None:
    assert gsa_client._checked("dataset", value) == value


def test_an_error_body_is_not_taken_for_an_id() -> None:
    # This test used to hand `_identifier_from` a Python dict, because it was
    # written against the same wrong belief as the code: that the reply is
    # JSON. It is `text/plain`. So the test could only ever confirm the
    # assumption, never the service -- which is how the upload feature
    # shipped unable to start an analysis. It now gets the *text* of an
    # error body, which is what would actually arrive.
    with pytest.raises(gsa_client.GsaError, match="not a valid identifier"):
        gsa_client._identifier_from('{"detail": "Bad Request"}', "analysis id")


def test_a_real_id_response_is_accepted() -> None:
    assert (
        gsa_client._identifier_from(
            "d8f825e8-b5f2-11f1-9de2-863bc094cc6c", "analysis id"
        )
        == "d8f825e8-b5f2-11f1-9de2-863bc094cc6c"
    )


def test_status_finished_covers_failure_not_just_success() -> None:
    # Measured: a submission returned 200 and then reached
    # status=failed with "CONNECTION_FORCED - broker forced connection
    # closure". A poll loop that waits for "complete" alone never exits.
    running = gsa_client.AnalysisStatus("running", "Permutation 900 / 1000", 0.6)
    failed = gsa_client.AnalysisStatus("failed", "Failed to analyse dataset", 1.0)
    done = gsa_client.AnalysisStatus("complete", "Analysis done", 1.0)

    assert not running.finished
    assert failed.finished
    assert failed.failed
    assert done.finished
    assert not done.failed


def test_summary_exposes_the_groups_a_user_must_choose_between() -> None:
    # The real shape: sample_metadata is a list of named factors, each with
    # one value per sample.
    summary = gsa_client.DatasetSummary(
        dataset_id="EXAMPLE_MEL_RNA",
        title="Melanoma RNA-seq example",
        type="rnaseq_counts",
        samples=[f"S{i}" for i in range(4)],
        factors={"condition": ["MCM", "MOCK", "MCM", "MOCK"]},
    )
    assert summary.groups("condition") == ["MCM", "MOCK"]
    assert summary.groups("no-such-factor") == []
