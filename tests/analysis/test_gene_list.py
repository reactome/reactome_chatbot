"""A gene list typed into the chat: recognised, submitted, described.

The phrasing sets are sized rather than a couple of examples, because the
recogniser is a heuristic and a heuristic's failures are in its edges.
"""

import asyncio
from collections.abc import Callable

import gene_list_phrases as phrases
import httpx
import pytest

from analysis import client as analysis_client
from analysis.gene_list import (
    MAX_PROPOSED_LISTED,
    describe_overrepresentation,
    describe_proposal,
    gene_list_request,
    identifiers_in,
)

# The message that prompted this, verbatim.
ASKED = (
    "can we do a gsa analysis in the chat. I want to do it with genes TP53, "
    "ERBB2 and RUNX2"
)

REQUESTS = [
    (ASKED, ["TP53", "ERBB2", "RUNX2"]),
    ("perform ORA on\nTP53\nMDM2\nCDKN1A\n", ["TP53", "MDM2", "CDKN1A"]),
    ("run a GSEA with genes MYC, MAX", ["MYC", "MAX"]),
    ("do a pathway analysis for TP53, TP53, tp53, MDM2", ["TP53", "MDM2"]),
    ("run reactome gsa on P04637-2 and Q00987", ["P04637-2", "Q00987"]),
    # Words in capitals and accession-like strings are not submitted.
    (
        "run an enrichment on TP53, MDM2 in HUMAN NOT MOUSE, see GSE12345 chr17",
        ["TP53", "MDM2"],
    ),
    *phrases.REVIEW_REQUESTS,
    *phrases.HELD_OUT_REQUESTS,
]

NOT_REQUESTS = [
    "can we run gsa in this chat please",  # no genes: the how-to reply
    "what is GSEA?",
    "Compare GSEA and PADOG analysis methods",  # capitals, not genes
    "CAN YOU RUN A GSA ANALYSIS ON MY DATA PLEASE",  # shouting
    # Shouting, with capitals the stoplist does not know.
    "RUN AN ENRICHMENT ANALYSIS FROM MY EXPERIMENT TODAY",
    # Space-separated lower-case words after "for" are prose, not a list.
    "run an enrichment for mice given high doses",
    "run a pathway analysis on TP53",  # one gene is not a list
    "How do I run GSA on RNA-seq data from a TSV or CSV?",
    "what is the difference between ORA and GSEA",
    "",
    *phrases.REVIEW_QUESTIONS,
    *phrases.HELD_OUT_QUESTIONS,
]


@pytest.mark.parametrize(("text", "expected"), REQUESTS)
def test_a_request_with_genes_is_recognised(text: str, expected: list[str]) -> None:
    assert gene_list_request(text) == expected


@pytest.mark.parametrize("text", NOT_REQUESTS)
def test_other_messages_are_left_to_the_model(text: str) -> None:
    assert gene_list_request(text) is None


def test_the_sets_are_the_size_they_claim() -> None:
    # Sized sets are the point: a pass at n=2 says nothing about a rate.
    assert len(REQUESTS) >= 40
    assert len(NOT_REQUESTS) >= 70


def test_the_proposal_lists_what_will_be_submitted() -> None:
    proposal = describe_proposal(["TP53", "ERBB2", "RUNX2"])
    assert "over-representation" in proposal
    assert "**3 identifiers**" in proposal
    assert "`TP53`, `ERBB2`, `RUNX2`" in proposal
    many = describe_proposal([f"G{i}" for i in range(MAX_PROPOSED_LISTED + 5)])
    assert "and 5 more" in many


def test_the_words_around_the_genes_are_not_submitted() -> None:
    assert identifiers_in(ASKED) == ["TP53", "ERBB2", "RUNX2"]


# Shaped like the reply measured from beta on 2026-09-25 for TP53, ERBB2,
# RUNX2, NOTAGENE1 (pathways trimmed).
MEASURED = {
    "summary": {"token": "MjAyNjA5MjUxOTM5MzNfMTY%3D", "type": "OVERREPRESENTATION"},
    "identifiersNotFound": 1,
    "pathwaysFound": 188,
    "pathways": [
        {
            "stId": "R-HSA-6804754",
            "name": "Regulation of TP53 Expression",
            "entities": {"total": 4, "found": 2, "pValue": 2.18e-06, "fdr": 0.000233},
        },
        {
            "stId": "R-HSA-0000001",
            "name": "NOTCH1:M1580_K2555 | *odd* name",
            "entities": {"total": 90, "found": 1, "pValue": 0.01, "fdr": 0.04},
        },
    ],
}
SUBMITTED = ["TP53", "ERBB2", "RUNX2", "NOTAGENE1"]
URL = "https://beta.reactome.org/PathwayBrowser/#/DTAB=AN&ANALYSIS=MjAy"


def test_the_reply_reports_matches_pathways_and_the_link() -> None:
    reply = describe_overrepresentation(SUBMITTED, MEASURED, URL, ["NOTAGENE1"])
    assert reply.has_pathways
    assert "Over-representation analysis" in reply.text
    assert "Matched **3 of 4** identifiers" in reply.text
    assert "Top 2 of 188 pathways" in reply.text
    assert (
        "| Regulation of TP53 Expression (R-HSA-6804754) | 2 of 4 | 0.00023 |"
        in reply.text
    )
    assert URL in reply.text
    assert "Not found in Reactome: NOTAGENE1" in reply.text
    # Three matched genes: the caution is given.
    assert "pointers rather than findings" in reply.text


def test_pathway_names_cannot_break_the_table() -> None:
    reply = describe_overrepresentation(SUBMITTED, MEASURED, URL, None)
    row = next(line for line in reply.text.splitlines() if "R-HSA-0000001" in line)
    assert row.count("|") - row.count("\\|") == 4
    assert "M1580\\_K2555" in row
    assert "\\*odd\\*" in row


def test_no_caution_for_a_list_of_useful_size() -> None:
    genes = [f"G{i}" for i in range(12)]
    reply = describe_overrepresentation(
        genes, {**MEASURED, "identifiersNotFound": 0}, URL, None
    )
    assert "Matched **12 of 12**" in reply.text
    assert "pointers rather than findings" not in reply.text


def test_nothing_matched_says_so_and_offers_nothing_to_continue() -> None:
    empty = {
        "summary": {"token": "x"},
        "identifiersNotFound": 2,
        "pathwaysFound": 0,
        "pathways": [],
    }
    reply = describe_overrepresentation(["AAA1", "BBB2"], empty, URL, ["AAA1", "BBB2"])
    assert not reply.has_pathways
    assert "Matched **0 of 2**" in reply.text
    assert "No Reactome pathways" in reply.text
    assert URL not in reply.text


def test_the_remainder_of_unmatched_counts_all_of_them() -> None:
    # The notFound lookup is paged (50); the reply said "and 30 more" for 300.
    many = [f"G{i}" for i in range(300)]
    result = {**MEASURED, "identifiersNotFound": 300}
    reply = describe_overrepresentation(many, result, URL, many[:50])
    assert "and 280 more" in reply.text


def test_an_impossible_match_count_is_not_shown() -> None:
    result = {**MEASURED, "identifiersNotFound": 5}
    reply = describe_overrepresentation(["A1", "B2"], result, URL, None)
    assert "Matched" not in reply.text
    assert "-3" not in reply.text


def test_an_odd_entities_field_does_not_raise() -> None:
    odd = {**MEASURED, "pathways": [{"stId": "R-HSA-1", "name": "X", "entities": "?"}]}
    reply = describe_overrepresentation(SUBMITTED, odd, URL, None)
    assert "| X (R-HSA-1) | – of – | – |" in reply.text


def test_a_newline_in_a_name_stays_in_its_row() -> None:
    odd = {
        **MEASURED,
        "pathways": [{"stId": "R-HSA-1", "name": "A\nB", "entities": {}}],
    }
    reply = describe_overrepresentation(SUBMITTED, odd, URL, None)
    assert "| A B (R-HSA-1) |" in reply.text


# --- submission, stubbed at the transport -----------------------------------


def _submit(
    handler: Callable[[httpx.Request], httpx.Response], ids: list[str] = SUBMITTED
) -> analysis_client.Submitted | None:
    async def go() -> analysis_client.Submitted | None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await analysis_client.submit_identifiers(ids, client=http)

    return asyncio.run(go())


def test_submission_posts_the_list_and_returns_the_token() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = request.content.decode()
        seen["type"] = request.headers["content-type"]
        seen["sort"] = request.url.params.get("sortBy")
        return httpx.Response(200, json=MEASURED)

    submitted = _submit(handler)
    assert submitted is not None
    assert submitted.token == MEASURED["summary"]["token"]  # type: ignore[index]
    assert seen == {
        "path": "/AnalysisService/identifiers/projection",
        "body": "TP53\nERBB2\nRUNX2\nNOTAGENE1",
        "type": "text/plain",
        "sort": "ENTITIES_FDR",
    }


def test_submission_is_bounded() -> None:
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sizes.append(len(request.content.decode().split("\n")))
        return httpx.Response(200, json=MEASURED)

    many = [f"G{i}" for i in range(analysis_client.MAX_SUBMITTED_IDENTIFIERS + 50)]
    assert _submit(handler, many) is not None
    assert sizes == [analysis_client.MAX_SUBMITTED_IDENTIFIERS]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="boom"),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"summary": {}}),
        httpx.Response(200, json={"summary": {"token": "../../etc"}}),
        httpx.Response(200, json=["a list"]),
    ],
)
def test_submission_that_yields_no_usable_token_is_none(
    response: httpx.Response,
) -> None:
    assert _submit(lambda request: response) is None


def test_a_transport_failure_is_none_not_an_exception() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow", request=request)

    assert _submit(handler) is None


def test_the_link_is_to_the_service_that_holds_the_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANALYSIS_BASE_URL", "https://beta.reactome.org/AnalysisService")
    assert analysis_client.pathway_browser_url("abc%3D") == (
        "https://beta.reactome.org/PathwayBrowser/#/DTAB=AN&ANALYSIS=abc%3D"
    )
