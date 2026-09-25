"""A gene list typed into the chat: recognised, submitted, described.

The phrasing sets are sized rather than a couple of examples, because the
recogniser is a heuristic and a heuristic's failures are in its edges.
"""

import asyncio
from collections.abc import Callable

import httpx
import pytest

from analysis import client as analysis_client
from analysis.gene_list import (
    describe_overrepresentation,
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
    ("run a pathway analysis on EGFR KRAS BRAF PTEN", ["EGFR", "KRAS", "BRAF", "PTEN"]),
    (
        "Please run an enrichment for P04637, Q9Y6K9 and ENSG00000141510",
        ["P04637", "Q9Y6K9", "ENSG00000141510"],
    ),
    ("analyse these genes: tp53 mdm2 cdkn1a", ["tp53", "mdm2", "cdkn1a"]),
    (
        "Can you do an over-representation analysis with BRCA1, BRCA2, PALB2?",
        ["BRCA1", "BRCA2", "PALB2"],
    ),
    ("perform ORA on\nTP53\nMDM2\nCDKN1A\n", ["TP53", "MDM2", "CDKN1A"]),
    ("run a GSEA with genes MYC, MAX", ["MYC", "MAX"]),
    ("I would like to run an enrichment analysis for Trp53, Mdm2", ["Trp53", "Mdm2"]),
    ("could you analyse my gene list: HLA-A, HLA-B, B2M", ["HLA-A", "HLA-B", "B2M"]),
    ("do a pathway analysis for TP53, TP53, tp53, MDM2", ["TP53", "MDM2"]),
    ("run reactome gsa on P04637-2 and Q00987", ["P04637-2", "Q00987"]),
    (
        "Please do a gene set analysis for SMAD2, SMAD3, SMAD4, TGFB1",
        ["SMAD2", "SMAD3", "SMAD4", "TGFB1"],
    ),
]

NOT_REQUESTS = [
    "can we run gsa in this chat please",  # no genes: how-to reply
    "what is GSEA?",
    "can you explain how TP53 and MDM2 interact?",  # genes, no analysis term
    "What pathways are BRCA1 and BRCA2 in?",
    "Compare GSEA and PADOG analysis methods",  # capitals, not genes
    "CAN YOU RUN A GSA ANALYSIS ON MY DATA PLEASE",  # shouting
    "Is TP53 enriched in apoptosis?",  # one gene is not a list
    "run a pathway analysis on TP53",
    "How do I run GSA on RNA-seq data from a TSV or CSV?",
    "what does an FDR mean in an ORA result?",
    "Tell me about the role of CDK5 in neurons",
    "what is the difference between ORA and GSEA",
    "",
]


@pytest.mark.parametrize(("text", "expected"), REQUESTS)
def test_a_request_with_genes_is_recognised(text: str, expected: list[str]) -> None:
    assert gene_list_request(text) == expected


@pytest.mark.parametrize("text", NOT_REQUESTS)
def test_other_messages_are_left_to_the_model(text: str) -> None:
    assert gene_list_request(text) is None


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
    assert "over-representation" in reply.text
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
