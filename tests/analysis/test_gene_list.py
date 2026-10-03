"""A gene list typed into the chat: recognised, submitted, described.

The phrasing sets are sized rather than a couple of examples, because the
recogniser is a heuristic and a heuristic's failures are in its edges.
"""

import asyncio
import re
import time
from collections.abc import Callable

import gene_list_phrases as phrases
import httpx
import pytest

from analysis import client as analysis_client
from analysis.client import MAX_SUBMITTED_IDENTIFIERS
from analysis.gene_list import (
    MAX_PROPOSED_LISTED,
    answer_to_invitation,
    confirms,
    describe_overrepresentation,
    describe_proposal,
    gene_list_request,
    identifiers_in,
    listed,
    read_message,
    refers_back,
)

# The message that prompted this, verbatim.
ASKED = (
    "can we do a gsa analysis in the chat. I want to do it with genes TP53, "
    "ERBB2 and RUNX2"
)

#: Reported 2026-09-29: sent straight after the chat said "include the genes
#: in your message", and answered by the model instead.
HANDED_OVER = "here is my gene list TP53, ERBB3 and JAX9"

REQUESTS = [
    (ASKED, ["TP53", "ERBB2", "RUNX2"]),
    (HANDED_OVER, ["TP53", "ERBB3", "JAX9"]),
    ("my genes are TP53, MDM2, CDKN1A", ["TP53", "MDM2", "CDKN1A"]),
    ("Here are my genes:\nTP53\nMDM2", ["TP53", "MDM2"]),
    ("these are my genes: egfr, kras, braf", ["egfr", "kras", "braf"]),
    ("gene list: SOX2, POU5F1, NANOG", ["SOX2", "POU5F1", "NANOG"]),
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
    *phrases.SECOND_REVIEW_REQUESTS,
    *phrases.HELD_OUT_3_REQUESTS,
    *phrases.THIRD_REVIEW_REQUESTS,
    *phrases.HELD_OUT_4_REQUESTS,
    *phrases.TUNED_LATER,
    *(
        (phrases.TRAILING_BASE + tail, ["TP53", "MDM2", "CDKN1A"])
        for tail in phrases.TRAILING
    ),
    (
        "Run an enrichment on up-regulated genes: TP53, MDM2; down-regulated: MYC, CCND1",
        ["TP53", "MDM2", "MYC", "CCND1"],
    ),
    ("Run ORA on\r\nTP53\r\nMDM2", ["TP53", "MDM2"]),
    # Round three: a lower-case word after a list is the sentence going on.
    ("run ORA on TP53, MDM2, then show me the top hits", ["TP53", "MDM2"]),
    ("run ORA on TP53, MDM2 or similar", ["TP53", "MDM2"]),
    ("run an enrichment on TP53, MDM2, CDKN1A, cheers", ["TP53", "MDM2", "CDKN1A"]),
    (
        "run an enrichment on TP53, MDM2, CDKN1A, and plot it",
        ["TP53", "MDM2", "CDKN1A"],
    ),
    ("analyze my genes - CTNNB1, APC, AXIN2", ["CTNNB1", "APC", "AXIN2"]),
    ("run enrichment in STAT1, STAT2, IRF9", ["STAT1", "STAT2", "IRF9"]),
    # A list can open the message, in lower case.
    ("egfr, kras, braf - run ORA on these", ["egfr", "kras", "braf"]),
    # The verb is not the list's first member.
    ("submit TLR4, MYD88 for pathway analysis", ["TLR4", "MYD88"]),
    ("run ORA on TP53\u00a0MDM2\u00a0CDKN1A", ["TP53", "MDM2", "CDKN1A"]),
    ("run ORA on TP53\uff0cMDM2\uff0cCDKN1A", ["TP53", "MDM2", "CDKN1A"]),
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
    *phrases.SECOND_REVIEW_QUESTIONS,
    *phrases.HELD_OUT_3_QUESTIONS,
    *phrases.THIRD_REVIEW_QUESTIONS,
    *phrases.HELD_OUT_4_QUESTIONS,
    "please run GSA on my matrix, columns are sample1, sample2, sample3",
    "Map EGFR mutations L858R and T790M to pathways",
]


@pytest.mark.parametrize(("text", "current"), phrases.KNOWN_LIMITS)
def test_known_limits_are_as_recorded(text: str, current: list[str] | None) -> None:
    # Wrong, and pinned: if one of these changes, update the record and the
    # measured rate in gene_list_phrases.py.
    assert gene_list_request(text) == current


@pytest.mark.parametrize(
    "text",
    [
        # Under the cap, so the parser reads all of it: the first rewrite
        # took 38 s on 4,000 lines. (8,000 lines are over the cap, and would
        # pass on the old code unread.)
        "run ORA on:\n" + "\n".join(f"GENE{i}" for i in range(6000)),
        "run ORA on TP53, MDM2" + "\n" * 50_000 + "x",
        "run ORA on TP53, MDM2" + "?" * 50_000 + "x",
        "run ORA on: TP53, MDM2" + " " * 50_000 + "!MYC",
        "run ORA on " + "of " * 15_000 + "TP53, MDM2",
        "do " + "word " * 11_000 + "analysis on TP53, MDM2",
        # Over the cap: refused unread, however list-like.
        "run ORA on: " + "TP53, " * 500_000,
    ],
    ids=[
        "column",
        "newlines",
        "question-marks",
        "spaces",
        "openers",
        "request",
        "huge",
    ],
)
def test_reading_a_message_is_fast_whatever_it_holds(text: str) -> None:
    # It runs on the event loop every session shares. The first rewrite took
    # 38 s on a 4,000-line column and 24 s on 40,000 newlines.
    started = time.perf_counter()
    gene_list_request(text)
    assert time.perf_counter() - started < 0.5


@pytest.mark.parametrize(
    "text", ["yes", "Yes please", "ok", "run it", "Go ahead!", "sure."]
)
def test_a_typed_yes_confirms(text: str) -> None:
    assert confirms(text)


@pytest.mark.parametrize(
    "text", ["yes but first explain ORA", "no", "what is TP53?", "", "yesterday"]
)
def test_anything_else_does_not(text: str) -> None:
    assert not confirms(text)


def test_the_proposal_says_when_the_list_will_be_cut() -> None:
    long = describe_proposal([f"G{i}" for i in range(MAX_SUBMITTED_IDENTIFIERS + 1)])
    assert f"Only the first {MAX_SUBMITTED_IDENTIFIERS:,} will be submitted" in long
    assert "will be submitted" not in describe_proposal(["TP53", "MDM2"])


@pytest.mark.parametrize(("text", "expected"), REQUESTS)
def test_a_request_with_genes_is_recognised(text: str, expected: list[str]) -> None:
    assert gene_list_request(text) == expected


@pytest.mark.parametrize("text", NOT_REQUESTS)
def test_other_messages_are_left_to_the_model(text: str) -> None:
    assert gene_list_request(text) is None


def test_the_sets_are_the_size_they_claim() -> None:
    # Sized sets are the point: a pass at n=2 says nothing about a rate.
    assert len(REQUESTS) >= 100
    assert len(NOT_REQUESTS) >= 120


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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("TP53, ERBB3, JAX9", ["TP53", "ERBB3", "JAX9"]),
        (HANDED_OVER, ["TP53", "ERBB3", "JAX9"]),
        ("TP53\nMDM2\nCDKN1A", ["TP53", "MDM2", "CDKN1A"]),
        ("ok: egfr, kras, braf", ["egfr", "kras", "braf"]),
    ],
)
def test_after_an_invitation_a_bare_list_is_the_reply(
    text: str, expected: list[str]
) -> None:
    assert answer_to_invitation(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "how do TP53 and MDM2 interact?",
        "thanks",
        "TP53",
        "what does the 20 MB limit mean?",
        "Can you explain the difference between TP53, MDM2 and CDKN1A?",
    ],
)
def test_after_an_invitation_other_replies_are_not_lists(text: str) -> None:
    assert answer_to_invitation(text) is None


def test_a_bare_list_without_an_invitation_is_left_alone() -> None:
    # The invitation is what makes a bare list a request.
    assert gene_list_request("TP53, ERBB3, JAX9") is None


#: Reported 2026-09-29, after the list had been sent in an earlier message.
REFERS_BACK = "can you analyze the gene list that I gave you"


@pytest.mark.parametrize(
    "text",
    [
        REFERS_BACK,
        "please run an enrichment on those genes",
        "run ORA on the list I sent earlier",
        "analyse my gene list",
    ],
)
def test_a_request_about_an_earlier_list_refers_back(text: str) -> None:
    assert refers_back(text)


@pytest.mark.parametrize(
    "text",
    [
        "what is an enrichment analysis?",  # a question
        "can you explain the list of pathways above?",  # about, not a request
        "run ORA on TP53, MDM2",  # the list is here, not earlier
        "tell me about TP53",
        "",
        # "them" could be anything; a gene list must be named.
        "can you do a pathway analysis of them?",
        # Pointing back at something that is not a gene list (review, round 4).
        "analyse the pathways above",
        "analyze the results above",
        "can you perform the analysis the website mentioned",
        "run a gene set analysis on my expression data I sent before",
        "analyze them with GSEA instead",
        "run GSA on it again with the samples I listed",
        "analyse the file I uploaded earlier",
        # A gene list is named, but it is a file or a GSA -- the guard for these.
        "analyse the genes in the file I uploaded",
        "run GSEA on those genes",
        "run a GSA on the gene list from my expression matrix",
    ],
)
def test_other_messages_do_not(text: str) -> None:
    assert not refers_back(text)


def test_a_list_is_remembered_from_any_message() -> None:
    assert listed("What do TP53, MDM2 and CDKN1A have in common?") == [
        "TP53",
        "MDM2",
        "CDKN1A",
    ]
    assert listed("What does TP53 do?") is None


def test_an_offer_of_an_earlier_list_says_so() -> None:
    assert "in your earlier message" in describe_proposal(
        ["TP53", "MDM2"], earlier=True
    )
    assert "in your message" in describe_proposal(["TP53", "MDM2"])


def test_asking_for_a_list_when_none_was_sent_invites_one_that_would_work() -> None:
    # The example in the reply, sent as the next message, must be offered.
    from analysis.gene_list import NO_LIST_YET

    example = re.search(r"\*([^*]+)\*", NO_LIST_YET)
    assert example is not None
    assert answer_to_invitation(example.group(1)) == ["TP53", "ERBB2", "RUNX2"]


@pytest.mark.parametrize(
    "text",
    [
        # The how-to's own step-2 example, typed straight after it.
        "control, control, treated, treated",
        "yes, great",
        "sure, sounds good",
        "nope, later",
        "ctrl, trt",
        "WT, KO",
        "day0, day3, day7",
        "Sample1, Sample2, Sample3",
        "Rep1 Rep2 Rep3",
        "HeLa, U2OS cells",
        "Treated, Untreated",
        "GeneSymbol, Sample1, Sample2",
    ],
)
def test_after_an_invitation_replies_that_are_not_genes_are_not_offered(
    text: str,
) -> None:
    assert answer_to_invitation(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "hmm, interesting",
        "I see, makes sense",
        "great, now tell me more",
        "What is PD-1 PD-L1 checkpoint blockade",
        "SARS-CoV-2 ACE2 TMPRSS2 entry pathway",
    ],
)
def test_chat_is_not_remembered_as_a_list(text: str) -> None:
    assert listed(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("my genes are TP53, MDM2, CDKN1A", ["TP53", "MDM2", "CDKN1A"]),
        ("my genes are egfr, kras, braf", ["egfr", "kras", "braf"]),
    ],
)
def test_a_hand_over_reads_the_list_after_the_phrase(
    text: str, expected: list[str]
) -> None:
    assert gene_list_request(text) == expected


def test_are_does_not_open_a_list_in_general() -> None:
    # Review, round 4: "are" opening lists made mid-sentence pairs requests.
    assert (
        gene_list_request("run an analysis on RNA-seq where the controls are WT, KO")
        is None
    )


# --- the whole decision, as the handler takes it ------------------------------


def test_reading_the_reported_first_conversation() -> None:
    how_to = read_message("can we run a gsa experiment", invited=False)
    assert how_to.offer is None
    assert how_to.refers_back is False
    handed = read_message(HANDED_OVER, invited=True)
    assert handed.offer == ["TP53", "ERBB3", "JAX9"]
    assert handed.listed == ["TP53", "ERBB3", "JAX9"]


def test_reading_the_reported_second_conversation() -> None:
    asked = read_message("What do TP53, ERBB3 and MDM2 have in common?", invited=False)
    assert asked.offer is None
    assert asked.listed == ["TP53", "ERBB3", "MDM2"]
    back = read_message(REFERS_BACK, invited=False)
    assert back.refers_back is True
    assert back.offer is None


def test_an_invitation_only_widens_what_is_offered() -> None:
    assert read_message("TP53, ERBB3, MDM2", invited=False).offer is None
    assert read_message("TP53, ERBB3, MDM2", invited=True).offer == [
        "TP53",
        "ERBB3",
        "MDM2",
    ]
    assert (
        read_message("control, control, treated, treated", invited=True).offer is None
    )


def test_a_matrix_request_is_left_to_the_gsa_how_to() -> None:
    reading = read_message(
        "run a gene set analysis on my expression data I sent before", invited=False
    )
    assert reading.offer is None
    assert reading.refers_back is False


@pytest.mark.parametrize(("text", "invited", "offer", "back"), phrases.HELD_OUT_5)
def test_the_new_routes_on_a_held_out_set(
    text: str, invited: bool, offer: list[str] | None, back: bool
) -> None:
    reading = read_message(text, invited=invited)
    assert reading.offer == offer
    assert reading.refers_back == back


@pytest.mark.parametrize(("text", "current"), phrases.KNOWN_LIMITS_INVITED)
def test_known_limits_after_an_invitation(text: str, current: list[str] | None) -> None:
    assert read_message(text, invited=True).offer == current


@pytest.mark.parametrize("invited", [False, True])
def test_reading_a_whole_message_is_fast(invited: bool) -> None:
    # Off the event loop, but still one worker thread per message.
    for text in ("a b " * 15_000, "run analysis on them " + "a " * 29_000):
        started = time.perf_counter()
        read_message(text, invited=invited)
        assert time.perf_counter() - started < 2.0


def test_the_model_never_gets_the_link_that_carries_the_token() -> None:
    # The Pathway Browser link embeds the analysis token; anyone holding it can
    # fetch the result. It is for the reader, never the model (review, 1a).
    token = MEASURED["summary"]["token"]  # type: ignore[index]
    url = f"https://beta.reactome.org/PathwayBrowser/#/DTAB=AN&ANALYSIS={token}"
    reply = describe_overrepresentation(SUBMITTED, MEASURED, url, ["NOTAGENE1"])
    assert url in reply.text
    assert token not in reply.for_model
    assert "Regulation of TP53 Expression" in reply.for_model


def test_a_padded_message_cannot_stall_the_yes_check() -> None:
    # "y" plus spaces backtracked quadratically: ~90 minutes at 990K on the
    # shared event loop (review, area 2). Bounded and possessive now.
    started = time.perf_counter()
    assert not confirms("y" + " " * 990_000 + "x")
    assert not confirms("yes" + " " * 30 + "x")
    assert time.perf_counter() - started < 0.1
    assert confirms("yes  !")
