"""The client against HTTP responses shaped like the real service's.

Every earlier test of this package stubbed `GsaClient` at the method level,
returning Python values. That tested everything except the client itself,
and the client was where the bug was: `submit` called `response.json()` on a
`text/plain` body, so the upload feature could never start an analysis. It
was deployed that way, and 67 tests, CI and the answer sweep all passed.

Found by a headless browser driving the deployed image, not by any of those.

These tests go through `httpx.MockTransport`, so the request is built, sent
and parsed by the same code that runs in production. The response bodies are
the service's own: a bare UUID as `text/plain`, measured from
`POST /analysis` on 2026-09-21 and declared in its swagger
(`produces: text/plain`, example `Analysis00371643`).
"""

import asyncio
import json
from typing import Any

import httpx
import pytest

from gsa.client import GsaClient, GsaError

ANALYSIS_ID = "30790e1a-b5f2-11f1-b43e-026f8a266be5"


def client_answering(handler: Any) -> GsaClient:
    return GsaClient("https://gsa.test/0.1", transport=httpx.MockTransport(handler))


def submit(client: GsaClient) -> str:
    return asyncio.run(
        client.submit(
            method="PADOG",
            dataset_name="uploaded",
            dataset_type="rnaseq_counts",
            matrix="\tS1\tS2\nENSG1\t1\t2\n",
            samples=["S1", "S2"],
            analysis_group=["A", "B"],
            group1="A",
            group2="B",
        )
    )


def plain(text: str, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status, text=text, headers={"content-type": "text/plain; charset=utf-8"}
    )


class TestSubmit:
    def test_reads_the_bare_id_the_service_actually_returns(self) -> None:
        # The bug. This is the body `POST /analysis` sends.
        assert submit(client_answering(lambda _: plain(ANALYSIS_ID))) == ANALYSIS_ID

    def test_tolerates_a_trailing_newline(self) -> None:
        assert (
            submit(client_answering(lambda _: plain(ANALYSIS_ID + "\n"))) == ANALYSIS_ID
        )

    def test_would_also_accept_a_json_quoted_id(self) -> None:
        # Not what the service does today. Accepted so a change of framing
        # upstream does not silently break the feature a second time.
        assert (
            submit(client_answering(lambda _: plain(json.dumps(ANALYSIS_ID))))
            == ANALYSIS_ID
        )

    @pytest.mark.parametrize(
        "body",
        [
            "",
            '{"detail": "Bad Request"}',
            "<html><body>502 Bad Gateway</body></html>",
            "Analysis/../../status",
        ],
    )
    def test_refuses_a_body_that_is_not_an_identifier(self, body: str) -> None:
        # Each of these would otherwise become a URL path segment and fail
        # later as a 404 that reads like a missing analysis.
        with pytest.raises(GsaError, match="not a valid identifier"):
            submit(client_answering(lambda _, b=body: plain(b)))

    def test_sends_the_matrix_inline_and_the_design_the_service_needs(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["body"] = json.loads(request.content)
            return plain(ANALYSIS_ID)

        submit(client_answering(handler))

        assert seen["path"] == "/0.1/analysis"
        dataset = seen["body"]["datasets"][0]
        assert dataset["data"].startswith("\tS1\tS2")
        assert dataset["design"]["comparison"] == {"group1": "A", "group2": "B"}
        assert dataset["name"] == "uploaded"

    def test_a_non_200_is_an_error_with_the_status(self) -> None:
        with pytest.raises(GsaError, match="500"):
            submit(
                client_answering(lambda _: plain("Internal Server Error", status=500))
            )


class TestLoad:
    def test_reads_the_bare_loading_id(self) -> None:
        # `POST /data/load/{resourceId}` is also `text/plain`, and had the
        # same bug. It has no chat route yet, so nobody would have found it
        # until one was added.
        client = client_answering(
            lambda _: plain("160b3166-b5f2-11f1-b383-e618a8785587")
        )
        loading_id = asyncio.run(
            client.load_public_dataset("example_datasets", "EXAMPLE_MEL_RNA")
        )
        assert loading_id == "160b3166-b5f2-11f1-b383-e618a8785587"


class TestTheJsonEndpointsStayJson:
    def test_status_is_still_read_as_json(self) -> None:
        # The fix must not over-correct: /status, /data/summary and /result
        # really are JSON, measured.
        body = {
            "id": ANALYSIS_ID,
            "status": "running",
            "description": "Permutation 60 / 1000",
            "completed": 0.6,
        }
        client = client_answering(lambda _: httpx.Response(200, json=body))
        status = asyncio.run(client.analysis_status(ANALYSIS_ID))
        assert status.status == "running"
        assert status.completed == 0.6
