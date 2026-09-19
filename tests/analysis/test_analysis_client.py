"""Reading a result, and the three status codes that are not faults."""

import asyncio

import httpx
import pytest

from analysis import client as analysis_client

RESULT = {"summary": {"type": "OVERREPRESENTATION"}, "pathways": []}
GSA = {"summary": {"type": "GSA_REGULATION", "gsaMethod": "Camera"}}


def _client(handler) -> httpx.AsyncClient:  # type: ignore[no-untyped-def]
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# Shaped like a real one. S105 flags any string bound to a name containing
# "token"; an analysis token addresses a user's uploaded result but is not a
# credential, and this one is for a throwaway analysis of public gene names.
SAMPLE_TOKEN = "MjAyNjA5MTkxODExNDJfMTE"  # noqa: S105


async def _fetch(status: int, body: object = None) -> object:
    token = SAMPLE_TOKEN

    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(body, dict):
            return httpx.Response(status, json=body)
        return httpx.Response(status, text="" if body is None else str(body))

    async with _client(handler) as http:
        return await analysis_client.fetch_result(token, client=http)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (404, "not_found"),
        # 410 stays distinct: the result was deleted by a release, so the
        # reader can re-run. 404 is a dead end; collapsing them wastes
        # information the service went to the trouble of giving us.
        (410, "gone"),
        # Undocumented, and what a malformed token actually returns --
        # measured against beta, where `x` and `%20` both give 500. Treating
        # it as a fault would mean a `failed` state or a retry loop against a
        # service that will answer identically every time.
        (500, "not_found"),
        (503, "failed"),
    ],
)
def test_status_codes_map_to_outcomes(status: int, expected: str) -> None:
    assert asyncio.run(_fetch(status)).outcome == expected  # type: ignore[attr-defined]


def test_a_result_comes_back_on_200() -> None:
    fetched = asyncio.run(_fetch(200, RESULT))
    assert fetched.outcome == "ok"  # type: ignore[attr-defined]
    assert fetched.result == RESULT  # type: ignore[attr-defined]


def test_a_reactomegsa_result_is_declined_not_summarised() -> None:
    # A separate service with its own result shape. Recognising it costs one
    # field check and prevents a confident summary of something unmodelled.
    assert asyncio.run(_fetch(200, GSA)).outcome == "unsupported"  # type: ignore[attr-defined]


def test_a_non_json_200_is_a_failure_not_a_crash() -> None:
    # What the automation block returns: 200 with an HTML body.
    assert asyncio.run(_fetch(200, "<html>blocked</html>")).outcome == "failed"  # type: ignore[attr-defined]


def test_the_request_carries_a_browser_user_agent() -> None:
    # Measured: a library user-agent gets 403 with an HTML body on every
    # endpoint, and it looks exactly like an auth failure.
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(200, json=RESULT)

    async def go() -> None:
        async with _client(handler) as http:
            await analysis_client.fetch_result("T", client=http)

    asyncio.run(go())
    assert "Mozilla/5.0" in seen["user-agent"]
    assert "python" not in seen["user-agent"].lower()


def test_the_default_base_url_is_beta_and_never_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(analysis_client.BASE_URL_ENV, raising=False)
    assert analysis_client.base_url() == "https://beta.reactome.org/AnalysisService"
    assert "//reactome.org" not in analysis_client.DEFAULT_BASE_URL


def test_the_release_is_read_not_hardcoded() -> None:
    # Principle V, and it matters twice: the release is reported to the
    # reader *and* is the cache key, because the service deletes results on a
    # release change.
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/database/version")
        return httpx.Response(200, text="97\n")

    async def go() -> str | None:
        async with _client(handler) as http:
            return await analysis_client.current_release(client=http)

    assert asyncio.run(go()) == "97"


def test_a_release_lookup_failure_is_none_not_an_exception() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    async def go() -> str | None:
        async with _client(handler) as http:
            return await analysis_client.current_release(client=http)

    assert asyncio.run(go()) is None


def test_the_release_request_does_not_demand_json() -> None:
    # `/database/version` answers with the bare number as text/plain and
    # rejects `Accept: application/json` with 406. The first version of this
    # client did exactly that, and every mocked test passed -- a mock cannot
    # refuse a header it was never told about. Found by calling the real
    # service; pinned here as the property rather than the status code.
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(200, text="97")

    async def go() -> None:
        async with _client(handler) as http:
            await analysis_client.current_release(client=http)

    asyncio.run(go())
    assert seen["accept"] != "application/json"
    assert "text/plain" in seen["accept"]


# Every one of these escapes `/token/{token}` when interpolated into a path.
# The first is the one that matters: it addresses the endpoint returning the
# user's unmatched identifiers -- the identifier tier -- for a caller who
# asked only for an aggregate summary. Demonstrated against beta 2026-09-19.
ESCAPES = (
    f"{SAMPLE_TOKEN}%3D/notFound",
    "../database/version",
    "a/b/c",
    "x?pageSize=9999",
    "..%2f..%2fadmin",
    "",
)


@pytest.mark.parametrize("token", ESCAPES)
def test_a_token_that_escapes_the_endpoint_never_reaches_the_network(
    token: str,
) -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(200, json=RESULT)

    async def go() -> object:
        async with _client(handler) as http:
            return await analysis_client.fetch_result(token, client=http)

    fetched = asyncio.run(go())
    assert fetched.outcome == "not_found"  # type: ignore[attr-defined]
    assert requested == [], f"{token!r} was sent to {requested}"


def test_a_real_token_is_still_accepted() -> None:
    # The guard above is worthless if it also rejects valid tokens, and the
    # issued form carries percent-encoded padding.
    assert analysis_client.is_well_formed(f"{SAMPLE_TOKEN}%3D")
    assert analysis_client.is_well_formed(SAMPLE_TOKEN)
    assert analysis_client.is_well_formed("MjAyNjA5MTkxODIyMTFfMTI=")


def test_a_list_body_is_an_outcome_not_an_exception() -> None:
    # `/token/{t}/notFound` answers with a list. Before this, a body that was
    # not an object raised AttributeError straight out of a function whose
    # whole contract is that it never raises.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"id": "SMITH_LAB_SECRET_GENE_001"}])

    async def go() -> object:
        async with _client(handler) as http:
            return await analysis_client.fetch_result(SAMPLE_TOKEN, client=http)

    assert asyncio.run(go()).outcome == "failed"  # type: ignore[attr-defined]
