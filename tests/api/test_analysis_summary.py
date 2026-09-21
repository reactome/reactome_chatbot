"""The analysis-summary endpoint, driven over HTTP.

Principle I: mounting order, the router prefix and the captcha exemption
interact only on the served path. Spec 010's route check found a problem there
that isolated tests could not.

Neither the Analysis Service nor a model is called: the client is replaced and
so is the LLM, because what is asserted here is refusal, event shape, and that
no model call happens without a person -- none of which depends on content.
"""

import asyncio
import json
import re
import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, cast

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import FastAPI
from fastapi.testclient import TestClient

from analysis.client import Fetched
from analysis.store import SummaryStore
from api.analysis_summary import SummaryRequest, analysis_summary, router
from util.caller_token import DEFAULT_AUDIENCE
from util.rate_limit import SlidingWindowLimiter

PREFIX = "/chat/guest/api"

# Shaped like a real analysis token. S106 flags any string passed to an
# argument named `token`; an analysis token addresses a user's result but is
# not a credential.
SAMPLE_TOKEN = "MjAyNjA5MTkxODExNDJfMTE"  # noqa: S105

RESULT: dict[str, Any] = {
    "summary": {"type": "OVERREPRESENTATION", "fileName": "smith_unpublished.txt"},
    "identifiersNotFound": 2,
    "pathwaysFound": 3,
    "warnings": [],
    "pathways": [
        {
            "stId": "R-HSA-109581",
            "name": "Apoptosis",
            "entities": {"found": 4, "total": 11, "pValue": 1e-7, "fdr": 4e-5},
        },
        {
            "stId": "R-HSA-1640170",
            "name": "Cell Cycle",
            "entities": {"found": 2, "total": 90, "pValue": 0.2, "fdr": 0.4},
        },
    ],
}


class _Counter:
    """Counts model calls, so "no model call" is asserted, not assumed."""

    def __init__(self) -> None:
        self.calls = 0

    async def astream(self, _messages: Any) -> AsyncIterator[Any]:
        self.calls += 1
        for piece in ("Four pathways ", "pass correction."):
            yield type("Chunk", (), {"content": piece})()


@pytest.fixture(scope="module")
def keys() -> tuple[str, str]:
    private = Ed25519PrivateKey.generate()
    return (
        private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode(),
        private.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode(),
    )


@pytest.fixture(autouse=True)
def wired(monkeypatch: pytest.MonkeyPatch) -> _Counter:
    counter = _Counter()
    monkeypatch.setattr(
        "api.analysis_summary._limiter",
        SlidingWindowLimiter(limit=10_000, window=600.0),
    )
    # A private store per test. It is module state that outlives a request by
    # design, so without this one test serves another its cached summary and
    # the model is never called -- which looks like the feature being broken
    # and is the tests interfering.
    monkeypatch.setattr("api.analysis_summary._store", SummaryStore())
    monkeypatch.setattr("api.analysis_summary.get_llm", lambda *a, **k: counter)
    monkeypatch.setattr(
        "api.analysis_summary.resolve_llm_model", lambda _c: ("openai", "m", None)
    )

    async def _fetch(_token: str) -> Fetched:
        return Fetched("ok", RESULT)

    async def _release() -> str:
        return "97"

    monkeypatch.setattr("api.analysis_summary.fetch_result", _fetch)
    monkeypatch.setattr("api.analysis_summary.current_release", _release)
    return counter


def _client(public_pem: str) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix=PREFIX)
    app.state.caller_token_key = public_pem
    return TestClient(app)


def _token(private_pem: str, omit: tuple[str, ...] = (), **claims: object) -> str:
    """`omit` removes a claim entirely, which is different from setting it to
    None -- and the difference is the whole point of the absence cases."""
    payload: dict[str, object] = {
        "iss": "reactome-website",
        "aud": DEFAULT_AUDIENCE,
        "exp": int(time.time()) + 300,
        "sub": "visit-1",
        "human": True,
        "human_iat": int(time.time()) - 10,
    }
    payload.update(claims)
    for key in omit:
        payload.pop(key, None)
    return jwt.encode(payload, private_pem, algorithm="EdDSA")


def _events(text: str) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(ln.split(": ", 1) for ln in block.splitlines() if ": " in ln)
        if "event" in lines:
            out.append((lines["event"], json.loads(lines.get("data", "{}"))))
    return out


def _post(public: str, **body: object) -> Any:
    payload: dict[str, object] = {
        "token": "MjAyNjA5MTkxODExNDJfMTE",
        "disclosure": "aggregate",
    }
    payload.update(body)
    return _client(public).post(f"{PREFIX}/analysis-summary", json=payload)


def test_a_verified_human_caller_gets_a_summary(keys: tuple[str, str]) -> None:
    private, public = keys
    response = _post(public, caller_token=_token(private))
    assert response.status_code == 200
    events = _events(response.text)
    kinds = [k for k, _ in events]
    assert kinds[0] == "start"
    assert kinds[-1] == "done"
    assert "citation" in kinds
    assert "token" in kinds
    start = events[0][1]
    assert start["release"] == 97
    assert start["analysis_type"] == "OVERREPRESENTATION"
    assert start["cached"] is False
    assert events[-1][1]["state"] == "summarised"


@pytest.mark.parametrize(
    ("claims", "reason"),
    [
        ({"human": None}, "no_human"),
        ({"human": False}, "no_human"),
        # The only refusal a reader can act on, so it gets its own reason.
        ({"human_iat": int(time.time()) - 1801}, "stale_human"),
        ({"human_iat": "not-a-number"}, "no_human"),
    ],
)
def test_a_caller_without_a_fresh_person_is_refused_with_no_model_call(
    keys: tuple[str, str], wired: _Counter, claims: dict[str, Any], reason: str
) -> None:
    # SC-004, counted on a patched model rather than inferred from timing.
    private, public = keys
    response = _post(public, caller_token=_token(private, **claims))
    assert response.status_code == 200
    state, payload = _events(response.text)[-1]
    assert payload["state"] == "refused"
    assert payload["reason"] == reason
    assert wired.calls == 0


def test_the_agreed_freshness_bound_is_inclusive_in_whole_seconds(
    keys: tuple[str, str],
) -> None:
    # Agreed with the website as `now - human_iat <= 1800`. It was first
    # agreed as 1800.000 against 1800.001, which `human_iat` cannot represent:
    # it is epoch seconds derived from a cookie expiry minus a constant TTL,
    # so it arrives already rounded. Both sides pin the whole-second edge.
    private, public = keys
    now = int(time.time())
    accepted = _post(public, caller_token=_token(private, human_iat=now - 1800))
    refused = _post(public, caller_token=_token(private, human_iat=now - 1801))
    assert _events(accepted.text)[-1][1]["state"] == "summarised"
    assert _events(refused.text)[-1][1]["reason"] == "stale_human"


def test_no_caller_token_means_no_model_call(
    keys: tuple[str, str], wired: _Counter
) -> None:
    _, public = keys
    payload = _events(_post(public).text)[-1][1]
    assert payload["state"] == "refused"
    assert payload["reason"] == "no_caller"
    assert wired.calls == 0


def test_every_cited_identifier_is_in_the_result(keys: tuple[str, str]) -> None:
    # SC-003, mechanically: citations are emitted from the result rather than
    # parsed out of the model's prose, so an invented identifier is impossible
    # by construction. This asserts the construction holds.
    private, public = keys
    events = _events(_post(public, caller_token=_token(private)).text)
    cited = {p["st_id"] for k, p in events if k == "citation"}
    assert cited
    assert cited <= {p["stId"] for p in RESULT["pathways"]}


def test_the_users_filename_never_reaches_the_model(keys: tuple[str, str]) -> None:
    # The allow-list is unit-tested; this checks it is actually applied on the
    # served path, which is a different claim.
    private, public = keys
    sent: list[Any] = []

    class _Recording(_Counter):
        async def astream(self, messages: Any) -> AsyncIterator[Any]:
            sent.append(messages)
            async for chunk in super().astream(messages):
                yield chunk

    recording = _Recording()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("api.analysis_summary.get_llm", lambda *a, **k: recording)
        _post(public, caller_token=_token(private))
    assert sent, "the model was never called, so this proves nothing"
    assert "smith_unpublished" not in json.dumps(sent, default=str)


def test_an_unknown_token_is_a_terminal_state_not_an_error(
    keys: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _missing(_token: str) -> Fetched:
        return Fetched("not_found")

    monkeypatch.setattr("api.analysis_summary.fetch_result", _missing)
    private, public = keys
    response = _post(public, caller_token=_token(private))
    assert response.status_code == 200
    assert _events(response.text)[-1][1]["state"] == "not_found"


def test_gone_is_distinct_from_not_found(
    keys: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # One is a dead end; the other has an action attached -- re-run the
    # analysis, because a release deleted the result.
    async def _gone(_token: str) -> Fetched:
        return Fetched("gone")

    monkeypatch.setattr("api.analysis_summary.fetch_result", _gone)
    private, public = keys
    assert (
        _events(_post(public, caller_token=_token(private)).text)[-1][1]["state"]
        == "gone"
    )


def test_a_missing_disclosure_choice_is_rejected(keys: tuple[str, str]) -> None:
    # Required with no default: a default is not a choice (FR-012).
    private, public = keys
    response = _client(public).post(
        f"{PREFIX}/analysis-summary",
        json={"token": "MjAyNjA5MTkxODExNDJfMTE", "caller_token": _token(private)},
    )
    assert response.status_code == 422


def _with_not_found_spy(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Records every request for the reader's own identifiers."""
    asked: list[str] = []

    async def _spy(token: str, **_kwargs: Any) -> list[str]:
        asked.append(token)
        return ["SMITH_LAB_SECRET_GENE_001", "PATIENT_004_MARKER"]

    monkeypatch.setattr("api.analysis_summary.fetch_not_found", _spy)
    return asked


def test_the_aggregate_tier_never_asks_for_the_users_identifiers(
    keys: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # T019, and the single most important assertion in this feature. Asserted
    # by recording the outbound call, not by reading the summary and seeing
    # nothing alarming -- a model that simply did not mention them would pass
    # the second check while the identifiers had already left the service.
    asked = _with_not_found_spy(monkeypatch)
    private, public = keys
    events = _events(_post(public, caller_token=_token(private)).text)
    assert events[-1][1]["state"] == "summarised"
    assert asked == [], "the aggregate tier fetched the user's identifiers"


def test_the_identifier_tier_asks_only_when_it_was_chosen(
    keys: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    asked = _with_not_found_spy(monkeypatch)
    private, public = keys
    events = _events(
        _post(public, caller_token=_token(private), disclosure="identifiers").text
    )
    assert events[-1][1]["state"] == "summarised"
    assert len(asked) == 1, "the chosen tier did not fetch what it promised"


def test_the_identifiers_reach_the_model_only_on_the_disclosing_tier(
    keys: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The guarantee stated as the reader would understand it: on the default
    # choice their identifiers are not sent anywhere, and on the other choice
    # they are -- which is the whole point of offering a choice.
    _with_not_found_spy(monkeypatch)
    sent: list[Any] = []

    class _Recording(_Counter):
        async def astream(self, messages: Any) -> AsyncIterator[Any]:
            sent.append(messages)
            async for chunk in super().astream(messages):
                yield chunk

    private, public = keys
    for tier, expected in (("aggregate", False), ("identifiers", True)):
        sent.clear()
        recording = _Recording()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                "api.analysis_summary.get_llm",
                lambda *a, _r=recording, **k: _r,
            )
            _post(public, caller_token=_token(private), disclosure=tier)
        assert sent, f"{tier}: the model was never called, so this proves nothing"
        leaked = "SMITH_LAB_SECRET_GENE_001" in json.dumps(sent, default=str)
        assert leaked is expected, f"{tier} tier sent identifiers: {leaked}"


def test_a_rate_limited_caller_is_not_told_it_is_unverified(
    keys: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # It used to answer `no_caller`, which an interface renders as "not
    # verified" to a reader who is verified and merely asked too often.
    monkeypatch.setattr(
        "api.analysis_summary._limiter", SlidingWindowLimiter(limit=0, window=600.0)
    )
    private, public = keys
    payload = _events(_post(public, caller_token=_token(private)).text)[-1][1]
    assert payload["reason"] == "rate_limited"


def test_the_disclosing_tier_is_told_to_name_what_it_was_given(
    keys: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Measured 2026-09-20: the first version sent the identifiers and never
    # mentioned them, because nothing asked it to. The reader chose to
    # disclose and got the aggregate summary back -- disclosure with no
    # benefit, which is worse than not offering the choice.
    _with_not_found_spy(monkeypatch)
    sent: list[Any] = []

    class _Recording(_Counter):
        async def astream(self, messages: Any) -> AsyncIterator[Any]:
            sent.append(messages)
            async for chunk in super().astream(messages):
                yield chunk

    private, public = keys
    for tier, expected in (("identifiers", True), ("aggregate", False)):
        sent.clear()
        recording = _Recording()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                "api.analysis_summary.get_llm",
                lambda *a, _r=recording, **k: _r,
            )
            _post(public, caller_token=_token(private), disclosure=tier)
        told = "Name them" in json.dumps(sent, default=str)
        assert told is expected, f"{tier}: instruction to name identifiers {told}"


def test_a_disclosure_that_could_not_be_honoured_is_reported(
    keys: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The failure path of the bug this phase fixed. The reader chose to
    # disclose, the lookup failed, and they got the aggregate summary with
    # nothing saying the disclosure had not happened -- disclosure with no
    # benefit again, arriving down the error path instead of the prompt.
    async def _fails(_token: str, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr("api.analysis_summary.fetch_not_found", _fails)
    private, public = keys
    start = _events(
        _post(public, caller_token=_token(private), disclosure="identifiers").text
    )[0][1]
    assert start["disclosure"] == "aggregate", "the caller was not told"


def test_the_applied_disclosure_matches_the_request_when_it_is_honoured(
    keys: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_not_found_spy(monkeypatch)
    private, public = keys
    for tier in ("aggregate", "identifiers"):
        start = _events(
            _post(public, caller_token=_token(private), disclosure=tier).text
        )[0][1]
        assert start["disclosure"] == tier


def test_nothing_to_disclose_is_not_a_failed_disclosure(
    keys: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A result where every identifier matched has nothing to retrieve. That
    # is the disclosing tier honoured, not denied -- reporting it as a
    # downgrade would tell the reader something untrue.
    async def _empty(_token: str, **_kwargs: Any) -> list[str]:
        return []

    monkeypatch.setattr("api.analysis_summary.fetch_not_found", _empty)
    monkeypatch.setitem(RESULT, "identifiersNotFound", 0)
    private, public = keys
    start = _events(
        _post(public, caller_token=_token(private), disclosure="identifiers").text
    )[0][1]
    assert start["disclosure"] == "identifiers"


@pytest.mark.parametrize(
    ("analysis_type", "expected", "forbidden"),
    [
        ("EXPRESSION", "across", "orthology"),
        ("SPECIES_COMPARISON", "orthology", "columns"),
        ("OVERREPRESENTATION", "up, down", "orthology"),
    ],
)
def test_each_type_gets_its_own_reading_on_the_served_path(
    keys: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
    analysis_type: str,
    expected: str,
    forbidden: str,
) -> None:
    # US4's independent test: submit each type and check neither summary is
    # told the other's reading. Asserted on what reaches the model, over
    # HTTP, because the branch is in the endpoint and not in `summarise`.
    monkeypatch.setitem(RESULT["summary"], "type", analysis_type)
    sent: list[Any] = []

    class _Recording(_Counter):
        async def astream(self, messages: Any) -> AsyncIterator[Any]:
            sent.append(messages)
            async for chunk in super().astream(messages):
                yield chunk

    recording = _Recording()
    private, public = keys
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("api.analysis_summary.get_llm", lambda *a, **k: recording)
        response = _post(public, caller_token=_token(private))
    assert _events(response.text)[0][1]["analysis_type"] == analysis_type
    assert sent, "the model was never called, so this proves nothing"
    prompt = json.dumps(sent, default=str)
    assert expected in prompt, f"{analysis_type} was not given its own reading"
    assert forbidden not in prompt, f"{analysis_type} was given another's"


def _prose(response: Any) -> str:
    return "".join(
        payload["text"] for kind, payload in _events(response.text) if kind == "token"
    )


def test_a_second_request_is_byte_identical_and_calls_no_model(
    keys: tuple[str, str], wired: _Counter
) -> None:
    # FR-014 on the served path. The generator is not deterministic, so if
    # the second request reached it the text would differ -- which is why
    # the assertion is on the bytes and the call count together.
    private, public = keys
    first = _post(public, caller_token=_token(private))
    calls_after_first = wired.calls
    second = _post(public, caller_token=_token(private))

    assert _prose(first) == _prose(second)
    assert _prose(first), "no prose was produced, so this proves nothing"
    assert wired.calls == calls_after_first, "the model ran again"
    assert _events(first.text)[0][1]["cached"] is False
    assert _events(second.text)[0][1]["cached"] is True


def test_a_release_change_regenerates(
    keys: tuple[str, str], wired: _Counter, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The service deletes results on a release, so a summary from the old one
    # describes something that no longer exists.
    private, public = keys
    _post(public, caller_token=_token(private))
    before = wired.calls

    async def _next_release() -> str:
        return "98"

    monkeypatch.setattr("api.analysis_summary.current_release", _next_release)
    second = _post(public, caller_token=_token(private))
    assert _events(second.text)[0][1]["cached"] is False
    assert wired.calls == before + 1


def test_the_two_tiers_do_not_share_a_cached_summary(
    keys: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Serving the aggregate summary to someone who chose to disclose, or the
    # disclosing one to someone who did not, are both failures -- and the
    # second is a disclosure nobody asked for.
    _with_not_found_spy(monkeypatch)
    private, public = keys
    _post(public, caller_token=_token(private), disclosure="aggregate")
    start = _events(
        _post(public, caller_token=_token(private), disclosure="identifiers").text
    )[0][1]
    assert start["cached"] is False, "the disclosing tier reused the aggregate summary"


def test_citations_come_back_with_a_cached_summary(keys: tuple[str, str]) -> None:
    # A reused summary that lost its chips would look like a summary citing
    # nothing, and the caller cannot tell that from a result with no pathways.
    private, public = keys
    first = _post(public, caller_token=_token(private))
    second = _post(public, caller_token=_token(private))

    def cited(response: Any) -> list[str]:
        return [p["st_id"] for k, p in _events(response.text) if k == "citation"]

    assert cited(second) == cited(first)
    assert cited(second), "no citations at all"


def test_a_cached_disclosing_summary_does_not_refetch_the_identifiers(
    keys: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The first version looked the cache up *after* the disclosure fetch, so
    # every reload of an already-summarised analysis went and asked the
    # Analysis Service for the reader's identifiers again and discarded
    # them. Pointless, and on the one endpoint where pointless requests are
    # worth avoiding.
    asked = _with_not_found_spy(monkeypatch)
    private, public = keys
    _post(public, caller_token=_token(private), disclosure="identifiers")
    assert len(asked) == 1
    second = _post(public, caller_token=_token(private), disclosure="identifiers")
    assert _events(second.text)[0][1]["cached"] is True
    assert len(asked) == 1, "a cached summary refetched the user's identifiers"


def test_a_failed_disclosure_does_not_poison_later_requests(
    keys: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A request whose disclosure failed stores an aggregate summary under
    # `aggregate`. The next disclosing request must miss and try again,
    # rather than being served that fallback forever -- which is why the
    # lookup uses the requested tier and the write uses the applied one.
    failing = {"on": True}

    async def _sometimes(_token: str, **_kwargs: Any) -> list[str] | None:
        return None if failing["on"] else ["SMITH_LAB_SECRET_GENE_001"]

    monkeypatch.setattr("api.analysis_summary.fetch_not_found", _sometimes)
    private, public = keys
    first = _post(public, caller_token=_token(private), disclosure="identifiers")
    assert _events(first.text)[0][1]["disclosure"] == "aggregate"

    failing["on"] = False
    second = _post(public, caller_token=_token(private), disclosure="identifiers")
    assert _events(second.text)[0][1]["cached"] is False, "served the fallback"
    assert _events(second.text)[0][1]["disclosure"] == "identifiers"


def test_a_deleted_result_is_reported_even_when_a_summary_is_stored(
    keys: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The result is fetched on every request, before the cache is consulted,
    # and that is what keeps `gone` correct: a release deletes the analysis,
    # and a reader must be told to re-run rather than handed a confident
    # summary of something that no longer exists.
    private, public = keys
    _post(public, caller_token=_token(private))

    async def _gone(_token: str) -> Fetched:
        return Fetched("gone")

    monkeypatch.setattr("api.analysis_summary.fetch_result", _gone)
    assert (
        _events(_post(public, caller_token=_token(private)).text)[-1][1]["state"]
        == "gone"
    )


class _HangingModel(_Counter):
    """Stands in for a model that has stopped answering."""

    async def astream(self, _messages: Any) -> AsyncIterator[Any]:
        self.calls += 1
        yield type("Chunk", (), {"content": "starting"})()
        # Finite, so a regression fails in five seconds rather than hanging
        # the suite for the full timeout.
        await asyncio.sleep(5)
        yield type("Chunk", (), {"content": "never arrives"})()


def test_a_stuck_model_still_ends_the_stream(
    keys: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # T034. FR-008 says a failure must be a terminal state the caller can
    # render, never a broken panel -- which means never an open connection
    # either. An analysis page must not hang because this service did.
    monkeypatch.setattr("api.analysis_summary.get_llm", lambda *a, **k: _HangingModel())
    monkeypatch.setattr("api.analysis_summary.SUMMARY_TIMEOUT_SECONDS", 0.25)
    private, public = keys

    started = time.monotonic()
    response = _post(public, caller_token=_token(private))
    elapsed = time.monotonic() - started

    assert response.status_code == 200
    assert _events(response.text)[-1][1]["state"] == "failed"
    assert elapsed < 2, f"stream ran {elapsed:.1f}s; the bound did not fire"


def test_a_stuck_summary_is_not_stored(
    keys: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The partial text of a timed-out generation must not become the summary
    # served forever after. `put` refuses empty text, but this one is not
    # empty -- it is worse, being a plausible fragment that ends mid-sentence.
    store = SummaryStore()
    monkeypatch.setattr("api.analysis_summary._store", store)
    monkeypatch.setattr("api.analysis_summary.get_llm", lambda *a, **k: _HangingModel())
    monkeypatch.setattr("api.analysis_summary.SUMMARY_TIMEOUT_SECONDS", 0.25)
    private, public = keys
    _post(public, caller_token=_token(private))
    assert len(store) == 0, "a truncated summary was stored"


def test_an_abandoned_summary_stream_is_recorded(
    keys: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # T035. A caller that hangs up mid-summary leaves no other trace: the
    # request 200s, tokens flow, and then nothing more happens -- identical
    # to a healthy stream in every signal. The website already hit this on
    # the answer route, where a keystroke unmounted their panel mid-answer.
    #
    # Driven through the response iterator rather than a client, because the
    # point is to close it mid-stream and a TestClient will not.
    from types import SimpleNamespace

    class _SlowModel(_Counter):
        async def astream(self, _messages: Any) -> AsyncIterator[Any]:
            self.calls += 1
            for index in range(50):
                await asyncio.sleep(0.01)
                yield type("Chunk", (), {"content": f"t{index} "})()

    monkeypatch.setattr("api.analysis_summary.get_llm", lambda *a, **k: _SlowModel())
    private, public = keys
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(caller_token_key=public))
    )

    async def drive() -> None:
        response = await analysis_summary(
            SummaryRequest(
                token=SAMPLE_TOKEN,
                caller_token=_token(private),
                disclosure="aggregate",
            ),
            request,  # type: ignore[arg-type]
        )
        iterator = cast("AsyncGenerator[str, None]", response.body_iterator)
        seen = 0
        async for _chunk in iterator:
            seen += 1
            if seen == 4:
                break
        await iterator.aclose()

    with caplog.at_level("INFO", logger="api.analysis_summary"):
        asyncio.run(drive())

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "abandoned" in message for message in messages
    ), f"no record of the abandoned stream; logged: {messages}"


def test_expression_values_reach_the_model_on_the_served_path(
    keys: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # End to end, because the bug lived in the join between two layers that
    # each had passing tests: the allow-list kept `exp` and `prompt_input`
    # dropped it. Asserted on what the model receives, which is the only
    # place the join is visible.
    monkeypatch.setitem(RESULT["summary"], "type", "EXPRESSION")
    monkeypatch.setitem(RESULT["pathways"][0]["entities"], "exp", [0.7, 0.25, 0.25])
    monkeypatch.setitem(RESULT["pathways"][1]["entities"], "exp", [1.1, -0.4, 2.0])
    sent: list[Any] = []

    class _Recording(_Counter):
        async def astream(self, messages: Any) -> AsyncIterator[Any]:
            sent.append(messages)
            async for chunk in super().astream(messages):
                yield chunk

    recording = _Recording()
    private, public = keys
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("api.analysis_summary.get_llm", lambda *a, **k: recording)
        _post(public, caller_token=_token(private))

    assert sent, "the model was never called, so this proves nothing"
    prompt = json.dumps(sent, default=str)
    assert "0.25" in prompt, "the per-column values never reached the model"
    # The payload is JSON inside JSON, so the inner quotes are escaped.
    # Matching on the unescaped form silently never matches -- which is how
    # this assertion first passed review while testing nothing.
    assert "expression_columns" in prompt
    assert re.search(r'expression_columns\\?":\s*3', prompt), prompt[-200:]


@pytest.mark.parametrize(
    ("omit", "claims", "expected_log"),
    [
        # Absent, which is what the website's hand-built claim effectively
        # was: they sent `subject` and `solvedAt`, so neither agreed claim
        # was there at all.
        (("human",), {}, "no `human` claim present"),
        ((), {"human": "yes"}, "present but not true"),
        (("human_iat",), {"human": True}, "no `human_iat`"),
        ((), {"human": True, "human_iat": "solvedAt"}, "not a number"),
    ],
)
def test_the_specific_presence_failure_is_logged_but_never_returned(
    keys: tuple[str, str],
    caplog: pytest.LogCaptureFixture,
    omit: tuple[str, ...],
    claims: dict[str, Any],
    expected_log: str,
) -> None:
    # The caller gets a coarse `no_human` whatever went wrong, so this side
    # cannot be probed for what a valid claim looks like. The cost of that
    # was paid on 2026-09-21: the website hand-built a claim with their
    # cookie's field names, got `no_human` twice, and nearly concluded our
    # gate was rejecting their valid tokens.
    #
    # So the distinction lives in the log. Both halves are asserted, because
    # either alone is the bug: a coarse log is undiagnosable and a detailed
    # response is a probe.
    private, public = keys
    with caplog.at_level("INFO", logger="api.analysis_summary"):
        response = _post(public, caller_token=_token(private, omit=omit, **claims))

    payload = _events(response.text)[-1][1]
    assert payload["reason"] == "no_human"
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert expected_log in logged, f"not diagnosable from the log: {logged}"
    assert expected_log not in response.text, "the detail reached the caller"


def test_release_is_a_number_as_the_answer_endpoint_sends_it(
    keys: tuple[str, str],
) -> None:
    # The Analysis Service answers `/database/version` as text, so this
    # arrived as a string while `/api/answer` sends an int -- the same field
    # in the same event shape with a different type on each endpoint. A
    # consumer required a number, got null, and did not notice, because null
    # is legitimate here.
    private, public = keys
    start = _events(_post(public, caller_token=_token(private)).text)[0][1]
    assert start["release"] == 97
    assert isinstance(start["release"], int)


def test_a_non_numeric_release_is_null_rather_than_a_string(
    keys: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Null is already a legitimate value for this field, so degrading to it
    # keeps the type honest. Emitting the raw string would put a second type
    # back on the wire for the case nobody tests.
    async def _odd() -> str:
        return "97-beta"

    monkeypatch.setattr("api.analysis_summary.current_release", _odd)
    private, public = keys
    start = _events(_post(public, caller_token=_token(private)).text)[0][1]
    assert start["release"] is None
