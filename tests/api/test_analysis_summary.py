"""The analysis-summary endpoint, driven over HTTP.

Principle I: mounting order, the router prefix and the captcha exemption
interact only on the served path. Spec 010's route check found a problem there
that isolated tests could not.

Neither the Analysis Service nor a model is called: the client is replaced and
so is the LLM, because what is asserted here is refusal, event shape, and that
no model call happens without a person -- none of which depends on content.
"""

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import FastAPI
from fastapi.testclient import TestClient

from analysis.client import Fetched
from api.analysis_summary import router
from util.caller_token import DEFAULT_AUDIENCE
from util.rate_limit import SlidingWindowLimiter

PREFIX = "/chat/guest/api"

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


def _token(private_pem: str, **claims: object) -> str:
    payload: dict[str, object] = {
        "iss": "reactome-website",
        "aud": DEFAULT_AUDIENCE,
        "exp": int(time.time()) + 300,
        "sub": "visit-1",
        "human": True,
        "human_iat": int(time.time()) - 10,
    }
    payload.update(claims)
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
    assert start["release"] == "97"
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


def test_the_unbuilt_identifier_tier_is_refused_not_quietly_downgraded(
    keys: tuple[str, str], wired: _Counter
) -> None:
    # `identifiers` is agreed and not built. Serving an aggregate summary for
    # it would disclose nothing extra and still be wrong: the reader chose the
    # disclosing option and would get the other, with nothing saying so. A
    # choice quietly overridden is worse than one refused, because it looks
    # like it was honoured.
    private, public = keys
    payload = _events(
        _post(public, caller_token=_token(private), disclosure="identifiers").text
    )[-1][1]
    assert payload["state"] == "refused"
    assert payload["reason"] == "unsupported_tier"
    assert wired.calls == 0


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
