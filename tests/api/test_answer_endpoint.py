"""The answer endpoint, driven over HTTP.

Principle I: the served path is what matters, and mounting order, the router
prefix and the captcha middleware only interact there. Calling the handler
directly would pass while the route was unreachable behind a redirect.

The graph is replaced with a stand-in. Building a real one takes about 52
seconds, and none of what is asserted here depends on the answer's content --
only on refusal happening before any model call, on the stream's shape, and on
a failure ending with a terminal event rather than a hang.
"""

import time
from collections.abc import AsyncIterator
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.graph import AnswerEvent
from api.answer import router

PREFIX = "/chat/guest/api"


class _StubGraph:
    """Counts calls, so "no model call on refusal" is asserted, not assumed."""

    def __init__(self, events: list[AnswerEvent] | None = None) -> None:
        self.calls = 0
        self._events = events or [
            AnswerEvent(kind="citation", st_id="R-HSA-1", display_name="Apoptosis"),
            AnswerEvent(kind="token", text="CDK5 "),
            AnswerEvent(kind="token", text="phosphorylates tau."),
            AnswerEvent(kind="done", state="answered"),
        ]

    async def astream_answer(self, *_a: Any, **_k: Any) -> AsyncIterator[AnswerEvent]:
        self.calls += 1
        for event in self._events:
            yield event


class _ExplodingGraph(_StubGraph):
    async def astream_answer(self, *_a: Any, **_k: Any) -> AsyncIterator[AnswerEvent]:
        self.calls += 1
        yield AnswerEvent(kind="token", text="partial ")
        raise RuntimeError("upstream died mid-answer")


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


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> _StubGraph:
    graph = _StubGraph()
    monkeypatch.setattr("api.answer.get_graph", lambda: graph)
    return graph


def _client(public_pem: str) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix=PREFIX)
    app.state.human_token_key = public_pem
    return TestClient(app)


def _token(private_pem: str, seconds: int = 300) -> str:
    return jwt.encode(
        {"exp": int(time.time()) + seconds}, private_pem, algorithm="EdDSA"
    )


def _events(text: str) -> list[tuple[str, str]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in lines:
            out.append((lines["event"], lines.get("data", "")))
    return out


def test_a_verified_request_streams_tokens_and_citations(
    keys: tuple[str, str], stub: _StubGraph
) -> None:
    private_pem, public_pem = keys
    response = _client(public_pem).post(
        f"{PREFIX}/answer",
        json={
            "question": "what does CDK5 phosphorylate?",
            "human_token": _token(private_pem),
        },
    )
    assert response.status_code == 200
    kinds = [kind for kind, _ in _events(response.text)]
    assert kinds[0] == "start"
    assert kinds[-1] == "done"
    assert kinds.count("token") == 2, "tokens must stream, not arrive as one blob"
    assert "citation" in kinds
    assert '"state": "answered"' in response.text


def test_no_token_means_no_model_call(keys: tuple[str, str], stub: _StubGraph) -> None:
    """Search pages are crawled; every crawled search reaching the model is a bill."""
    _, public_pem = keys
    response = _client(public_pem).post(
        f"{PREFIX}/answer", json={"question": "anything", "human_token": ""}
    )
    assert response.status_code == 200
    assert _events(response.text) == [("done", '{"state": "refused"}')]
    assert stub.calls == 0, "the graph must not be touched for a refused request"


def test_an_expired_token_makes_no_model_call(
    keys: tuple[str, str], stub: _StubGraph
) -> None:
    private_pem, public_pem = keys
    response = _client(public_pem).post(
        f"{PREFIX}/answer",
        json={"question": "anything", "human_token": _token(private_pem, seconds=-10)},
    )
    assert '"state": "refused"' in response.text
    assert stub.calls == 0


def test_a_failure_mid_stream_still_ends_with_done(
    keys: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """FR-006: the caller must be able to stop waiting, whatever went wrong."""
    private_pem, public_pem = keys
    monkeypatch.setattr("api.answer.get_graph", lambda: _ExplodingGraph())
    response = _client(public_pem).post(
        f"{PREFIX}/answer",
        json={"question": "boom", "human_token": _token(private_pem)},
    )
    assert response.status_code == 200
    kinds = [kind for kind, _ in _events(response.text)]
    assert kinds[-1] == "done"
    assert '"state": "failed"' in response.text


def test_an_empty_question_is_rejected_by_validation(
    keys: tuple[str, str], stub: _StubGraph
) -> None:
    private_pem, public_pem = keys
    response = _client(public_pem).post(
        f"{PREFIX}/answer", json={"question": "", "human_token": _token(private_pem)}
    )
    assert response.status_code == 422
    assert stub.calls == 0
