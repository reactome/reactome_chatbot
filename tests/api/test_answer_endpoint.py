"""The answer endpoint, driven over HTTP.

Principle I: the served path is what matters, and mounting order, the router
prefix and the captcha middleware only interact there. Calling the handler
directly would pass while the route was unreachable behind a redirect.

The graph is replaced with a stand-in. Building a real one takes about 52
seconds, and none of what is asserted here depends on the answer's content --
only on refusal happening before any model call, on the stream's shape, and on
a failure ending with a terminal event rather than a hang.
"""

import asyncio
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

from agent.graph import AnswerEvent
from api.answer import router
from util.caller_token import DEFAULT_AUDIENCE
from util.rate_limit import SlidingWindowLimiter

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


@pytest.fixture(autouse=True)
def _fresh_limiter(monkeypatch: pytest.MonkeyPatch) -> None:
    """A private limiter per test.

    `_limiter` is module state shared by the whole process, so without this the
    fifty requests below would exhaust the real limit and refuse later tests --
    and which tests failed would depend on the order they ran in.
    """
    monkeypatch.setattr(
        "api.answer._limiter", SlidingWindowLimiter(limit=10_000, window=600.0)
    )


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> _StubGraph:
    graph = _StubGraph()
    monkeypatch.setattr("api.answer.get_graph", lambda: graph)
    return graph


def _client(public_pem: str) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix=PREFIX)
    app.state.caller_token_key = public_pem
    return TestClient(app)


def _token(private_pem: str, seconds: int = 300, **claims: object) -> str:
    """A token shaped like the website's: iss, aud, exp, and a per-visit sub."""
    payload: dict[str, object] = {
        "iss": "reactome-website",
        "aud": DEFAULT_AUDIENCE,
        "exp": int(time.time()) + seconds,
        "sub": "visit-1",
    }
    payload.update(claims)
    return jwt.encode(payload, private_pem, algorithm="EdDSA")


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
            "caller_token": _token(private_pem),
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
        f"{PREFIX}/answer", json={"question": "anything", "caller_token": ""}
    )
    assert response.status_code == 200
    events = _events(response.text)
    assert [name for name, _ in events] == ["done"]
    assert json.loads(events[0][1])["state"] == "refused"
    assert stub.calls == 0, "the graph must not be touched for a refused request"


def test_an_expired_token_makes_no_model_call(
    keys: tuple[str, str], stub: _StubGraph
) -> None:
    private_pem, public_pem = keys
    response = _client(public_pem).post(
        f"{PREFIX}/answer",
        json={"question": "anything", "caller_token": _token(private_pem, seconds=-10)},
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
        json={"question": "boom", "caller_token": _token(private_pem)},
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
        f"{PREFIX}/answer", json={"question": "", "caller_token": _token(private_pem)}
    )
    assert response.status_code == 422
    assert stub.calls == 0


class _ThreadRecordingGraph(_StubGraph):
    """Records the thread_id each request runs under."""

    def __init__(self) -> None:
        super().__init__()
        self.thread_ids: list[str] = []

    async def astream_answer(
        self, *_a: Any, **kwargs: Any
    ) -> AsyncIterator[AnswerEvent]:
        self.thread_ids.append(kwargs["thread_id"])
        for event in self._events:
            yield event


def test_separate_requests_do_not_share_a_thread(
    keys: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two askers must not land on one checkpointer thread.

    `chat_history` is checkpointed state annotated with `add_messages`, and the
    rephraser injects it via a MessagesPlaceholder. Sharing a thread_id means one
    stranger's question and answer become the context that rephrases the next
    stranger's question.

    Why 50 requests and not 2: the original bug keyed the thread on `id(body)`,
    and two sequential requests usually get distinct addresses, so a two-request
    version of this test passed against the broken code. At 50 it fails every
    time -- measured over 200 requests, the broken version produced 38 distinct
    threads and put 192 of them on a shared one.
    """
    private, public = keys
    graph = _ThreadRecordingGraph()
    monkeypatch.setattr("api.answer.get_graph", lambda: graph)
    client = _client(public)

    requests = 50
    for index in range(requests):
        response = client.post(
            f"{PREFIX}/answer",
            json={"question": f"question {index}", "caller_token": _token(private)},
        )
        assert response.status_code == 200

    assert len(graph.thread_ids) == requests
    assert (
        len(set(graph.thread_ids)) == requests
    ), f"{requests} requests shared {requests - len(set(graph.thread_ids))} threads"


def test_start_carries_the_release_and_done_carries_seconds(
    keys: tuple[str, str], stub: _StubGraph
) -> None:
    """The contract's fields, pinned.

    Both were specified in contracts/answer_endpoint.md and both were missing
    from the first implementation. The website codes against that document, so a
    field it promises and we never send is a defect on their side, not ours.
    `release` is what makes FR-007 cache invalidation possible at all.
    """
    private, public = keys
    app = FastAPI()
    app.include_router(router, prefix=PREFIX)
    app.state.caller_token_key = public
    app.state.release = 97
    client = TestClient(app)

    response = client.post(
        f"{PREFIX}/answer",
        json={"question": "what is CDK5", "caller_token": _token(private)},
    )
    events = dict(_events(response.text))

    assert json.loads(events["start"])["release"] == 97
    done = json.loads(events["done"])
    assert done["state"] == "answered"
    assert isinstance(done["seconds"], float)


def test_a_refusal_has_the_same_done_shape_as_an_answer(
    keys: tuple[str, str], stub: _StubGraph
) -> None:
    """One shape, so the caller parses `done` one way."""
    _, public = keys
    client = _client(public)

    response = client.post(f"{PREFIX}/answer", json={"question": "what is CDK5"})
    done = json.loads(dict(_events(response.text))["done"])

    assert done["state"] == "refused"
    assert set(done) == {"state", "seconds"}
    assert stub.calls == 0


class _HangingGraph(_StubGraph):
    """Never yields its second event, like an upstream that stopped responding."""

    async def astream_answer(self, *_a: Any, **_k: Any) -> AsyncIterator[AnswerEvent]:
        self.calls += 1
        yield AnswerEvent(kind="token", text="partial ")
        await asyncio.sleep(
            5
        )  # finite, so a regression fails fast instead of hanging CI


def test_a_hanging_upstream_still_ends_the_stream(
    keys: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """FR-006 names timeout alongside error, and nothing implemented it.

    The only bound was the LLM client's 360s request_timeout, once per model call
    and six calls per answer, so a stuck upstream could hold the connection for
    over half an hour and never send `done`. The search page must not depend on
    this service being up.

    The timeout is patched to 0.25s so the test is quick, and the stand-in hangs
    for a finite 5s so that a regression fails in five seconds rather than hanging
    the suite for an hour.
    """
    private, public = keys
    graph = _HangingGraph()
    monkeypatch.setattr("api.answer.get_graph", lambda: graph)
    monkeypatch.setattr("api.answer.ANSWER_TIMEOUT_SECONDS", 0.25)
    client = _client(public)

    started = time.monotonic()
    response = client.post(
        f"{PREFIX}/answer",
        json={"question": "what is CDK5", "caller_token": _token(private)},
    )
    elapsed = time.monotonic() - started

    events = dict(_events(response.text))
    assert json.loads(events["done"])["state"] == "failed"
    assert elapsed < 2, f"stream ran {elapsed:.1f}s; the timeout did not fire"


def test_a_caller_over_the_limit_is_refused_before_the_model(
    keys: tuple[str, str], stub: _StubGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FR-008. A backstop: the website enforces the real budget upstream.

    Refused like any other refusal -- one `done` shape, no HTTP error -- so the
    page renders no panel rather than a broken one.
    """
    private, public = keys
    monkeypatch.setattr(
        "api.answer._limiter", SlidingWindowLimiter(limit=2, window=600.0)
    )
    client = _client(public)
    token = _token(private)

    states = []
    for _ in range(4):
        response = client.post(
            f"{PREFIX}/answer", json={"question": "what is CDK5", "caller_token": token}
        )
        assert response.status_code == 200
        states.append(json.loads(dict(_events(response.text))["done"])["state"])

    assert states == ["answered", "answered", "refused", "refused"]
    assert stub.calls == 2, "a refused request still reached the model"


def test_the_limit_is_per_caller(
    keys: tuple[str, str], stub: _StubGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One visitor exhausting their budget must not silence the page for others."""
    private, public = keys
    monkeypatch.setattr(
        "api.answer._limiter", SlidingWindowLimiter(limit=1, window=600.0)
    )
    client = _client(public)

    # Two visits, not two tokens. The limiter keys on `sub` -- an opaque
    # per-visit id from the website -- so a second token for the SAME visit is
    # the same caller, which is the point of keying on it rather than on the
    # token. An earlier version of this test used two tokens differing only in
    # expiry and expected them to count separately; once `sub` arrived that
    # became wrong.
    first = _token(private, sub="visit-1")
    second = _token(private, sub="visit-2")

    def ask(token: str) -> str:
        response = client.post(
            f"{PREFIX}/answer", json={"question": "what is CDK5", "caller_token": token}
        )
        return str(json.loads(dict(_events(response.text))["done"])["state"])

    assert ask(first) == "answered"
    assert ask(first) == "refused"
    assert (
        ask(second) == "answered"
    ), "a different visit was refused someone else's budget"

    # A newly minted token for the first visit must not buy a fresh allowance --
    # they mint one per request, so keying on the token would make the limit
    # meaningless.
    assert ask(_token(private, sub="visit-1", seconds=301)) == "refused"


def test_the_search_page_does_not_pay_for_a_web_search(
    keys: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The postprocess node runs a Tavily search whose result this path drops.

    `astream_answer` has no event carrying `additional_content`, so with the
    default the endpoint paid for a web search, discarded it, and delayed `done`
    by its duration. The chat UI renders those results; the search page has no
    place for them.
    """
    private, public = keys
    seen: dict[str, object] = {}

    class _RecordingGraph(_StubGraph):
        async def astream_answer(
            self, *_a: Any, **kwargs: Any
        ) -> AsyncIterator[AnswerEvent]:
            seen.update(kwargs)
            for event in self._events:
                yield event

    monkeypatch.setattr("api.answer.get_graph", lambda: _RecordingGraph())
    client = _client(public)
    client.post(
        f"{PREFIX}/answer",
        json={"question": "what is CDK5", "caller_token": _token(private)},
    )

    assert seen["enable_postprocess"] is False
