"""`POST /api/handoff`: mint a Continue-in-chat handoff (spec 013).

The endpoint's one real decision is what it refuses. It mints a handoff only
for a summary that already exists at the requested tier, which is how it
enforces both "continue what the reader saw" (FR-002) and "never wider than
they agreed to" (FR-003) with a single lookup.
"""

import time

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import FastAPI
from fastapi.testclient import TestClient

from analysis.store import SummaryStore
from api import handoff as endpoint
from handoff.store import AnalysisHandoff, HandoffStore, SearchHandoff
from util.caller_token import DEFAULT_AUDIENCE
from util.rate_limit import SlidingWindowLimiter

PREFIX = "/chat/guest/api"
ANALYSIS = "MjAyNjA5MjExOTMyNTlfNjE%3D"


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
def stores(monkeypatch: pytest.MonkeyPatch) -> tuple[SummaryStore, HandoffStore]:
    # Private stores per test: both are module state that outlives a request.
    summaries = SummaryStore()
    handoffs = HandoffStore()
    monkeypatch.setattr("api.analysis_summary._store", summaries)
    monkeypatch.setattr("api.handoff.handoffs", handoffs)
    monkeypatch.setattr(
        "api.handoff._limiter", SlidingWindowLimiter(limit=10_000, window=600.0)
    )

    async def _release() -> str:
        return "97"

    monkeypatch.setattr("api.handoff.current_release", _release)
    monkeypatch.setenv("CHAINLIT_URI", "/chat/guest")
    return summaries, handoffs


def client(public_pem: str) -> TestClient:
    app = FastAPI()
    app.include_router(endpoint.router, prefix=PREFIX)
    app.state.caller_token_key = public_pem
    return TestClient(app)


def caller(private_pem: str, **claims: object) -> str:
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


def mint(keys: tuple[str, str], disclosure: str = "aggregate", **claims: object):  # type: ignore[no-untyped-def]
    private, public = keys
    return client(public).post(
        f"{PREFIX}/handoff",
        json={
            "kind": "analysis",
            "token": ANALYSIS,
            "disclosure": disclosure,
            "caller_token": caller(private, **claims),
        },
    )


def test_mints_a_handoff_for_a_summary_the_reader_saw(
    keys: tuple[str, str], stores: tuple[SummaryStore, HandoffStore]
) -> None:
    summaries, handoffs = stores
    summaries.put(ANALYSIS, "97", "aggregate", "Four pathways pass correction.", ())

    response = mint(keys)

    assert response.status_code == 200
    body = response.json()
    assert body["path"] == f"/chat/guest/#handoff={body['id']}"
    stored = handoffs.get(body["id"])
    assert isinstance(stored, AnalysisHandoff)
    # A copy of the text the reader saw, not a reference to regenerate from.
    assert stored.summary == "Four pathways pass correction."
    assert stored.tier == "aggregate"


def test_the_id_travels_in_the_fragment_never_the_query(
    keys: tuple[str, str], stores: tuple[SummaryStore, HandoffStore]
) -> None:
    # A fragment is never sent to a server, so the ID cannot reach nginx
    # logs or a Referer. A query string would reach both.
    stores[0].put(ANALYSIS, "97", "aggregate", "Summary.", ())
    path = mint(keys).json()["path"]
    assert "#handoff=" in path
    assert "?" not in path


def test_the_path_is_this_processs_own_chat(
    keys: tuple[str, str],
    stores: tuple[SummaryStore, HandoffStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The store is in memory, and guest and logged-in chats are separate
    processes, so a handoff can only be claimed by the chat that minted it.
    The first version also returned a `personal_path` from the guest
    deployment -- a link that could only ever open on "couldn't load"."""
    stores[0].put(ANALYSIS, "97", "aggregate", "Summary.", ())
    monkeypatch.setenv("CHAINLIT_URI", "/chat/personal")
    body = mint(keys).json()
    assert body["path"].startswith("/chat/personal/#handoff=")
    assert "personal_path" not in body
    assert "guest_path" not in body


def test_refuses_when_no_summary_was_generated(keys: tuple[str, str]) -> None:
    # Nothing the reader has seen, so nothing to continue -- and generating
    # one here would break FR-002.
    response = mint(keys)
    assert response.status_code == 404
    assert response.json() == {"reason": "no_summary"}


def test_cannot_widen_the_disclosure_tier(
    keys: tuple[str, str], stores: tuple[SummaryStore, HandoffStore]
) -> None:
    """FR-003. The reader saw only the aggregate summary, so a request to
    continue at the identifiers tier must fail: no such summary exists."""
    stores[0].put(ANALYSIS, "97", "aggregate", "Aggregate summary.", ())
    response = mint(keys, disclosure="identifiers")
    assert response.status_code == 404
    assert stores[1].get("anything") is None


def test_a_summary_from_another_release_is_not_handed_off(
    keys: tuple[str, str], stores: tuple[SummaryStore, HandoffStore]
) -> None:
    # The Analysis Service deletes results on a new release; a summary of one
    # describes an analysis that no longer exists.
    stores[0].put(ANALYSIS, "96", "aggregate", "Old release.", ())
    assert mint(keys).status_code == 404


def test_refuses_without_evidence_of_a_person(
    keys: tuple[str, str], stores: tuple[SummaryStore, HandoffStore]
) -> None:
    # Same bar as the summary endpoint: what this releases is a reader's
    # analysis summary.
    stores[0].put(ANALYSIS, "97", "aggregate", "Summary.", ())
    response = mint(keys, human=False)
    assert response.status_code == 403


def test_refuses_a_token_signed_by_someone_else(
    keys: tuple[str, str], stores: tuple[SummaryStore, HandoffStore]
) -> None:
    stores[0].put(ANALYSIS, "97", "aggregate", "Summary.", ())
    stranger = (
        Ed25519PrivateKey.generate()
        .private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        .decode()
    )
    response = client(keys[1]).post(
        f"{PREFIX}/handoff",
        json={
            "kind": "analysis",
            "token": ANALYSIS,
            "disclosure": "aggregate",
            "caller_token": caller(stranger),
        },
    )
    assert response.status_code == 403
    assert response.json() == {"reason": "no_caller"}


def test_refuses_an_unknown_kind(keys: tuple[str, str]) -> None:
    private, public = keys
    response = client(public).post(
        f"{PREFIX}/handoff",
        json={
            "kind": "search",
            "token": ANALYSIS,
            "disclosure": "aggregate",
            "caller_token": caller(private),
        },
    )
    assert response.status_code == 422


class TestSearchHandoffs:
    """Story 2: continue a search-page answer."""

    @pytest.fixture(autouse=True)
    def _answers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from api.answer_store import AnswerStore

        self.answers = AnswerStore()
        monkeypatch.setattr("api.handoff.answers", self.answers)

    def keep(self, text: str = "CDK5 phosphorylates tau.") -> str:
        from api.answer_store import StoredAnswer

        answer_id = self.answers.put(
            StoredAnswer(
                question="what does CDK5 phosphorylate?",
                text=text,
                citations=(("R-HSA-1", "Apoptosis"),),
                release=97,
                created_at=time.time(),
            )
        )
        assert answer_id is not None
        return answer_id

    def mint(self, keys: tuple[str, str], answer_id: str, **claims: object):  # type: ignore[no-untyped-def]
        private, public = keys
        return client(public).post(
            f"{PREFIX}/handoff",
            json={
                "kind": "search",
                "answer_id": answer_id,
                "caller_token": caller(private, **claims),
            },
        )

    def test_mints_for_an_answer_the_reader_saw(
        self, keys: tuple[str, str], stores: tuple[SummaryStore, HandoffStore]
    ) -> None:
        response = self.mint(keys, self.keep())
        assert response.status_code == 200
        body = response.json()
        assert body["path"] == f"/chat/guest/#handoff={body['id']}"
        handoff = stores[1].get(body["id"])
        assert isinstance(handoff, SearchHandoff)
        # A copy of what the page showed, question included.
        assert handoff.summary == "CDK5 phosphorylates tau."
        assert handoff.question == "what does CDK5 phosphorylate?"

    def test_does_not_require_proof_a_person_is_present(
        self, keys: tuple[str, str]
    ) -> None:
        # Same bar as /api/answer, which produced it: public pathway text,
        # and the search page may have no presence claim to send. The
        # analysis handoff does require it; that asymmetry is deliberate.
        response = self.mint(keys, self.keep(), human=False)
        assert response.status_code == 200

    def test_an_analysis_handoff_still_does(
        self, keys: tuple[str, str], stores: tuple[SummaryStore, HandoffStore]
    ) -> None:
        # The control for the test above: the relaxation must be search-only.
        stores[0].put(ANALYSIS, "97", "aggregate", "Summary.", ())
        assert mint(keys, human=False).status_code == 403

    def test_refuses_an_unknown_answer(self, keys: tuple[str, str]) -> None:
        response = self.mint(keys, "never-issued-0000000000")
        assert response.status_code == 404
        assert response.json() == {"reason": "no_answer"}

    def test_still_requires_a_valid_caller(self, keys: tuple[str, str]) -> None:
        answer_id = self.keep()
        response = client(keys[1]).post(
            f"{PREFIX}/handoff",
            json={
                "kind": "search",
                "answer_id": answer_id,
                "caller_token": "not-a-token",
            },
        )
        assert response.status_code == 403

    def test_a_search_request_cannot_smuggle_an_analysis_field(
        self, keys: tuple[str, str]
    ) -> None:
        # Discriminated by `kind`: a search request carrying `disclosure` or
        # `token` is not quietly treated as an analysis one.
        private, public = keys
        response = client(public).post(
            f"{PREFIX}/handoff",
            json={
                "kind": "search",
                "token": ANALYSIS,
                "disclosure": "identifiers",
                "caller_token": caller(private),
            },
        )
        assert response.status_code == 422
