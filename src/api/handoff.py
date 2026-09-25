"""Mint a "Continue in chat" handoff for a summary the reader has seen.

Spec 013. The website calls this when the reader clicks "Continue in chat"
beside an analysis summary, then opens `/chat/guest/#handoff=<id>` (or the
personal path) in a new tab. The ID travels in the URL fragment, which
browsers never send to a server.

**Only a summary that already exists can be handed off.** The summary is
looked up in the analysis-summary cache under the tier the reader chose. If
it is not there, this refuses rather than generating one. That enforces two
requirements with one check:

- FR-002, continue what the reader *saw*: a summary that was never
  generated was never seen.
- FR-003, never wider than they agreed to: a handoff at the `identifiers`
  tier exists only if an `identifiers` summary does, which exists only if
  the reader chose it on the website.

Same authorisation bar as the summary endpoint -- caller token plus
evidence that a person is present -- because what it releases is that
reader's analysis summary.

Plain JSON, not SSE: minting is a lookup, not a generation.
"""

import os
import time
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from analysis.client import current_release
from analysis.disclosure import Tier
from api.analysis_summary import stored_summary
from api.answer_store import StoredAnswer, answers
from handoff.store import (
    DEFAULT_TTL_SECONDS,
    AnalysisHandoff,
    Handoff,
    SearchHandoff,
    handoffs,
)
from util.caller_token import (
    TokenRejectedError,
    human_presence_detail,
    human_presence_reason,
    verify,
)
from util.logging import logging
from util.rate_limit import identity_of, limiter_from_env

logger = logging.getLogger(__name__)

router = APIRouter()

_limiter = limiter_from_env()


class AnalysisHandoffRequest(BaseModel):
    kind: Literal["analysis"]
    token: str = Field(min_length=1, max_length=256)
    #: The tier of the summary the reader was shown. Required, no default.
    disclosure: Tier
    caller_token: str = ""


class SearchHandoffRequest(BaseModel):
    kind: Literal["search"]
    #: From the `done` event of the `/api/answer` stream the page rendered.
    answer_id: str = Field(min_length=1, max_length=128)
    caller_token: str = ""


HandoffRequest = Annotated[
    AnalysisHandoffRequest | SearchHandoffRequest, Field(discriminator="kind")
]


def _refuse(status: int, reason: str, log: str) -> JSONResponse:
    logger.info("handoff refused", extra={"reason": reason, "detail": log})
    return JSONResponse(status_code=status, content={"reason": reason})


def stored_answer(answer_id: str) -> StoredAnswer | None:
    """An answer `/api/answer` showed a reader, if it is still kept."""
    return answers.get(answer_id)


def own_chat_path() -> str:
    """Where this process's chat is mounted, e.g. `/chat/guest`."""
    return (os.getenv("CHAINLIT_URI") or "/chat").rstrip("/")


@router.post("/handoff")
async def create_handoff(body: HandoffRequest, request: Request) -> JSONResponse:
    verifying_key = getattr(request.app.state, "caller_token_key", None)
    if not verifying_key:
        return _refuse(403, "no_caller", "no verifying key on the app")
    try:
        claims = verify(body.caller_token, verifying_key)
    except TokenRejectedError as rejected:
        return _refuse(403, "no_caller", rejected.reason)

    # Human presence for an analysis only. It releases a reader's own
    # analysis summary; a search answer is public pathway text, and
    # `/api/answer` -- which produced it -- deliberately does not require
    # presence either, so the search page may not have the claim to send.
    presence = (
        human_presence_reason(claims, time.time()) if body.kind == "analysis" else None
    )
    if presence:
        detail = (
            human_presence_detail(claims, time.time())
            if presence == "no_human"
            else presence
        )
        return _refuse(403, presence, detail)

    human_sub = claims.get("human_sub")
    key = f"human:{human_sub}" if isinstance(human_sub, str) and human_sub else None
    if not _limiter.allow(key or identity_of(claims, body.caller_token)):
        return _refuse(429, "rate_limited", "rate limited")

    handoff: Handoff
    if isinstance(body, SearchHandoffRequest):
        answer = stored_answer(body.answer_id)
        if answer is None:
            # Unknown, expired, or never answered: nothing the reader saw.
            return _refuse(404, "no_answer", "no stored answer for that id")
        handoff = SearchHandoff(
            kind="search",
            question=answer.question,
            summary=answer.text,
            citations=answer.citations,
            created_at=time.time(),
        )
    else:
        release = await current_release()
        if not release:
            # Summaries are cached per release, so without knowing the release
            # there is no way to find the one the reader saw. Refusing is the
            # honest answer; guessing a release could hand off a summary of a
            # result the Analysis Service has since deleted.
            return _refuse(503, "no_release", "current release unknown")
        stored = stored_summary(body.token, release, body.disclosure)
        if stored is None:
            # Not generated, evicted, from a previous release, or requested at
            # a tier the reader never chose. In every case there is nothing the
            # reader has seen to continue, and generating one here would break
            # both FR-002 and FR-003.
            return _refuse(
                404, "no_summary", f"no stored summary at tier {body.disclosure}"
            )
        handoff = AnalysisHandoff(
            kind="analysis",
            token=body.token,
            release=release,
            tier=body.disclosure,
            summary=stored.text,
            citations=stored.citations,
            created_at=time.time(),
        )

    handoff_id = handoffs.put(handoff)
    payload: dict[str, Any] = {
        "id": handoff_id,
        "expires_in": int(DEFAULT_TTL_SECONDS),
        # The chat served by *this* process, and only that one. The store is
        # in memory, and the guest and logged-in chats are separate
        # processes, so a handoff minted here can only be claimed here. The
        # first version also returned a `personal_path` from the guest
        # deployment: a link that would always open on "couldn't load the
        # summary", because the logged-in process had never heard of the ID.
        #
        # A fragment, not a query string: never sent to a server (FR-004).
        "path": f"{own_chat_path()}/#handoff={handoff_id}",
    }
    return JSONResponse(status_code=200, content=payload)
