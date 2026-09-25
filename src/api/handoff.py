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
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from analysis.client import current_release
from analysis.disclosure import Tier
from api.analysis_summary import stored_summary
from handoff.store import DEFAULT_TTL_SECONDS, Handoff, handoffs
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


class HandoffRequest(BaseModel):
    kind: str = Field(pattern="^analysis$")
    token: str = Field(min_length=1, max_length=256)
    #: The tier of the summary the reader was shown. Required, no default.
    disclosure: Tier
    caller_token: str = ""


def _refuse(status: int, reason: str, log: str) -> JSONResponse:
    logger.info("handoff refused", extra={"reason": reason, "detail": log})
    return JSONResponse(status_code=status, content={"reason": reason})


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

    presence = human_presence_reason(claims, time.time())
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

    release = await current_release()
    if not release:
        # Summaries are cached per release, so without knowing the release
        # there is no way to find the one the reader saw. Refusing is the
        # honest answer; guessing a release could hand off a summary of a
        # result the Analysis Service has since deleted.
        return _refuse(503, "no_release", "current release unknown")
    stored = stored_summary(body.token, release, body.disclosure)
    if stored is None:
        # Not generated, evicted, from a previous release, or requested at a
        # tier the reader never chose. In every case there is nothing the
        # reader has seen to continue, and generating one here would break
        # both FR-002 and FR-003.
        return _refuse(
            404, "no_summary", f"no stored summary at tier {body.disclosure}"
        )

    handoff_id = handoffs.put(
        Handoff(
            kind="analysis",
            token=body.token,
            release=release,
            tier=body.disclosure,
            summary=stored.text,
            citations=stored.citations,
            created_at=time.time(),
        )
    )
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
