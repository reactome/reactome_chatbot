"""The answer endpoint the website's search page calls.

Implements specs/010-search-page-answers/contracts/answer_endpoint.md. Server-sent
events, because the answer takes tens of seconds and a search page cannot wait for a
whole one.

Two properties matter more than the wire format, and both come from the contract:

**Refuse before any model call.** Search pages are crawled, and every crawled search
reaching the model is an unbounded bill. Verification happens before the graph is
touched.

**Fail invisibly.** Any error, timeout or refusal ends with a `done` event carrying a
non-`answered` state, so the caller renders no panel. A search page must not break
because this service did. That is not in tension with failing loudly: a *runtime*
failure answering one question is quiet, while *misconfiguration* -- a missing
verifying key -- stops the process at startup.
"""

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.registry import get_graph
from util.human_token import TokenRejectedError, verify
from util.logging import logging

logger = logging.getLogger(__name__)

router = APIRouter()

PROFILE = "react-to-me"


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    human_token: str = ""


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def _refusal(reason: str) -> StreamingResponse:
    """A refusal is a well-formed stream, not an HTTP error.

    The caller renders "no panel" from the done state. Returning 401 with a body
    would make every refusal a special case in the website's error handling, and
    the contract promises one shape.
    """

    async def body() -> AsyncIterator[str]:
        yield _sse("done", {"state": "refused"})

    logger.info("answer request refused: %s", reason)
    return StreamingResponse(body(), media_type="text/event-stream")


@router.post("/answer")
async def answer(request: Request, body: AnswerRequest) -> StreamingResponse:
    verifying_key = getattr(request.app.state, "human_token_key", None)
    if not verifying_key:
        # Should be unreachable: startup refuses without a key. If it happens,
        # refuse rather than answer.
        return _refusal("no verifying key on the app")

    try:
        verify(body.human_token, verifying_key)
    except TokenRejectedError as rejected:
        return _refusal(rejected.reason)

    graph = get_graph()

    async def stream() -> AsyncIterator[str]:
        yield _sse("start", {"answered": True})
        state = "failed"
        try:
            async for event in graph.astream_answer(
                body.question, PROFILE, thread_id=f"search-{id(body)}"
            ):
                if event.kind == "token":
                    yield _sse("token", {"text": event.text})
                elif event.kind == "citation":
                    yield _sse(
                        "citation",
                        {"st_id": event.st_id, "display_name": event.display_name},
                    )
                elif event.kind == "done":
                    state = event.state or "failed"
        except Exception:
            # Deliberately broad, and deliberately not re-raised: a half-written
            # SSE stream cannot become an HTTP error code, and the caller needs a
            # terminal event to stop waiting.
            logger.exception("answering %r failed", body.question[:80])
            state = "failed"
        yield _sse("done", {"state": state})

    return StreamingResponse(stream(), media_type="text/event-stream")
