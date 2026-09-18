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

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.registry import get_graph
from util.anchor_strip import AnchorStripper
from util.caller_token import TokenRejectedError, verify
from util.logging import logging
from util.rate_limit import identity_of, limiter_from_env

logger = logging.getLogger(__name__)

router = APIRouter()

PROFILE = "react-to-me"

# One limiter for the process, built at import so the window is not reset by a
# request. FR-008: a backstop behind the website's own budget.
_limiter = limiter_from_env()

# FR-006 names timeout alongside error, and nothing here implemented it. The only
# bound was the LLM client's `request_timeout=360.0` -- six minutes per model call,
# and six calls run around one answer, so a pathological request could hold a
# connection for over half an hour and never send `done`.
#
# 120s is about twice the worst complete answer measured (max 31.5s, with the first
# answer token near 36s on a heavy question), so it does not cut off answers that
# were going to arrive; it converts an unbounded hang into a bounded one.
ANSWER_TIMEOUT_SECONDS = 120.0


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    caller_token: str = ""


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def _refusal(reason: str) -> StreamingResponse:
    """A refusal is a well-formed stream, not an HTTP error.

    The caller renders "no panel" from the done state. Returning 401 with a body
    would make every refusal a special case in the website's error handling, and
    the contract promises one shape.
    """

    async def body() -> AsyncIterator[str]:
        # `seconds` too: a refusal is a `done` like any other, and the caller
        # parses one shape. It is ~0 because refusal precedes the graph.
        yield _sse("done", {"state": "refused", "seconds": 0.0})

    logger.info("answer request refused: %s", reason)
    return StreamingResponse(body(), media_type="text/event-stream")


@router.post("/answer")
async def answer(request: Request, body: AnswerRequest) -> StreamingResponse:
    verifying_key = getattr(request.app.state, "caller_token_key", None)
    if not verifying_key:
        # Should be unreachable: startup refuses without a key. If it happens,
        # refuse rather than answer.
        return _refusal("no verifying key on the app")

    try:
        claims = verify(body.caller_token, verifying_key)
    except TokenRejectedError as rejected:
        return _refusal(rejected.reason)

    # After verification, so an unsigned token cannot consume someone else's
    # budget by claiming their `sub`, and before the graph, so a caller over the
    # limit costs nothing.
    if not _limiter.allow(identity_of(claims, body.caller_token)):
        return _refusal("rate limited")

    graph = get_graph()

    # A fresh thread per request, never `id(body)`. `chat_history` is checkpointed
    # state annotated with `add_messages`, and the rephraser injects it through a
    # MessagesPlaceholder, so two requests sharing a thread means one stranger's
    # question and answer rephrase the next stranger's question. CPython reuses the
    # address of a freed object immediately: measured over 200 requests, `id(body)`
    # produced 38 distinct threads and put 192 of them on a shared one.

    # Resolved at startup, not per request: the graph is built from the same
    # bundles, so startup is what is actually being served. None if unknown --
    # a missing release costs the caller cache invalidation, not an answer.
    release = getattr(request.app.state, "release", None)

    async def stream() -> AsyncIterator[str]:
        yield _sse("start", {"release": release, "answered": True})
        started = time.monotonic()
        state = "failed"
        # The `react-to-me` prompt is the chat UI's and asks for inline anchors.
        # The contract promises this caller prose without them, citations being
        # separate events, so they come out here -- across fragment boundaries,
        # because one anchor arrives as twenty-odd fragments.
        stripper = AnchorStripper()
        tokens_sent = 0
        try:
            async with asyncio.timeout(ANSWER_TIMEOUT_SECONDS):
                async for event in graph.astream_answer(
                    body.question,
                    PROFILE,
                    thread_id=f"search-{uuid.uuid4()}",
                    # The postprocess node runs a Tavily web search after the
                    # answer, and `astream_answer` has no event to carry the
                    # result -- so on this path it was paid for and discarded,
                    # delaying `done` by the length of a web search. The chat UI
                    # renders those results; the search page has no place for
                    # them.
                    enable_postprocess=False,
                ):
                    if event.kind == "token":
                        text = stripper.feed(event.text)
                        if text:
                            tokens_sent += 1
                            yield _sse("token", {"text": text})
                    elif event.kind == "citation":
                        yield _sse(
                            "citation",
                            {
                                "st_id": event.st_id,
                                "display_name": event.display_name,
                            },
                        )
                    elif event.kind == "done":
                        state = event.state or "failed"
        except (asyncio.CancelledError, GeneratorExit):
            # The caller hung up. Both forms are caught because a hang-up
            # arrives as either, depending on who notices first: Starlette
            # cancelling the task raises CancelledError, while closing the
            # generator raises GeneratorExit. A test that only closed the
            # generator passed against a handler catching only CancelledError,
            # which is how this was found.
            #
            # Either way the graph is cancelled and the model call stops -- that
            # part already worked and is measured. What was missing was a record.
            #
            # Worth a log line because an abandoned stream is indistinguishable
            # from a healthy one in every other signal: the request 200s, tokens
            # flow, and then nothing. The website found a bug on their side where
            # a keystroke unmounted the panel mid-answer, and from here it would
            # have looked like ordinary traffic. A rate of these is the symptom
            # of a caller that starts answers it does not want.
            #
            # Re-raised, never swallowed: cancellation is not an error to report
            # to a caller who has already gone, and suppressing it would leave
            # the task pretending to still be running.
            logger.info(
                "answer abandoned by the caller after %.1fs and %d token events",
                time.monotonic() - started,
                tokens_sent,
            )
            raise
        except TimeoutError:
            # Distinct from the crash below so an operator can tell a slow
            # upstream from a broken one.
            logger.warning(
                "answering %r exceeded %.0fs",
                body.question[:80],
                ANSWER_TIMEOUT_SECONDS,
            )
            state = "failed"
        except Exception:
            # Deliberately broad, and deliberately not re-raised: a half-written
            # SSE stream cannot become an HTTP error code, and the caller needs a
            # terminal event to stop waiting.
            logger.exception("answering %r failed", body.question[:80])
            state = "failed"
        held = stripper.flush()
        if held:
            yield _sse("token", {"text": held})
        yield _sse(
            "done", {"state": state, "seconds": round(time.monotonic() - started, 1)}
        )

    return StreamingResponse(stream(), media_type="text/event-stream")
