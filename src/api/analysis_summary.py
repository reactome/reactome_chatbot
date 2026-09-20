"""Summarise a completed Reactome analysis, for the website's analysis page.

Contract: specs/011-summarise-analysis-results/contracts/summary_endpoint.md

Shaped like `/api/answer` -- SSE, citations as their own events, failure as a
terminal state and never an HTTP error -- because a summary takes comparable
time and an analysis page must not break because this service had a problem.

The authorisation bar is stricter, though. `/api/answer` verifies *caller
identity* and deliberately says nothing about a person. This discloses a
user's own uploaded analysis to a model provider, and the choice of what to
disclose is only meaningful if a person made it, so it additionally requires
evidence that one is present.
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

from agent.graph import resolve_llm_model
from agent.models import get_llm
from analysis.client import current_release, fetch_not_found, fetch_result
from analysis.disclosure import Tier, for_tier
from analysis.store import SummaryStore
from analysis.summarise import (
    INEXACT_COUNT_INSTRUCTION,
    NAMED_UNMATCHED_INSTRUCTION,
    STATISTICS_INSTRUCTION,
    TYPE_INSTRUCTION,
    UNMATCHED_INSTRUCTION,
    VERDICT_INSTRUCTION,
    prompt_input,
)
from util.caller_token import TokenRejectedError, human_presence_reason, verify
from util.logging import logging
from util.rate_limit import identity_of, limiter_from_env

logger = logging.getLogger(__name__)

router = APIRouter()

SUMMARY_TIMEOUT_SECONDS = 120.0

#: Bounded for the same reason the answer endpoint is: a stuck upstream must
#: not hold a connection open.
_limiter = limiter_from_env()

#: Process-local, lost on deploy (research D4). Module state so it outlives
#: a request, as the limiter does.
_store = SummaryStore()

SYSTEM_PROMPT = """
You explain a completed Reactome pathway-analysis result to the researcher who
ran it.

Rules, in order of importance:
1. Every quantitative claim must come from the data below. Never state a
   statistic it does not contain.
2. Follow the verdict instruction exactly. It is computed from the data, not
   guessed, and it overrides any impression the numbers give you.
3. Whenever you call a pathway significant, say whether that is before or
   after multiple-testing correction.
4. Do not name a pathway that is not in the data below.
5. Never state how many pathways were significant overall unless the data
   says that count is exact. Only the highest-ranked are included.
6. Do not list sources or citations at the end. The interface renders them
   from structured events; a list here is a duplicate.
7. Four short paragraphs at most. Plain prose for a working scientist.
""".strip()


IMPLEMENTED_TIERS = ("aggregate", "identifiers")


class SummaryRequest(BaseModel):
    token: str = Field(min_length=1, max_length=256)
    caller_token: str = ""
    #: Required, with no default: a default is not a choice (FR-012).
    disclosure: Tier


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def _done(state: str, seconds: float, reason: str | None = None) -> str:
    payload: dict[str, Any] = {"state": state, "seconds": round(seconds, 1)}
    if reason:
        payload["reason"] = reason
    return _sse("done", payload)


def _refusal(reason: str, log: str) -> StreamingResponse:
    """Always HTTP 200. An analysis page must not break because of us."""
    logger.info("analysis summary refused: %s", log)

    async def stream() -> AsyncIterator[str]:
        yield _done("refused", 0.0, reason)

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.post("/analysis-summary")
async def analysis_summary(body: SummaryRequest, request: Request) -> StreamingResponse:
    started = time.monotonic()

    verifying_key = getattr(request.app.state, "caller_token_key", None)
    if not verifying_key:
        # Unreachable: startup refuses without a key. Refuse rather than answer.
        return _refusal("no_caller", "no verifying key on the app")
    try:
        claims = verify(body.caller_token, verifying_key)
    except TokenRejectedError as rejected:
        return _refusal("no_caller", rejected.reason)

    # Stricter than the answer endpoint, and checked before any model call.
    presence = human_presence_reason(claims, time.time())
    if presence:
        return _refusal(presence, presence)

    if body.disclosure not in IMPLEMENTED_TIERS:
        return _refusal("unsupported_tier", f"tier {body.disclosure} is not built")

    # Per person where the website tells us who, else per caller. `human_sub`
    # is the identity cookie's subject; `sub` is the caller token's own.
    human_sub = claims.get("human_sub")
    key = f"human:{human_sub}" if isinstance(human_sub, str) and human_sub else None
    if not _limiter.allow(key or identity_of(claims, body.caller_token)):
        # Its own reason: a caller told `no_caller` would render "not
        # verified" to a reader who is verified and simply asked too often.
        return _refusal("rate_limited", "rate limited")

    async def stream() -> AsyncIterator[str]:
        state = "failed"
        try:
            async with asyncio.timeout(SUMMARY_TIMEOUT_SECONDS):
                fetched = await fetch_result(body.token)
                if fetched.outcome != "ok" or fetched.result is None:
                    yield _done(fetched.outcome, time.monotonic() - started)
                    return

                payload = for_tier(fetched.result, body.disclosure)
                model_input = prompt_input(payload)
                release = await current_release()

                # Looked up BEFORE the disclosure fetch, and under the tier
                # the reader *asked* for. The first version looked it up
                # after, so a cache hit on the disclosing tier still went and
                # fetched the reader's identifiers and then discarded them --
                # a pointless request to the endpoint that returns their data,
                # on every reload of a summary we already had.
                #
                # Stored under the tier that *applied*, which is not always
                # the same. The asymmetry is deliberate: a request whose
                # disclosure failed stores an aggregate summary under
                # `aggregate`, so the next disclosing request misses and gets
                # another chance at the identifiers rather than being served
                # the fallback forever.
                cached = (
                    _store.get(body.token, release, body.disclosure)
                    if release
                    else None
                )

                # Which tier the answer was actually built from.
                applied: Tier = body.disclosure
                if cached is None and body.disclosure == "identifiers":
                    # The only place this service asks for the reader's own
                    # identifiers, reached only because they chose it and
                    # only when there is nothing to reuse. A separate call
                    # taking a separate decision, never a flag with a
                    # default, and never on the aggregate path.
                    unmatched = await fetch_not_found(body.token)
                    if unmatched:
                        model_input["identifiers_not_found_names"] = unmatched
                    elif model_input.get("identifiers_not_found"):
                        # There were unmatched identifiers and we could not
                        # retrieve them, so the summary is the aggregate one.
                        # Saying so is the point: a reader who chose to
                        # disclose and silently got the other summary has
                        # been told nothing and given nothing.
                        applied = "aggregate"
                        logger.warning(
                            "identifier tier requested but the not-found "
                            "lookup returned nothing; serving aggregate"
                        )
                yield _sse(
                    "start",
                    {
                        "release": release,
                        "analysis_type": model_input.get("analysis_type"),
                        # Stability is reuse, not determinism (FR-015). This
                        # is how the interface knows which it is looking at.
                        "cached": cached is not None,
                        # What the summary was built from. Equal to the
                        # request's `disclosure` except when the disclosing
                        # tier could not be honoured.
                        "disclosure": applied,
                    },
                )

                # From the result, never from the model's prose. An invented
                # or mismatched identifier is impossible by construction
                # rather than by checking afterwards (SC-003).
                citations: tuple[tuple[str, str], ...] = (
                    cached.citations
                    if cached
                    else tuple(
                        (p["st_id"], p.get("name") or p["st_id"])
                        for p in model_input["pathways"]
                        if p.get("st_id")
                    )
                )
                for st_id, display_name in citations:
                    yield _sse(
                        "citation", {"st_id": st_id, "display_name": display_name}
                    )

                if cached:
                    # Byte-identical, and in one event: re-streaming it token
                    # by token would imitate generation that is not happening.
                    yield _sse("token", {"text": cached.text})
                    yield _done("summarised", time.monotonic() - started)
                    return

                provider, model, base_url = resolve_llm_model(None)
                llm = get_llm(provider, model, base_url=base_url, request_timeout=90.0)
                instruction = VERDICT_INSTRUCTION[model_input["verdict"]]
                if not model_input["significant_count_is_exact"]:
                    instruction = f"{instruction} {INEXACT_COUNT_INSTRUCTION}"
                by_type = TYPE_INSTRUCTION.get(
                    str(model_input.get("analysis_type") or "").upper()
                )
                if by_type:
                    instruction = f"{instruction} {by_type}"
                instruction = f"{instruction} {STATISTICS_INSTRUCTION}"
                instruction = f"{instruction} {UNMATCHED_INSTRUCTION}"
                if model_input.get("identifiers_not_found_names"):
                    instruction = f"{instruction} {NAMED_UNMATCHED_INSTRUCTION}"
                messages = [
                    ("system", SYSTEM_PROMPT),
                    (
                        "human",
                        f"Verdict instruction: {instruction}\n\n"
                        f"Data:\n{json.dumps(model_input, default=str)}",
                    ),
                ]
                produced: list[str] = []
                async for chunk in llm.astream(messages):
                    text = getattr(chunk, "content", "")
                    if isinstance(text, str) and text:
                        produced.append(text)
                        yield _sse("token", {"text": text})
                if release:
                    _store.put(
                        body.token, release, applied, "".join(produced), citations
                    )
                state = "summarised"
        except (asyncio.CancelledError, GeneratorExit):
            logger.info(
                "analysis summary abandoned after %.1fs", time.monotonic() - started
            )
            raise
        except Exception:
            # Deliberately broad and never re-raised: a half-written SSE
            # stream cannot become a status code, and the caller needs a
            # terminal event to stop waiting.
            logger.exception("summarising an analysis failed")
            state = "failed"
        yield _done(state, time.monotonic() - started)

    return StreamingResponse(stream(), media_type="text/event-stream")


def new_thread_id() -> str:
    """Unused by the stream, kept for parity with the answer endpoint's ids."""
    return f"summary-{uuid.uuid4()}"
