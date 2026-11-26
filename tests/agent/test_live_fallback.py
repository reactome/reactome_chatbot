"""Falling back to retrieval when the live services cannot be reached."""

import asyncio
from typing import Any, cast

import pytest
from langchain_core.runnables import RunnableConfig

import agent.profiles.react_to_me as profile
from agent.profiles.react_to_me import ReactToMeGraphBuilder, ReactToMeState


class _RecordingRag:
    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, _inputs: dict[str, Any], _config: Any) -> dict[str, Any]:
        self.calls += 1
        return {"answer": "an answer", "context": []}


def test_an_unreachable_mcp_is_not_reported_as_a_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed MCP start is remembered, so a retry could never succeed."""
    rag = _RecordingRag()
    builder = ReactToMeGraphBuilder.__new__(ReactToMeGraphBuilder)
    builder.rags = cast(Any, {"reactome": rag})

    async def _no_tools() -> None:
        return None

    monkeypatch.setattr(profile, "get_mcp_tools", _no_tools)
    state = ReactToMeState(
        user_input="q",
        rephrased_input="q",
        detected_language="English",
        chat_history=[],
        active_sources=["live"],
    )

    result = asyncio.run(builder._answer_from_live_services(state, RunnableConfig()))

    assert result.get("live_tool_failed") is not True
    assert rag.calls == 1, "the fallback never reached retrieval"
