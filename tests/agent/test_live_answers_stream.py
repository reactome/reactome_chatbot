"""A live-routed question must reach the caller.

Questions like "what species are in reactome" are answered from the MCP live
lookup, which uses tools and no retriever. `astream_answer` opened its token
boundary on `on_retriever_end`, so on that path the boundary never opened: every
token was discarded and the caller was told `nothing_found`, for questions the
chat answers correctly.

Nothing caught it. It only happens where MCP is configured -- beta, not a
developer's machine, where these questions fall back to the vector store and
retrieve normally -- and the chat UI and the answer sweep both use `ainvoke`,
which takes the final answer and never reads this stream.

The event sequences below are copied from a real run against beta's MCP sibling.
"""

import asyncio
from typing import Any, cast

from agent.graph import AgentGraph


def _chunk(text: str) -> Any:
    class Chunk:
        content = text

    return Chunk()


def _model_stream(text: str, node: str = "model") -> dict[str, Any]:
    return {
        "event": "on_chat_model_stream",
        "name": "ChatOpenAI",
        "metadata": {"langgraph_node": node},
        "data": {"chunk": _chunk(text)},
    }


def _preprocess_end(sources: list[str]) -> dict[str, Any]:
    return {
        "event": "on_chain_end",
        "name": "preprocess",
        "metadata": {"langgraph_node": "preprocess"},
        "data": {"output": {"active_sources": sources}},
    }


class _FakeCompiled:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events

    async def astream_events(self, *_a: Any, **_k: Any) -> Any:
        for event in self._events:
            yield event


def _drive(events: list[dict[str, Any]]) -> tuple[str, list[str]]:
    graph = AgentGraph.__new__(AgentGraph)
    # A stand-in for the compiled graph: astream_answer only calls
    # astream_events on it, and building a real one costs about a minute.
    graph.graph = cast("dict[str, Any]", {"react-to-me": _FakeCompiled(events)})

    async def run() -> tuple[str, list[str]]:
        text: list[str] = []
        state = "never-set"
        async for event in graph.astream_answer(
            "what species are in reactome", "react-to-me", thread_id="t"
        ):
            if event.kind == "token":
                text.append(event.text)
            elif event.kind == "done":
                state = event.state or "none"
        return state, text

    return asyncio.run(run())


LIVE_SEQUENCE = [
    # The rephraser and the rest, at the preprocess node -- never the answer.
    _model_stream("What species are represented", node="preprocess"),
    _preprocess_end(["live"]),
    # The live answer. No retriever ran, and none will.
    _model_stream("React"),
    _model_stream("ome contains 15 species."),
]


def test_a_live_routed_question_reaches_the_caller() -> None:
    state, text = _drive(LIVE_SEQUENCE)
    assert state == "answered", "a live answer was reported as nothing_found"
    assert "".join(text) == "Reactome contains 15 species."


def test_preprocess_tokens_are_still_excluded_on_the_live_path() -> None:
    """Opening the boundary must not let the rephraser's output through.

    It is safe here only because the query expander lives inside the retriever,
    so nothing but the answer streams at the answer node on this path.
    """
    _state, text = _drive(LIVE_SEQUENCE)
    assert "What species are represented" not in "".join(text)


def test_a_non_live_route_still_waits_for_retrieval() -> None:
    """The original boundary must survive: with a retriever on the path, the
    expander's tokens run at the answer node too and only retrieval separates
    them."""
    events = [
        _preprocess_end(["reactome"]),
        _model_stream("expanded query variant"),  # the query expander
        {
            "event": "on_retriever_end",
            "name": "retriever",
            "metadata": {},
            "data": {"output": []},
        },
        _model_stream("The actual answer."),
    ]
    state, text = _drive(events)
    assert state == "answered"
    assert "".join(text) == "The actual answer."
    assert "expanded query" not in "".join(text)
