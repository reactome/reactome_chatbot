"""The stream must carry the answer, and nothing that merely looks like one.

Six model calls run around one answer -- rephrase, safety, language detection,
intent, query expansion, and the answer. Measured 2026-09-17: the first streamed
token of any kind arrives at 3.0s and belongs to the rephraser; the answer's own
first token arrives at 36.1s. Streaming everything would put the safety check and
an expanded query into the caller's panel.

The expander and the answer cannot be told apart by metadata -- both run at
`langgraph_node == "model"`, with identical keys and no distinguishing tags. Only
order separates them: the expander runs inside retrieval, the answer after it.
These tests pin that boundary, because a change to the graph could move it
silently and the symptom would be an answer panel showing "Which proteins...".
"""

import asyncio
from typing import Any, cast

from langgraph.graph.state import CompiledStateGraph

from agent.graph import ANSWER_NODE, MAX_CITATIONS, AgentGraph


class _Doc:
    def __init__(self, st_id: str, name: str = "") -> None:
        self.metadata = {"st_id": st_id, "display_name": name}


class _Chunk:
    def __init__(self, content: str) -> None:
        self.content = content


def _token(text: str, node: str = ANSWER_NODE) -> dict[str, Any]:
    return {
        "event": "on_chat_model_stream",
        "metadata": {"langgraph_node": node},
        "data": {"chunk": _Chunk(text)},
    }


def _retrieved(docs: list[_Doc]) -> dict[str, Any]:
    return {"event": "on_retriever_end", "data": {"output": docs}}


class _FakeCompiled:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events

    async def astream_events(self, *_a: Any, **_k: Any) -> Any:
        for event in self._events:
            yield event


def _graph(events: list[dict[str, Any]]) -> AgentGraph:
    """An AgentGraph with a scripted event stream and no real construction.

    `__new__` rather than `__init__` on purpose: building one takes about 85
    seconds and would make these tests useless as a fast guard. The cast names
    the substitution once -- _FakeCompiled is not a CompiledStateGraph and mypy
    is right to say so.
    """
    graph = AgentGraph.__new__(AgentGraph)
    graph.graph = cast(
        "dict[str, CompiledStateGraph[Any, None, Any, Any]]",
        {"react-to-me": _FakeCompiled(events)},
    )
    return graph


async def _collect(graph: AgentGraph) -> list[Any]:
    return [
        event async for event in graph.astream_answer("q", "react-to-me", thread_id="t")
    ]


def test_tokens_before_retrieval_are_not_the_answer() -> None:
    """The rephraser streams first. It must not reach the caller."""
    events = _collect_sync(
        [
            _token("What", node="preprocess"),
            _token("Which proteins"),  # the query expander, before retrieval
            _retrieved([_Doc("R-HSA-1")]),
            _token("In the context of"),
            _token(" Alzheimer disease"),
        ]
    )
    text = "".join(e.text for e in events if e.kind == "token")
    assert text == "In the context of Alzheimer disease"
    assert "Which" not in text


def _collect_sync(events: list[dict[str, Any]]) -> list[Any]:
    return asyncio.run(_collect(_graph(events)))


def test_citations_come_from_retrieved_documents() -> None:
    events = _collect_sync(
        [_retrieved([_Doc("R-HSA-1", "Apoptosis"), _Doc("R-HSA-2")]), _token("hi")]
    )
    cites = [(e.st_id, e.display_name) for e in events if e.kind == "citation"]
    assert cites == [("R-HSA-1", "Apoptosis"), ("R-HSA-2", "")]


def test_citations_are_deduplicated_and_capped() -> None:
    """Uncapped this emitted 315 for one question: every sub-retriever's output."""
    many = [_Doc(f"R-HSA-{i}") for i in range(100)]
    events = _collect_sync([_retrieved(many), _retrieved(many), _token("answer")])
    cites = [e.st_id for e in events if e.kind == "citation"]
    assert len(cites) == MAX_CITATIONS
    assert len(set(cites)) == len(cites)


def test_an_answer_with_no_tokens_reports_nothing_found() -> None:
    events = _collect_sync([_retrieved([_Doc("R-HSA-1")])])
    assert events[-1].kind == "done"
    assert events[-1].state == "nothing_found"


def test_a_streamed_answer_reports_answered() -> None:
    events = _collect_sync([_retrieved([]), _token("text")])
    assert events[-1].state == "answered"


def test_an_unknown_profile_fails_rather_than_hanging() -> None:
    import asyncio

    graph = _graph([])
    got = asyncio.run(_collect_unknown(graph))
    assert got[-1].kind == "done"
    assert got[-1].state == "failed"


async def _collect_unknown(graph: AgentGraph) -> list[Any]:
    return [e async for e in graph.astream_answer("q", "nope", thread_id="t")]
