"""Bounds on what one conversation or one query can cost (review, area 3)."""

import asyncio
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agent import history
from agent.graph import AgentGraph


def test_recent_keeps_the_latest_turns_by_count() -> None:
    turns = [HumanMessage(f"q{i}") for i in range(100)]
    kept = history.recent(turns)
    assert len(kept) == history.MAX_MESSAGES
    assert kept[-1].content == "q99"


def test_recent_keeps_within_a_size_budget() -> None:
    # 8,000-character messages overflowed the context window at turn 47.
    turns = [HumanMessage("x" * 8000) for _ in range(30)]
    kept = history.recent(turns)
    assert sum(len(str(m.content)) for m in kept) <= history.MAX_CHARS
    assert kept
    assert kept[-1] is turns[-1]


def test_a_seeded_first_turn_is_always_kept() -> None:
    seed: list[BaseMessage] = [
        HumanMessage("Summarise my analysis."),
        AIMessage("The summary.", additional_kwargs={history.SEED_MARK: True}),
    ]
    later: list[BaseMessage] = [HumanMessage(f"q{i}") for i in range(100)]
    kept = history.recent(seed + later)
    assert kept[:2] == seed
    assert kept[-1].content == "q99"


def test_recent_of_nothing_is_nothing() -> None:
    assert history.recent(None) == []


def test_bm25_scores_a_bounded_deduplicated_query() -> None:
    # ~45 ms of CPU per query token over Release 97, repeats included.
    from retrievers.csv_chroma import MAX_QUERY_TOKENS, BoundedBM25Retriever

    docs = [Document(page_content=t) for t in ("cdk5 tau", "apoptosis", "cell cycle")]
    retriever = BoundedBM25Retriever.from_documents(docs, preprocess_func=str.split)
    seen: list[list[str]] = []
    real = retriever.vectorizer.get_top_n

    def spy(tokens: list[str], documents: Any, n: int) -> Any:
        seen.append(list(tokens))
        return real(tokens, documents, n=n)

    retriever.vectorizer.get_top_n = spy
    words = " ".join(f"w{i}" for i in range(500))
    retriever.invoke("cdk5 cdk5 cdk5 " + words)
    assert len(seen[-1]) == MAX_QUERY_TOKENS
    assert seen[-1].count("cdk5") == 1
    # An ordinary question is scored exactly as before.
    assert retriever.invoke("cdk5 tau")[0].page_content == "cdk5 tau"


def test_embedding_requests_have_a_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent.models import EMBEDDING_TIMEOUT_SECONDS, get_embedding

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    embeddings = get_embedding("openai", "text-embedding-3-large")
    assert getattr(embeddings, "request_timeout", None) == EMBEDDING_TIMEOUT_SECONDS


def test_the_graph_is_compiled_once_under_concurrency() -> None:
    graph = AgentGraph.__new__(AgentGraph)
    graph.graph = None
    calls: list[int] = []

    async def initialize() -> dict[str, Any]:
        calls.append(1)
        await asyncio.sleep(0.01)
        return {"p": object()}

    graph.initialize = initialize  # type: ignore[method-assign]

    async def many() -> None:
        await asyncio.gather(*(graph._ensure_graph() for _ in range(5)))

    asyncio.run(many())
    assert calls == [1]
