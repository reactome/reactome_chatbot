"""Selecting collections within the `reactome` bundle.

Measured 2026-09-19 (specs/009-collection-routing/research.md): narrowing every
tracked question to `reactions` + `summations` took the sweep from 13/13 to
10/13, and a wrong narrow removed the answer rather than degrading it. So the
asymmetry these tests defend is: an empty selection must always mean *all*, and
nothing may narrow a source that has no collections.
"""

import asyncio
from typing import Any

from langchain_core.runnables import RunnableConfig

from agent.profiles.react_to_me import ReactToMeGraphBuilder, ReactToMeState
from agent.tasks.intent_classifier import QueryIntent, build_classifier_message
from retrievers.csv_chroma import selected_collections
from retrievers.reactome.metadata_info import reactome_descriptions_info


def test_an_omitted_selection_means_every_collection() -> None:
    # The field is optional in the schema, so a model that ignores it entirely
    # must get the behaviour that existed before this feature.
    assert QueryIntent(source="reactome").collections == []


def test_every_collection_in_the_bundle_is_described_to_the_classifier() -> None:
    # Sourced from the metadata, not a literal list: a collection added to the
    # bundle but missing from the prompt is never chosen, and nothing fails.
    message = build_classifier_message(frozenset({"reactome", "userguide"}))
    for name, description in reactome_descriptions_info.items():
        assert f"**{name}**" in message, f"{name} is not offered to the classifier"
        assert description.strip()[:40] in message


def test_the_userguide_prompt_alone_offers_no_collections() -> None:
    message = build_classifier_message(frozenset({"userguide"}))
    assert "disease_variants" not in message


class _RecordingRag:
    """Captures what the retriever would have been told to search."""

    def __init__(self) -> None:
        self.seen: list[list[str] | None] = []

    async def ainvoke(self, _inputs: dict[str, Any], _config: Any) -> dict[str, Any]:
        self.seen.append(selected_collections.get())
        return {"answer": "an answer", "context": []}


def _builder(rag: _RecordingRag, source: str) -> ReactToMeGraphBuilder:
    builder = ReactToMeGraphBuilder.__new__(ReactToMeGraphBuilder)
    builder.rags = {source: rag}  # type: ignore[attr-defined]
    return builder


def _state(source: str, collections: list[str]) -> ReactToMeState:
    return ReactToMeState(
        user_input="q",
        rephrased_input="q",
        detected_language="English",
        chat_history=[],
        active_sources=[source],  # type: ignore[list-item]
        collections=collections,
    )


def test_the_selection_reaches_retrieval() -> None:
    rag = _RecordingRag()
    asyncio.run(
        _builder(rag, "reactome").generate_answer(
            _state("reactome", ["disease_variants", "summations"]), RunnableConfig()
        )
    )
    assert rag.seen == [["disease_variants", "summations"]]


def test_the_userguide_is_never_narrowed_by_a_reactome_selection() -> None:
    # The dangerous direction. `userguide` is a different bundle with one
    # collection; a leaked selection names collections it does not have, and
    # `resolve_collections` would widen it back silently -- correct, but only
    # by accident, and with a WARNING for every question.
    rag = _RecordingRag()
    asyncio.run(
        _builder(rag, "userguide").generate_answer(
            _state("userguide", ["disease_variants"]), RunnableConfig()
        )
    )
    assert rag.seen == [None], "a reactome selection leaked into the userguide"


def test_the_selection_does_not_outlive_the_question() -> None:
    # The graph reuses one asyncio task across turns, so a selection left set
    # would narrow the *next* question -- which nothing downstream could
    # detect, because narrowing produces a confident answer from less.
    rag = _RecordingRag()
    asyncio.run(
        _builder(rag, "reactome").generate_answer(
            _state("reactome", ["ewas"]), RunnableConfig()
        )
    )
    assert selected_collections.get() is None


def test_a_state_without_the_field_searches_everything() -> None:
    # BaseState is total=False and this field is new, so a thread checkpointed
    # before it existed resumes without the key. Subscripting would raise;
    # missing must mean "all", which is how the graph behaved before routing.
    rag = _RecordingRag()
    state = _state("reactome", [])
    del state["collections"]  # type: ignore[misc]
    asyncio.run(_builder(rag, "reactome").generate_answer(state, RunnableConfig()))
    assert rag.seen == [[]], "a pre-routing checkpoint must not crash or narrow"
