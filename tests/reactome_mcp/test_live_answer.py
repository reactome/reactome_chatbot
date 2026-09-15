"""Answering from the live services instead of the vector store.

Why this exists: asked "What species are included in the Reactome database?",
retrieval routed correctly, pulled 40 documents, and answered "primarily Homo
sapiens ... no indications of other species being included". Reactome has 96.
The documents were all human, and it reported what it had. A sample of the
database's content cannot describe the database's scope, so no improvement to
retrieval fixes it.
"""

import asyncio
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from reactome_mcp.answer import MAX_TOOL_ROUNDS, answer_from_live_services


@tool
async def reactome_species() -> str:
    """List the species Reactome covers."""
    return "## Reactome Species\n**Total:** 96"


@tool
async def failing_tool() -> str:
    """A tool that raises."""
    raise RuntimeError("the service is down")


class _FakeLLM:
    """Replays a scripted sequence of replies, recording what it was sent."""

    def __init__(self, replies: list[AIMessage]) -> None:
        self._replies = list(replies)
        self.seen: list[list[Any]] = []
        self.bound_tools: list[Any] = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_FakeLLM":
        self.bound_tools = tools
        return self

    async def ainvoke(self, messages: Any, *args: Any, **kwargs: Any) -> AIMessage:
        self.seen.append(list(messages))
        return self._replies.pop(0) if self._replies else AIMessage("out of replies")


def _call(name: str, call_id: str = "1", args: dict | None = None) -> dict:
    return {"name": name, "args": args or {}, "id": call_id, "type": "tool_call"}


def test_a_tool_result_reaches_the_answer() -> None:
    llm = _FakeLLM(
        [
            AIMessage("", tool_calls=[_call("reactome_species")]),
            AIMessage("Reactome covers 96 species."),
        ]
    )
    answer = asyncio.run(
        answer_from_live_services(
            llm, [reactome_species], "what species are in reactome"
        )
    )

    assert "96" in answer
    # The tool output must actually have been given back to the model.
    tool_messages = [m for m in llm.seen[-1] if isinstance(m, ToolMessage)]
    assert any("96" in str(m.content) for m in tool_messages)


def test_an_answer_with_no_tool_call_is_returned_as_is() -> None:
    llm = _FakeLLM([AIMessage("I could not determine that.")])
    answer = asyncio.run(
        answer_from_live_services(llm, [reactome_species], "something")
    )
    assert answer == "I could not determine that."


def test_a_failing_tool_is_reported_to_the_model_not_raised() -> None:
    """A failed lookup is information. The model must be able to say it could
    not find out, rather than the turn dying or an answer being invented."""
    llm = _FakeLLM(
        [
            AIMessage("", tool_calls=[_call("failing_tool")]),
            AIMessage("I could not look that up just now."),
        ]
    )
    answer = asyncio.run(answer_from_live_services(llm, [failing_tool], "anything"))

    assert "could not" in answer.lower()
    tool_messages = [m for m in llm.seen[-1] if isinstance(m, ToolMessage)]
    assert any("the service is down" in str(m.content) for m in tool_messages)


def test_an_invented_tool_name_is_survivable() -> None:
    llm = _FakeLLM(
        [
            AIMessage("", tool_calls=[_call("reactome_teleport")]),
            AIMessage("Reactome covers 96 species."),
        ]
    )
    answer = asyncio.run(answer_from_live_services(llm, [reactome_species], "x"))

    assert "96" in answer
    tool_messages = [m for m in llm.seen[-1] if isinstance(m, ToolMessage)]
    assert any("No such tool" in str(m.content) for m in tool_messages)


def test_the_loop_is_bounded() -> None:
    """A model that keeps calling tools must not spend the user's afternoon."""
    # Every reply is another tool call: this model never stops on its own.
    llm = _FakeLLM(
        [
            AIMessage("", tool_calls=[_call("reactome_species", str(i))])
            for i in range(10)
        ]
    )
    answer = asyncio.run(answer_from_live_services(llm, [reactome_species], "x"))

    # MAX_TOOL_ROUNDS bound invocations, plus the forced final answer.
    assert len(llm.seen) <= MAX_TOOL_ROUNDS + 1
    # And it still says something: returning "" would render as the assistant
    # having nothing at all to say.
    assert answer.strip()
    assert "could not complete" in answer.lower()


def test_the_cap_still_answers_when_the_model_stops_in_time() -> None:
    llm = _FakeLLM(
        [
            AIMessage("", tool_calls=[_call("reactome_species")]),
            AIMessage("Reactome covers 96 species."),
        ]
    )
    assert "96" in asyncio.run(answer_from_live_services(llm, [reactome_species], "x"))


def test_the_requested_language_is_passed_through() -> None:
    llm = _FakeLLM([AIMessage("96 especes.")])
    asyncio.run(
        answer_from_live_services(
            llm, [reactome_species], "quelles especes?", language="French"
        )
    )
    system = llm.seen[0][0]
    assert "French" in str(system.content)


def test_content_blocks_are_flattened() -> None:
    """Some providers return a list of blocks rather than a string."""
    llm = _FakeLLM([AIMessage([{"type": "text", "text": "96 species."}])])
    answer = asyncio.run(answer_from_live_services(llm, [reactome_species], "x"))
    assert answer == "96 species."
