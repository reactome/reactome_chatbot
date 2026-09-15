"""Answer a question about the Reactome database using the live services.

This exists because retrieval cannot. Asked which species Reactome includes,
the vector store answered "primarily Homo sapiens ... no indications of other
species" -- from forty documents that were all human pathways. It reported what
it had. Reactome covers 96 species, and no amount of reranking puts that in a
corpus which does not contain it.

One bounded tool-calling loop, not an agent: the model may call tools, sees the
results, and answers. Two rounds is enough for "look it up, then say what it
found", and a cap means a confused model cannot spend a user's afternoon.
"""

import logging
from typing import Any, Protocol, runtime_checkable

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 2


@runtime_checkable
class ToolCallingModel(Protocol):
    """What this loop needs of a model: bind tools, then be invoked.

    A Protocol rather than BaseChatModel so the loop can be tested against a
    scripted fake without a network call or a subclass of half of LangChain.
    """

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any: ...

    async def ainvoke(self, messages: Any, *args: Any, **kwargs: Any) -> Any: ...


SYSTEM_PROMPT = """You answer questions about the Reactome database using live lookups.

You have tools that query Reactome directly. Use them -- do not answer from memory, and
do not guess at numbers. If a tool gives you the answer, report it exactly: if Reactome
covers 96 species, say 96.

Answer in {language}.

Be direct and brief. Give the figure or the fact asked for, then any short context that
helps. Do not describe the tools you used or narrate your process.

If the tools do not answer the question, say plainly what you could not find out. Do not
substitute a plausible answer."""


async def answer_from_live_services(
    llm: ToolCallingModel,
    tools: list[BaseTool],
    question: str,
    language: str = "English",
    chat_history: list[BaseMessage] | None = None,
) -> str:
    """Run the tool-calling loop and return the answer text."""
    by_name = {tool.name: tool for tool in tools}
    bound = llm.bind_tools(tools)

    messages: list[BaseMessage] = [
        SystemMessage(SYSTEM_PROMPT.format(language=language)),
        *(chat_history or []),
        HumanMessage(question),
    ]

    for _round in range(MAX_TOOL_ROUNDS):
        reply = await bound.ainvoke(messages)
        messages.append(reply)

        calls = getattr(reply, "tool_calls", None) or []
        if not calls:
            break

        for call in calls:
            tool = by_name.get(call["name"])
            if tool is None:
                # The model invented a tool name. Tell it so, rather than
                # failing the turn: it can recover by calling a real one.
                logger.warning("model called unknown tool %r", call["name"])
                messages.append(
                    ToolMessage(
                        content=f"No such tool: {call['name']}. Available: {', '.join(by_name)}",
                        tool_call_id=call["id"],
                    )
                )
                continue

            try:
                result = await tool.ainvoke(call["args"])
            except Exception as exc:
                # A failed lookup is information, not a crash. The model needs
                # to say it could not find out, rather than invent.
                logger.warning("live tool %s failed: %s", call["name"], exc)
                result = f"This lookup failed: {exc}"
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    else:
        # Out of rounds with tool calls still pending: answer from what we have
        # rather than looping.
        logger.info("live answer hit the %d-round tool cap", MAX_TOOL_ROUNDS)
        messages.append(
            HumanMessage(
                "Answer now, from the tool results above. Do not call more tools."
            )
        )
        reply = await llm.ainvoke(messages)

    content: Any = reply.content
    if isinstance(content, list):
        # Some providers return content blocks rather than a string.
        text = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    else:
        text = str(content)

    if not text.strip():
        # A model still trying to call tools after the cap leaves no text.
        # Returning "" would render as the assistant having nothing to say.
        logger.warning("live answer produced no text after %d rounds", MAX_TOOL_ROUNDS)
        return (
            "I could not complete that lookup against the live Reactome services. "
            "Please try rephrasing the question."
        )
    return text
