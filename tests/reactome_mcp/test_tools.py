"""The tool wrappers: names, arguments, and the size of the surface."""

import asyncio
from typing import Any

from reactome_mcp.tools import create_mcp_tools


class _RecordingClient:
    """Records what the wrapper asked the MCP server for."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> str:
        self.calls.append((name, arguments or {}))
        return f"result of {name}"


def test_the_surface_is_five_tools_and_that_is_deliberate() -> None:
    """reactome-mcp exposes 53; the chatbot wraps five.

    Every tool description is spent from the model's context before it answers
    anything, and a model choosing between 53 similar names chooses worse than
    one choosing between five. If this number changes, it should be because a
    question was getting answered wrongly without the new tool.
    """
    names = [t.name for t in create_mcp_tools(_RecordingClient())]

    assert names == [
        "reactome_search",
        "reactome_get_pathway",
        "reactome_analyze_identifiers",
        "reactome_database_info",
        "reactome_species",
    ]


def test_every_tool_describes_itself_to_the_model() -> None:
    """A tool description is a prompt, not documentation."""
    for tool in create_mcp_tools(_RecordingClient()):
        assert tool.description, f"{tool.name} has no description"
        assert len(tool.description) > 40, f"{tool.name}'s description is too thin"


def test_stable_id_is_sent_as_the_id_the_server_expects() -> None:
    """The wrapper renames the argument; the wire format must not change.

    `id` shadows a builtin and "stable ID" is Reactome's own term, so the model
    sees `stable_id` -- but the MCP tool's parameter is `id`, and getting this
    mapping wrong would fail only at runtime, on a real question.
    """
    client = _RecordingClient()
    tools = {t.name: t for t in create_mcp_tools(client)}

    asyncio.run(tools["reactome_get_pathway"].ainvoke({"stable_id": "R-HSA-109582"}))

    assert client.calls == [("reactome_get_pathway", {"id": "R-HSA-109582"})]


def test_identifiers_are_passed_through_as_a_list() -> None:
    client = _RecordingClient()
    tools = {t.name: t for t in create_mcp_tools(client)}

    asyncio.run(
        tools["reactome_analyze_identifiers"].ainvoke(
            {"identifiers": ["TP53", "BRCA1"]}
        )
    )

    assert client.calls == [
        ("reactome_analyze_identifiers", {"identifiers": ["TP53", "BRCA1"]})
    ]


def test_the_no_argument_tools_send_no_arguments() -> None:
    client = _RecordingClient()
    tools = {t.name: t for t in create_mcp_tools(client)}

    asyncio.run(tools["reactome_species"].ainvoke({}))
    asyncio.run(tools["reactome_database_info"].ainvoke({}))

    assert client.calls == [("reactome_species", {}), ("reactome_database_info", {})]


def test_analysis_tool_tells_the_model_not_to_answer_from_retrieval() -> None:
    """Spec 007's edge case: a gene list answered from the vector store looks
    like an analysis and is not one."""
    tools = {t.name: t for t in create_mcp_tools(_RecordingClient())}
    description = tools["reactome_analyze_identifiers"].description.lower()

    assert "enrich" in description
    assert "retriev" in description
