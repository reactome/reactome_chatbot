"""LangChain tool wrappers over the MCP server.

Harvested from #137 by @GovindhKishore.

**Five tools, not fifty-three.** reactome-mcp exposes 53; this exposes the five
below. That is a decision, not an accident of what was needed first. Every tool
description is spent from the model's context before it has answered anything,
and a model choosing between 53 similarly-named tools chooses worse than one
choosing between five. The five cover what the chatbot cannot already do from
the bundle: live search, live pathway lookup, enrichment analysis, and the two
metadata questions ("what release is this?", "what species?") the bundle cannot
answer because it is a snapshot.

Add to this list when there is a question the chatbot gets wrong without the
tool -- not because the tool exists.
"""

import logging
from typing import Any, Protocol

from langchain_core.tools import BaseTool, tool

logger = logging.getLogger(__name__)


class ToolCaller(Protocol):
    """What these wrappers need: one method that calls an MCP tool."""

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> str: ...


def create_mcp_tools(client: ToolCaller) -> list[BaseTool]:
    """Wrap the curated MCP tools as LangChain tools bound to one client."""

    @tool
    async def reactome_search(query: str) -> str:
        """Search live Reactome for pathways, reactions, proteins, and genes.

        Use when the question names something that may be newer than the local
        knowledge base, or when an exact identifier is given.
        """
        return await client.call_tool("reactome_search", {"query": query})

    @tool
    async def reactome_get_pathway(stable_id: str) -> str:
        """Get details of one Reactome pathway or reaction by its stable ID.

        A stable ID looks like R-HSA-109582. Use after a search has found one.
        """
        # `stable_id` here, `id` on the wire: the MCP tool's parameter is `id`,
        # but that shadows a builtin and "stable ID" is Reactome's own term, so
        # it is the clearer thing to show the model.
        return await client.call_tool("reactome_get_pathway", {"id": stable_id})

    @tool
    async def reactome_analyze_identifiers(identifiers: list[str]) -> str:
        """Run pathway enrichment analysis over a list of genes or proteins.

        Use when the user supplies several identifiers and asks which pathways
        are enriched, over-represented, or implicated. This is a real analysis
        run by Reactome, not a lookup: do not answer such a question from
        retrieved documents instead.
        """
        return await client.call_tool(
            "reactome_analyze_identifiers", {"identifiers": identifiers}
        )

    @tool
    async def reactome_database_info() -> str:
        """Get the current Reactome release version and date.

        The local knowledge base is a snapshot and cannot answer this.
        """
        return await client.call_tool("reactome_database_info", {})

    @tool
    async def reactome_species() -> str:
        """List every species Reactome has pathway data for, with taxonomy IDs.

        Use when asked whether Reactome covers an organism, or which organisms
        it covers. Most Reactome content is human; other species are largely
        inferred by orthology, which is worth saying when it matters.
        """
        return await client.call_tool("reactome_species", {})

    return [
        reactome_search,
        reactome_get_pathway,
        reactome_analyze_identifiers,
        reactome_database_info,
        reactome_species,
    ]
