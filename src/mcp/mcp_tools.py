from langchain_core.tools import tool
from mcp.mcp_process_manager import MCPProcessManager
from mcp.mcp_client import MCPClient


async def create_mcp_tools(mcp_server_path: str | None):
    """
    Start the MCP server and return LangChain tool wrappers + manager.
    Returns ([], None) if no server path provided.
    """
    if not mcp_server_path:
        return [], None

    manager = MCPProcessManager(mcp_server_path)
    process = await manager.start()
    client = MCPClient(process)

    @tool
    async def search_reactome(query: str) -> str:
        """Search Reactome for pathways, proteins, genes, and biological entities."""
        return await client.call_tool("reactome_search", {"query": query})

    @tool
    async def get_pathway(id: str) -> str:
        """Get detailed information about a specific Reactome pathway using its stable ID (e.g. R-HSA-109582)."""
        return await client.call_tool("reactome_get_pathway", {"id": id})

    @tool
    async def analyze_identifiers(identifiers: list[str]) -> str:
        """Run pathway enrichment analysis on a list of gene or protein identifiers (e.g. TP53, BRCA1)."""
        return await client.call_tool(
            "reactome_analyze_identifiers", {"identifiers": identifiers}
        )

    @tool
    async def get_database_info() -> str:
        """Get current Reactome database version and release information."""
        return await client.call_tool("reactome_database_info", {})

    @tool
    async def get_species() -> str:
        """Get the list of all species available in the Reactome database."""
        return await client.call_tool("reactome_species", {})

    tools = [search_reactome, get_pathway, analyze_identifiers, get_database_info, get_species]

    return tools, manager