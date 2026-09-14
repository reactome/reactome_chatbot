"""Client for the Reactome MCP server.

Named `reactome_mcp`, not `mcp`. The official MCP Python SDK is published on
PyPI as `mcp`; a local package of that name shadows it, and the failure only
appears the day someone adds the dependency. Harvested from #127/#137 by
@GovindhKishore, where it was `src/mcp/`.
"""

from reactome_mcp.client import MCPClient, MCPToolError
from reactome_mcp.process import MCPConnectionError, MCPProcessManager

__all__ = [
    "MCPClient",
    "MCPConnectionError",
    "MCPProcessManager",
    "MCPToolError",
]
