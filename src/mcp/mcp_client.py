import asyncio
import json


class MCPToolError(Exception):
    """Raised when MCP server returns a JSON-RPC error response."""
    pass


class MCPClient:
    """
    JSON-RPC client for communicating with the Reactome MCP server over stdin/stdout.

    Args:
        process: Running MCP server subprocess from MCPProcessManager.
        timeout: Seconds to wait for a response before raising TimeoutError.
    """

    def __init__(self, process: asyncio.subprocess.Process, timeout: float = 30.0):
        self.process = process
        self.timeout = timeout
        self.request_id = 0

    async def call(self, method: str, params: dict | None = None) -> dict:
        """
        Send a JSON-RPC request and return the result.

        Raises:
            MCPToolError: If the server returns an error response.
            asyncio.TimeoutError: If no response within timeout seconds.
            RuntimeError: If the server closes the connection or returns invalid JSON.
        """
        if params is None:
            params = {}

        self.request_id += 1

        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params,
        }

        message = json.dumps(request) + "\n"
        self.process.stdin.write(message.encode("utf-8"))
        await self.process.stdin.drain()

        # Wait for response with timeout so chatbot never hangs indefinitely
        response_line = await asyncio.wait_for(
            self.process.stdout.readline(),
            timeout=self.timeout,
        )

        if not response_line:
            raise RuntimeError("MCP server closed the connection.")

        try:
            response = json.loads(response_line.decode("utf-8").strip())
        except json.JSONDecodeError as e:
            raise RuntimeError(f"MCP server returned invalid JSON: {e}")

        # JSON-RPC error response — server understood request but returned an error
        if "error" in response:
            error = response["error"]
            raise MCPToolError(
                f"MCP error {error.get('code')}: {error.get('message')}"
            )

        return response.get("result", {})

    async def call_tool(self, tool_name: str, arguments: dict | None = None) -> str:
        """
        Call a specific MCP tool and return its text output.

        Args:
            tool_name: Name of the tool e.g. 'reactome_search'.
            arguments: Tool arguments as key-value pairs.
        """
        if arguments is None:
            arguments = {}

        result = await self.call(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
        )

        # MCP returns content as list of typed blocks — extract text blocks
        content = result.get("content", [])
        text_parts = [block["text"] for block in content if block.get("type") == "text"]
        return "\n".join(text_parts)