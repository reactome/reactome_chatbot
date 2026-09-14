"""JSON-RPC over the MCP server's stdio.

Harvested from #127 by @GovindhKishore, with three changes.

**It performs the MCP initialize handshake.** The original went straight to
`tools/call`. The server accepts that today because the SDK is lenient, but the
protocol requires initialize first, and relying on leniency means the day an SDK
release enforces it, every call fails at once. Doing it properly also removes
the arbitrary `sleep(1)` the original used to decide the server had started: a
successful initialize *is* the readiness check.

**It matches responses to requests by id.** The original returned the next line
on stdout, whatever it was. A server notification arriving between request and
response would have been read as the answer, and every later call would be one
reply out of step -- returning the previous question's answer, with nothing
raising.

**One call at a time.** A lock serialises the write/read pair, because two
coroutines interleaving on one pipe is the same desync by another route.
"""

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "2024-11-05"


class MCPToolError(RuntimeError):
    """The server returned a JSON-RPC error."""


class MCPClient:
    def __init__(
        self,
        process: asyncio.subprocess.Process,
        timeout: float = 30.0,
    ) -> None:
        self.process = process
        self.timeout = timeout
        self._next_id = 0
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self, client_name: str = "reactome-chatbot") -> dict[str, Any]:
        """Complete the handshake. Doubles as the readiness check."""
        result = await self.call(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": client_name, "version": "1"},
            },
        )
        await self._notify("notifications/initialized")
        self._initialized = True
        server = result.get("serverInfo", {})
        logger.info(
            "MCP server ready: %s %s",
            server.get("name", "unknown"),
            server.get("version", ""),
        )
        return result

    async def _write(self, payload: dict[str, Any]) -> None:
        if self.process.stdin is None:
            raise MCPToolError("MCP server stdin is closed")
        self.process.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
        await self.process.stdin.drain()

    async def _notify(self, method: str) -> None:
        """A notification has no id and gets no reply."""
        async with self._lock:
            await self._write({"jsonrpc": "2.0", "method": method})

    async def _read_reply(self, request_id: int) -> dict[str, Any]:
        """Read until the reply to this request arrives.

        Anything that is not a reply to us -- a notification, a stray line --
        is logged and skipped rather than returned. Returning it would answer
        the caller's question with someone else's answer.
        """
        if self.process.stdout is None:
            raise MCPToolError("MCP server stdout is closed")

        while True:
            line = await self.process.stdout.readline()
            if not line:
                raise MCPToolError("MCP server closed the connection")

            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue

            try:
                message = json.loads(text)
            except json.JSONDecodeError:
                logger.debug("ignoring non-JSON line from MCP server: %.200s", text)
                continue

            if message.get("id") != request_id:
                logger.debug("ignoring MCP message not addressed to %s", request_id)
                continue

            if "error" in message:
                error = message["error"]
                raise MCPToolError(
                    f"MCP error {error.get('code')}: {error.get('message')}"
                )

            # Checked, not asserted. json.loads gives Any, and casting it to
            # the shape we hoped for is how reactome-mcp shipped ten formatters
            # that read fields the API never returned.
            result = message.get("result", {})
            if not isinstance(result, dict):
                raise MCPToolError(
                    f"MCP returned a {type(result).__name__} result for "
                    f"request {request_id}, expected an object"
                )
            return result

    async def call(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        async with self._lock:
            self._next_id += 1
            request_id = self._next_id
            await self._write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params or {},
                }
            )
            return await asyncio.wait_for(
                self._read_reply(request_id), timeout=self.timeout
            )

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> str:
        """Call a tool and return its text, joined across content blocks."""
        if not self._initialized:
            await self.initialize()

        result = await self.call(
            "tools/call", {"name": name, "arguments": arguments or {}}
        )
        blocks = result.get("content", [])
        if not isinstance(blocks, list):
            raise MCPToolError(f"{name} returned no content blocks")
        text = "\n".join(
            str(block.get("text", ""))
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if result.get("isError"):
            raise MCPToolError(text or f"{name} failed with no message")
        return text

    async def list_tools(self) -> list[dict[str, Any]]:
        if not self._initialized:
            await self.initialize()
        result = await self.call("tools/list")
        tools = result.get("tools", [])
        if not isinstance(tools, list):
            raise MCPToolError(
                f"MCP returned a {type(tools).__name__} tool list, expected an array"
            )
        return [tool for tool in tools if isinstance(tool, dict)]
