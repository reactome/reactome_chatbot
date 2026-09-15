"""Talk to a reactome-mcp server over Streamable HTTP.

The stdio client spawns `node` and speaks over a pipe. That works on a
developer's machine and cannot work where the chatbot actually runs: the
deployed image is Python, has no node, and does not mount reactome-mcp. So
`REACTOME_MCP_SERVER` can never be satisfied in the container -- the live
destination worked everywhere I tested it and nowhere it ships.

reactome-mcp also serves Streamable HTTP, which is what a sibling container or
a hosted instance offers. This speaks that.

Two things about the protocol worth stating, because both are easy to get wrong
and neither fails loudly:

  - The session id arrives in the `mcp-session-id` **response header** of the
    initialize call, and must be sent on every request afterwards. Omit it and
    the server answers 400, not a hint.
  - Responses come back **SSE-framed** (`content-type: text/event-stream`,
    `event: message` / `data: {...}`) rather than as bare JSON, even for a
    single reply to a POST. Parsing the body as JSON gets a decode error on
    text that is perfectly valid.
"""

import json
import logging
from typing import Any

import httpx

from reactome_mcp.client import PROTOCOL_VERSION, MCPToolError

logger = logging.getLogger(__name__)

ACCEPT = "application/json, text/event-stream"


class MCPHttpClient:
    """The same surface as the stdio `MCPClient`, over HTTP."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/mcp"
        self.timeout = timeout
        self._session_id: str | None = None
        self._client = httpx.AsyncClient(timeout=timeout)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": ACCEPT}
        if self._session_id:
            headers["mcp-session-id"] = self._session_id
        return headers

    @staticmethod
    def _parse(response: httpx.Response) -> dict[str, Any]:
        """Read one JSON-RPC message out of a JSON or SSE body."""
        body = response.text
        if "text/event-stream" in response.headers.get("content-type", ""):
            # event: message \n data: {...}. Take the last data line: a stream
            # may carry progress notifications before the reply.
            payloads = [
                line[len("data:") :].strip()
                for line in body.splitlines()
                if line.startswith("data:")
            ]
            if not payloads:
                raise MCPToolError(f"no data in SSE response: {body[:200]}")
            body = payloads[-1]

        try:
            message = json.loads(body)
        except json.JSONDecodeError as exc:
            raise MCPToolError(f"MCP returned invalid JSON: {exc}") from exc

        if not isinstance(message, dict):
            raise MCPToolError(
                f"MCP returned a {type(message).__name__}, expected an object"
            )

        if "error" in message:
            error = message["error"]
            raise MCPToolError(f"MCP error {error.get('code')}: {error.get('message')}")

        result = message.get("result", {})
        if not isinstance(result, dict):
            raise MCPToolError(
                f"MCP returned a {type(result).__name__} result, expected an object"
            )
        return result

    async def initialize(self, client_name: str = "reactome-chatbot") -> dict[str, Any]:
        response = await self._client.post(
            self.endpoint,
            headers=self._headers(),
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": client_name, "version": "1"},
                },
            },
        )
        response.raise_for_status()

        self._session_id = response.headers.get("mcp-session-id")
        if not self._session_id:
            raise MCPToolError(
                "the server issued no mcp-session-id on initialize; every later "
                "request would be refused"
            )
        result = self._parse(response)

        # A notification: no id, no reply expected.
        await self._client.post(
            self.endpoint,
            headers=self._headers(),
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )

        server = result.get("serverInfo", {})
        logger.info(
            "MCP server ready over HTTP at %s: %s %s",
            self.base_url,
            server.get("name", "unknown"),
            server.get("version", ""),
        )
        return result

    async def call(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        response = await self._client.post(
            self.endpoint,
            headers=self._headers(),
            json={"jsonrpc": "2.0", "id": 2, "method": method, "params": params or {}},
        )
        response.raise_for_status()
        return self._parse(response)

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> str:
        if self._session_id is None:
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
        if self._session_id is None:
            await self.initialize()
        tools = (await self.call("tools/list")).get("tools", [])
        if not isinstance(tools, list):
            raise MCPToolError(f"MCP returned a {type(tools).__name__} tool list")
        return [tool for tool in tools if isinstance(tool, dict)]

    async def aclose(self) -> None:
        if self._session_id:
            try:
                await self._client.delete(self.endpoint, headers=self._headers())
            except Exception as exc:
                logger.debug("could not end the MCP session cleanly: %s", exc)
        await self._client.aclose()
        self._session_id = None
