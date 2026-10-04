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

import asyncio
import itertools
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
        # A fresh id for every request. Every call used to send id 2 on the
        # one session the whole process shares, and the server matches
        # replies by id: two users' overlapping lookups got each other's
        # results -- one of them another reader's identifiers and analysis
        # token -- and the other call hung (review, area 3).
        self._ids = itertools.count(2)
        self._init_lock = asyncio.Lock()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": ACCEPT}
        if self._session_id:
            headers["mcp-session-id"] = self._session_id
        return headers

    @staticmethod
    def _parse(
        response: httpx.Response, expected_id: int | None = None
    ) -> dict[str, Any]:
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

        if expected_id is not None and message.get("id") != expected_id:
            # Never someone else's answer: refuse rather than hand it on.
            raise MCPToolError(
                f"MCP replied to request {message.get('id')!r}, not {expected_id}"
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

    async def _ensure_session(self) -> None:
        async with self._init_lock:
            if self._session_id is None:
                await self.initialize()

    @staticmethod
    def _session_lost(response: httpx.Response) -> bool:
        """The server no longer knows our session -- it restarted, or evicted
        it. Answered 400 "No valid session" by reactome-mcp, 404 by the spec."""
        if response.status_code == 404:
            return True
        return response.status_code == 400 and "session" in response.text.lower()

    async def call(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        await self._ensure_session()
        for attempt in (1, 2):
            request_id = next(self._ids)
            response = await self._client.post(
                self.endpoint,
                headers=self._headers(),
                json={
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params or {},
                },
            )
            if attempt == 1 and self._session_lost(response):
                # Re-initialize once. Without this, one MCP restart turned
                # live lookups off until the chatbot restarted (review, 3).
                logger.info("MCP session lost; initializing a new one")
                async with self._init_lock:
                    self._session_id = None
                    await self.initialize()
                continue
            response.raise_for_status()
            return self._parse(response, request_id)
        raise MCPToolError("unreachable")

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> str:
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
