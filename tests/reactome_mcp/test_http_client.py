"""The HTTP transport, against a fake server.

This exists because the stdio client cannot work where the chatbot runs. The
deployed image is Python, has no `node`, and does not mount reactome-mcp, so
`REACTOME_MCP_SERVER` can never be satisfied in the container -- the live
destination worked on every machine it was tested on and none that it ships to.

Two protocol details are pinned here because both are easy to get wrong and
neither fails in a way that points at the cause: the session id arrives in a
response header, and replies come back SSE-framed rather than as bare JSON.
"""

import asyncio
import json
from typing import Any

import httpx
import pytest

from reactome_mcp.client import MCPToolError
from reactome_mcp.http_client import MCPHttpClient


def sse(payload: dict[str, Any]) -> str:
    """How the server actually frames a reply to a POST."""
    return f"event: message\ndata: {json.dumps(payload)}\n\n"


def _client(handler: Any, url: str = "http://mcp.test") -> MCPHttpClient:
    client = MCPHttpClient(url)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


def test_the_session_id_is_read_from_the_header_and_resent() -> None:
    """Omit it on later requests and the server answers 400, not a hint."""
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("mcp-session-id"))
        body = json.loads(request.content)
        if body.get("method") == "initialize":
            return httpx.Response(
                200,
                headers={
                    "content-type": "text/event-stream",
                    "mcp-session-id": "abc-123",
                },
                text=sse(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "result": {"serverInfo": {"name": "reactome"}},
                    }
                ),
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=sse(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "result": {"content": [{"type": "text", "text": "96"}]},
                }
            ),
        )

    client = _client(handler)
    assert asyncio.run(client.call_tool("reactome_species")) == "96"

    # initialize carries none; everything after carries the issued id.
    assert seen[0] is None
    assert all(s == "abc-123" for s in seen[1:]), seen


def test_an_sse_framed_reply_is_parsed() -> None:
    """Parsing the body as JSON gets a decode error on text that is valid."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        result: dict[str, Any] = (
            {"serverInfo": {"name": "reactome"}}
            if body.get("method") == "initialize"
            else {"content": [{"type": "text", "text": "hello"}]}
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream", "mcp-session-id": "s"},
            text=sse({"jsonrpc": "2.0", "id": 1, "result": result}),
        )

    assert asyncio.run(_client(handler).call_tool("x")) == "hello"


def test_a_plain_json_reply_is_also_accepted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        result: dict[str, Any] = (
            {"serverInfo": {}}
            if body.get("method") == "initialize"
            else {"content": [{"type": "text", "text": "json"}]}
        )
        return httpx.Response(
            200,
            headers={"content-type": "application/json", "mcp-session-id": "s"},
            json={"jsonrpc": "2.0", "id": 1, "result": result},
        )

    assert asyncio.run(_client(handler).call_tool("x")) == "json"


def test_a_missing_session_id_is_refused_up_front() -> None:
    """Rather than sending every later request to be rejected one at a time."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=sse({"jsonrpc": "2.0", "id": 1, "result": {}}),
        )

    with pytest.raises(MCPToolError, match="no mcp-session-id"):
        asyncio.run(_client(handler).initialize())


def test_a_jsonrpc_error_is_raised() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("method") == "initialize":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream", "mcp-session-id": "s"},
                text=sse({"jsonrpc": "2.0", "id": 1, "result": {}}),
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=sse(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "error": {"code": -32601, "message": "no such tool"},
                }
            ),
        )

    with pytest.raises(MCPToolError, match="no such tool"):
        asyncio.run(_client(handler).call_tool("nope"))


def test_a_tool_error_result_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        result = (
            {"serverInfo": {}}
            if body.get("method") == "initialize"
            else {
                "isError": True,
                "content": [{"type": "text", "text": "pathway not found"}],
            }
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream", "mcp-session-id": "s"},
            text=sse({"jsonrpc": "2.0", "id": 1, "result": result}),
        )

    with pytest.raises(MCPToolError, match="pathway not found"):
        asyncio.run(_client(handler).call_tool("reactome_get_pathway", {"id": "nope"}))


def test_an_http_failure_surfaces() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream is down")

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(_client(handler).initialize())


def test_the_last_data_line_wins() -> None:
    """A stream may carry progress notifications before the reply."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("method") == "initialize":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream", "mcp-session-id": "s"},
                text=sse({"jsonrpc": "2.0", "id": 1, "result": {}}),
            )
        stream = sse({"jsonrpc": "2.0", "method": "notifications/progress"}) + sse(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"content": [{"type": "text", "text": "the answer"}]},
            }
        )
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=stream
        )

    assert asyncio.run(_client(handler).call_tool("x")) == "the answer"
