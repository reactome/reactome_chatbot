"""The MCP client's failure modes, without a server or a network.

The protocol handling is what is pinned here. It is easy to get subtly wrong in
a way that does not raise: returning the previous question's answer is the one
that matters, because nothing about it looks like an error.
"""

import asyncio
import json
from typing import Any, cast

import pytest

from reactome_mcp.client import MCPClient, MCPToolError
from reactome_mcp.process import MCPConnectionError, MCPProcessManager


def run(coro: Any) -> Any:
    """Drive a coroutine from a sync test.

    This repo has no pytest-asyncio and no other async tests; adding the
    dependency for this file alone is not worth a lockfile change.
    """
    return asyncio.run(coro)


class _FakeStdin:
    def __init__(self) -> None:
        self.written: list[dict[str, Any]] = []

    def write(self, data: bytes) -> None:
        self.written.append(json.loads(data.decode()))

    async def drain(self) -> None:
        return None


class _FakeStdout:
    """Replays a scripted sequence of lines, one per readline()."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = list(lines)

    async def readline(self) -> bytes:
        if not self._lines:
            return b""
        return (self._lines.pop(0) + "\n").encode()


class _FakeProcess:
    def __init__(self, lines: list[str]) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout(lines)
        self.stderr = None
        self.returncode = None


def _client(lines: list[str], initialized: bool = True) -> MCPClient:
    # cast: _FakeProcess stands in for asyncio.subprocess.Process, which cannot
    # be constructed without actually spawning something.
    client = MCPClient(cast(Any, _FakeProcess(lines)), timeout=5.0)
    client._initialized = initialized
    return client


def test_call_tool_returns_the_text_blocks() -> None:
    client = _client(
        [
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"content": [{"type": "text", "text": "hello"}]},
                }
            )
        ]
    )
    assert run(client.call_tool("reactome_species")) == "hello"


def test_a_notification_is_not_mistaken_for_the_answer() -> None:
    """The bug this client exists to avoid.

    Reading "the next line on stdout" returns a server notification as if it
    were the reply. Nothing raises: the caller gets a plausible object, and
    every later call is one reply out of step — answering each question with
    the previous question's answer.
    """
    client = _client(
        [
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/message",
                    "params": {"level": "info"},
                }
            ),
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "content": [{"type": "text", "text": "the real answer"}]
                    },
                }
            ),
        ]
    )
    assert run(client.call_tool("reactome_species")) == "the real answer"


def test_a_reply_to_a_different_request_is_skipped() -> None:
    client = _client(
        [
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 99,
                    "result": {"content": [{"type": "text", "text": "someone else's"}]},
                }
            ),
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"content": [{"type": "text", "text": "mine"}]},
                }
            ),
        ]
    )
    assert run(client.call_tool("reactome_species")) == "mine"


def test_non_json_output_is_skipped_rather_than_parsed() -> None:
    client = _client(
        [
            "npm WARN something on stdout",
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"content": [{"type": "text", "text": "ok"}]},
                }
            ),
        ]
    )
    assert run(client.call_tool("reactome_species")) == "ok"


def test_a_jsonrpc_error_is_raised_not_returned() -> None:
    client = _client(
        [
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "error": {"code": -32601, "message": "no such tool"},
                }
            )
        ]
    )
    with pytest.raises(MCPToolError, match="no such tool"):
        run(client.call_tool("nope"))


def test_a_closed_connection_raises() -> None:
    client = _client([])
    with pytest.raises(MCPToolError, match="closed the connection"):
        run(client.call_tool("reactome_species"))


def test_a_tool_error_result_raises() -> None:
    """isError means the call failed, even though the transport succeeded."""
    client = _client(
        [
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "isError": True,
                        "content": [{"type": "text", "text": "pathway not found"}],
                    },
                }
            )
        ]
    )
    with pytest.raises(MCPToolError, match="pathway not found"):
        run(client.call_tool("reactome_get_pathway", {"id": "nope"}))


def test_a_result_of_the_wrong_shape_is_rejected() -> None:
    """Checked, not asserted -- the lesson from reactome-mcp's formatter bugs."""
    client = _client(
        [json.dumps({"jsonrpc": "2.0", "id": 1, "result": ["not", "an", "object"]})]
    )
    with pytest.raises(MCPToolError, match="expected an object"):
        run(client.call_tool("reactome_species"))


def test_initialize_handshake_is_sent_before_the_first_tool_call() -> None:
    """The protocol requires it. The server is lenient today; that is not a guarantee."""
    client = _client(
        [
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"serverInfo": {"name": "reactome", "version": "1.4.0"}},
                }
            ),
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "result": {"content": [{"type": "text", "text": "ok"}]},
                }
            ),
        ],
        initialized=False,
    )
    run(client.call_tool("reactome_species"))

    sent = cast(Any, client.process.stdin).written
    assert sent[0]["method"] == "initialize"
    assert sent[1]["method"] == "notifications/initialized"
    assert "id" not in sent[1]  # a notification carries no id
    assert sent[2]["method"] == "tools/call"


def test_concurrent_calls_do_not_interleave_on_the_pipe() -> None:
    client = _client(
        [
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"content": [{"type": "text", "text": "first"}]},
                }
            ),
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "result": {"content": [{"type": "text", "text": "second"}]},
                }
            ),
        ]
    )

    async def both() -> tuple[str, str]:
        first, second = await asyncio.gather(
            client.call_tool("reactome_species"),
            client.call_tool("reactome_database_info"),
        )
        return first, second

    a, b = run(both())
    assert {a, b} == {"first", "second"}


def test_a_missing_server_says_how_to_build_it() -> None:
    manager = MCPProcessManager("/nowhere/dist/index.js")
    with pytest.raises(MCPConnectionError, match="npm ci"):
        run(manager.start())


def test_stopping_a_server_that_never_started_is_harmless() -> None:
    run(MCPProcessManager("/nowhere/dist/index.js").stop())
