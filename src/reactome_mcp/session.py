"""One MCP server per process, started on first use.

The alternative was to start it in `AgentGraph.initialize()` and stop it in
`close_pool()`. That is tidier in principle and worse in practice: the graph
builders are constructed synchronously in `__init__`, so the tools would have
to be attached to an already-built graph, and every caller that builds a graph
outside the app -- the evaluator, `bin/retrieval_baseline`, tests -- would need
teaching about a lifecycle it does not care about.

Lazy instead. The first question that needs the MCP pays about a second for the
handshake; every later one is free. The subprocess is terminated at interpreter
exit.

Absent `REACTOME_MCP_SERVER`, everything here returns None and the chatbot
behaves exactly as it did before -- which is the point, not a fallback.
"""

import asyncio
import atexit
import contextlib
import logging
import os
import time
from pathlib import Path

from langchain_core.tools import BaseTool

from reactome_mcp.client import MCPClient
from reactome_mcp.http_client import MCPHttpClient
from reactome_mcp.process import MCPConnectionError, MCPProcessManager
from reactome_mcp.tools import ToolCaller, create_mcp_tools

logger = logging.getLogger(__name__)

_lock = asyncio.Lock()
_manager: MCPProcessManager | None = None
_http: MCPHttpClient | None = None
_tools: list[BaseTool] | None = None
#: When starting the MCP last failed. Retried after `RETRY_AFTER_SECONDS`:
#: remembered for the life of the process, one failed first connect -- the
#: MCP not up yet after a deploy -- turned live lookups off until the
#: chatbot restarted, while the router kept sending questions to them
#: (review, area 3).
_failed_at: float | None = None
RETRY_AFTER_SECONDS = 60.0


def mcp_server_path() -> Path | None:
    """A local server to spawn, if one is configured."""
    configured = os.getenv("REACTOME_MCP_SERVER")
    return Path(configured) if configured else None


def mcp_server_url() -> str | None:
    """A running server to connect to, if one is configured."""
    return os.getenv("REACTOME_MCP_URL") or None


def is_configured() -> bool:
    return mcp_server_url() is not None or mcp_server_path() is not None


def _recently_failed() -> bool:
    return (
        _failed_at is not None and time.monotonic() - _failed_at < RETRY_AFTER_SECONDS
    )


async def get_mcp_tools() -> list[BaseTool] | None:
    """The MCP tools, starting the server if it is not already running.

    Returns None when no server is configured, and also when starting one
    failed. A failure is logged once and remembered: retrying the spawn on
    every question would turn one misconfiguration into a stall on every
    request.
    """
    global _manager, _http, _tools, _failed_at

    if _tools is not None:
        return _tools
    if not is_configured() or _recently_failed():
        return None

    async with _lock:
        # Another coroutine may have finished while this one waited.
        if _tools is not None:
            return _tools
        if _recently_failed():
            return None

        url = mcp_server_url()
        server_path = mcp_server_path()
        # HTTP first. The deployed image is Python with no node and does not
        # mount reactome-mcp, so spawning one cannot work there -- stdio is for
        # a developer's machine, where the repo and node both exist.
        where = url or str(server_path)
        try:
            client: ToolCaller
            if url is not None:
                http = MCPHttpClient(url)
                # The handshake is the readiness check: if this returns, the
                # server is up and answering.
                await http.initialize()
                _http, client = http, http
            elif server_path is not None:
                manager = MCPProcessManager(server_path)
                await manager.start()
                if manager.process is None:
                    raise MCPConnectionError("server did not start")
                stdio = MCPClient(manager.process)
                await stdio.initialize()
                _manager, client = manager, stdio
            else:
                return None
            _tools = create_mcp_tools(client)
        except Exception as exc:
            _failed_at = time.monotonic()
            stderr = ""
            with contextlib.suppress(Exception):
                # Best effort: this runs while reporting another failure and
                # must not replace it with one of its own.
                if _manager is not None:
                    stderr = await _manager.stderr_tail()
            logger.warning(
                "MCP server unavailable (%s); live Reactome tools are off for "
                "%.0fs, then retried. Checked %s.%s",
                exc,
                RETRY_AFTER_SECONDS,
                where,
                f" Server said: {stderr}" if stderr else "",
            )
            return None

        logger.info("MCP server ready; %d live Reactome tools available", len(_tools))
        return _tools


async def shutdown() -> None:
    global _manager, _http, _tools
    if _manager is not None:
        await _manager.stop()
    if _http is not None:
        await _http.aclose()
    _manager, _http, _tools = None, None, None


def _terminate_at_exit() -> None:
    """Stop the subprocess when the interpreter exits.

    Without this a reload or a crash leaves an orphaned node process holding a
    pipe nobody reads.
    """
    if _manager is None or _manager.process is None:
        return
    # The interpreter is going away; there is nowhere useful to report a
    # failure to terminate, and raising here would mask the real exit.
    with contextlib.suppress(Exception):
        _manager.process.terminate()


atexit.register(_terminate_at_exit)
