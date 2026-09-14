"""Lifecycle of the reactome-mcp server process.

The server is spawned as a subprocess and spoken to over stdio. That is a
deliberate choice for now rather than a permanent one: reactome-mcp also serves
Streamable HTTP, so a hosted instance can be talked to instead once there is
one. Keeping the transport behind this class is what makes that swap small.

Harvested from #127 by @GovindhKishore.
"""

import asyncio
import logging
from pathlib import Path
from types import TracebackType

logger = logging.getLogger(__name__)


class MCPConnectionError(RuntimeError):
    """The MCP server could not be started, or died."""


class MCPProcessManager:
    """Start and stop the MCP server, and own its process."""

    def __init__(self, server_path: str | Path) -> None:
        self.server_path = Path(server_path)
        self.process: asyncio.subprocess.Process | None = None

    async def start(self) -> asyncio.subprocess.Process:
        if not self.server_path.exists():
            raise MCPConnectionError(
                f"MCP server not found at {self.server_path}. Clone reactome-mcp "
                "and run `npm ci && npm run build`, then point "
                "REACTOME_MCP_SERVER at its dist/index.js."
            )

        logger.info("starting MCP server: node %s", self.server_path)
        self.process = await asyncio.create_subprocess_exec(
            "node",
            str(self.server_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            # The server logs to stderr and keeps stdout for JSON-RPC, so this
            # must not be merged into stdout or every log line corrupts a
            # response.
            stderr=asyncio.subprocess.PIPE,
        )
        return self.process

    async def stop(self) -> None:
        process, self.process = self.process, None
        if process is None or process.returncode is not None:
            return

        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except TimeoutError:
            logger.warning("MCP server did not terminate in 5s; killing it")
            process.kill()
            await process.wait()

    async def stderr_tail(self, limit: int = 2000) -> str:
        """Whatever the server complained about, for an error message.

        Read without blocking: if the process is alive and silent, there is
        nothing to read and waiting for some would hang the caller.
        """
        if self.process is None or self.process.stderr is None:
            return ""
        try:
            data = await asyncio.wait_for(self.process.stderr.read(limit), timeout=1.0)
        # Best-effort: this runs while reporting another failure, and must not
        # replace it with one of its own.
        except Exception:
            return ""
        return data.decode("utf-8", errors="replace").strip()

    async def __aenter__(self) -> "MCPProcessManager":
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        await self.stop()
        return False
