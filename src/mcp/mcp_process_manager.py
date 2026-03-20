import asyncio
from pathlib import Path


class MCPConnectionError(Exception):
    """Raised when MCP server fails to start or crashes."""
    pass


class MCPProcessManager:
    """
    Manages the lifecycle of the Reactome MCP server subprocess.

    Spawns a Node.js process communicating over stdio via JSON-RPC.
    Supports async context manager for automatic cleanup.

    Args:
        mcp_server_path: Path to compiled server entry point (reactome-mcp/dist/index.js).
                         Run 'npm run build' in reactome-mcp repo first.

    Raises:
        FileNotFoundError: If the server path does not exist.
    """

    def __init__(self, mcp_server_path: str):
        self.mcp_server_path = Path(mcp_server_path)
        if not self.mcp_server_path.exists():
            raise FileNotFoundError(
                f"MCP server not found at: {self.mcp_server_path}\n"
                f"Make sure reactome-mcp is cloned and built with 'npm run build'"
            )
        self.process = None

    async def start(self) -> asyncio.subprocess.Process:
        """
        Spawn the MCP server and verify it started successfully.

        Returns:
            The running asyncio subprocess instance.

        Raises:
            MCPConnectionError: If the process exits immediately after launch.
        """
        self.process = await asyncio.create_subprocess_exec(
            "node",
            str(self.mcp_server_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Allow server time to initialize before checking if it survived
        await asyncio.sleep(1)

        if self.process.returncode is not None:
            # Process already exited — read stderr to find out why
            stderr_output = await self.process.stderr.read()
            raise MCPConnectionError(
                f"MCP server failed to start:\n{stderr_output.decode('utf-8')}"
            )

        return self.process

    async def stop(self) -> None:
        """
        Shut down the server gracefully.

        Tries terminate first, falls back to force kill after 5 seconds.
        Safe to call if process was never started.
        """
        if not self.process:
            return

        try:
            self.process.terminate()
            await asyncio.wait_for(self.process.wait(), timeout=5.0)

        except asyncio.TimeoutError:
            self.process.kill()
            await self.process.wait()

        finally:
            self.process = None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
        return False