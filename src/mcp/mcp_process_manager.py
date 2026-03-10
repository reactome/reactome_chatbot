import asyncio
from pathlib import Path


class MCPConnectionError(Exception):
    """Raised when MCP server fails to start or crashes."""
    pass


class MCPProcessManager:
    """Manages lifecycle of the Reactome MCP server process."""

    def __init__(self, mcp_server_path: str):
        self.mcp_server_path = Path(mcp_server_path)
        if not self.mcp_server_path.exists():
            raise FileNotFoundError(
                f"MCP server not found at: {self.mcp_server_path}\n"
                f"Make sure reactome-mcp is cloned and built with 'npm run build'"
            )
        self.process = None

    async def start(self) -> asyncio.subprocess.Process:
        """Start the MCP server process."""
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
        """Stop the MCP server — graceful terminate, falls back to kill."""
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