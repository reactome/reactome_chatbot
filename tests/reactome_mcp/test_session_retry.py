"""A failed MCP start is retried, not remembered for the life of the process."""

import asyncio
from typing import Any

import pytest

from reactome_mcp import session


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session, "_tools", None)
    monkeypatch.setattr(session, "_failed_at", None)
    monkeypatch.setenv("REACTOME_MCP_URL", "http://mcp.test")
    monkeypatch.delenv("REACTOME_MCP_SERVER", raising=False)


def test_a_failed_start_is_retried_after_a_while(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[int] = []
    up = {"now": False}

    class _Client:
        def __init__(self, _url: str) -> None:
            pass

        async def initialize(self) -> None:
            attempts.append(1)
            if not up["now"]:
                raise ConnectionError("not up yet")

    monkeypatch.setattr(session, "MCPHttpClient", _Client)
    monkeypatch.setattr(session, "create_mcp_tools", lambda _c: [object()])
    clock = {"t": 1000.0}
    monkeypatch.setattr("reactome_mcp.session.time.monotonic", lambda: clock["t"])

    async def get() -> Any:
        return await session.get_mcp_tools()

    assert asyncio.run(get()) is None  # the MCP is not up yet after a deploy
    assert asyncio.run(get()) is None  # not hammered on every question
    assert len(attempts) == 1
    up["now"] = True
    clock["t"] += session.RETRY_AFTER_SECONDS + 1
    assert asyncio.run(get()) is not None  # and back, without a restart
    assert len(attempts) == 2
