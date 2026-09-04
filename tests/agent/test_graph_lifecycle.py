"""AgentGraph.__del__ must never raise, whatever the loop state.

It used to call asyncio.run() unconditionally. That raises RuntimeError when a
loop is already running, and __del__ can fire at any moment -- including inside
the running server. Reported independently in PRs #147 and #156.
"""

import asyncio

from agent.graph import AgentGraph


class _FakePool:
    """Stands in for AsyncConnectionPool; records whether it was closed."""

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _graph_with_pool() -> tuple[AgentGraph, _FakePool]:
    """Build an AgentGraph without running __init__, which would need an LLM."""
    graph = AgentGraph.__new__(AgentGraph)
    pool = _FakePool()
    graph.pool = pool  # type: ignore[assignment]
    graph.graph = None
    return graph, pool


def test_del_closes_the_pool_when_no_loop_is_running() -> None:
    graph, pool = _graph_with_pool()
    graph.__del__()
    assert pool.closed is True


def test_del_does_not_raise_inside_a_running_loop() -> None:
    """The regression: asyncio.run() here raised RuntimeError."""

    async def collect() -> None:
        graph, pool = _graph_with_pool()
        graph.__del__()  # must not raise
        # The pool is deliberately left open rather than closed unreliably;
        # scheduling from __del__ is not guaranteed to run.
        assert pool.closed is False

    asyncio.run(collect())


def test_del_is_a_no_op_without_a_pool() -> None:
    graph = AgentGraph.__new__(AgentGraph)
    graph.pool = None
    graph.__del__()


def test_del_survives_a_failing_close() -> None:
    """__del__ must swallow errors; exceptions raised there are printed, not raised."""

    class _Boom(_FakePool):
        async def close(self) -> None:
            raise RuntimeError("pool already gone")

    graph = AgentGraph.__new__(AgentGraph)
    graph.pool = _Boom()  # type: ignore[assignment]
    graph.__del__()
    # Detach the pool so the object does not retry at interpreter shutdown, when
    # logging's handlers are closed and the warning itself fails. That failure
    # mode is the reason __del__ is the wrong place for this work at all.
    graph.pool = None
