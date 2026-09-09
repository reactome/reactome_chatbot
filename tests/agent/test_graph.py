"""AgentGraph construction and teardown."""

import pytest

pytest.importorskip("langgraph", reason="agent stack not installed")

from agent.graph import AgentGraph  # noqa: E402


def test_agent_graph_del_survives_a_failed_init() -> None:
    """__del__ must not raise when __init__ never finished.

    A chromadb error during __init__ left `pool` unset, and __del__ then raised
    `AttributeError: 'AgentGraph' object has no attribute 'pool'` -- which is
    what got printed, while the error that actually stopped startup did not.
    """
    orphan = AgentGraph.__new__(AgentGraph)  # __init__ deliberately not run
    orphan.__del__()  # must not raise
