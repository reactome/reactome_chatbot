"""One graph, shared by every surface, and not built at import time.

`bin/chat-chainlit.py` used to build the graph at module scope. `mount_chainlit`
imports that module, so importing the FastAPI app constructed the graph and every
BM25 index before the module finished loading: **85 seconds**, measured on
2026-09-17. That is why the container needs a three-minute startup wait, and it is
why nothing could load the app in a test.

Spec 010 adds a second surface, and two graphs would pay that twice and could answer
the same question differently -- which SC-003 calls a defect.
"""

from typing import cast

import pytest

from agent import registry
from agent.graph import AgentGraph


class _FakeGraph:
    """A stand-in. Constructing a real AgentGraph takes about 85 seconds."""

    def __init__(self, tag: str = "one") -> None:
        self.tag = tag


def _fake(tag: str = "one") -> AgentGraph:
    # cast rather than a loosened comparison: the tests below assert object
    # identity, and mypy's strict_equality is right that the two types do not
    # overlap. The substitution is the lie, so it is named here once.
    return cast(AgentGraph, _FakeGraph(tag))


@pytest.fixture(autouse=True)
def _clean() -> None:
    registry.reset_for_tests()


def test_get_graph_returns_what_startup_installed() -> None:
    graph = _fake()
    registry.set_graph(graph)
    assert registry.get_graph() is graph


def test_the_same_instance_is_returned_every_time() -> None:
    """Two surfaces asking must get one object, or they can answer differently."""
    registry.set_graph(_fake())
    assert registry.get_graph() is registry.get_graph()


def test_it_builds_on_demand_when_startup_did_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`chainlit run bin/chat-chainlit.py` is a documented dev path with no lifespan.

    Without this fallback that path would fail with an unset graph rather than
    simply being slower.
    """
    built: list[int] = []

    def _build() -> AgentGraph:
        built.append(1)
        return _fake("built-on-demand")

    monkeypatch.setattr(registry, "build_graph", _build)
    first = registry.get_graph()
    second = registry.get_graph()

    assert first is second
    assert len(built) == 1, "the expensive build must happen once, not per call"


def test_a_later_set_graph_replaces_an_on_demand_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry, "build_graph", lambda: _fake("lazy"))
    registry.get_graph()
    installed = _fake("startup")
    registry.set_graph(installed)
    assert registry.get_graph() is installed
