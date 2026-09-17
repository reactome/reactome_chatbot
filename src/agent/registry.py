"""The one AgentGraph, shared by every surface that answers questions.

Two things made this necessary.

`bin/chat-chainlit.py` built the graph at module scope, and `mount_chainlit`
imports that module, so importing the FastAPI app constructed the graph and every
BM25 index: **85 seconds**, measured. That is why the container needs a three-minute
startup wait, and it makes any test that loads the app impractical.

And spec 010 adds a second surface. Two graphs would pay that cost twice and, worse,
could answer the same question differently -- which SC-003 calls a defect rather than
a feature.

So: one instance, built once, reachable from both. The FastAPI lifespan warms it so
no user waits for it; `get_graph()` builds on demand if nothing did, because
`chainlit run bin/chat-chainlit.py` is a documented dev path with no lifespan.
"""

import logging
import threading

from agent.graph import AgentGraph
from agent.profile_names import ProfileName
from util.config_yml import Config

logger = logging.getLogger(__name__)

_graph: AgentGraph | None = None
# Construction is synchronous and slow. Two concurrent first requests would
# otherwise each build one, and the loser's copy would be silently discarded
# after paying the full cost.
_lock = threading.Lock()


def set_graph(graph: AgentGraph) -> None:
    """Install the graph built at startup."""
    global _graph
    with _lock:
        _graph = graph


def get_graph() -> AgentGraph:
    """The shared graph, building it if startup did not."""
    global _graph
    if _graph is not None:
        return _graph
    with _lock:
        if _graph is None:
            logger.info("Building the agent graph on demand; no startup hook set it.")
            _graph = build_graph()
    return _graph


def build_graph() -> AgentGraph:
    """Construct a graph from config. Slow: about 85 seconds, mostly BM25."""
    config: Config | None = Config.from_yaml()
    profiles: list[ProfileName] = (
        config.profiles if config else [ProfileName.React_to_Me]
    )
    return AgentGraph(profiles, llm_config=config.llm if config else None)


def reset_for_tests() -> None:
    """Drop the shared graph. Tests only; nothing in the app calls this."""
    global _graph
    with _lock:
        _graph = None
