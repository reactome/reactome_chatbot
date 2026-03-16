import os
from typing import Any

from langchain_core.callbacks.base import Callbacks
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph, StateGraph
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from agent.models import get_embedding, get_llm
from agent.profiles import ProfileName, create_profile_graphs
from agent.profiles.base import InputState, OutputState
from util.logging import logging


def _build_db_uri() -> str | None:
    """Lazily construct the Postgres URI only when all required env vars are present."""
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    db = os.getenv("POSTGRES_LANGGRAPH_DB")
    if not all([user, password, db]):
        return None
    return f"postgresql://{user}:{password}@postgres:5432/{db}?sslmode=disable"


class AgentGraph:
    def __init__(
        self,
        profiles: list[ProfileName],
        llm_config: str = "openai/gpt-4o-mini",
        embedding_config: str = "openai/text-embedding-3-large",
    ) -> None:
        provider, _, model = llm_config.partition("/")
        emb_provider, _, emb_model = embedding_config.partition("/")

        llm: BaseChatModel = get_llm(provider, model)
        embedding: Embeddings = get_embedding(emb_provider, emb_model)

        self.uncompiled_graph: dict[str, StateGraph] = create_profile_graphs(
            profiles, llm, embedding
        )

        # Set asynchronously by initialize() / async context manager
        self.graph: dict[str, CompiledStateGraph] | None = None
        self.pool: AsyncConnectionPool[AsyncConnection[dict[str, Any]]] | None = None

    # ------------------------------------------------------------------ #
    # Async context manager — preferred lifecycle                          #
    # ------------------------------------------------------------------ #

    async def __aenter__(self) -> "AgentGraph":
        self.graph = await self.initialize()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.shutdown()

    # ------------------------------------------------------------------ #
    # Initialisation helpers                                               #
    # ------------------------------------------------------------------ #

    async def initialize(self) -> dict[str, CompiledStateGraph]:
        checkpointer: BaseCheckpointSaver[str] = await self.create_checkpointer()
        return {
            profile: graph.compile(checkpointer=checkpointer)
            for profile, graph in self.uncompiled_graph.items()
        }

    async def create_checkpointer(self) -> BaseCheckpointSaver[str]:
        uri = _build_db_uri()
        if uri is None:
            logging.warning("POSTGRES_LANGGRAPH_DB undefined; falling back to MemorySaver.")
            return MemorySaver()
        self.pool = AsyncConnectionPool(
            conninfo=uri,
            max_size=20,
            open=False,
            timeout=30,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
            },
        )
        await self.pool.open()
        checkpointer = AsyncPostgresSaver(self.pool)
        await checkpointer.setup()
        return checkpointer

    async def shutdown(self) -> None:
        """Explicit lifecycle teardown — call this instead of relying on __del__."""
        if self.pool:
            await self.pool.close()
            self.pool = None

    # ------------------------------------------------------------------ #
    # Invocation                                                           #
    # ------------------------------------------------------------------ #

    async def ainvoke(
        self,
        user_input: str,
        profile: str,
        *,
        callbacks: Callbacks,
        thread_id: str,
        enable_postprocess: bool = True,
    ) -> OutputState:
        if self.graph is None:
            self.graph = await self.initialize()
        if profile not in self.graph:
            return OutputState()
        result: OutputState = await self.graph[profile].ainvoke(
            InputState(user_input=user_input),
            config=RunnableConfig(
                callbacks=callbacks,
                configurable={
                    "thread_id": thread_id,
                    "enable_postprocess": enable_postprocess,
                },
            ),
        )
        return result
