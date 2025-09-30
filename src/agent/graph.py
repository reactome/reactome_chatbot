import asyncio
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

LANGGRAPH_DB_URI = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@postgres:5432/{os.getenv('POSTGRES_LANGGRAPH_DB')}?sslmode=disable"

if not os.getenv("POSTGRES_LANGGRAPH_DB"):
    logging.warning("POSTGRES_LANGGRAPH_DB undefined; falling back to MemorySaver.")


class AgentGraph:
    def __init__(
        self,
        profiles: list[ProfileName],
    ) -> None:
        # Get base models
        llm: BaseChatModel = get_llm("openai", "gpt-4o-mini")
        embedding: Embeddings = get_embedding("openai", "text-embedding-3-large")

        self.uncompiled_graph: dict[str, StateGraph] = create_profile_graphs(
            profiles, llm, embedding
        )

        # The following are set asynchronously by calling initialize()
        self.graph: dict[str, CompiledStateGraph] | None = None
        self.pool: AsyncConnectionPool[AsyncConnection[dict[str, Any]]] | None = None

    def __del__(self) -> None:
        if self.pool:
            asyncio.run(self.close_pool())

    async def initialize(self) -> dict[str, CompiledStateGraph]:
        checkpointer: BaseCheckpointSaver[str] = await self.create_checkpointer()
        return {
            profile: graph.compile(checkpointer=checkpointer)
            for profile, graph in self.uncompiled_graph.items()
        }

    async def create_checkpointer(self) -> BaseCheckpointSaver[str]:
        if not os.getenv("POSTGRES_LANGGRAPH_DB"):
            return MemorySaver()
        self.pool = AsyncConnectionPool(
            conninfo=LANGGRAPH_DB_URI,
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

    async def close_pool(self) -> None:
        if self.pool:
            await self.pool.close()

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