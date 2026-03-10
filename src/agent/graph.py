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
from agent.profiles import ProfileName
from agent.profiles.base import InputState, OutputState
from agent.profiles.cross_database import create_cross_database_graph
from agent.profiles.react_to_me import create_reactome_graph
from mcp.mcp_tools import create_mcp_tools
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

        self.llm = llm
        self.embedding = embedding
        self.profiles = profiles

        # The following are set asynchronously by calling initialize()
        self.graph: dict[str, CompiledStateGraph] | None = None
        self.pool: AsyncConnectionPool[AsyncConnection[dict[str, Any]]] | None = None

    def __del__(self) -> None:
        if self.pool:
            asyncio.run(self.close_pool())

    async def initialize(self) -> dict[str, CompiledStateGraph]:

        mcp_tools, self.mcp_manager = await create_mcp_tools(
            os.getenv("MCP_SERVER_PATH")
        )

        uncompiled_graphs: dict[str, StateGraph] = {}
        for profile in map(str.lower, self.profiles):
            if profile == ProfileName.React_to_Me.lower():
                uncompiled_graphs[profile] = create_reactome_graph(
                    self.llm, self.embedding, mcp_tools
                )
            elif profile == ProfileName.Cross_Database_Prototype.lower():
                uncompiled_graphs[profile] = create_cross_database_graph(
                    self.llm, self.embedding
                )

        checkpointer: BaseCheckpointSaver[str] = await self.create_checkpointer()

        return {
            profile: graph.compile(checkpointer=checkpointer)
            for profile, graph in uncompiled_graphs.items()
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
        if self.mcp_manager:
            await self.mcp_manager.stop()

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
