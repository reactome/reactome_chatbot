"""
Agent graph module with proper async lifecycle management.

This module provides the AgentGraph class for managing LangGraph agents
with PostgreSQL checkpointing and connection pool management.
"""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, ClassVar

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


def _build_langgraph_db_uri() -> str | None:
    """
    Build the PostgreSQL connection URI lazily.
    
    Returns None if required environment variables are not set,
    avoiding construction of invalid URIs with 'None' values.
    """
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    db = os.getenv("POSTGRES_LANGGRAPH_DB")
    
    if not all([user, password, db]):
        logging.warning(
            "PostgreSQL environment variables not fully configured. "
            "POSTGRES_USER, POSTGRES_PASSWORD, and POSTGRES_LANGGRAPH_DB are required. "
            "Falling back to MemorySaver."
        )
        return None
    
    return f"postgresql://{user}:{password}@postgres:5432/{db}?sslmode=disable"


class AgentGraph:
    """
    LangGraph agent with PostgreSQL checkpointing and proper resource management.
    
    This class manages the lifecycle of LangGraph agents with optional PostgreSQL
    checkpointing. It supports both async context manager usage (recommended) and
    explicit initialization/cleanup.
    
    Usage (recommended - async context manager):
    ```python
    async with AgentGraph(profiles=["cross_database"]) as graph:
        result = await graph.ainvoke(user_input, profile, ...)
    ```
    
    Usage (explicit):
    ```python
    graph = AgentGraph(profiles=["cross_database"])
    try:
        await graph.initialize()
        result = await graph.ainvoke(user_input, profile, ...)
    finally:
        await graph.close()
    ```
    """
    
    # Class-level tracking for debugging
    _instance_count: ClassVar[int] = 0
    
    def __init__(
        self,
        profiles: list[ProfileName],
    ) -> None:
        """
        Initialize the agent graph.
        
        Args:
            profiles: List of profile names to create graphs for.
        """
        AgentGraph._instance_count += 1
        self._instance_id = AgentGraph._instance_count
        
        # Get base models
        llm: BaseChatModel = get_llm("openai", "gpt-4o-mini")
        embedding: Embeddings = get_embedding("openai", "text-embedding-3-large")

        self.uncompiled_graph: dict[str, StateGraph] = create_profile_graphs(
            profiles, llm, embedding
        )

        # The following are set asynchronously by calling initialize()
        self.graph: dict[str, CompiledStateGraph] | None = None
        self.pool: AsyncConnectionPool[AsyncConnection[dict[str, Any]]] | None = None
        self._initialized: bool = False
        self._closed: bool = False
        
        logging.debug(f"AgentGraph instance {self._instance_id} created")

    
    # REMOVED: __del__ with asyncio.run() — this was causing production crashes!
    # 
    # def __del__(self) -> None:
    #     if self.pool:
    #         asyncio.run(self.close_pool())  # RuntimeError in async context!
    

    async def __aenter__(self) -> "AgentGraph":
        """Async context manager entry — initializes the graph."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit — closes resources."""
        await self.close()

    async def initialize(self) -> dict[str, CompiledStateGraph]:
        """
        Initialize the graph with checkpointing.
        
        This method is idempotent — calling it multiple times is safe.
        
        Returns:
            Dict mapping profile names to compiled state graphs.
        """
        if self._initialized:
            return self.graph or {}
        
        if self._closed:
            raise RuntimeError("Cannot initialize a closed AgentGraph")
        
        checkpointer: BaseCheckpointSaver[str] = await self._create_checkpointer()
        self.graph = {
            profile: graph.compile(checkpointer=checkpointer)
            for profile, graph in self.uncompiled_graph.items()
        }
        self._initialized = True
        
        logging.debug(f"AgentGraph instance {self._instance_id} initialized")
        return self.graph

    async def _create_checkpointer(self) -> BaseCheckpointSaver[str]:
        """
        Create the appropriate checkpointer based on environment configuration.
        
        Returns:
            MemorySaver if PostgreSQL is not configured, otherwise AsyncPostgresSaver.
        """
        db_uri = _build_langgraph_db_uri()
        
        if db_uri is None:
            return MemorySaver()
        
        self.pool = AsyncConnectionPool(
            conninfo=db_uri,
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
        
        logging.debug(f"PostgreSQL checkpointer created for instance {self._instance_id}")
        return checkpointer

    async def close(self) -> None:
        """
        Close the connection pool and release resources.
        
        This method is idempotent — calling it multiple times is safe.
        """
        if self._closed:
            return
        
        if self.pool:
            try:
                await self.pool.close()
                logging.debug(f"Connection pool closed for instance {self._instance_id}")
            except Exception as e:
                logging.warning(f"Error closing connection pool: {e}")
            finally:
                self.pool = None
        
        self._closed = True
        logging.debug(f"AgentGraph instance {self._instance_id} closed")

    # Backwards compatibility alias
    async def close_pool(self) -> None:
        """Alias for close() — kept for backwards compatibility."""
        await self.close()

    @property
    def is_initialized(self) -> bool:
        """Check if the graph has been initialized."""
        return self._initialized

    @property
    def is_closed(self) -> bool:
        """Check if the graph has been closed."""
        return self._closed

    async def ainvoke(
        self,
        user_input: str,
        profile: str,
        *,
        callbacks: Callbacks,
        thread_id: str,
        enable_postprocess: bool = True,
    ) -> OutputState:
        """
        Invoke the agent with a user query.
        
        Args:
            user_input: The user's query text.
            profile: Which agent profile to use.
            callbacks: LangChain callbacks for streaming/logging.
            thread_id: Unique identifier for the conversation thread.
            enable_postprocess: Whether to run postprocessing (web search fallback).
            
        Returns:
            OutputState containing the agent's response and metadata.
            
        Raises:
            RuntimeError: If the graph is closed or not initialized.
        """
        if self._closed:
            raise RuntimeError("Cannot invoke on a closed AgentGraph")
        
        if self.graph is None:
            self.graph = await self.initialize()
        
        if profile not in self.graph:
            logging.warning(f"Profile '{profile}' not found, returning empty response")
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


@asynccontextmanager
async def create_agent_graph(
    profiles: list[ProfileName],
) -> AsyncIterator[AgentGraph]:
    """
    Factory function to create an AgentGraph with automatic resource management.
    
    This is the recommended way to create and use an AgentGraph:
    
    ```python
    async with create_agent_graph(["cross_database"]) as graph:
        result = await graph.ainvoke(...)
    ```
    
    Args:
        profiles: List of profile names to create graphs for.
        
    Yields:
        An initialized AgentGraph instance.
    """
    graph = AgentGraph(profiles=profiles)
    try:
        await graph.initialize()
        yield graph
    finally:
        await graph.close()