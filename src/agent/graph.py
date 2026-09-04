import asyncio
import os
from typing import Any, cast

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
from util.embedding_environment import EmbeddingEnvironment
from util.logging import logging

LANGGRAPH_DB_URI = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@postgres:5432/{os.getenv('POSTGRES_LANGGRAPH_DB')}?sslmode=disable"

if not os.getenv("POSTGRES_LANGGRAPH_DB"):
    logging.warning("POSTGRES_LANGGRAPH_DB undefined; falling back to MemorySaver.")


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"


def resolve_embedding_model() -> str:
    """Pick the embedding model, defaulting to whatever built the installed bundle.

    A query is embedded with this model and compared against vectors produced by
    whichever model built the bundle. If they differ the comparison is
    meaningless, so the bundle is the right source of truth rather than a
    constant that has to be kept in sync by hand.

    The default used to be a literal "bge-m3", which OpenAI has no such model
    for -- so any deployment that did not set EMBEDDING_MODEL got a 404 on its
    first query.
    """
    configured = os.getenv("EMBEDDING_MODEL")
    try:
        installed = EmbeddingEnvironment.get_model("reactome")
    except KeyError:
        # get_model raises when the database is not installed, unlike get_dir
        # which returns None for the same condition.
        return configured or DEFAULT_EMBEDDING_MODEL

    # get_model returns "<provider>/<model>"; the provider is supplied separately.
    bundle_model = installed.split("/", 1)[-1]
    if configured and configured != bundle_model:
        logging.error(
            f"EMBEDDING_MODEL is {configured!r} but the installed bundle was built "
            f"with {bundle_model!r}. Queries will be embedded with a different "
            "model than the stored vectors, so retrieval results will be "
            "meaningless. Unset EMBEDDING_MODEL, or install a matching bundle."
        )
    return configured or bundle_model


class AgentGraph:
    def __init__(
        self,
        profiles: list[ProfileName],
    ) -> None:
        # Get base models
        embedding_model = resolve_embedding_model()
        llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        llm_base_url = os.getenv("LLM_BASE_URL", None)
        llm: BaseChatModel = get_llm(
            "openai", llm_model, base_url=llm_base_url, request_timeout=360.0
        )
        embedding_base_url = os.getenv("OPENAI_BASE_URL", None)
        embedding: Embeddings = get_embedding(
            "openai", embedding_model, base_url=embedding_base_url
        )

        self.uncompiled_graph: dict[str, StateGraph] = create_profile_graphs(
            profiles, llm, embedding
        )

        # The following are set asynchronously by calling initialize()
        self.graph: dict[str, CompiledStateGraph] | None = None
        self.pool: AsyncConnectionPool[AsyncConnection[dict[str, Any]]] | None = None

    def __del__(self) -> None:
        """Close the connection pool if nothing else did.

        This used to call asyncio.run() unconditionally, which raises
        RuntimeError when a loop is already running -- and __del__ can fire at
        any point, including inside the running server. Exceptions in __del__ are
        swallowed and printed, so it surfaced as noise in production logs with the
        pool still open.

        Scheduling the close with loop.create_task() is not a fix either: the
        task is not guaranteed to run if the loop is shutting down, which is
        exactly when a graph is usually collected.

        Nothing calls close_pool() explicitly today, so this is the only cleanup
        there is. The real fix is an explicit lifecycle -- close the pool from the
        application's shutdown hook -- which belongs with the agent-API work.
        """
        if self.pool is None:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No loop running, so we can drive the close to completion.
            try:
                asyncio.run(self.close_pool())
            except Exception as e:
                logging.warning(f"Could not close the connection pool: {e}")
            return
        logging.warning(
            "AgentGraph was garbage-collected while an event loop is running; "
            "its Postgres pool is still open. Close it explicitly from the "
            "application shutdown hook."
        )

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
        # ainvoke is typed dict[str, Any] | Any; the graph's output schema is
        # OutputState.
        result: OutputState = cast(
            "OutputState",
            await self.graph[profile].ainvoke(
                InputState(user_input=user_input),
                config=RunnableConfig(
                    callbacks=callbacks,
                    configurable={
                        "thread_id": thread_id,
                        "enable_postprocess": enable_postprocess,
                    },
                ),
            ),
        )
        return result
