import asyncio
import os
from typing import Annotated, Any, TypedDict

from langchain_core.callbacks.base import Callbacks
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph, StateGraph
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from agent.models import get_embedding, get_llm
from agent.tasks.rephrase import create_rephrase_chain
from retrievers.reactome.rag import create_reactome_rag
from tools.external_search.state import WebSearchResult
from tools.external_search.workflow import create_search_workflow
from util.logging import logging

LANGGRAPH_DB_URI = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@postgres:5432/{os.getenv('POSTGRES_LANGGRAPH_DB')}?sslmode=disable"

if not os.getenv("POSTGRES_LANGGRAPH_DB"):
    logging.warning("POSTGRES_LANGGRAPH_DB undefined; falling back to MemorySaver.")


class AdditionalContent(TypedDict):
    search_results: list[WebSearchResult]


class AgentState(TypedDict):
    user_input: str  # User input text
    rephrased_input: str  # LLM-generated query from user input
    chat_history: Annotated[list[BaseMessage], add_messages]
    context: list[Document]
    answer: str  # primary LLM response that is streamed to the user
    additional_content: AdditionalContent  # sends on graph completion


class AgentGraph:
    def __init__(self) -> None:
        # Get base models
        llm: BaseChatModel = get_llm("openai", "gpt-4o-mini")
        embedding: Embeddings = get_embedding("openai", "text-embedding-3-large")

        # Create runnables (tasks & tools)
        self.reactome_rag: Runnable = create_reactome_rag(
            llm, embedding, streaming=True
        )
        self.rephrase_chain: Runnable = create_rephrase_chain(llm)
        self.search_workflow: CompiledStateGraph = create_search_workflow(llm)

        # Create graph
        state_graph = StateGraph(AgentState)
        # Set up nodes
        state_graph.add_node("preprocess", self.preprocess)
        state_graph.add_node("model", self.call_model)
        state_graph.add_node("postprocess", self.postprocess)
        # Set up edges
        state_graph.set_entry_point("preprocess")
        state_graph.add_edge("preprocess", "model")
        state_graph.add_edge("model", "postprocess")
        state_graph.set_finish_point("postprocess")

        self.uncompiled_graph: StateGraph = state_graph

        # The following are set asynchronously by calling initialize()
        self.graph: CompiledStateGraph | None = None
        self.pool: AsyncConnectionPool[AsyncConnection[dict[str, Any]]] | None = None

    def __del__(self) -> None:
        if self.pool:
            asyncio.run(self.close_pool())

    async def initialize(self) -> CompiledStateGraph:
        checkpointer: BaseCheckpointSaver[str] = await self.create_checkpointer()
        return self.uncompiled_graph.compile(checkpointer=checkpointer)

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

    async def preprocess(
        self, state: AgentState, config: RunnableConfig
    ) -> dict[str, str]:
        query: str = await self.rephrase_chain.ainvoke(state, config)
        return {"rephrased_input": query}

    async def call_model(
        self, state: AgentState, config: RunnableConfig
    ) -> dict[str, Any]:
        result: dict[str, Any] = await self.reactome_rag.ainvoke(
            {
                "input": state["rephrased_input"],
                "user_input": state["user_input"],
                "chat_history": state["chat_history"],
            },
            config,
        )
        return {
            "chat_history": [
                HumanMessage(state["user_input"]),
                AIMessage(result["answer"]),
            ],
            "context": result["context"],
            "answer": result["answer"],
        }

    async def postprocess(
        self, state: AgentState, config: RunnableConfig
    ) -> dict[str, dict[str, list[WebSearchResult]]]:
        search_results: list[WebSearchResult] = []
        if config["configurable"]["enable_postprocess"]:
            result: dict[str, Any] = await self.search_workflow.ainvoke(
                {"question": state["rephrased_input"], "generation": state["answer"]},
                config=RunnableConfig(callbacks=config["callbacks"]),
            )
            search_results = result["search_results"]
        return {
            "additional_content": {"search_results": search_results},
        }

    async def ainvoke(
        self,
        user_input: str,
        *,
        callbacks: Callbacks,
        thread_id: str,
        enable_postprocess: bool = True,
    ) -> dict[str, Any]:
        if self.graph is None:
            self.graph = await self.initialize()
        result: dict[str, Any] = await self.graph.ainvoke(
            {"user_input": user_input},
            config=RunnableConfig(
                callbacks=callbacks,
                configurable={
                    "thread_id": thread_id,
                    "enable_postprocess": enable_postprocess,
                },
            ),
        )
        return result
