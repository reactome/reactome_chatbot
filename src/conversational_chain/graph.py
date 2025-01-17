import asyncio
import os
from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.callbacks.base import Callbacks
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph, StateGraph
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from conversational_chain.chain import create_rag_chain
from external_search.workflow import create_search_workflow
from util.logging import logging

LANGGRAPH_DB_URI = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@postgres:5432/{os.getenv('POSTGRES_LANGGRAPH_DB')}?sslmode=disable"

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

if not os.getenv("POSTGRES_LANGGRAPH_DB"):
    logging.warning("POSTGRES_LANGGRAPH_DB undefined; falling back to MemorySaver.")


class ChatState(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: list[Document]
    answer: str  # primary LLM response that is streamed to the user
    additional_text: str  # additional text to send after graph completes


class RAGGraphWithMemory:
    def __init__(self, retriever: BaseRetriever, llm: BaseChatModel) -> None:
        # Set up runnables
        self.rag_chain: Runnable = create_rag_chain(llm, retriever)
        self.search_workflow: CompiledStateGraph = create_search_workflow(llm)

        # Create graph
        state_graph: StateGraph = StateGraph(ChatState)
        # Set up nodes
        state_graph.add_node("model", self.call_model)
        state_graph.add_node("postprocess", self.postprocess)
        # Set up edges
        state_graph.set_entry_point("model")
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
            kwargs=connection_kwargs,
        )
        await self.pool.open()
        checkpointer = AsyncPostgresSaver(self.pool)
        await checkpointer.setup()
        return checkpointer

    async def close_pool(self) -> None:
        if self.pool:
            await self.pool.close()

    async def call_model(
        self, state: ChatState, config: RunnableConfig
    ) -> dict[str, Any]:
        result = await self.rag_chain.ainvoke(state, config)
        return {
            "chat_history": [
                HumanMessage(state["input"]),
                AIMessage(result["answer"]),
            ],
            "context": result["context"],
            "answer": result["answer"],
        }

    async def postprocess(
        self, state: ChatState, config: RunnableConfig
    ) -> dict[str, str]:
        additional_text: str = ""
        if config["configurable"]["enable_postprocess"]:
            result: dict[str, Any] = await self.search_workflow.ainvoke({
                "question": state["input"],
                "generation": state["answer"]
            })
            additional_text = result["search_results"]
        return {
            "additional_text": additional_text,
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
            {"input": user_input},
            config=RunnableConfig(
                callbacks=callbacks,
                configurable={
                    "thread_id": thread_id,
                    "enable_postprocess": enable_postprocess
                },
            ),
        )
        return result
