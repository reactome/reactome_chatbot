import asyncio
import os
from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.callbacks.base import Callbacks
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph, StateGraph
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from conversational_chain.chain import RAGChainWithMemory
from util.logging import logging

LANGGRAPH_DB_URI = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@postgres:5432/{os.getenv('POSTGRES_LANGGRAPH_DB')}?sslmode=disable"

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

if not os.getenv("POSTGRES_LANGGRAPH_DB"):
    logging.warning("POSTGRES_LANGGRAPH_DB undefined; falling back to MemorySaver.")


class ChatResponse(TypedDict):
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str


class ChatState(ChatResponse):
    input: str


class RAGGraphWithMemory(RAGChainWithMemory):
    def __init__(self, **chain_kwargs) -> None:
        super().__init__(**chain_kwargs)
        state_graph: StateGraph = StateGraph(ChatState)
        state_graph.add_node("model", self.call_model)
        state_graph.set_entry_point("model")
        state_graph.set_finish_point("model")
        self.uncompiled_graph: StateGraph = state_graph
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
    ) -> ChatResponse:
        response = await self.rag_chain.ainvoke(state, config)
        return {
            "chat_history": [
                HumanMessage(state["input"]),
                AIMessage(response["answer"]),
            ],
            "context": response["context"],
            "answer": response["answer"],
        }

    async def ainvoke(
        self, user_input: str, callbacks: Callbacks, thread_id: str
    ) -> str:
        if self.graph is None:
            self.graph = await self.initialize()
        response: dict[str, Any] = await self.graph.ainvoke(
            {"input": user_input},
            config=RunnableConfig(
                callbacks=callbacks,
                configurable={"thread_id": thread_id},
            ),
        )
        return response["answer"]
