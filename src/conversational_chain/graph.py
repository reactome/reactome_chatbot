import os
from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.callbacks.base import Callbacks
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph, StateGraph

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from psycopg import AsyncConnection

from conversational_chain.chain import RAGChainWithMemory

DB_URI = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@postgres:5432/{os.getenv('POSTGRES_DB')}?sslmode=disable"

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

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


    async def initialize(self) -> None:
        """Initialize the connection pool asynchronously."""
        await self.create_conn_pool()


    async def create_conn_pool(self) -> None:
        self.pool = AsyncConnectionPool(
            conninfo=DB_URI,
            max_size=20,
            timeout=30,
            kwargs=connection_kwargs,
        )
        checkpointer = AsyncPostgresSaver(self.pool)
        await checkpointer.setup()
        self.graph = self.uncompiled_graph.compile(checkpointer=checkpointer)


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
        self, user_input: str, callbacks: Callbacks,
        configurable: dict[str, Any]
    ) -> str:
        if self.graph is not None:
            response: dict[str, Any] = await self.graph.ainvoke(
                {"input": user_input},
                config = RunnableConfig(
                    callbacks = callbacks,
                    configurable = configurable,
                )
            )
        return response["answer"]


    def __del__(self) -> None:
        """Close the connection pool when the object is destroyed."""
        if self.pool is not None:
            await self.pool.close()
