from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.graph.message import add_messages

from conversational_chain.chain import RAGChainWithMemory


class ChatResponse(TypedDict):
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str


class ChatState(ChatResponse):
    input: str


class RAGGraphWithMemory(RAGChainWithMemory):
    def __init__(self, **chain_kwargs):
        super().__init__(**chain_kwargs)

        # Single-node graph (for now)
        graph: StateGraph = StateGraph(ChatState)
        graph.add_node("model", self.call_model)
        graph.set_entry_point("model")
        graph.set_finish_point("model")

        memory = MemorySaver()
        self.graph: CompiledStateGraph = graph.compile(checkpointer=memory)

    async def call_model(self, state: ChatState, config: RunnableConfig) -> ChatResponse:
        response = await self.rag_chain.ainvoke(state, config)
        return {
            "chat_history": [
                HumanMessage(state["input"]),
                AIMessage(response["answer"]),
            ],
            "context": response["context"],
            "answer": response["answer"],
        }

    async def ainvoke(self, user_input: str, **kwargs) -> str:
        response: ChatResponse = await self.graph.ainvoke(
            {"input": user_input},
            config = RunnableConfig(**kwargs)
        )
        return response["answer"]
