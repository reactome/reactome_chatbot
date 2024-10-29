from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import START, CompiledStateGraph, StateGraph
from langgraph.graph.message import add_messages

from conversational_chain.chain import RAGChainWithMemory


class ChatResponse(TypedDict):
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str


class ChatState(ChatResponse):
    input: str


class RAGGraphWithMemory(RAGChainWithMemory):
    def __init__(self, *chain_args):
        super().__init__(*chain_args)

        # Single-node graph (for now)
        workflow: StateGraph = StateGraph(ChatState)
        workflow.add_edge(START, "model")
        workflow.add_node("model", self.call_model)

        self.app: CompiledStateGraph = workflow.compile(checkpointer=MemorySaver())

    def call_model(self, state: ChatState) -> ChatResponse:
        response = self.rag_chain.invoke(state)
        return {
            "chat_history": [
                HumanMessage(state["input"]),
                AIMessage(response["answer"]),
            ],
            "context": response["context"],
            "answer": response["answer"],
        }
