from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph.state import StateGraph

from agent.profiles.base import BaseGraphBuilder, BaseState
from retrievers.reactome.rag import create_reactome_rag


class ReactToMeState(BaseState):
    pass


class ReactToMeGraphBuilder(BaseGraphBuilder):
    def __init__(
        self,
        llm: BaseChatModel,
        embedding: Embeddings,
    ) -> None:
        super().__init__(llm, embedding)

        # Create runnables (tasks & tools)
        self.reactome_rag: Runnable = create_reactome_rag(
            llm, embedding, streaming=True
        )

        # Create graph
        state_graph = StateGraph(ReactToMeState)
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

    async def call_model(
        self, state: ReactToMeState, config: RunnableConfig
    ) -> ReactToMeState:
        result: dict[str, Any] = await self.reactome_rag.ainvoke(
            {
                "input": state["rephrased_input"],
                "chat_history": state["chat_history"],
            },
            config,
        )
        return ReactToMeState(
            chat_history=[
                HumanMessage(state["user_input"]),
                AIMessage(result["answer"]),
            ],
            answer=result["answer"],
        )


def create_reactome_graph(
    llm: BaseChatModel,
    embedding: Embeddings,
) -> StateGraph:
    return ReactToMeGraphBuilder(llm, embedding).uncompiled_graph
