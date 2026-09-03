from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph.state import StateGraph

from agent.profiles.base import BaseGraphBuilder, BaseState
from agent.tasks.unsafe_question import create_unsafe_answer_generator
from retrievers.plantreactome.rag import create_plantreactome_rag


class PlantReactomeState(BaseState):
    pass


class PlantReactomeGraphBuilder(BaseGraphBuilder):
    def __init__(
        self,
        llm: BaseChatModel,
        embedding: Embeddings,
    ) -> None:
        super().__init__(llm, embedding)

        # Create runnables (tasks & tools)
        self.unsafe_answer_generator: Runnable = create_unsafe_answer_generator(
            llm, streaming=True
        )
        self.plantreactome_rag: Runnable = create_plantreactome_rag(
            llm, embedding, streaming=True
        )

        # Create graph
        state_graph = StateGraph(PlantReactomeState)
        # Set up nodes
        state_graph.add_node("preprocess", self.preprocess)
        state_graph.add_node("model", self.call_model)
        state_graph.add_node("generate_unsafe_response", self.generate_unsafe_response)
        state_graph.add_node("postprocess", self.postprocess)
        # Set up edges
        state_graph.set_entry_point("preprocess")
        state_graph.add_conditional_edges(
            "preprocess",
            self.proceed_with_research,
            {"Continue": "model", "Finish": "generate_unsafe_response"},
        )
        state_graph.add_edge("model", "postprocess")
        state_graph.add_edge("generate_unsafe_response", "postprocess")
        state_graph.set_finish_point("postprocess")

        self.uncompiled_graph: StateGraph = state_graph

    async def generate_unsafe_response(
        self, state: PlantReactomeState, config: RunnableConfig
    ) -> PlantReactomeState:
        answer: str = await self.unsafe_answer_generator.ainvoke(
            {
                "language": state["detected_language"],
                "user_input": state["rephrased_input"],
                "reason_unsafe": state["reason_unsafe"],
            },
            config,
        )
        return PlantReactomeState(
            chat_history=[
                HumanMessage(state["user_input"]),
                AIMessage(answer),
            ],
            answer=answer,
        )

    async def call_model(
        self, state: PlantReactomeState, config: RunnableConfig
    ) -> PlantReactomeState:
        result: dict[str, Any] = await self.plantreactome_rag.ainvoke(
            {
                "input": state["rephrased_input"],
                "chat_history": (
                    state["chat_history"]
                    if state["chat_history"]
                    else [HumanMessage(state["user_input"])]
                ),
            },
            config,
        )
        return PlantReactomeState(
            chat_history=[
                HumanMessage(state["user_input"]),
                AIMessage(result["answer"]),
            ],
            answer=result["answer"],
        )


def create_plantreactome_graph(
    llm: BaseChatModel,
    embedding: Embeddings,
) -> StateGraph:
    return PlantReactomeGraphBuilder(llm, embedding).uncompiled_graph
