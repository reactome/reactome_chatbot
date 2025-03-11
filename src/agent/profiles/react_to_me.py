from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph.state import CompiledStateGraph, StateGraph

from agent.profiles.base import (AdditionalContent, BaseGraphBuilder,
                                 BaseState, InputState, OutputState)
from retrievers.reactome.rag import create_reactome_rag
from tools.external_search.state import WebSearchResult
from tools.external_search.workflow import create_search_workflow


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
        self.search_workflow: CompiledStateGraph = create_search_workflow(llm)

        # Create graph
        state_graph = StateGraph(ReactToMeState, input=InputState, output=OutputState)
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

    async def postprocess(
        self, state: ReactToMeState, config: RunnableConfig
    ) -> ReactToMeState:
        search_results: list[WebSearchResult] = []
        if config["configurable"]["enable_postprocess"]:
            result: dict[str, Any] = await self.search_workflow.ainvoke(
                {"question": state["rephrased_input"], "generation": state["answer"]},
                config=RunnableConfig(callbacks=config["callbacks"]),
            )
            search_results = result["search_results"]
        return ReactToMeState(
            additional_content=AdditionalContent(search_results=search_results)
        )


def create_reactome_graph(
    llm: BaseChatModel,
    embedding: Embeddings,
) -> StateGraph:
    return ReactToMeGraphBuilder(llm, embedding).uncompiled_graph
