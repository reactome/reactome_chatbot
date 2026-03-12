from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph.state import StateGraph

from agent.profiles.base import BaseGraphBuilder, BaseState
from agent.tasks.unsafe_question import create_unsafe_answer_generator
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
        self.unsafe_answer_generator: Runnable = create_unsafe_answer_generator(
            llm, streaming=True
        )
        self.reactome_rag: Runnable = create_reactome_rag(
            llm, embedding, streaming=True
        )

        # Create graph
        state_graph = StateGraph(ReactToMeState)
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
        self, state: ReactToMeState, config: RunnableConfig
    ) -> ReactToMeState:
        answer: str = await self.unsafe_answer_generator.ainvoke(
            {
                "language": state["detected_language"],
                "user_input": state["rephrased_input"],
                "reason_unsafe": state["reason_unsafe"],
            },
            config,
        )
        return ReactToMeState(
            chat_history=[
                HumanMessage(state["user_input"]),
                AIMessage(answer),
            ],
            answer=answer,
        )

    async def call_model(
        self, state: ReactToMeState, config: RunnableConfig
    ) -> ReactToMeState:
        # Build the query, injecting a language instruction for non-English users.
        # Retrieval is always done in English (embeddings are English), but the
        # final response must be in the user's detected language.
        query = state["rephrased_input"]
        detected_language = state.get("detected_language", "English")

        if detected_language.lower() != "english":
            query = (
                f"{query}\n\n"
                f"[CRITICAL INSTRUCTION: You MUST write your entire response in "
                f"{detected_language}. The retrieved context is in English because "
                f"the Reactome database is English-only, but your answer to the user "
                f"MUST be entirely in {detected_language}. "
                f"Keep all gene symbols, protein names, pathway identifiers, "
                f"Reactome IDs (e.g. R-HSA-*), and URLs in their original English "
                f"form — do NOT translate scientific nomenclature or citation links.]"
            )

        result: dict[str, Any] = await self.reactome_rag.ainvoke(
            {
                "input": query,
                "chat_history": (
                    state["chat_history"]
                    if state["chat_history"]
                    else [HumanMessage(state["user_input"])]
                ),
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