from typing import Any, Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph.state import StateGraph

from agent.profiles.base import (
    DEFAULT_LANGUAGE,
    SAFETY_SAFE,
    SAFETY_UNSAFE,
    BaseGraphBuilder,
    BaseState,
)
from agent.tasks.final_answer_generation.unsafe_question import (
    create_unsafe_answer_generator,
)
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
        streaming_llm = llm.model_copy(update={"streaming": True})

        self.unsafe_answer_generator = create_unsafe_answer_generator(streaming_llm)
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

    async def proceed_with_research(
        self, state: ReactToMeState
    ) -> Literal["Continue", "Finish"]:
        return "Continue" if state.get("safety") == SAFETY_SAFE else "Finish"

    async def generate_unsafe_response(
        self, state: ReactToMeState, config: RunnableConfig
    ) -> ReactToMeState:
        final_answer_message = await self.unsafe_answer_generator.ainvoke(
            {
                "language": state.get("detected_language", DEFAULT_LANGUAGE),
                "user_input": state.get("rephrased_input", state["user_input"]),
                "reason_unsafe": state.get("reason_unsafe", ""),
            },
            config,
        )

        final_answer = (
            final_answer_message.content
            if hasattr(final_answer_message, "content")
            else str(final_answer_message)
        )

        updated_state = dict(state)
        updated_state.update(
            chat_history=[
                HumanMessage(state["user_input"]),
                (
                    final_answer_message
                    if hasattr(final_answer_message, "content")
                    else AIMessage(final_answer)
                ),
            ],
            answer=final_answer,
            safety=SAFETY_UNSAFE,
            additional_content={"search_results": []},
        )
        return ReactToMeState(**updated_state)

    async def call_model(
        self, state: ReactToMeState, config: RunnableConfig
    ) -> ReactToMeState:
        result: dict[str, Any] = await self.reactome_rag.ainvoke(
            {
                "input": state["rephrased_input"],
                "expanded_queries": state.get("expanded_queries", []),
                "chat_history": (
                    state.get("chat_history")
                    if state.get("chat_history")
                    else [HumanMessage(state["user_input"])]
                ),
            },
            config,
        )
        updated_state = dict(state)
        updated_state.update(
            chat_history=[
                HumanMessage(state["user_input"]),
                AIMessage(result["answer"]),
            ],
            answer=result["answer"],
        )
        return ReactToMeState(**updated_state)


def create_reactome_graph(
    llm: BaseChatModel,
    embedding: Embeddings,
) -> StateGraph:
    return ReactToMeGraphBuilder(llm, embedding).uncompiled_graph
