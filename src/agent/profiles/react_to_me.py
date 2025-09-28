from typing import Any, Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph.state import StateGraph

from agent.profiles.base import (SAFETY_SAFE, SAFETY_UNSAFE, BaseGraphBuilder,
                                 BaseState)
from agent.tasks.final_answer_generation.unsafe_question import \
    create_unsafe_answer_generator
from retrievers.reactome.rag import create_reactome_rag


class ReactToMeState(BaseState):
    """ReactToMe state extends BaseState with all preprocessing results."""

    pass


class ReactToMeGraphBuilder(BaseGraphBuilder):
    """Graph builder for ReactToMe profile with Reactome-specific functionality."""

    def __init__(
        self, 
        llm: BaseChatModel, 
        embedding: Embeddings
        ) -> None:
        """Initialize ReactToMe graph builder with required components."""
        super().__init__(llm, embedding)

        # Create a streaming LLM instance only for final answer generation
        streaming_llm = ChatOpenAI(
            model=llm.model_name if hasattr(llm, "model_name") else "gpt-4o-mini",
            temperature=0.0,
            streaming=True,
        )

        self.unsafe_answer_generator = create_unsafe_answer_generator(streaming_llm)
        self.reactome_rag: Runnable = create_reactome_rag(
            streaming_llm, embedding, streaming=True
        )

        self.uncompiled_graph: StateGraph = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build and configure the ReactToMe workflow graph."""
        state_graph = StateGraph(ReactToMeState)

        # Add workflow nodes
        state_graph.add_node("preprocess", self.preprocess)
        state_graph.add_node("model", self.call_model)
        state_graph.add_node("generate_unsafe_response", self.generate_unsafe_response)
        state_graph.add_node("postprocess", self.postprocess)

        # Configure workflow edges
        state_graph.set_entry_point("preprocess")
        state_graph.add_conditional_edges(
            "preprocess",
            self.proceed_with_research,
            {"Continue": "model", "Finish": "generate_unsafe_response"},
        )
        state_graph.add_edge("model", "postprocess")
        state_graph.add_edge("generate_unsafe_response", "postprocess")
        state_graph.set_finish_point("postprocess")

        return state_graph

    async def preprocess(
        self, 
        state: ReactToMeState, 
        config: RunnableConfig
    ) -> ReactToMeState:
        """Run preprocessing workflow."""
        result = await super().preprocess(state, config)
        return ReactToMeState(**result)

    async def proceed_with_research(
        self, 
        state: ReactToMeState
    ) -> Literal["Continue", "Finish"]:
        """Determine whether to proceed with research based on safety check."""
        return "Continue" if state["safety"] == SAFETY_SAFE else "Finish"

    async def generate_unsafe_response(
        self, 
        state: ReactToMeState, 
        config: RunnableConfig
    ) -> ReactToMeState:
        """Generate appropriate refusal response for unsafe queries."""
        final_answer_message = await self.unsafe_answer_generator.ainvoke(
            {
                "language": state["detected_language"],
                "user_input": state["rephrased_input"],
                "reason_unsafe": state["reason_unsafe"],
            },
            config,
        )

        final_answer = (
            final_answer_message.content
            if hasattr(final_answer_message, "content")
            else str(final_answer_message)
        )

        return ReactToMeState(
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

    async def call_model(
        self, 
        state: ReactToMeState, 
        config: RunnableConfig
    ) -> ReactToMeState:
        """Generate response using Reactome RAG for safe queries."""
        result: dict[str, Any] = await self.reactome_rag.ainvoke(
            {
                "input": state["rephrased_input"],
                "expanded_queries": state.get("expanded_queries", []),
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
    embedding: Embeddings
    ) -> StateGraph:
    """Create and return the ReactToMe workflow graph."""
    return ReactToMeGraphBuilder(llm, embedding).uncompiled_graph
