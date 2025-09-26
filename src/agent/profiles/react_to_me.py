from typing import Any, Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph.state import StateGraph

from agent.profiles.base import BaseGraphBuilder, BaseState
from agent.tasks.final_answer_generators import create_unsafe_answer_generator
from agent.tasks.query_expansion import create_query_expander
from agent.tasks.safety_checker import SafetyCheck, create_safety_checker
from retrievers.reactome.rag import create_reactome_rag


class ReactToMeState(BaseState):
    safety: str = "true"
    reason_unsafe: str = ""
    expanded_queries: list[str] = []


class ReactToMeGraphBuilder(BaseGraphBuilder):
    def __init__(self, llm: BaseChatModel, embedding: Embeddings) -> None:
        super().__init__(llm, embedding)

        self.safety_checker = create_safety_checker(llm)
        self.query_expander = create_query_expander(llm)
        self.unsafe_answer_generator = create_unsafe_answer_generator(llm)
        self.reactome_rag: Runnable = create_reactome_rag(
            llm, embedding, streaming=True
        )

        state_graph = StateGraph(ReactToMeState)

        # Set up nodes
        state_graph.add_node("preprocess", self.preprocess)
        state_graph.add_node("check_question_safety", self.check_question_safety)
        state_graph.add_node("query_expansion", self.query_expansion)
        state_graph.add_node("model", self.call_model)
        state_graph.add_node("generate_unsafe_response", self.generate_unsafe_response)
        state_graph.add_node("postprocess", self.postprocess)

        # Set up edges
        state_graph.set_entry_point("preprocess")
        state_graph.add_edge("preprocess", "check_question_safety")
        state_graph.add_edge("preprocess", "query_expansion")
        state_graph.add_conditional_edges(
            "check_question_safety",
            self.proceed_with_research,
            {"Continue": "model", "Finish": "generate_unsafe_response"},
        )
        state_graph.add_edge("model", "postprocess")
        state_graph.add_edge("generate_unsafe_response", "postprocess")
        state_graph.set_finish_point("postprocess")

        self.uncompiled_graph: StateGraph = state_graph

    async def preprocess(
        self, state: ReactToMeState, config: RunnableConfig
    ) -> ReactToMeState:
        rephrased_input: str = await self.rephrase_chain.ainvoke(
            {
                "user_input": state["user_input"],
                "chat_history": state["chat_history"],
            },
            config,
        )
        return ReactToMeState(rephrased_input=rephrased_input)

    async def check_question_safety(
        self, state: ReactToMeState, config: RunnableConfig
    ) -> ReactToMeState:
        result: SafetyCheck = await self.safety_checker.ainvoke(
            {"rephrased_input": state["rephrased_input"]},
            config,
        )

        if result.safety == "false":
            return ReactToMeState(
                safety=result.safety,
                reason_unsafe=result.reason_unsafe,
            )
        else:
            return ReactToMeState(
                safety=result.safety, reason_unsafe=result.reason_unsafe
            )

    async def query_expansion(
        self, state: ReactToMeState, config: RunnableConfig
    ) -> ReactToMeState:
        expanded_queries: list[str] = await self.query_expander.ainvoke(
            {"rephrased_input": state["rephrased_input"]}, config
        )
        return ReactToMeState(expanded_queries=expanded_queries)

    async def proceed_with_research(
        self, state: ReactToMeState
    ) -> Literal["Continue", "Finish"]:
        if state["safety"] == "true":
            return "Continue"
        else:
            return "Finish"

    async def generate_unsafe_response(
        self, state: ReactToMeState, config: RunnableConfig
    ) -> ReactToMeState:
        final_answer: str = await self.unsafe_answer_generator.ainvoke(
            {
                "language": "English",
                "user_input": state["rephrased_input"],
                "reason_unsafe": state["reason_unsafe"],
            },
            config,
        )

        return ReactToMeState(
            chat_history=[
                HumanMessage(state["user_input"]),
                AIMessage(final_answer),
            ],
            answer=final_answer,
        )

    async def call_model(
        self, state: ReactToMeState, config: RunnableConfig
    ) -> ReactToMeState:
        expanded_queries = state.get("expanded_queries", [])
        primary_query = state["rephrased_input"]

        result: dict[str, Any] = await self.reactome_rag.ainvoke(
            {
                "input": primary_query,
                "expanded_queries": expanded_queries,
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
