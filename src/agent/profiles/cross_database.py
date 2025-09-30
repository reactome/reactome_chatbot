from typing import Any, Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph.state import StateGraph

from agent.profiles.base import BaseGraphBuilder, BaseState
from agent.tasks.completeness_grader import (CompletenessGrade,
                                             create_completeness_grader)
from agent.tasks.cross_database.rewrite_reactome_with_uniprot import \
    create_reactome_rewriter_w_uniprot
from agent.tasks.cross_database.rewrite_uniprot_with_reactome import \
    create_uniprot_rewriter_w_reactome
from agent.tasks.cross_database.summarize_reactome_uniprot import \
    create_reactome_uniprot_summarizer
from agent.tasks.preprocessing.detect_language import create_language_detector
from agent.tasks.preprocessing.safety_checker import SafetyCheck, create_safety_checker
from retrievers.reactome.rag import create_reactome_rag
from retrievers.uniprot.rag import create_uniprot_rag


class CrossDatabaseState(BaseState):
    safety: str  # LLM-assessed safety level of the user input
    query_language: str  # language of the user input

    reactome_query: str  # LLM-generated query for Reactome
    reactome_answer: str  # LLM-generated answer from Reactome
    reactome_completeness: str  # LLM-assessed completeness of the Reactome answer

    uniprot_query: str  # LLM-generated query for UniProt
    uniprot_answer: str  # LLM-generated answer from UniProt
    uniprot_completeness: str  # LLM-assessed completeness of the UniProt answer


class CrossDatabaseGraphBuilder(BaseGraphBuilder):
    def __init__(
        self,
        llm: BaseChatModel,
        embedding: Embeddings,
    ) -> None:
        super().__init__(llm, embedding)

        # Create runnables (tasks & tools)
        self.reactome_rag: Runnable = create_reactome_rag(llm, embedding)
        self.uniprot_rag: Runnable = create_uniprot_rag(llm, embedding)

        self.safety_checker = create_safety_checker(llm)
        self.completeness_checker = create_completeness_grader(llm)
        self.detect_language = create_language_detector(llm)
        self.write_reactome_query = create_reactome_rewriter_w_uniprot(llm)
        self.write_uniprot_query = create_uniprot_rewriter_w_reactome(llm)
        self.summarize_final_answer = create_reactome_uniprot_summarizer(
            llm.model_copy(update={"streaming": True})
        )

        # Create graph
        state_graph = StateGraph(CrossDatabaseState)
        # Set up nodes
        state_graph.add_node("check_question_safety", self.check_question_safety)
        state_graph.add_node("preprocess_question", self.preprocess)
        state_graph.add_node("identify_query_language", self.identify_query_language)
        state_graph.add_node("conduct_research", self.conduct_research)
        state_graph.add_node("generate_reactome_answer", self.generate_reactome_answer)
        state_graph.add_node("rewrite_reactome_query", self.rewrite_reactome_query)
        state_graph.add_node("rewrite_reactome_answer", self.rewrite_reactome_answer)
        state_graph.add_node("generate_uniprot_answer", self.generate_uniprot_answer)
        state_graph.add_node("rewrite_uniprot_query", self.rewrite_uniprot_query)
        state_graph.add_node("rewrite_uniprot_answer", self.rewrite_uniprot_answer)
        state_graph.add_node("assess_completeness", self.assess_completeness)
        state_graph.add_node("decide_next_steps", self.decide_next_steps)
        state_graph.add_node("generate_final_response", self.generate_final_response)
        state_graph.add_node("postprocess", self.postprocess)
        # Set up edges
        state_graph.set_entry_point("preprocess_question")
        state_graph.add_edge("preprocess_question", "identify_query_language")
        state_graph.add_edge("preprocess_question", "check_question_safety")
        state_graph.add_conditional_edges(
            "check_question_safety",
            self.proceed_with_research,
            {"Continue": "conduct_research", "Finish": "generate_final_response"},
        )
        state_graph.add_edge("conduct_research", "generate_reactome_answer")
        state_graph.add_edge("conduct_research", "generate_uniprot_answer")
        state_graph.add_edge("generate_reactome_answer", "assess_completeness")
        state_graph.add_edge("generate_uniprot_answer", "assess_completeness")
        state_graph.add_conditional_edges(
            "assess_completeness",
            self.decide_next_steps,
            {
                "generate_final_response": "generate_final_response",
                "perform_web_search": "generate_final_response",
                "rewrite_reactome_query": "rewrite_reactome_query",
                "rewrite_uniprot_query": "rewrite_uniprot_query",
            },
        )
        state_graph.add_edge("rewrite_reactome_query", "rewrite_reactome_answer")
        state_graph.add_edge("rewrite_uniprot_query", "rewrite_uniprot_answer")
        state_graph.add_edge("rewrite_reactome_answer", "generate_final_response")
        state_graph.add_edge("rewrite_uniprot_answer", "generate_final_response")
        state_graph.add_edge("generate_final_response", "postprocess")
        state_graph.set_finish_point("postprocess")

        self.uncompiled_graph: StateGraph = state_graph

    async def check_question_safety(
        self, state: CrossDatabaseState, config: RunnableConfig
    ) -> CrossDatabaseState:
        result: SafetyCheck = await self.safety_checker.ainvoke(
            {"input": state["rephrased_input"]},
            config,
        )
        if result.binary_score == "No":
            inappropriate_input = f"This is the user's question and it is NOT appropriate for you to answer: {state["user_input"]}. \n\n explain that you are unable to answer the question but you can answer questions about topics related to the Reactome Pathway Knowledgebase or UniProt Knowledgebas."
            return CrossDatabaseState(
                safety=result.binary_score,
                user_input=inappropriate_input,
                reactome_answer="",
                uniprot_answer="",
            )
        else:
            return CrossDatabaseState(safety=result.binary_score)

    async def proceed_with_research(
        self, state: CrossDatabaseState
    ) -> Literal["Continue", "Finish"]:
        if state["safety"] == "Yes":
            return "Continue"
        else:
            return "Finish"

    async def identify_query_language(
        self, state: CrossDatabaseState, config: RunnableConfig
    ) -> CrossDatabaseState:
        query_language: str = await self.detect_language.ainvoke(
            {"user_input": state["user_input"]}, config
        )
        return CrossDatabaseState(query_language=query_language)

    async def conduct_research(
        self, state: CrossDatabaseState, config: RunnableConfig
    ) -> CrossDatabaseState:
        return CrossDatabaseState()

    async def generate_reactome_answer(
        self, state: CrossDatabaseState, config: RunnableConfig
    ) -> CrossDatabaseState:
        reactome_answer: dict[str, Any] = await self.reactome_rag.ainvoke(
            {
                "input": state["rephrased_input"],
                "chat_history": state["chat_history"],
            },
            config,
        )
        return CrossDatabaseState(reactome_answer=reactome_answer["answer"])

    async def generate_uniprot_answer(
        self, state: CrossDatabaseState, config: RunnableConfig
    ) -> CrossDatabaseState:
        uniprot_answer: dict[str, Any] = await self.uniprot_rag.ainvoke(
            {
                "input": state["rephrased_input"],
                "chat_history": state["chat_history"],
            },
            config,
        )
        return CrossDatabaseState(uniprot_answer=uniprot_answer["answer"])

    async def rewrite_reactome_query(
        self, state: CrossDatabaseState, config: RunnableConfig
    ) -> CrossDatabaseState:
        reactome_query: str = await self.write_reactome_query.ainvoke(
            {
                "input": state["rephrased_input"],
                "uniprot_answer": state["uniprot_answer"],
            },
            config,
        )
        return CrossDatabaseState(reactome_query=reactome_query)

    async def rewrite_uniprot_query(
        self, state: CrossDatabaseState, config: RunnableConfig
    ) -> CrossDatabaseState:
        uniprot_query: str = await self.write_uniprot_query.ainvoke(
            {
                "input": state["rephrased_input"],
                "reactome_answer": state["reactome_answer"],
            },
            config,
        )
        return CrossDatabaseState(uniprot_query=uniprot_query)

    async def rewrite_reactome_answer(
        self, state: CrossDatabaseState, config: RunnableConfig
    ) -> CrossDatabaseState:
        rewritten_answer: dict[str, Any] = await self.reactome_rag.ainvoke(
            {
                "input": state["reactome_query"],
                "chat_history": state["chat_history"],
            },
            config,
        )
        return CrossDatabaseState(reactome_answer=rewritten_answer["answer"])

    async def rewrite_uniprot_answer(
        self, state: CrossDatabaseState, config: RunnableConfig
    ) -> CrossDatabaseState:
        rewritten_answer: dict[str, Any] = await self.uniprot_rag.ainvoke(
            {
                "input": state["uniprot_query"],
                "chat_history": state["chat_history"],
            },
            config,
        )
        return CrossDatabaseState(uniprot_answer=rewritten_answer["answer"])

    async def assess_completeness(
        self, state: CrossDatabaseState, config: RunnableConfig
    ) -> CrossDatabaseState:
        reactome_completeness_async = self.completeness_checker.ainvoke(
            {"input": state["rephrased_input"], "generation": state["reactome_answer"]},
            config,
        )
        uniprot_completeness_async = self.completeness_checker.ainvoke(
            {"input": state["rephrased_input"], "generation": state["uniprot_answer"]},
            config,
        )
        reactome_completeness: CompletenessGrade = await reactome_completeness_async
        uniprot_completeness: CompletenessGrade = await uniprot_completeness_async
        return CrossDatabaseState(
            reactome_completeness=reactome_completeness.binary_score,
            uniprot_completeness=uniprot_completeness.binary_score,
        )

    async def decide_next_steps(self, state: CrossDatabaseState) -> Literal[
        "generate_final_response",
        "perform_web_search",
        "rewrite_reactome_query",
        "rewrite_uniprot_query",
    ]:
        reactome_complete = state["reactome_completeness"] != "No"
        uniprot_complete = state["uniprot_completeness"] != "No"
        if reactome_complete and uniprot_complete:
            return "generate_final_response"
        elif not reactome_complete and uniprot_complete:
            return "rewrite_reactome_query"
        elif reactome_complete and not uniprot_complete:
            return "rewrite_uniprot_query"
        else:
            return "perform_web_search"

    async def generate_final_response(
        self, state: CrossDatabaseState, config: RunnableConfig
    ) -> CrossDatabaseState:
        final_response: str = await self.summarize_final_answer.ainvoke(
            {
                "input": state["rephrased_input"],
                "query_language": state["query_language"],
                "reactome_answer": state["reactome_answer"],
                "uniprot_answer": state["uniprot_answer"],
            },
            config,
        )
        return CrossDatabaseState(
            chat_history=[
                HumanMessage(state["user_input"]),
                AIMessage(final_response),
            ],
            answer=final_response,
        )


def create_cross_database_graph(
    llm: BaseChatModel,
    embedding: Embeddings,
) -> StateGraph:
    return CrossDatabaseGraphBuilder(llm, embedding).uncompiled_graph
