import asyncio
import os
from typing import Annotated, Any, TypedDict

from langchain_core.callbacks.base import Callbacks
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph, StateGraph
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from conversational_chain.chain import create_rag_chain, create_rephrase_chain
from external_search.state import WebSearchResult
from external_search.workflow import create_search_workflow
from util.logging import logging
from langchain_core.prompts import ChatPromptTemplate

from src.system_prompt.prerocess_prompt import SafetyChecker, detect_language
from src.system_prompt.reactome_prompt import rewrite_reactome_query
from src.system_prompt.uniprot_prompt import rewrite_uniprot_query
from src.system_prompt.completeness_grader import CompletenessGrader
from src.system_prompt.summary_generator import create_summarization_chain


LANGGRAPH_DB_URI = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@postgres:5432/{os.getenv('POSTGRES_LANGGRAPH_DB')}?sslmode=disable"

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

if not os.getenv("POSTGRES_LANGGRAPH_DB"):
    logging.warning("POSTGRES_LANGGRAPH_DB undefined; falling back to MemorySaver.")


class AdditionalContent(TypedDict):
    search_results: list[WebSearchResult]


class ChatState(TypedDict):
    user_input: str  # User input text
    safety: str # LLM-assessed safety level of the user input
    input: str  # LLM-generated query from user input
    query_language: str # language of the user input
    chat_history: Annotated[list[BaseMessage], add_messages]
    
    reactome_query: str # LLm-generated query for Reactome
    reactome_answer: str # LLm-generated answer from Reactome
    reactome_completeness: str # LLm-assessed completeness of the Reactome answer
    
    uniprot_query: str # LLm-generated query for UniProt
    uniprot_answer: str # LLm-generated answer from UniProt
    uniprot_completeness: str # LLm-assessed completeness of the UniProt answer
    
    context: list[Document]
    answer: str  # final LLM response that is streamed to the user
    additional_content: (
        AdditionalContent  # additional content to send after graph completes
    )

class CrossDatabaseRAGGraph:
    def __init__(self, llm: BaseChatModel, reactome_retriever: BaseRetriever, uniprot_retriever: BaseRetriever, reactome_prompt: ChatPromptTemplate, uniprot_prompt: ChatPromptTemplate) -> None:
        print("Initializing CrossDatabaseRAGGraph...")
        # Set up runnables
        self.rephrase_chain: Runnable = create_rephrase_chain(llm)
        self.search_workflow: CompiledStateGraph = create_search_workflow(llm)
        self.reactome_chain: Runnable = create_rag_chain(llm, retriever = reactome_retriever, qa_prompt = reactome_prompt)
        self.uniprot_chain: Runnable = create_rag_chain(llm, retriever = uniprot_retriever, qa_prompt = uniprot_prompt)

        self.safety_checker = SafetyChecker(llm)
        self.completeness_checker = CompletenessGrader(llm)
        self.detect_language = detect_language(llm)
        self.write_reactome_query = rewrite_reactome_query(llm)
        self.write_uniprot_query = rewrite_uniprot_query(llm)
        self.summarize_final_answer = create_summarization_chain(llm.model_copy(update={"streaming": True}))
        print("Runnables initialized.")
        # Create graph
        state_graph: StateGraph = StateGraph(ChatState)

        # Define nodes
        state_graph.add_node("check_question_safety", self.check_question_safety)
        state_graph.add_node("preprocess_question", self.preprocess_question)
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
        # Define edges
        state_graph.set_entry_point("preprocess_question")
        state_graph.add_edge("preprocess_question", "identify_query_language")
        state_graph.add_edge("preprocess_question", "check_question_safety")
        state_graph.add_conditional_edges("check_question_safety", self.proceed_with_research, 
            {
                "Continue": "conduct_research", 
                "finish": "generate_final_response"
                }
        )
        # state_graph.add_edge("preprocess_question", "identify_query_language")
        state_graph.add_edge("conduct_research", "generate_reactome_answer")
        state_graph.add_edge("conduct_research", "generate_uniprot_answer")
        state_graph.add_edge("generate_reactome_answer", "assess_completeness")
        state_graph.add_edge("generate_uniprot_answer", "assess_completeness")
        state_graph.add_conditional_edges("assess_completeness", self.decide_next_steps, 
            {
                "generate_final_response": "generate_final_response",
                "perform_web_search": "generate_final_response",
                "rewrite_reactome_query":"rewrite_reactome_query", 
                "rewrite_uniprot_query": "rewrite_uniprot_query"
            }
        )
        state_graph.add_edge("rewrite_reactome_query", "rewrite_reactome_answer")
        state_graph.add_edge("rewrite_uniprot_query", "rewrite_uniprot_answer")
        state_graph.add_edge("rewrite_reactome_answer", "generate_final_response")
        state_graph.add_edge("rewrite_uniprot_answer", "generate_final_response")
        state_graph.add_edge("generate_final_response", "postprocess")
        state_graph.set_finish_point("postprocess")

        self.uncompiled_graph: StateGraph = state_graph

        # The following are set asynchronously by calling initialize()
        self.graph: CompiledStateGraph | None = None
        self.pool: AsyncConnectionPool[AsyncConnection[dict[str, Any]]] | None = None
    def __del__(self) -> None:
        if self.pool:
            asyncio.run(self.close_pool())

    async def initialize(self) -> CompiledStateGraph:
        checkpointer: BaseCheckpointSaver[str] = await self.create_checkpointer()
        return self.uncompiled_graph.compile(checkpointer=checkpointer)

    async def create_checkpointer(self) -> BaseCheckpointSaver[str]:
        if not os.getenv("POSTGRES_LANGGRAPH_DB"):
            return MemorySaver()
        self.pool = AsyncConnectionPool(
            conninfo=LANGGRAPH_DB_URI,
            max_size=20,
            open=False,
            timeout=30,
            kwargs=connection_kwargs,
        )
        await self.pool.open()
        checkpointer = AsyncPostgresSaver(self.pool)
        await checkpointer.setup()
        return checkpointer

    async def close_pool(self) -> None:
        if self.pool:
            await self.pool.close()

    async def preprocess_question(
            self, state: ChatState, config: RunnableConfig
    ) -> dict[str, str]:
        """ Rephrases the user’s question for clarity. """
        rephrased_input : str = await self.rephrase_chain.ainvoke(state, config)
        print("rephrased_input:", rephrased_input)
        return {"input": rephrased_input}
    
    async def check_question_safety(
            self, state: ChatState, config: RunnableConfig) -> dict:
        """ Checks if the user question is appropriate. """
        print("input:", state["input"])
        result = await self.safety_checker.ainvoke(state, config)
        print("safety", result["safety"])
        if result["safety"] == "No":
            inappropriate_input = f"This is the user's question and it is NOT appropriate for you to answer: {state["user_input"]}. \n\n explain that you are unable to answer the question but you can answer questions about topics related to the Reactome Pathway Knowledgebase or UniProt Knowledgebas."
            return {"safety": result["safety"], "input": inappropriate_input, "reactome_answer": "", "uniprot_answer": ""}
        else:
            return {"safety": result["safety"]}
    
    async def proceed_with_research(
            self, state: ChatState, config: RunnableConfig) -> dict:
        """ Determines if the user question is appropriate for research. """
        if state["safety"] == "Yes":
            return "Continue"
        else:
            return "finish"

    async def identify_query_language(
            self, state: ChatState, config: RunnableConfig) -> dict:
        """ Identifies the language of the user input. """
        query_language: str = await self.detect_language.ainvoke(
            {"user_input": state["user_input"]},
            config)
        print("query_language:", query_language)
        return {"query_language": query_language}
    

    async def conduct_research(
            self, state: ChatState, config: RunnableConfig
    ):
        return {}

    async def generate_reactome_answer(
            self, state: ChatState, config: RunnableConfig
    ) -> dict[str, str]:
        """ Generates an answer to the user's question using Reactome. """
        reactome_answer : dict[str, Any] = await self.reactome_chain.ainvoke(
            {
                "input": state["input"],
                "chat_history": state["chat_history"],
            },
            config,
        )
        print("reactome_answer:", reactome_answer["answer"])
        return {"reactome_answer": reactome_answer["answer"]}
    
    async def rewrite_reactome_query(
            self, state: ChatState, config: RunnableConfig
    ) -> dict[str, str]:
        """ Rewrites the query for Reactome. """
        reactome_query : str = await self.write_reactome_query.ainvoke(
            {
                "input": state["input"],
                "uniprot_answer": state["uniprot_answer"]
            },
            config
            )
        print("reactome_query:", reactome_query)
        return {"reactome_query": reactome_query}
    
    async def rewrite_reactome_answer(
            self, state: ChatState, config: RunnableConfig
    ) -> dict[str, str]:
        """ Rewrites the answer from Reactome. """
        rewritten_answer : dict[str, Any] = await self.reactome_chain.ainvoke(
            {
                "input": state["reactome_query"],
                "chat_history": state["chat_history"],
            },
            config
            )
        print("re-written reactome_answer:", rewritten_answer["answer"])
        return {"reactome_answer": rewritten_answer["answer"]}
    
    async def generate_uniprot_answer(
            self, state: ChatState, config: RunnableConfig
    ) -> dict[str, str]:
        """ Generates an answer to the user's question using UniProt. """
        uniprot_answer : dict[str, Any] = await self.uniprot_chain.ainvoke(
            {
                "input": state["input"],
                "chat_history": state["chat_history"],
            },
            config,
        )
        print("uniprot_answer:", uniprot_answer["answer"])
        return {"uniprot_answer": uniprot_answer["answer"]}
    
    async def rewrite_uniprot_query(
            self, state: ChatState, config: RunnableConfig
    ) -> dict[str, str]:
        """ Rewrites the query for UniProt. """
        uniprot_query : str = await self.write_uniprot_query.ainvoke(
            {
                "input": state["input"],
                "reactome_answer": state["reactome_answer"]
            },
            config
            )
        print("uniprot_query:", uniprot_query)
        return {"uniprot_query": uniprot_query}
    
    async def rewrite_uniprot_answer(
            self, state: ChatState, config: RunnableConfig
    ) -> dict[str, str]:
        """ Rewrites the answer from UniProt. """
        rewritten_answer : dict[str, Any] = await self.uniprot_chain.ainvoke(
            {
                "input": state["uniprot_query"],
                "chat_history": state["chat_history"],
            },
            config
            )
        print("re-written uniprot_answer:", rewritten_answer["answer"])
        return {"uniprot_answer": rewritten_answer["answer"]}
    
    async def assess_completeness(
            self, state: ChatState, config: RunnableConfig
    ) -> dict[str, str]:
        """ Assesses the completeness of the answers from Reactome and UniProt. """
        reactome_completeness = await self.completeness_checker.ainvoke(
            {
                "input": state["input"], 
                "generation": state["reactome_answer"]
            },
            config
        )
        uniprot_completeness = await self.completeness_checker.ainvoke(
            {
                "input": state["input"], 
                "generation": state["uniprot_answer"]
            },
            config
        )
        print("reactome_completeness:", reactome_completeness["complete"], "\n\n uniprot_completeness:", uniprot_completeness["complete"])
        return {"reactome_completeness": reactome_completeness["complete"], "uniprot_completeness": uniprot_completeness["complete"]}
    
    async def decide_next_steps(
            self, state: ChatState, config: RunnableConfig
    ) -> dict[str, str]:
        """ Decides the next steps based on the completeness of the answers. """
        print("deciding next steps...")
        print("reactome_completeness:", state["reactome_completeness"], "uniprot_completeness:", state["uniprot_completeness"])
        if state["reactome_completeness"] == "Yes" and state["uniprot_completeness"] == "Yes":
            print("both answer --> generate_final_response")
            return "generate_final_response"
        elif state["reactome_completeness"] == "No" and state["uniprot_completeness"] == "No":
            print("neither answer --> perform_web_search")
            return "perform_web_search"
        elif state["reactome_completeness"] == "No" and state["uniprot_completeness"] == "Yes":
            print("rewrite_reactome_query")
            return "rewrite_reactome_query"
        elif state["reactome_completeness"] == "Yes" and state["uniprot_completeness"] == "No":
            print("rewrite_uniprot_query")
            return "rewrite_uniprot_query"

    async def generate_final_response(
            self, state: ChatState, config: RunnableConfig
    ) -> dict[str, str]:
        """ Generates the final response to the user. """
        print("final response...")
        final_response = await self.summarize_final_answer.ainvoke(
            {
                "input": state["input"],
                "query_language": state["query_language"],
                "reactome_answer": state["reactome_answer"],
                "uniprot_answer": state["uniprot_answer"]
            },
            config,
        )
        print("final_response:", final_response)
        return {"answer": final_response}
    
    
    async def postprocess(
        self, state: ChatState, config: RunnableConfig
    ) -> dict[str, dict[str, list[WebSearchResult]]]:
        search_results: list[WebSearchResult] = []
        if config["configurable"]["enable_postprocess"]:
            result: dict[str, Any] = await self.search_workflow.ainvoke(
                {"input": state["input"], "generation": state["answer"]},
                config=RunnableConfig(callbacks=config["callbacks"]),
            )
            search_results = result["search_results"]
        return {
            "additional_content": {"search_results": search_results},
        } 
    
    async def ainvoke(
        self,
        user_input: str,
        *,
        callbacks: Callbacks,
        thread_id: str,
        enable_postprocess: bool = True,
    ) -> dict[str, Any]:
        if self.graph is None:
            self.graph = await self.initialize()
        result: dict[str, Any] = await self.graph.ainvoke(
            {"user_input": user_input},
            config=RunnableConfig(
                callbacks=callbacks,
                configurable={
                    "thread_id": thread_id,
                    "enable_postprocess": enable_postprocess,
                },
            ),
        )
        return result