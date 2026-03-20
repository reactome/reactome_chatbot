from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph.state import StateGraph

from agent.profiles.base import BaseGraphBuilder, BaseState
from agent.tasks.unsafe_question import create_unsafe_answer_generator
from retrievers.reactome.rag import create_reactome_rag
from agent.models import get_llm
from mcp.query_router import create_query_router, ROUTE_RAG, ROUTE_MCP_SEARCH, ROUTE_MCP_ANALYSIS


class ReactToMeState(BaseState):
    pass


class ReactToMeGraphBuilder(BaseGraphBuilder):
    def __init__(
        self,
        llm: BaseChatModel,
        embedding: Embeddings,
        mcp_tools: list | None = None,
    ) -> None:
        super().__init__(llm, embedding)

        self.llm = llm

        # optional MCP tools - if provided the LLM can call them instead of using RAG
        self.mcp_tools = mcp_tools or []

        # pre-bind two route-specific tool subsets to LLM once at init time - search tools for
        # lookup/retrieval routes and analysis tool for enrichment routes - avoids rebinding on every message
        # produces two separate LLM instances: llm_with_search_tools and llm_with_analysis_tools
        search_tools = [t for t in self.mcp_tools if t.name in (
            "search_reactome", "get_pathway", "get_database_info", "get_species"
        )]
        analysis_tools = [t for t in self.mcp_tools if t.name == "analyze_identifiers"]

        self.llm_with_search_tools = self.llm.bind_tools(search_tools) if search_tools else None
        self.llm_with_analysis_tools = self.llm.bind_tools(analysis_tools) if analysis_tools else None

        # create router with cheap model - only used when mcp_tools available
        self.query_router = create_query_router(get_llm("openai", "gpt-4o-mini")) if self.mcp_tools else None

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
        # no MCP tools - fall back to existing RAG behaviour unchanged
        if not self.mcp_tools:
            result: dict[str, Any] = await self.reactome_rag.ainvoke(
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
            return ReactToMeState(
                chat_history=[
                    HumanMessage(state["user_input"]),
                    AIMessage(result["answer"]),
                ],
                answer=result["answer"],
            )

        # route question to correct path
        route = await self.query_router(state["rephrased_input"])

        if route == ROUTE_RAG:
            # question is general knowledge, use RAG directly, no tools needed
            result: dict[str, Any] = await self.reactome_rag.ainvoke(
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
            return ReactToMeState(
                chat_history=[
                    HumanMessage(state["user_input"]),
                    AIMessage(result["answer"]),
                ],
                answer=result["answer"],
            )

        if route == ROUTE_MCP_SEARCH:
            llm_with_tools = self.llm_with_search_tools
        elif route == ROUTE_MCP_ANALYSIS:
            llm_with_tools = self.llm_with_analysis_tools
        else:
            llm_with_tools = self.llm_with_search_tools

        messages = list(state["chat_history"] or []) + [
            HumanMessage(state["rephrased_input"])
        ]

        response = await llm_with_tools.ainvoke(messages, config)

        # tool calling loop - max 15 iterations to prevent infinite loop
        max_iterations = 15
        iteration = 0
        while response.tool_calls and iteration < max_iterations:
            iteration += 1
            tool_results = []

            for tool_call in response.tool_calls:
                # find matching tool and execute it - triggers MCP client to Reactome API
                tool = next(
                    t for t in self.mcp_tools if t.name == tool_call["name"]
                )

                result = await tool.ainvoke(tool_call["args"])

                # tool_call_id links this result back to the specific request the LLM made
                tool_results.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"],
                    )
                )

            # send tool results back to LLM for final answer
            messages = messages + [response] + tool_results
            response = await llm_with_tools.ainvoke(messages, config)

        # loop hit max iterations - LLM never gave direct answer
        if response.tool_calls:
            answer = "I was unable to complete the research in time. Please try rephrasing your question."
            return ReactToMeState(
                chat_history=[
                    HumanMessage(state["user_input"]),
                    AIMessage(answer),
                ],
                answer=answer,
            )

        # LLM gave direct answer
        return ReactToMeState(
            chat_history=[
                HumanMessage(state["user_input"]),
                AIMessage(response.content),
            ],
            answer=response.content,
        )


def create_reactome_graph(
    llm: BaseChatModel,
    embedding: Embeddings,
    mcp_tools: list | None = None
) -> StateGraph:
    return ReactToMeGraphBuilder(llm, embedding, mcp_tools).uncompiled_graph
