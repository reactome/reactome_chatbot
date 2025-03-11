from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.utils.runnable import RunnableLike

from agent.tasks.completeness_grader import (CompletenessGrade,
                                             create_completeness_grader)
from tools.external_search.state import GraphState
from tools.external_search.tavily_wrapper import TavilyWrapper


def decide_next_steps(state: GraphState) -> Literal["perform_web_search", "no_search"]:
    if state["complete"] == "No":
        return "perform_web_search"
    else:
        return "no_search"


def no_search(_) -> GraphState:
    return GraphState(search_results=[])


def run_completeness_grader(grader: Runnable) -> RunnableLike:
    async def _run_completeness_grader(
        state: GraphState, config: RunnableConfig
    ) -> GraphState:
        result: CompletenessGrade = await grader.ainvoke(
            {
                "input": state["input"],
                "generation": state["generation"],
            },
            config,
        )
        return GraphState(complete=result.binary_score)

    return _run_completeness_grader


def create_search_workflow(
    llm: BaseChatModel, max_results: int = 3
) -> CompiledStateGraph:
    completeness_grader: Runnable = create_completeness_grader(llm)
    tavily_wrapper = TavilyWrapper(max_results=max_results)

    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node(
        "assess_completeness", run_completeness_grader(completeness_grader)
    )
    workflow.add_node("perform_web_search", tavily_wrapper.ainvoke)
    workflow.add_node("no_search", no_search)

    # Add edges
    workflow.set_entry_point("assess_completeness")
    workflow.add_conditional_edges(
        "assess_completeness",
        decide_next_steps,
        {"perform_web_search": "perform_web_search", "no_search": "no_search"},
    )

    workflow.set_finish_point("perform_web_search")
    workflow.set_finish_point("no_search")
    return workflow.compile()
