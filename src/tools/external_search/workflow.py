from typing import Any, Literal, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.tasks.completeness_grader import (
    CompletenessGrade,
    create_completeness_grader,
)
from tools.external_search.state import SearchState
from tools.external_search.tavily_wrapper import TavilyWrapper


def decide_next_steps(state: SearchState) -> Literal["perform_web_search", "no_search"]:
    if state["complete"] == "No":
        return "perform_web_search"
    return "no_search"


def no_search(state: SearchState) -> SearchState:
    return SearchState(search_results=[])


class SearchNode(Protocol):
    """The shape langgraph 1.0 requires of a graph node taking a config.

    Spelled out rather than imported. This was
    `langgraph.utils.runnable.RunnableLike` -- a private module, and a union too
    wide for langgraph 1.0's stricter `add_node`.

    It has to be a Protocol and not a Callable alias: langgraph matches nodes
    structurally on the *parameter name* `state`, which a Callable alias cannot
    express. That is also why `no_search` below takes `state` rather than `_`.
    """

    def __call__(self, state: SearchState, config: RunnableConfig) -> Any: ...


def run_completeness_grader(grader: Runnable) -> SearchNode:
    async def _run_completeness_grader(
        state: SearchState, config: RunnableConfig
    ) -> SearchState:
        result: CompletenessGrade = await grader.ainvoke(
            {
                "input": state["input"],
                "generation": state["generation"],
            },
            config,
        )
        return SearchState(complete=result.binary_score)

    return _run_completeness_grader


def create_search_workflow(
    llm: BaseChatModel, max_results: int = 3
) -> CompiledStateGraph:
    completeness_grader: Runnable = create_completeness_grader(llm)
    tavily_wrapper = TavilyWrapper(max_results=max_results)

    workflow = StateGraph(SearchState)

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
