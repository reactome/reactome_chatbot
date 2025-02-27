from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from tools.external_search.completeness_grader import CompletenessGrader
from tools.external_search.state import GraphState
from tools.external_search.tavily_wrapper import TavilyWrapper


def decide_next_steps(state: GraphState) -> str:
    if state["complete"] == "No":
        return "perform_web_search"
    else:
        return "no_search"


def no_search(_) -> dict[str, list]:
    return {"search_results": []}


def create_search_workflow(
    llm: BaseChatModel, max_results: int = 3
) -> CompiledStateGraph:
    completeness_grader = CompletenessGrader(llm)
    tavily_wrapper = TavilyWrapper(max_results=max_results)

    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("assess_completeness", completeness_grader.ainvoke)
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
