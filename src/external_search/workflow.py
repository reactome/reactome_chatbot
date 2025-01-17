from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from external_search.completeness_grader import CompletenessGrader
from external_search.state import GraphState
from external_search.tavily_wrapper import TavilyWrapper


def decide_next_steps(state):
    external_search = state['external_search']
    if external_search == "Yes":
        return "perform_web_search"
    else:
        return "finish"

def create_search_workflow(llm: BaseChatModel) -> CompiledStateGraph:
    completeness_grader = CompletenessGrader(llm)
    tavily_wrapper = TavilyWrapper()

    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("assess_completeness", completeness_grader.runnable)
    workflow.add_node("perform_web_search", tavily_wrapper.ainvoke)

    # Add edges
    workflow.set_entry_point("assess_completeness")
    workflow.add_conditional_edges(
        "assess_completeness",
        decide_next_steps,
        {
            "perform_web_search": "perform_web_search",
            "finish": END
        },
    )

    workflow.set_finish_point("perform_web_search")
    return workflow.compile()
