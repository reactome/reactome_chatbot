from typing import List

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from nodes import (generate, 
                   grade_completeness, 
                   perform_web_search, 
                   decide_to_search_web)


class GraphState(TypedDict):
    """
    Represents the state of graph.

    Attributes:
        question: User question
        generation: LLM generated reponse to the user question.
        pmc_search: whether or not to search PMC
        pmc_question: LLM generated PMC query
        pmc_search_results: results from searching PMC.

    """

    question: str
    generation: str
    pmc_search: str
    pmc_question: str
    pmc_search_results: List[str]
    web_search: str
    web_search_results: List[str]


def build_workflow():
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("generate", generate)
    workflow.add_node("grade_completeness", grade_completeness)
    workflow.add_node("perform_web_search", perform_web_search)


## Add edges
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", "grade_completeness")
    workflow.add_conditional_edges("grade_completeness", decide_to_search_web, 
                  {"search_web": "perform_web_search",
                   "finish": END
                  },
                             )

    workflow.add_edge("perform_web_search", END) 
    return workflow



if __name__ == "__main__":
    build_workflow()
