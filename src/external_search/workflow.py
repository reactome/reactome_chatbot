from typing import List

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from nodes import (generate_response, 
                   assess_completeness,  
                   perform_web_search,
                   format_external_results,
                   decide_next_steps
                  )


class GraphState(TypedDict):
    """
    Represents the state of graph. 

    Attributes:
        question: User question 
        generation: LLM generated reponse to the user question. 
        external_search: whether or not to search for external resources.
        pmc_question: LLM generated PMC query
        pmc_search_results: results from searching PMC. 
        web_search_results: results from searching the web

    """
    question: str
    generation: str
    external_search: str
    web_search_results: List[str]


def build_workflow():
    workflow = StateGraph(GraphState)

    ## Add nodes
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("assess_completeness", assess_completeness)
    workflow.add_node("perform_web_search", perform_web_search)
    workflow.add_node("format_external_results", format_external_results)


    ## Add edges
    workflow.add_edge(START, "generate_response")
    workflow.add_edge("generate_response", "assess_completeness")
    workflow.add_conditional_edges("assess_completeness", decide_next_steps, 
                    {"perform_web_search": "perform_web_search",
                    "finish": END
                    },
                                )

    workflow.add_edge("perform_web_search", "format_external_results")

    workflow.add_edge("format_external_results", END) 
    return workflow



if __name__ == "__main__":
    build_workflow()
