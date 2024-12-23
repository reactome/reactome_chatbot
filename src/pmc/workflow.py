from typing import List

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from nodes import (decide_to_search_pmc, generate, grade_completeness,
                           perform_pmc_search, transform_query)


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


def build_workflow():
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("generate", generate)
    workflow.add_node("grade_completeness", grade_completeness)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("perform_pmc_search", perform_pmc_search)

    # Add edges
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", "grade_completeness")
    workflow.add_conditional_edges(
        "grade_completeness",
        decide_to_search_pmc,
        {"transform_query": "transform_query", "finish": END},
    )
    workflow.add_edge("transform_query", "perform_pmc_search")
    workflow.add_edge("perform_pmc_search", END)

    return workflow


def run_workflow():
    from pprint import pprint

    # Build the workflow
    workflow = build_workflow()
    app = workflow.compile()

    # Get user input from terminal
    question = input("Enter your question: ")

    # Initialize the state
    inputs = {"question": question}

    # Run the workflow and stream outputs
    for output in app.stream(inputs):
        for key, value in output.items():
            # Print node name
            pprint(f"Node '{key}':")
            # Print optional full state at each node
            # pprint(value, indent=2, width=80, depth=None)
        pprint("\n---\n")

    # Final generation
    pprint(value.get("generation", "No response generated"))


if __name__ == "__main__":
    run_workflow()
