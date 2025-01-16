from typing_extensions import TypedDict

class GraphState(TypedDict):
    """
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
    web_search_results: list[str]
