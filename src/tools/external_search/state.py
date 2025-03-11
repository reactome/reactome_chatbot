from typing import TypedDict

from util.safe_typeddict import safe_typeddict


@safe_typeddict
class WebSearchResult(TypedDict):
    title: str
    url: str
    content: str


@safe_typeddict
class GraphState(TypedDict, total=False):
    input: str  # LLM enhanced User question
    generation: str  # LLM generated reponse to the user question
    complete: str  # "Yes" or "No" to search for external resources
    search_results: list[WebSearchResult]  # Results from searching the web
