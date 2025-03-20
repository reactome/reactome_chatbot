from typing import TypedDict


class WebSearchResult(TypedDict):
    title: str
    url: str
    content: str


class SearchState(TypedDict, total=False):
    input: str  # LLM enhanced User question
    generation: str  # LLM generated reponse to the user question
    complete: str  # "Yes" or "No" to search for external resources
    search_results: list[WebSearchResult]  # Results from searching the web
