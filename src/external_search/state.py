from typing import TypedDict


class WebSearchResult(TypedDict):
    title: str
    url: str
    content: str


class GraphState(TypedDict):
    question: str  # User question
    generation: str  # LLM generated reponse to the user question
    external_search: str  # "Yes" or "No" to search for external resources
    search_results: list[WebSearchResult]  # Results from searching the web
