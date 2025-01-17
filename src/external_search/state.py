from typing import TypedDict

class GraphState(TypedDict):
    question: str  # User question
    generation: str  # LLM generated reponse to the user question
    search_results: str  # Formatted results from searching the web
