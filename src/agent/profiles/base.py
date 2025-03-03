from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from tools.external_search.state import WebSearchResult


class AdditionalContent(TypedDict):
    search_results: list[WebSearchResult]


class BaseState(TypedDict):
    # (Everything the Chainlit layer uses should be included here)

    user_input: str  # User input text
    chat_history: Annotated[list[BaseMessage], add_messages]
    context: list[Document]
    answer: str  # primary LLM response that is streamed to the user
    additional_content: AdditionalContent  # sends on graph completion


class BaseGraphBuilder:
    pass  # NOTE: Anything that is common to all graph builders goes here
