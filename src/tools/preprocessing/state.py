from typing import TypedDict

from langchain_core.messages import BaseMessage


class PreprocessingState(TypedDict, total=False):
    """State for the preprocessing workflow."""

    # Inputs
    user_input: str
    chat_history: list[BaseMessage]

    # Task outputs
    rephrased_input: str
    safety: str
    reason_unsafe: str
    expanded_queries: list[str]
    detected_language: str


