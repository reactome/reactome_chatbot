from typing import TypedDict

from langchain_core.messages import BaseMessage


class PreprocessingState(TypedDict, total=False):
    """State for the preprocessing workflow."""

    # Input
    user_input: str  # Original user input
    chat_history: list[BaseMessage]  # Conversation history

    # Step 1: Rephrase and incorporate conversation history
    rephrased_input: str  # Standalone question with context

    # Step 2: Parallel processing
    safety: str  # "true" or "false" from safety check
    reason_unsafe: str  # Reason if unsafe
    expanded_queries: list[str]  # Alternative queries for better retrieval
    detected_language: str  # Detected language (e.g., "English", "French")
