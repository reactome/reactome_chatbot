from typing import Annotated, Literal, TypedDict

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph.message import add_messages

from tools.external_search.state import SearchState, WebSearchResult
from tools.external_search.workflow import create_search_workflow
from tools.preprocessing.state import PreprocessingState
from tools.preprocessing.workflow import create_preprocessing_workflow

# Constants
SAFETY_SAFE: Literal["true"] = "true"
SAFETY_UNSAFE: Literal["false"] = "false"
DEFAULT_LANGUAGE: str = "English"


class AdditionalContent(TypedDict, total=False):
    """Additional content sent on graph completion."""

    search_results: list[WebSearchResult]


class InputState(TypedDict, total=False):
    """Input state for user queries."""

    user_input: str


class OutputState(TypedDict, total=False):
    """Output state for responses."""

    answer: str
    additional_content: AdditionalContent


class BaseState(InputState, OutputState, total=False):
    """Base state containing all common fields for agent workflows."""

    rephrased_input: str
    chat_history: Annotated[list[BaseMessage], add_messages]

    # Preprocessing results
    safety: str
    reason_unsafe: str
    expanded_queries: list[str]
    detected_language: str


class BaseGraphBuilder:
    """Base class for all graph builders with common preprocessing and postprocessing."""

    def __init__(self, llm: BaseChatModel, embedding: Embeddings) -> None:
        """Initialize with LLM and embedding models."""
        self.preprocessing_workflow: Runnable = create_preprocessing_workflow(llm)
        self.search_workflow: Runnable = create_search_workflow(llm)

    async def preprocess(self, state: BaseState, config: RunnableConfig) -> BaseState:
        """Run the complete preprocessing workflow and map results to state."""
        result: PreprocessingState = await self.preprocessing_workflow.ainvoke(
            PreprocessingState(
                user_input=state["user_input"],
                chat_history=state["chat_history"],
            ),
            config,
        )

        return self._map_preprocessing_result(result)

    def _map_preprocessing_result(self, result: PreprocessingState) -> BaseState:
        """Map preprocessing results to BaseState with defaults."""
        return BaseState(
            rephrased_input=result["rephrased_input"],
            safety=result.get("safety", SAFETY_SAFE),
            reason_unsafe=result.get("reason_unsafe", ""),
            expanded_queries=result.get("expanded_queries", []),
            detected_language=result.get("detected_language", DEFAULT_LANGUAGE),
        )

    async def postprocess(self, state: BaseState, config: RunnableConfig) -> BaseState:
        """Postprocess that preserves existing state and conditionally adds search results."""
        search_results: list[WebSearchResult] = []

        # Only run external search for safe questions
        if (
            state.get("safety") == SAFETY_SAFE
            and config["configurable"]["enable_postprocess"]
        ):
            result: SearchState = await self.search_workflow.ainvoke(
                SearchState(
                    input=state["rephrased_input"],
                    generation=state["answer"],
                ),
                config=RunnableConfig(callbacks=config["callbacks"]),
            )
            search_results = result["search_results"]

        # Create new state with updated additional_content
        return BaseState(
            **{
                **state,
                "additional_content": AdditionalContent(search_results=search_results),
            }
        )
