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
from agent.tasks.completeness_grader import CompletenessGrade, create_completeness_grader

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

    standalone_query: str
    chat_history: Annotated[list[BaseMessage], add_messages]

    # Preprocessing results
    safety: str
    reason_unsafe: str
    expanded_queries: list[str]
    detected_language: str
    user_tone: str
    complexity: str
    
    # Postprocessing results
    final_complete: str  # LLM-assessed completeness of the final answer
    did_fallback: str    # Whether external search was used


class BaseGraphBuilder:
    """Base class for all graph builders with common preprocessing and postprocessing."""

    def __init__(self, llm: BaseChatModel, embedding: Embeddings) -> None:
        """Initialize with LLM and embedding models."""
        self.preprocessing_workflow: Runnable = create_preprocessing_workflow(llm)
        self.search_workflow: Runnable = create_search_workflow(llm)
        self.completeness_checker = create_completeness_grader(llm)

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
            standalone_query=result["standalone_query"],
            safety=result.get("safety", SAFETY_SAFE),
            reason_unsafe=result.get("reason_unsafe", ""),
            expanded_queries=result.get("expanded_queries", []),
            detected_language=result.get("detected_language", DEFAULT_LANGUAGE),
            user_tone=result.get("user_tone", "lay"),
            complexity=result.get("complexity", "simple"),
        )

    async def postprocess(self, state: BaseState, config: RunnableConfig) -> BaseState:
        """Enhanced postprocess with completeness assessment and intelligent external search."""
        search_results: list[WebSearchResult] = []
        safety: str = state.get("safety", SAFETY_SAFE)
        generation: str = state.get("answer", "") or ""
        
        print(f"🔄 Postprocessing - Safety: {safety}, Answer length: {len(generation)}")
        
        # First, assess completeness if we have an answer
        final_complete: str = "Yes"  # Default to complete
        if generation:
            print("🔍 Assessing completeness of generated answer")
            completeness_async = self.completeness_checker.ainvoke(
                {"input": state.get("standalone_query", ""), "generation": generation},
                config,
            )
            completeness: CompletenessGrade = await completeness_async
            final_complete = completeness.binary_score
            print(f"🔍 Completeness assessment: {final_complete}")
        
        # Only run external search for safe questions that are incomplete
        if (
            safety == SAFETY_SAFE 
            and config["configurable"]["enable_postprocess"] 
            and final_complete == "No"
        ):
            print("🔍 Running external search due to incomplete answer")
            result: SearchState = await self.search_workflow.ainvoke(
                SearchState(
                    input=state.get("standalone_query", ""),
                    generation=state.get("answer", ""),
                ),
                config=RunnableConfig(callbacks=config["callbacks"]),
            )
            search_results = result["search_results"]
            return BaseState(
                **state,  # Copy existing state
                final_complete=final_complete,
                additional_content=AdditionalContent(search_results=search_results),
                did_fallback="Yes"
            )
        else:
            print("✅ No external search needed - answer is complete or question is unsafe")
            return BaseState(
                **state,  # Copy existing state
                final_complete=final_complete,
                additional_content=AdditionalContent(search_results=search_results),
                did_fallback="No"
            )