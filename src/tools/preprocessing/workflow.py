from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.utils.runnable import RunnableLike

from agent.tasks.preprocessing.tone_language import create_tone_language_detector
from agent.tasks.preprocessing.query_expansion import create_query_expander
from agent.tasks.preprocessing.rephrase import create_rephrase_chain
from agent.tasks.preprocessing.safety_checker import create_safety_checker
from tools.preprocessing.state import PreprocessingState


def create_task_wrapper(
    task: Runnable,
    input_mapper: Callable[[PreprocessingState], dict[str, Any]],
    output_mapper: Callable[[Any], PreprocessingState],
) -> RunnableLike:
    """Generic wrapper for preprocessing tasks."""

    async def _wrapper(
        state: PreprocessingState, config: RunnableConfig
    ) -> PreprocessingState:
        result = await task.ainvoke(input_mapper(state), config)
        return output_mapper(result)

    return _wrapper


def create_preprocessing_workflow(llm: BaseChatModel) -> CompiledStateGraph:
    """Create a preprocessing workflow with rephrasing and parallel processing."""

    # Task configurations
    tasks = {
        "rephrase_query": (
            create_rephrase_chain(llm),
            lambda state: {
                "user_input": state["user_input"],
                "chat_history": state["chat_history"],
            },
            lambda result: PreprocessingState(standalone_query=result),
        ),
        "safety_check": (
            create_safety_checker(llm),
            lambda state: {"standalone_query": state["standalone_query"]},
            lambda result: PreprocessingState(
                safety=result.safety, reason_unsafe=result.reason_unsafe
            ),
        ),
        "query_expansion": (
            create_query_expander(llm),
            lambda state: {"standalone_query": state["standalone_query"]},
            lambda result: PreprocessingState(expanded_queries=result),
        ),
        "detect_language": (
            create_tone_language_detector(llm),
            lambda state: {"user_input": state["user_input"]},
            lambda result: PreprocessingState(
                detected_language=result.language,
                user_tone=result.user_tone,
                complexity=result.complexity
            ),
        ),
    }

    workflow = StateGraph(PreprocessingState)

    # Add nodes
    for node_name, (task, input_mapper, output_mapper) in tasks.items():
        workflow.add_node(
            node_name, create_task_wrapper(task, input_mapper, output_mapper)
        )

    # Add a merge node that collects all results
    async def merge_results(state: PreprocessingState) -> PreprocessingState:
        """Merge all preprocessing results into a single state."""
        return state  # State is already merged by LangGraph's automatic merging

    workflow.add_node("merge_results", merge_results)

    # Configure workflow
    workflow.set_entry_point("rephrase_query")

    # Parallel execution after rephrasing
    workflow.add_edge("rephrase_query", "safety_check")
    workflow.add_edge("rephrase_query", "query_expansion") 
    workflow.add_edge("rephrase_query", "detect_language")
    
    # All parallel tasks feed into the merge node
    workflow.add_edge("safety_check", "merge_results")
    workflow.add_edge("query_expansion", "merge_results")
    workflow.add_edge("detect_language", "merge_results")
    
    # Single finish point
    workflow.set_finish_point("merge_results")

    return workflow.compile()
