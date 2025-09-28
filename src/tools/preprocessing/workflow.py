from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.utils.runnable import RunnableLike

from agent.tasks.detect_language import create_language_detector
from agent.tasks.query_expansion import create_query_expander
from agent.tasks.rephrase import create_rephrase_chain
from agent.tasks.safety_checker import create_safety_checker
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
            lambda result: PreprocessingState(rephrased_input=result),
        ),
        "safety_check": (
            create_safety_checker(llm),
            lambda state: {"rephrased_input": state["rephrased_input"]},
            lambda result: PreprocessingState(
                safety=result.safety, reason_unsafe=result.reason_unsafe
            ),
        ),
        "query_expansion": (
            create_query_expander(llm),
            lambda state: {"rephrased_input": state["rephrased_input"]},
            lambda result: PreprocessingState(expanded_queries=result),
        ),
        "detect_language": (
            create_language_detector(llm),
            lambda state: {"user_input": state["user_input"]},
            lambda result: PreprocessingState(detected_language=result),
        ),
    }

    workflow = StateGraph(PreprocessingState)

    # Add nodes
    for node_name, (task, input_mapper, output_mapper) in tasks.items():
        workflow.add_node(
            node_name, create_task_wrapper(task, input_mapper, output_mapper)
        )

    # Configure workflow
    workflow.set_entry_point("rephrase_query")

    # Parallel execution after rephrasing
    for parallel_node in ["safety_check", "query_expansion", "detect_language"]:
        workflow.add_edge("rephrase_query", parallel_node)
        workflow.set_finish_point(parallel_node)

    return workflow.compile()
