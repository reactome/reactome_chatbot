from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.utils.runnable import RunnableLike

from agent.tasks.rephrase import create_rephrase_chain
from agent.tasks.safety_checker import create_safety_checker
from tools.preprocessing.state import PreprocessingState


def create_task_wrapper(
    task: Runnable,
    input_mapper: Callable[[PreprocessingState], dict[str, Any]],
    output_mapper: Callable[[Any], PreprocessingState],
) -> RunnableLike:
    """Wrap a runnable with state mappers."""

    async def _wrapper(
        state: PreprocessingState, config: RunnableConfig
    ) -> PreprocessingState:
        result = await task.ainvoke(input_mapper(state), config)
        return output_mapper(result)

    return _wrapper


def create_preprocessing_workflow(llm: BaseChatModel) -> CompiledStateGraph:
    """Create the preprocessing workflow with rephrasing and safety checking."""

    tasks = {
        "rephrase_query": (
            create_rephrase_chain(llm),
            lambda state: {
                "user_input": state["user_input"],
                "chat_history": state.get("chat_history", []),
            },
            lambda result: PreprocessingState(rephrased_input=result),
        ),
        "safety_check": (
            create_safety_checker(llm),
            lambda state: {"rephrased_input": state["rephrased_input"]},
            lambda result: PreprocessingState(
                safety=result.safety.lower(),
                reason_unsafe=result.reason_unsafe,
            ),
        ),
    }

    workflow = StateGraph(PreprocessingState)

    for node_name, (task, input_mapper, output_mapper) in tasks.items():
        workflow.add_node(
            node_name, create_task_wrapper(task, input_mapper, output_mapper)
        )

    workflow.set_entry_point("rephrase_query")
    workflow.add_edge("rephrase_query", "safety_check")
    workflow.set_finish_point("safety_check")

    return workflow.compile()


