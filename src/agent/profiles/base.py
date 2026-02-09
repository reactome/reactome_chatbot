from typing import Annotated, Literal, TypedDict

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph.message import add_messages

from agent.tasks.detect_language import create_language_detector
from agent.tasks.rephrase import create_rephrase_chain
from agent.tasks.safety_checker import SafetyCheck, create_safety_checker
from tools.external_search.state import SearchState, WebSearchResult
from tools.external_search.workflow import create_search_workflow


class AdditionalContent(TypedDict, total=False):
    search_results: list[WebSearchResult]


class InputState(TypedDict, total=False):
    user_input: str  # User input text


class OutputState(TypedDict, total=False):
    answer: str  # primary LLM response that is streamed to the user
    additional_content: AdditionalContent  # sends on graph completion


class BaseState(InputState, OutputState, total=False):
    rephrased_input: str  # LLM-generated query from user input
    chat_history: Annotated[list[BaseMessage], add_messages]

    # Preprocessing results
    safety: str  # "true" or "false" from safety check
    reason_unsafe: str  # Reason if unsafe
    detected_language: str  # Detected language


class BaseGraphBuilder:
    # NOTE: Anything that is common to all graph builders goes here

    def __init__(
        self,
        llm: BaseChatModel,
        embedding: Embeddings,
    ) -> None:
        self.rephrase_chain: Runnable = create_rephrase_chain(llm)
        self.safety_checker: Runnable = create_safety_checker(llm)
        self.language_detector: Runnable = create_language_detector(llm)
        self.search_workflow: Runnable = create_search_workflow(llm)

    async def preprocess(self, state: BaseState, config: RunnableConfig) -> BaseState:
        rephrased_input: str = await self.rephrase_chain.ainvoke(
            {
                "user_input": state["user_input"],
                "chat_history": state.get("chat_history", []),
            },
            config,
        )
        safety_check: SafetyCheck = await self.safety_checker.ainvoke(
            {"rephrased_input": rephrased_input}, config
        )
        detected_language: str = await self.language_detector.ainvoke(
            {"user_input": state["user_input"]}, config
        )
        return BaseState(
            rephrased_input=rephrased_input,
            safety=safety_check.safety,
            reason_unsafe=safety_check.reason_unsafe,
            detected_language=detected_language,
        )

    def proceed_with_research(self, state: BaseState) -> Literal["Continue", "Finish"]:
        return "Continue" if state["safety"] == "true" else "Finish"

    async def postprocess(self, state: BaseState, config: RunnableConfig) -> BaseState:
        search_results: list[WebSearchResult] = []
        if (
            config["configurable"].get("enable_postprocess")
            and state["safety"] == "true"
        ):
            result: SearchState = await self.search_workflow.ainvoke(
                SearchState(
                    input=state["rephrased_input"],
                    generation=state["answer"],
                ),
                config=RunnableConfig(callbacks=config["callbacks"]),
            )
            search_results = result["search_results"]
        return BaseState(
            additional_content=AdditionalContent(search_results=search_results)
        )
