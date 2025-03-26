from typing import Annotated, TypedDict

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph.message import add_messages

from agent.tasks.rephrase import create_rephrase_chain
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


class BaseGraphBuilder:
    # NOTE: Anything that is common to all graph builders goes here

    def __init__(
        self,
        llm: BaseChatModel,
        embedding: Embeddings,
    ) -> None:
        self.rephrase_chain: Runnable = create_rephrase_chain(llm)
        self.search_workflow: Runnable = create_search_workflow(llm)

    async def preprocess(self, state: BaseState, config: RunnableConfig) -> BaseState:
        rephrased_input: str = await self.rephrase_chain.ainvoke(
            {
                "user_input": state["user_input"],
                "chat_history": state["chat_history"],
            },
            config,
        )
        return BaseState(rephrased_input=rephrased_input)

    async def postprocess(self, state: BaseState, config: RunnableConfig) -> BaseState:
        search_results: list[WebSearchResult] = []
        if config["configurable"]["enable_postprocess"]:
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
