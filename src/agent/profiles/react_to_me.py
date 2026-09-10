import logging
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph.state import StateGraph

from agent.profiles.base import BaseGraphBuilder, BaseState
from agent.tasks.intent_classifier import (
    QueryIntent,
    SourceName,
    create_intent_classifier,
    resolve_active_sources,
)
from agent.tasks.safety_checker import SafetyCheck
from agent.tasks.unsafe_question import create_unsafe_answer_generator
from retrievers.reactome.rag import create_reactome_rag
from retrievers.userguide.rag import create_userguide_rag
from util.embedding_environment import EmbeddingEnvironment

logger = logging.getLogger(__name__)


class ReactToMeState(BaseState):
    active_sources: list[SourceName]


class ReactToMeGraphBuilder(BaseGraphBuilder):
    def __init__(
        self,
        llm: BaseChatModel,
        embedding: Embeddings,
    ) -> None:
        super().__init__(llm, embedding)

        self.intent_classifier: Runnable = create_intent_classifier(llm)
        self.unsafe_answer_generator: Runnable = create_unsafe_answer_generator(
            llm, streaming=True
        )

        self.rags: dict[SourceName, Runnable] = {
            "reactome": create_reactome_rag(
                llm,
                embedding,
                EmbeddingEnvironment.require_dir("reactome"),
                streaming=True,
            ),
        }
        self._available_sources: frozenset[SourceName] = frozenset({"reactome"})
        self._register_userguide_rag(llm, embedding)

        state_graph = StateGraph(ReactToMeState)
        state_graph.add_node("preprocess", self.preprocess)
        state_graph.add_node("model", self.generate_answer)
        state_graph.add_node("generate_unsafe_response", self.generate_unsafe_response)
        state_graph.add_node("postprocess", self.postprocess)
        state_graph.set_entry_point("preprocess")
        state_graph.add_conditional_edges(
            "preprocess",
            self.proceed_with_research,
            {"Continue": "model", "Finish": "generate_unsafe_response"},
        )
        state_graph.add_edge("model", "postprocess")
        state_graph.add_edge("generate_unsafe_response", "postprocess")
        state_graph.set_finish_point("postprocess")

        self.uncompiled_graph: StateGraph = state_graph

    def _register_userguide_rag(
        self,
        llm: BaseChatModel,
        embedding: Embeddings,
    ) -> None:
        userguide_dir = EmbeddingEnvironment.get_dir("userguide")
        if userguide_dir is None:
            logger.info(
                "User guide embeddings not configured; routing will use reactome only."
            )
            return

        chroma_path = userguide_dir / "sections" / "chroma.sqlite3"
        if not chroma_path.exists():
            logger.warning(
                "User guide embeddings directory exists but Chroma DB is missing at %s",
                chroma_path,
            )
            return

        try:
            self.rags["userguide"] = create_userguide_rag(
                llm, embedding, userguide_dir, streaming=True
            )
            self._available_sources = frozenset(self.rags)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("User guide RAG unavailable: %s", exc)

    async def preprocess(
        self, state: ReactToMeState, config: RunnableConfig
    ) -> ReactToMeState:
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
        intent: QueryIntent = await self.intent_classifier.ainvoke(
            {"rephrased_input": rephrased_input}, config
        )
        active_sources = resolve_active_sources(intent.source, self._available_sources)
        if intent.source not in self._available_sources:
            logger.info(
                "Requested source %r unavailable; falling back to %r",
                intent.source,
                active_sources[0],
            )

        return ReactToMeState(
            rephrased_input=rephrased_input,
            safety=safety_check.safety,
            reason_unsafe=safety_check.reason_unsafe,
            detected_language=detected_language,
            active_sources=active_sources,
        )

    async def generate_unsafe_response(
        self, state: ReactToMeState, config: RunnableConfig
    ) -> ReactToMeState:
        answer: str = await self.unsafe_answer_generator.ainvoke(
            {
                "language": state["detected_language"],
                "user_input": state["rephrased_input"],
                "reason_unsafe": state["reason_unsafe"],
            },
            config,
        )
        return ReactToMeState(
            chat_history=[
                HumanMessage(state["user_input"]),
                AIMessage(answer),
            ],
            answer=answer,
        )

    async def generate_answer(
        self, state: ReactToMeState, config: RunnableConfig
    ) -> ReactToMeState:
        source = state["active_sources"][0]
        rag = self.rags[source]
        result: dict[str, Any] = await rag.ainvoke(
            {
                "input": state["rephrased_input"],
                # A separate variable, never concatenated into `input`:
                # create_retrieval_chain passes `input` alone to the retriever, so
                # anything folded into it reaches BM25 and the query expander.
                "detected_language": state["detected_language"],
                "chat_history": (
                    state["chat_history"]
                    if state["chat_history"]
                    else [HumanMessage(state["user_input"])]
                ),
            },
            config,
        )
        return ReactToMeState(
            chat_history=[
                HumanMessage(state["user_input"]),
                AIMessage(result["answer"]),
            ],
            answer=result["answer"],
        )


def create_reactome_graph(
    llm: BaseChatModel,
    embedding: Embeddings,
) -> StateGraph:
    return ReactToMeGraphBuilder(llm, embedding).uncompiled_graph
