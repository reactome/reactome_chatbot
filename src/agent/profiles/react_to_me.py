import asyncio
import logging
from typing import Any, NotRequired, cast

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
from reactome_mcp.answer import (
    LiveReport,
    ToolCallingModel,
    answer_from_live_services,
)
from reactome_mcp.session import get_mcp_tools, is_configured
from retrievers.csv_chroma import selected_collections
from retrievers.reactome.rag import create_reactome_rag
from retrievers.userguide.rag import create_userguide_rag
from util.embedding_environment import EmbeddingEnvironment

logger = logging.getLogger(__name__)


class ReactToMeState(BaseState):
    active_sources: list[SourceName]
    # NotRequired, and it matters: this subclass is total=True, so declaring
    # it plainly would make every existing construction of the state invalid
    # and -- worse -- would claim a guarantee the runtime does not have. A
    # thread resumed from a checkpoint written before this field existed has
    # no key, which is why `generate_answer` reads it with `.get`.
    #
    # Empty, or absent, means every collection: that is what
    # `resolve_collections` does with it, and it is how the graph behaved
    # before routing. Only meaningful when the active source is `reactome`.
    collections: NotRequired[list[str]]
    # True when a live tool call raised underneath this answer. Carried so a
    # caller can tell "upstream broke" from "there is genuinely nothing" --
    # which is impossible from the answer text, because the model paraphrases
    # the failure into the same prose a correct empty result produces.
    live_tool_failed: NotRequired[bool]


class ReactToMeGraphBuilder(BaseGraphBuilder):
    def __init__(
        self,
        llm: BaseChatModel,
        embedding: Embeddings,
    ) -> None:
        super().__init__(llm, embedding)
        # A streaming copy, the same way the RAG chains get theirs. The UI
        # displays only what the callback handler streams, so an answer
        # produced without streaming is invisible however correct it is.
        self.live_llm = llm.model_copy(update={"streaming": True})

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

        # `live` is offered only where an MCP server is configured. The server
        # itself is started on the first question that needs it, not here:
        # construction is synchronous, and every caller that builds a graph
        # outside the app would otherwise have to know about the lifecycle.
        if is_configured():
            self._available_sources |= {"live"}
            logger.info("MCP configured; live Reactome lookups are available.")

        # Built only once the sources are known, so the prompt describes just
        # the destinations that exist. Offering one that is not wired up is
        # worse than not having it: the model routes there, and the fallback
        # answers as though it had been asked instead.
        self.intent_classifier: Runnable = create_intent_classifier(
            llm, self._available_sources
        )

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
        # Two rounds, not four. This override used to run all four calls back to
        # back, discarding the overlap the base class documents. Only the
        # dependencies force an order: language detection reads the raw
        # `user_input` and so needs nothing from the rephraser, while safety and
        # intent both read `rephrased_input` and so must follow it -- but not
        # each other. Both always run regardless of the safety verdict, here and
        # before, so overlapping them changes no behaviour.
        rephrased_input: str
        detected_language: str
        rephrased_input, detected_language = await asyncio.gather(
            self.rephrase_chain.ainvoke(
                {
                    "user_input": state["user_input"],
                    "chat_history": state.get("chat_history", []),
                },
                config,
            ),
            self.language_detector.ainvoke({"user_input": state["user_input"]}, config),
        )
        safety_check: SafetyCheck
        intent: QueryIntent
        safety_check, intent = await asyncio.gather(
            self.safety_checker.ainvoke({"rephrased_input": rephrased_input}, config),
            self.intent_classifier.ainvoke(
                {"rephrased_input": rephrased_input}, config
            ),
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
            collections=intent.collections,
        )

    async def _answer_from_live_services(
        self, state: ReactToMeState, config: RunnableConfig
    ) -> ReactToMeState:
        """Answer from the MCP tools instead of the vector store.

        If the server is unreachable the question falls back to retrieval, with
        a line saying the live lookup failed -- because the alternative is what
        retrieval does unaided with these questions: answer "Reactome primarily
        focuses on Homo sapiens" when Reactome has 96 species. A wrong answer
        delivered confidently is worse than a slow one.
        """
        tools = await get_mcp_tools()
        if not tools:
            logger.warning(
                "Question routed to live lookup but no MCP tools are available; "
                "falling back to retrieval."
            )
            fallback = dict(state)
            fallback["active_sources"] = ["reactome"]
            # This fallback is retrieval over the whole bundle standing in for
            # a live lookup; a question routed to `live` chose no collections,
            # and narrowing to a selection nobody made would be worse than the
            # answer it replaces.
            fallback["collections"] = []
            # The real config, not a fresh one: it carries the callbacks the UI
            # streams through, and a fallback nobody can see is not a fallback.
            return await self.generate_answer(ReactToMeState(**fallback), config)

        report = LiveReport()
        answer = await answer_from_live_services(
            # BaseChatModel satisfies ToolCallingModel at runtime; its
            # bind_tools signature is wider than the Protocol restates.
            cast(ToolCallingModel, self.live_llm),
            tools,
            state["rephrased_input"],
            language=state["detected_language"],
            chat_history=state["chat_history"] or None,
            config=config,
            report=report,
        )
        return ReactToMeState(
            chat_history=[HumanMessage(state["user_input"]), AIMessage(answer)],
            answer=answer,
            live_tool_failed=report.upstream_failed,
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
        if source == "live":
            return await self._answer_from_live_services(state, config)
        rag = self.rags[source]
        # Set around the whole retrieval rather than passed down:
        # `create_retrieval_chain` gives the retriever no path for extra
        # arguments, and LangGraph copies the context into the tasks it
        # spawns, so a value set here reaches the retriever inside them.
        # Reset in `finally`, because the graph reuses this task across turns.
        #
        # Only `reactome` has collections. `userguide` is a separate bundle
        # with one, and leaving a stale selection set would narrow it to names
        # it does not have -- which `resolve_collections` widens back, but
        # silently and with a WARNING for every question.
        # `.get`, not `[...]`: BaseState is total=False, so a state resumed
        # from a checkpoint written before this field existed has no key at
        # all. Missing means empty means every collection -- the behaviour
        # from before routing, which is the right way to fail.
        token = selected_collections.set(
            (state.get("collections") or []) if source == "reactome" else None
        )
        try:
            result: dict[str, Any] = await rag.ainvoke(
                {
                    "input": state["rephrased_input"],
                    # A separate variable, never concatenated into `input`:
                    # create_retrieval_chain passes `input` alone to the
                    # retriever, so anything folded into it reaches BM25 and
                    # the query expander.
                    "detected_language": state["detected_language"],
                    "chat_history": (
                        state["chat_history"]
                        if state["chat_history"]
                        else [HumanMessage(state["user_input"])]
                    ),
                },
                config,
            )
        finally:
            selected_collections.reset(token)
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
