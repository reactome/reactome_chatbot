import asyncio
import os
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from langchain_core.callbacks.base import Callbacks
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph, StateGraph
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from agent.models import ModelParameters, get_embedding, get_llm
from agent.profile_names import ProfileName
from agent.profiles import create_profile_graphs
from agent.profiles.base import InputState, OutputState
from reactome_mcp.session import get_mcp_tools
from util.config_yml.models import LLMConfig
from util.embedding_environment import EmbeddingEnvironment
from util.logging import logging
from util.secrets import get_db_uri

# Built at call time, not import time: with Vault the password is a short-lived
# credential issued on demand, and a module-level f-string would pin whatever
# was valid when the module was first imported.

if not os.getenv("POSTGRES_LANGGRAPH_DB"):
    logging.warning("POSTGRES_LANGGRAPH_DB undefined; falling back to MemorySaver.")


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"


def resolve_embedding_model() -> str:
    """Pick the embedding model, defaulting to whatever built the installed bundles.

    A query is embedded with this model and compared against vectors already in
    Chroma, so it has to match the model that produced them. The bundle path
    records that -- `openai/text-embedding-3-large/reactome/Release95` -- which
    makes it the source of truth rather than a constant kept in sync by hand.

    The default used to be a literal "bge-m3". That is the right model for the
    Plant Reactome deployment, which serves it from a self-hosted
    OpenAI-compatible endpoint via OPENAI_BASE_URL, but it is wrong for every
    bundle published for Reactome: against api.openai.com it is a 404 on the
    first query. Hardcoding either one breaks the other deployment, so neither
    is hardcoded.

    AgentGraph builds a single embedding shared by every profile, so all
    installed bundles must agree on the model. Disagreement is a real
    misconfiguration and is reported rather than silently resolved.
    """
    configured = os.getenv("EMBEDDING_MODEL")

    # Bundle paths are "<provider>/<model>/<database>/<version>"; the provider is
    # supplied separately to get_embedding, so only the model is wanted here.
    installed = {
        bundle.parent.parent.name for bundle in EmbeddingEnvironment.get_dict().values()
    }

    if not installed:
        return configured or DEFAULT_EMBEDDING_MODEL

    if len(installed) > 1:
        logging.error(
            f"Installed bundles were built with different embedding models "
            f"({', '.join(sorted(installed))}), but one embedding is shared by "
            "every profile. Retrieval will be meaningless for whichever does not "
            "match. Install bundles built with the same model."
        )
        return configured or DEFAULT_EMBEDDING_MODEL

    bundle_model = installed.pop()
    if configured and configured != bundle_model:
        logging.error(
            f"EMBEDDING_MODEL is {configured!r} but the installed bundle was built "
            f"with {bundle_model!r}. Queries would be embedded with a different "
            "model than the stored vectors, making retrieval meaningless. Unset "
            "EMBEDDING_MODEL, or install a matching bundle."
        )
    return configured or bundle_model


DEFAULT_LLM_MODEL = "gpt-6-luna"


def resolve_llm_model(llm_config: "LLMConfig | None") -> tuple[str, str, str | None]:
    """Pick the answering model: environment, then config.yml, then the default.

    Returns (provider, model, base_url).

    Environment beats file here, which is the reverse of util/secrets.py, where a
    mounted Docker secret beats the environment. The two are the same rule seen
    from different sides -- the more specific source wins. A secret is mounted BY
    a deployment and should beat a file committed to the repository; LLM_MODEL is
    how an operator overrides a committed config.yml for one container without
    editing it. Stating this because the inconsistency looks like a bug until you
    see which way each one points.
    """
    provider = "openai"
    model = DEFAULT_LLM_MODEL
    base_url = os.getenv("LLM_BASE_URL")

    if llm_config is not None:
        provider = llm_config.provider
        model = llm_config.model or model
        base_url = base_url or llm_config.base_url

    return provider, os.getenv("LLM_MODEL", model), base_url


#: Sent to OpenAI on every call except the answer when config.yml sets no
#: `llm.reasoning_effort`. Those calls -- rephrase, safety, language, intent,
#: query expansion, grading -- return short or structured results and all run
#: before the first answer token, so reasoning there is latency a user waits
#: through. Accepted by every current reasoning model except chat-latest when
#: measured on 2026-11-03 (specs/015-openai-reasoning-effort research R1).
DEFAULT_REASONING_EFFORT = "low"

#: Sent to OpenAI on answer calls -- those whose text reaches the user as the
#: answer -- when config.yml sets no `llm.answer_reasoning_effort`.
DEFAULT_ANSWER_REASONING_EFFORT = "medium"


def _effort_setting(answer: bool) -> tuple[str, str]:
    """The config.yml key and default for one kind of call, for messages."""
    if answer:
        return "llm.answer_reasoning_effort", DEFAULT_ANSWER_REASONING_EFFORT
    return "llm.reasoning_effort", DEFAULT_REASONING_EFFORT


def resolve_model_parameters(
    provider: str, llm_config: "LLMConfig | None", *, answer: bool = False
) -> ModelParameters:
    """What every call of one kind sends: one rule per provider.

    OpenAI: no temperature, because current models accept only their own
    default, and a reasoning effort -- `llm.answer_reasoning_effort` for answer
    calls (`answer=True`), `llm.reasoning_effort` for everything else. Ollama: a
    temperature, as before, and no reasoning effort. The model name is not an
    input. Which levels a particular OpenAI model accepts varies, and that is
    checked against the API by `check_model` rather than kept in a table here;
    the table this replaced went stale (specs/015-openai-reasoning-effort
    research R3).

    A setting that cannot be honoured stops the process, whichever kind of call
    is being resolved: both kinds refuse the same configurations. Checks run in
    a fixed order, so an operator with two problems sees the first one first.
    """
    configured_temperature = llm_config.temperature if llm_config else None
    configured_efforts = {
        "llm.reasoning_effort": llm_config.reasoning_effort if llm_config else None,
        "llm.answer_reasoning_effort": (
            llm_config.answer_reasoning_effort if llm_config else None
        ),
    }
    key, default = _effort_setting(answer)
    override = os.getenv("LLM_TEMPERATURE")
    if override is not None and override.strip() == "":
        override = None

    if provider == "openai":
        if configured_temperature is not None:
            raise SystemExit(
                f"config.yml sets llm.temperature={configured_temperature} with "
                "provider openai. OpenAI models use only their own default "
                "temperature, so none is sent. Remove it; set "
                "llm.reasoning_effort to change how much the model reasons."
            )
        if override is not None:
            raise SystemExit(
                f"LLM_TEMPERATURE={override!r} is set with provider openai. OpenAI "
                "models use only their own default temperature, so none is sent. "
                "Remove it; set llm.reasoning_effort in config.yml to change how "
                "much the model reasons."
            )
        return ModelParameters(
            temperature=None,
            reasoning_effort=configured_efforts[key] or default,
        )

    for name, configured_effort in configured_efforts.items():
        if configured_effort is not None:
            raise SystemExit(
                f"config.yml sets {name}={configured_effort!r} with provider "
                f"{provider}. Reasoning effort applies only to OpenAI models. "
                "Remove it."
            )
    if override is not None:
        try:
            return ModelParameters(temperature=float(override), reasoning_effort=None)
        except ValueError:
            raise SystemExit(f"LLM_TEMPERATURE={override!r} is not a number.") from None
    return ModelParameters(
        temperature=(
            configured_temperature if configured_temperature is not None else 0.0
        ),
        reasoning_effort=None,
    )


def check_model(
    llm: BaseChatModel,
    *,
    provider: str,
    model: str,
    parameters: ModelParameters,
    answer: bool = False,
) -> None:
    """Send one request with exactly the parameters every call of one kind
    will send; `answer` says which kind, for the fix line.

    Whether a model accepts a reasoning effort at all, and which levels, varies
    by model and is known only to the API. Asking it once at startup turns a 400
    on a user's first question into a refusal to start (constitution IV).

    No output cap: a reasoning model can spend a small cap on reasoning and fail
    for a reason unrelated to the parameters (research R1). The API's error is
    always quoted; the fix line is a best guess from its text, so a wrong guess
    still leaves the operator with the API's own words.
    """
    try:
        llm.invoke("Reply with the single word OK.")
    except Exception as exc:
        error = str(exc)
        if "reasoning_effort" in error and "Unrecognized" in error:
            fix = (
                "This model takes no reasoning effort; it is one of the older "
                "models being retired. Choose a current model (the built-in default "
                f"is {DEFAULT_LLM_MODEL})."
            )
        elif "reasoning_effort" in error:
            key, default = _effort_setting(answer)
            fix = (
                f"Set {key} to a level this model accepts (listed above), or "
                f"remove it to use {default}."
            )
        else:
            fix = "Check OPENAI_API_KEY, LLM_BASE_URL and the model name."
        raise SystemExit(
            f"{provider}/{model} rejected a startup request sent with "
            f"{parameters.describe()}:\n  {error}\n{fix}"
        ) from None


def build_llm(
    provider: str,
    model: str,
    *,
    llm_config: "LLMConfig | None",
    base_url: str | None = None,
    request_timeout: float | None = None,
    check: bool,
    answer: bool = False,
) -> BaseChatModel:
    """The one way to build a chat model: resolve, build, and optionally check.

    Every place that builds a model comes through here, so the same provider and
    settings always send the same parameters and refuse the same way. `answer`
    builds the model for answer calls, at the answer's reasoning effort. `check`
    sends the startup request for OpenAI (`check_model`); Ollama is never
    checked.
    """
    parameters = resolve_model_parameters(provider, llm_config, answer=answer)
    llm = get_llm(
        provider,
        model,
        parameters=parameters,
        base_url=base_url,
        request_timeout=request_timeout,
    )
    if check and provider == "openai":
        check_model(
            llm, provider=provider, model=model, parameters=parameters, answer=answer
        )
    return llm


MAX_CITATIONS = 12


def _is_sub_retriever(event: Mapping[str, Any]) -> bool:
    """True for the per-collection searches inside the hybrid retriever.

    Identified by the tags this repository sets itself in
    `HybridRetriever.retrieve_documents` -- `{collection}-bm25-{i}` and
    `{collection}-vector-{i}` -- rather than by retriever class name. The name
    is not usable: the user guide's own chain-level retriever is a
    `VectorStoreRetriever` too, and filtering on that would silently drop every
    user-guide citation.
    """
    return any(
        "-bm25-" in str(tag) or "-vector-" in str(tag)
        for tag in (event.get("tags") or [])
    )


def _citation_for(document: "Document") -> "AnswerEvent | None":
    """One source, from whichever identifier the collection actually carries.

    Reactome and disease-variant documents have `st_id`, a stable identifier the
    caller resolves to a detail page. Userguide documents do not and never will:
    they are documentation pages, and their identity is the page URL.

    Until 2026-09-18 that meant a userguide-routed question returned **no
    citations at all** -- a good answer with no sources, and the fastest answers
    at that. Rather than mint a fake `R-` id, which would resolve to nothing or
    to the wrong entity, a userguide source is cited by `url`.

    Exactly one of `st_id` and `url` is set, so a caller that understands only
    `st_id` keeps working by skipping what it does not recognise.
    """
    metadata = document.metadata
    stable_id = metadata.get("st_id")
    if stable_id:
        return AnswerEvent(
            kind="citation",
            st_id=str(stable_id),
            # Disease-variant documents have no display_name; their name is
            # the variant ("ABCA1 W590S [plasma membrane]"). Every one of
            # their citations went out unlabelled (review, area 3).
            display_name=str(
                metadata.get("display_name") or metadata.get("variant") or ""
            ),
        )
    source = str(metadata.get("source") or "")
    # Only a real web URL. `source` is a generic LangChain field, and the CSV
    # loaders set it to the file they read -- so without this guard a Reactome
    # document without an `st_id` cited
    # "/home/awright/git/reactome_chatbot/embeddings/openai/...", leaking a server
    # path to the website. Caught by running a real question; the unit tests used
    # clean fixtures and never saw it.
    if source.startswith(("https://", "http://")):
        # Deduplicated by URL rather than by chunk: the userguide bundle is 98
        # chunks across 10 pages, so chunk-level citations would repeat the same
        # page up to 27 times. The page title is the right label for a page URL.
        return AnswerEvent(
            kind="citation",
            url=str(source),
            display_name=str(metadata.get("page_title") or ""),
        )
    return None


"""How many sources one answer may cite.

Without a cap this emitted **315** for one question. `HybridRetriever` queries
every collection for every expanded query, and each sub-retriever raises its own
`on_retriever_end`, so the union across them is far larger than the fused set the
model actually saw. A panel cannot show 315 sources, and a caller should not have
to guess which of them mattered.

First twelve by order of appearance, which is fusion order, so the best-ranked
survive. This is a presentation limit, not a claim about what the answer used --
see research.md R3: these are what retrieval found, not what the prose cited.
"""

ANSWER_NODE = "model"
PREPROCESS_NODE = "preprocess"
"""The graph node the final answer streams from.

`preprocess` carries the rephrase, safety, language and intent calls. `model`
carries both the query expander and the answer, which is why a retriever
completing is used as the boundary between them rather than this name alone.
"""


@dataclass(frozen=True)
class AnswerEvent:
    """One thing worth telling a caller about while an answer is produced."""

    kind: Literal["token", "citation", "done"]
    text: str = ""
    st_id: str = ""
    url: str = ""
    display_name: str = ""
    state: Literal["answered", "nothing_found", "refused", "failed"] | None = None


class AgentGraph:
    def __init__(
        self,
        profiles: list[ProfileName],
        llm_config: "LLMConfig | None" = None,
    ) -> None:
        # Get base models
        embedding_model = resolve_embedding_model()
        llm_provider, llm_model, llm_base_url = resolve_llm_model(llm_config)
        # Two models, one name: `answer_llm` for the calls whose text reaches the
        # user as the answer, `llm` for everything else. They differ only in
        # reasoning effort. Both are checked against the API before the ~85s of
        # BM25 below, so a model or level the API rejects stops startup in
        # seconds -- the second only when its parameters differ from the first.
        parameters = resolve_model_parameters(llm_provider, llm_config)
        answer_parameters = resolve_model_parameters(
            llm_provider, llm_config, answer=True
        )
        llm: BaseChatModel = build_llm(
            llm_provider,
            llm_model,
            llm_config=llm_config,
            base_url=llm_base_url,
            request_timeout=360.0,
            check=True,
        )
        answer_llm: BaseChatModel = build_llm(
            llm_provider,
            llm_model,
            llm_config=llm_config,
            base_url=llm_base_url,
            request_timeout=360.0,
            check=answer_parameters != parameters,
            answer=True,
        )
        # The name only. A model id is not a secret, but this is the line a key
        # would end up on if anyone ever widened it.
        logging.info(
            f"Answering with {llm_provider}/{llm_model}: "
            f"{parameters.describe()}; answer: {answer_parameters.describe()}"
            + (" (checked)" if llm_provider == "openai" else "")
        )
        embedding_base_url = os.getenv("OPENAI_BASE_URL", None)
        embedding: Embeddings = get_embedding(
            "openai", embedding_model, base_url=embedding_base_url
        )

        self.uncompiled_graph: dict[str, StateGraph] = create_profile_graphs(
            profiles, llm, answer_llm, embedding
        )

        # The following are set asynchronously by calling initialize()
        self.graph: dict[str, CompiledStateGraph] | None = None
        self.pool: AsyncConnectionPool[AsyncConnection[dict[str, Any]]] | None = None
        self.checkpointer: BaseCheckpointSaver[str] | None = None

    def __del__(self) -> None:
        """Close the connection pool if nothing else did.

        This used to call asyncio.run() unconditionally, which raises
        RuntimeError when a loop is already running -- and __del__ can fire at
        any point, including inside the running server. Exceptions in __del__ are
        swallowed and printed, so it surfaced as noise in production logs with the
        pool still open.

        Scheduling the close with loop.create_task() is not a fix either: the
        task is not guaranteed to run if the loop is shutting down, which is
        exactly when a graph is usually collected.

        Nothing calls close_pool() explicitly today, so this is the only cleanup
        there is. The real fix is an explicit lifecycle -- close the pool from the
        application's shutdown hook -- which belongs with the agent-API work.
        """
        # getattr, not self.pool: __del__ runs even when __init__ raised part
        # way through, and then the attribute does not exist yet. That turned a
        # readable startup error into "AttributeError: 'AgentGraph' object has no
        # attribute 'pool'" printed from __del__, which is where the real cause
        # went missing.
        if getattr(self, "pool", None) is None:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No loop running, so we can drive the close to completion.
            try:
                asyncio.run(self.close_pool())
            except Exception as e:
                logging.warning(f"Could not close the connection pool: {e}")
            return
        logging.warning(
            "AgentGraph was garbage-collected while an event loop is running; "
            "its Postgres pool is still open. Close it explicitly from the "
            "application shutdown hook."
        )

    async def _ensure_graph(self) -> dict[str, CompiledStateGraph]:
        """Compile once. Concurrent first requests each compiled the graph and
        opened their own Postgres pool, and close_pool closed only the last
        (review, area 3)."""
        if self.graph is not None:
            return self.graph
        lock = getattr(self, "_init_lock", None)
        if lock is None:
            lock = self._init_lock = asyncio.Lock()
        async with lock:
            if self.graph is None:
                self.graph = await self.initialize()
            return self.graph

    async def initialize(self) -> dict[str, CompiledStateGraph]:
        checkpointer: BaseCheckpointSaver[str] = await self.create_checkpointer()
        self.checkpointer = checkpointer
        return {
            profile: graph.compile(checkpointer=checkpointer)
            for profile, graph in self.uncompiled_graph.items()
        }

    async def create_checkpointer(self) -> BaseCheckpointSaver[str]:
        if not os.getenv("POSTGRES_LANGGRAPH_DB"):
            return MemorySaver()
        conninfo = get_db_uri(os.getenv("POSTGRES_LANGGRAPH_DB"))
        if conninfo is None:
            return MemorySaver()
        self.pool = AsyncConnectionPool(
            conninfo=conninfo,
            max_size=20,
            open=False,
            timeout=30,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
            },
        )
        await self.pool.open()
        checkpointer = AsyncPostgresSaver(self.pool)
        await checkpointer.setup()
        return checkpointer

    async def close_pool(self) -> None:
        if self.pool:
            await self.pool.close()

    async def thread_holds_analysis(self, profile: str, thread_id: str) -> bool:
        """Whether a handoff seeded a reader's analysis into this thread.

        Read from the thread's own history, so it holds for as long as the
        history does -- across reconnects and restarts.
        """
        # Built here rather than answering "no": the app's startup never
        # initializes the compiled graph, so after a restart this said "no
        # analysis" until something else did -- and web search ran on a
        # thread seeded with a reader's analysis (review, area 3).
        self.graph = await self._ensure_graph()
        if profile not in self.graph:
            return False
        state = await self.graph[profile].aget_state(
            RunnableConfig(configurable={"thread_id": thread_id})
        )
        history = (state.values or {}).get("chat_history") or []
        return any(
            getattr(m, "additional_kwargs", {}).get("reactome_analysis_seed")
            for m in history
        )

    async def forget_thread(self, thread_id: str) -> None:
        """Delete a one-shot thread's checkpoints.

        The search page's answers each run on a fresh thread that nothing
        reads again. Left in place, every answer kept ~15 KiB of checkpoints
        in the MemorySaver for the life of the process, which also serves the
        chat (review, area 1a: 600 answers grew RSS by 22 MiB).
        """
        # Initialized first: before anything else had built the graph this was
        # a silent no-op, and on Postgres the thread stayed (review, area 3).
        self.graph = await self._ensure_graph()
        if self.checkpointer is not None:
            await self.checkpointer.adelete_thread(thread_id)

    async def astream_answer(
        self,
        user_input: str,
        profile: str,
        *,
        thread_id: str,
        enable_postprocess: bool = True,
        callbacks: Callbacks = None,
    ) -> AsyncIterator["AnswerEvent"]:
        """Stream one answer: its tokens, then the sources retrieval found.

        `callbacks` are passed into the run, as `ainvoke` does, so a measurement
        can see every model call on the served path. The answer endpoint passes
        none.

        Why this is not simply "stream every model token": six model calls run
        around one answer -- rephrase, safety, language, intent, query expansion,
        and the answer itself. Measured 2026-09-17, the first token of any kind
        arrives at 3.0s and belongs to the rephraser; the answer's own first token
        arrives at 36.1s. Streaming the lot would render the safety check and an
        expanded query into the caller's panel as though they were the answer.

        The expander and the answer are not separable by metadata: both run at
        `langgraph_node == "model"` with identical keys and no distinguishing
        tags. What does separate them is order -- the expander runs *inside*
        retrieval, the answer after it. So a retriever completing is the boundary.
        """
        self.graph = await self._ensure_graph()
        if profile not in self.graph:
            yield AnswerEvent(kind="done", state="failed")
            return

        retrieval_done = False
        seen_citations: set[str] = set()
        answered = False

        async for event in self.graph[profile].astream_events(
            InputState(user_input=user_input),
            config=RunnableConfig(
                callbacks=callbacks,
                configurable={
                    "thread_id": thread_id,
                    "enable_postprocess": enable_postprocess,
                },
            ),
            version="v2",
        ):
            kind = event["event"]

            if kind == "on_chain_end" and event.get("name") == PREPROCESS_NODE:
                # A question routed to the live MCP lookup is answered from tools,
                # with no retriever anywhere on the path -- so `on_retriever_end`
                # never arrives and the boundary below never opens. Every token
                # was discarded and the caller was told `nothing_found`, for
                # questions the chat answers correctly.
                #
                # It only happened where MCP is configured, which is beta and not
                # a developer's machine: without it these questions fall back to
                # the vector store and retrieve normally. The chat UI and the
                # answer sweep both use `ainvoke`, which takes the final answer
                # and never looks at this stream, so nothing else could see it.
                #
                # Opening the boundary here is safe precisely because there is no
                # retrieval on this path: the query expander lives inside the
                # retriever, so nothing else is streaming at the answer node and
                # there is nothing to tell apart.
                output = event["data"].get("output")
                if (
                    isinstance(output, dict)
                    and "live" in (output.get("active_sources") or [])
                    # Only if live tools will really answer. Without them the
                    # same node falls back to retrieval, whose query expander
                    # streams first -- and the search page showed its
                    # alternate questions as the answer (review, area 3).
                    and await get_mcp_tools() is not None
                ):
                    retrieval_done = True
                continue

            if kind == "on_retriever_end" and _is_sub_retriever(event):
                # Skip. The hybrid retriever runs one BM25 and one vector search
                # per query variant per collection, and each fires its own
                # on_retriever_end -- 51 events for one question. Taking
                # citations from those meant taking them from whichever fired
                # first, which is always the first collection's BM25.
                #
                # Measured: 97 of 97 cited documents came from reactions.csv,
                # including for "which diseases involve variants of the PTEN
                # gene", where the disease_variants documents were retrieved and
                # could never be cited. The contract calls citations "the most
                # relevant few"; they were one sub-retriever's few.
                continue

            if kind == "on_retriever_end":
                # Skip the sub-retrievers. The Reactome ensemble fires 51 of
                # these -- one per collection per query variant per retriever --
                # and only the last is the fused result. Citations were taken
                # from the first events to arrive, so every one came from
                # whichever collection happened to be iterated first: measured,
                # 97 of 97 cited documents were from reactions.csv, including for
                # "which diseases involve variants of the PTEN gene", whose
                # disease_variants documents were retrieved and never cited.
                #
                # Keyed on the child tags this repo creates itself
                # ({collection}-bm25-{i}, {collection}-vector-{i}) rather than on
                # the retriever class: the user guide's top-level retriever is a
                # VectorStoreRetriever, so filtering by class would silently drop
                # its citations instead.
                tags = event.get("tags") or []
                if any("-bm25-" in tag or "-vector-" in tag for tag in tags):
                    continue
                retrieval_done = True
                for document in event["data"].get("output") or []:
                    if len(seen_citations) >= MAX_CITATIONS:
                        break
                    citation = _citation_for(document)
                    if citation is None:
                        continue
                    key = citation.st_id or citation.url
                    if key in seen_citations:
                        continue
                    seen_citations.add(key)
                    yield citation
                continue

            if kind != "on_chat_model_stream" or not retrieval_done:
                continue
            if event.get("metadata", {}).get("langgraph_node") != ANSWER_NODE:
                continue

            text = getattr(event["data"].get("chunk"), "content", "")
            if text:
                answered = True
                yield AnswerEvent(kind="token", text=str(text))

        yield AnswerEvent(
            kind="done", state="answered" if answered else "nothing_found"
        )

    async def seed_history(
        self, profile: str, *, thread_id: str, messages: list[BaseMessage]
    ) -> bool:
        """Put a turn into a thread's history as if it had been asked here.

        For spec 013's handoff: the reader's summary becomes the thread's
        previous turn, so their first message can be a follow-up. Recorded as
        having ended at `postprocess`, the graph's last node, so the next
        `ainvoke` starts an ordinary new turn with this history in place.

        Returns False, and changes nothing, for an unknown profile.
        """
        self.graph = await self._ensure_graph()
        if profile not in self.graph:
            return False
        await self.graph[profile].aupdate_state(
            RunnableConfig(configurable={"thread_id": thread_id}),
            {"chat_history": messages},
            as_node="postprocess",
        )
        return True

    async def ainvoke(
        self,
        user_input: str,
        profile: str,
        *,
        callbacks: Callbacks,
        thread_id: str,
        enable_postprocess: bool = True,
    ) -> OutputState:
        self.graph = await self._ensure_graph()
        if profile not in self.graph:
            return OutputState()
        # ainvoke is typed dict[str, Any] | Any; the graph's output schema is
        # OutputState.
        result: OutputState = cast(
            "OutputState",
            await self.graph[profile].ainvoke(
                InputState(user_input=user_input),
                config=RunnableConfig(
                    callbacks=callbacks,
                    configurable={
                        "thread_id": thread_id,
                        "enable_postprocess": enable_postprocess,
                    },
                ),
            ),
        )
        return result
