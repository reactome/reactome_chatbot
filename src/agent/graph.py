import asyncio
import os
import re
from typing import Any, cast

from langchain_core.callbacks.base import Callbacks
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph, StateGraph
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from agent.models import get_embedding, get_llm
from agent.profiles import ProfileName, create_profile_graphs
from agent.profiles.base import InputState, OutputState
from util.embedding_environment import EmbeddingEnvironment
from util.logging import logging

LANGGRAPH_DB_URI = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@postgres:5432/{os.getenv('POSTGRES_LANGGRAPH_DB')}?sslmode=disable"

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


# Models that accept only one temperature: their own default of 1. Any other
# value, 0.0 included, is a 400 on the first request rather than an error at
# construction:
#
#   Unsupported value: 'temperature' does not support 0.0 with this model.
#   Only the default (1) value is supported.
#
# Sending nothing is not an option -- ChatOpenAI supplies its own default of 0.7
# when the argument is omitted, and these models reject that too -- so the value
# has to be 1.0 explicitly.
#
# This is an exact-match set and not a name pattern, because the behaviour
# interleaves: gpt-5 refuses 0.0, gpt-5.1/5.2/5.4 accept it, and gpt-5.5/5.6
# refuse it again. A "gpt-5" prefix would have caught gpt-5.1 as well, and an
# earlier version of this file did exactly that -- and was wrong for eleven
# models, including the gpt-5 and o-series entries below.
#
# The list is empirical: no endpoint reports which values a model accepts, so it
# was measured. `./bin/probe_model_temperature` is the tool that measured it and
# prints this set; run it rather than reasoning about a name.
#
# Measured 2026-09-09. LLM_TEMPERATURE overrides, for a model added since.
FIXED_TEMPERATURE_MODELS = frozenset(
    {
        "chat-latest",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5.5",
        "gpt-5.6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-6-astra",
        "o3",
        "o4-mini",
    }
)
FIXED_TEMPERATURE = 1.0

# OpenAI pins dated snapshots of a model as `<model>-YYYY-MM-DD`. They behave as
# the model they pin, so the suffix is stripped rather than listing every one.
_SNAPSHOT_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def resolve_temperature(model: str) -> float:
    """The temperature to send for `model`.

    Returning 1.0 for the models above trades determinism for being able to use
    them at all. That trade is made here, once, rather than at each call site.
    """
    override = os.getenv("LLM_TEMPERATURE")
    if override is not None and override.strip() != "":
        try:
            return float(override)
        except ValueError:
            raise SystemExit(f"LLM_TEMPERATURE={override!r} is not a number.") from None
    if _SNAPSHOT_SUFFIX.sub("", model) in FIXED_TEMPERATURE_MODELS:
        return FIXED_TEMPERATURE
    return 0.0


class AgentGraph:
    def __init__(
        self,
        profiles: list[ProfileName],
    ) -> None:
        # Get base models
        embedding_model = resolve_embedding_model()
        llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        llm_base_url = os.getenv("LLM_BASE_URL", None)
        llm: BaseChatModel = get_llm(
            "openai",
            llm_model,
            base_url=llm_base_url,
            request_timeout=360.0,
            temperature=resolve_temperature(llm_model),
        )
        embedding_base_url = os.getenv("OPENAI_BASE_URL", None)
        embedding: Embeddings = get_embedding(
            "openai", embedding_model, base_url=embedding_base_url
        )

        self.uncompiled_graph: dict[str, StateGraph] = create_profile_graphs(
            profiles, llm, embedding
        )

        # The following are set asynchronously by calling initialize()
        self.graph: dict[str, CompiledStateGraph] | None = None
        self.pool: AsyncConnectionPool[AsyncConnection[dict[str, Any]]] | None = None

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

    async def initialize(self) -> dict[str, CompiledStateGraph]:
        checkpointer: BaseCheckpointSaver[str] = await self.create_checkpointer()
        return {
            profile: graph.compile(checkpointer=checkpointer)
            for profile, graph in self.uncompiled_graph.items()
        }

    async def create_checkpointer(self) -> BaseCheckpointSaver[str]:
        if not os.getenv("POSTGRES_LANGGRAPH_DB"):
            return MemorySaver()
        self.pool = AsyncConnectionPool(
            conninfo=LANGGRAPH_DB_URI,
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

    async def ainvoke(
        self,
        user_input: str,
        profile: str,
        *,
        callbacks: Callbacks,
        thread_id: str,
        enable_postprocess: bool = True,
    ) -> OutputState:
        if self.graph is None:
            self.graph = await self.initialize()
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
