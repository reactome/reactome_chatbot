#!/usr/bin/env python

import os
from typing import Any

import chainlit as cl
import chainlit.data as cl_data
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.types import ThreadDict
from dotenv import load_dotenv

from conversational_chain.graph import RAGGraphWithMemory
from retreival_chain import create_retrieval_chain
from util.chainlit_helpers import is_feature_enabled, static_messages
from util.config_yml import Config, TriggerEvent
from util.embedding_environment import EmbeddingEnvironment
from util.logging import logging

load_dotenv()
config: Config | None = Config.from_yaml()

ENV = os.getenv("CHAT_ENV", "reactome")
logging.info(f"Selected environment: {ENV}")

llm_graph: RAGGraphWithMemory = create_retrieval_chain(
    ENV,
    EmbeddingEnvironment.get_dir(ENV),
    False,
    False,
    hf_model=EmbeddingEnvironment.get_model(ENV),
)

if os.getenv("POSTGRES_CHAINLIT_DB"):
    CHAINLIT_DB_URI = f"postgresql+psycopg://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@postgres:5432/{os.getenv('POSTGRES_CHAINLIT_DB')}?sslmode=disable"
    cl_data._data_layer = SQLAlchemyDataLayer(conninfo=CHAINLIT_DB_URI)
else:
    logging.warning("POSTGRES_CHAINLIT_DB undefined; Chainlit persistence disabled.")

if os.getenv("OAUTH_AUTH0_CLIENT_ID"):

    @cl.oauth_callback
    def oauth_callback(
        provider_id: str,
        token: str,
        raw_user_data: dict[str, str],
        default_user: cl.User,
    ) -> cl.User | None:
        return default_user


@cl.set_chat_profiles
async def chat_profile() -> list[cl.ChatProfile]:
    return [
        cl.ChatProfile(
            name="React-to-me",
            markdown_description="An AI assistant specialized in exploring **Reactome** biological pathways and processes.",
        )
    ]


@cl.on_chat_start
async def start() -> None:
    thread_id: str = cl.user_session.get("id")
    cl.user_session.set("thread_id", thread_id)
    await static_messages(config, TriggerEvent.on_chat_start)


@cl.on_chat_resume
async def resume(thread: ThreadDict) -> None:
    await static_messages(config, TriggerEvent.on_chat_resume)


@cl.on_chat_end
async def end() -> None:
    await static_messages(config, TriggerEvent.on_chat_end)


@cl.on_message
async def main(message: cl.Message) -> None:
    await static_messages(config, TriggerEvent.on_message)

    message_count: int = cl.user_session.get("message_count", 0) + 1
    cl.user_session.set("message_count", message_count)

    thread_id: str = cl.user_session.get("thread_id")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        force_stream_final_answer=True,  # we're not using prefix tokens
    )
    enable_postprocess: bool = is_feature_enabled(config, "postprocessing")
    result: dict[str, Any] = await llm_graph.ainvoke(
        message.content,
        callbacks=[cb],
        thread_id=thread_id,
        enable_postprocess=enable_postprocess,
    )
    if (
        enable_postprocess
        and cb.final_stream
        and len(result["additional_content"]["search_results"]) > 0
    ):
        sent_message: cl.Message = cb.final_stream
        search_results_element = cl.CustomElement(
            name="SearchResults",
            props={"results": result["additional_content"]["search_results"]},
        )
        sent_message.elements = [search_results_element]  # type: ignore
        await sent_message.update()
    await static_messages(config, after_messages=message_count)
