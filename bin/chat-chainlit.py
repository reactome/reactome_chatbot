import os

import chainlit as cl
from chainlit.data.base import BaseDataLayer
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.types import ThreadDict
from dotenv import load_dotenv
from langchain_community.callbacks import OpenAICallbackHandler

from agent.graph import AgentGraph
from agent.profiles import ProfileName, get_chat_profiles
from agent.profiles.base import OutputState
from util.chainlit_helpers import (is_feature_enabled, message_rate_limited,
                                   save_openai_metrics, static_messages,
                                   update_search_results)
from util.config_yml import Config, TriggerEvent
from util.logging import logging

load_dotenv()
config: Config | None = Config.from_yaml()

profiles: list[ProfileName] = config.profiles if config else [ProfileName.React_to_Me]
llm_graph = AgentGraph(profiles)

if os.getenv("POSTGRES_CHAINLIT_DB"):
    CHAINLIT_DB_URI = f"postgresql+psycopg://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@postgres:5432/{os.getenv('POSTGRES_CHAINLIT_DB')}?sslmode=disable"

    @cl.data_layer
    def get_data_layer() -> BaseDataLayer:
        return SQLAlchemyDataLayer(conninfo=CHAINLIT_DB_URI)

else:
    logging.warning("POSTGRES_CHAINLIT_DB undefined; Chainlit persistence disabled.")

if os.getenv("CHAINLIT_AUTH_SECRET"):

    @cl.oauth_callback
    def oauth_callback(
        provider_id: str,
        token: str,
        raw_user_data: dict[str, str],
        default_user: cl.User,
    ) -> cl.User | None:
        return default_user


@cl.set_chat_profiles
async def chat_profiles() -> list[cl.ChatProfile]:
    return [
        cl.ChatProfile(
            name=profile.name,
            markdown_description=profile.description,
        )
        for profile in get_chat_profiles(profiles)
    ]


@cl.on_chat_start
async def start() -> None:
    if cl.user_session.get("thread_id") is None:
        cl.user_session.set("thread_id", cl.user_session.get("id"))
    await static_messages(config, TriggerEvent.on_chat_start)


@cl.on_chat_resume
async def resume(thread: ThreadDict) -> None:
    await static_messages(config, TriggerEvent.on_chat_resume)


@cl.on_chat_end
async def end() -> None:
    await static_messages(config, TriggerEvent.on_chat_end)


@cl.on_message
async def main(message: cl.Message) -> None:
    if await message_rate_limited(config):
        return

    await static_messages(config, TriggerEvent.on_message)

    message_count: int = cl.user_session.get("message_count", 0) + 1
    cl.user_session.set("message_count", message_count)

    chat_profile: str = cl.user_session.get("chat_profile")

    thread_id: str = cl.user_session.get("thread_id")

    chainlit_cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        force_stream_final_answer=True,  # we're not using prefix tokens
    )
    openai_cb = OpenAICallbackHandler()

    enable_postprocess: bool = is_feature_enabled(config, "postprocessing")
    result: OutputState = await llm_graph.ainvoke(
        message.content,
        chat_profile.lower(),
        callbacks=[chainlit_cb, openai_cb],
        thread_id=thread_id,
        enable_postprocess=enable_postprocess,
    )

    if (
        enable_postprocess
        and chainlit_cb.final_stream
        and len(result["additional_content"]["search_results"]) > 0
    ):
        await update_search_results(
            result["additional_content"]["search_results"],
            chainlit_cb.final_stream,
        )

    await static_messages(config, after_messages=message_count)

    save_openai_metrics(message.id, openai_cb)
