#!/usr/bin/env python

import os

import chainlit as cl
import chainlit.data as cl_data
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.types import ThreadDict
from dotenv import load_dotenv

from conversational_chain.graph import RAGGraphWithMemory
from retreival_chain import create_retrieval_chain
from util.embedding_environment import EmbeddingEnvironment
from util.logging import logging


load_dotenv()

ENV = os.getenv("CHAT_ENV", "reactome")
logging.info(f"Selected environment: {ENV}")

CHAINLIT_DB_URI = f"postgresql+psycopg://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@postgres:5432/{os.getenv('POSTGRES_CHAINLIT_DB')}?sslmode=disable"
cl_data._data_layer = SQLAlchemyDataLayer(conninfo=CHAINLIT_DB_URI)

llm_graph: RAGGraphWithMemory = create_retrieval_chain(
    ENV,
    EmbeddingEnvironment.get_dir(ENV),
    False,
    False,
    hf_model=EmbeddingEnvironment.get_model(ENV),
)


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
            markdown_description="An AI assistant specialized in exploring **Reactome** biological pathways and processes."
        )
    ]


@cl.on_chat_start
async def start() -> None:
    chat_profile: str = cl.user_session.get("chat_profile")
    initial_message = (
        f"Welcome to {chat_profile}, your interactive chatbot for exploring Reactome!"
        " Ask me about biological pathways and processes."
    )
    await cl.Message(content=initial_message).send()


@cl.on_chat_resume
async def resume(thread: ThreadDict) -> None:
    pass


@cl.on_message
async def main(message: cl.Message) -> None:
    session_id: str = cl.user_session.get("id")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        force_stream_final_answer=True,  # we're not using prefix tokens
    )
    await llm_graph.ainvoke(
        message.content,
        callbacks=[cb],
        thread_id=session_id,
    )
