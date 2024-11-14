#!/usr/bin/env python

import logging
import logging.config
import os
from typing import Optional

import chainlit as cl
import chainlit.data as cl_data
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from dotenv import load_dotenv

from retreival_chain import create_retrieval_chain
from util.embedding_environment import EmbeddingEnvironment
from conversational_chain.graph import RAGGraphWithMemory


load_dotenv()

DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": DEFAULT_LOG_LEVEL,  # Change to WARNING, ERROR, or CRITICAL
        },
    },
    "root": {
        "handlers": ["console"],
        "level": DEFAULT_LOG_LEVEL,  # Set the default log level for all loggers
    },
}
logging.config.dictConfig(LOGGING_CONFIG)

ENV = os.getenv("CHAT_ENV", "reactome")
logging.info(f"Selected environment: {ENV}")

if os.getenv("AUTH_AUTH0_CLIENT_ID"):
    @cl.oauth_callback
    def oauth_callback(
        provider_id: str,
        token: str,
        raw_user_data: dict[str, str],
        default_user: cl.User,
    ) -> Optional[cl.User]:
        return default_user

# f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@postgres:5432/{os.getenv('POSTGRES_DB')}?sslmode=disable"
CHAINLIT_DB_URI = f"postgresql+psycopg://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@postgres:5432/{os.getenv('POSTGRES_CHAINLIT_DB')}?sslmode=disable"
cl_data._data_layer = SQLAlchemyDataLayer(
    conninfo=CHAINLIT_DB_URI
)


@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="React-to-me",
            markdown_description="An AI assistant specialized in exploring **Reactome** biological pathways and processes.",
            icon="https://reactome.org/templates/favourite/favicon.ico",
        )
    ]


@cl.on_chat_start
async def start() -> None:
    chat_profile = cl.user_session.get("chat_profile")

    # Set up the embeddings directory
    embeddings_directory = EmbeddingEnvironment.get_dir(ENV)

    llm_graph = create_retrieval_chain(
        ENV,
        embeddings_directory,
        False,
        False,
        hf_model=EmbeddingEnvironment.get_model(ENV),
    )

    await llm_graph.initialize()
    cl.user_session.set("llm_graph", llm_graph)

    initial_message = (
        f"Welcome to {chat_profile}, your interactive chatbot for exploring Reactome!"
        " Ask me about biological pathways and processes."
    )
    await cl.Message(content=initial_message).send()


@cl.on_message
async def main(message: cl.Message) -> None:
    llm_graph: RAGGraphWithMemory = cl.user_session.get("llm_graph")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    res = await llm_graph.ainvoke(
        message.content,
        callbacks=[cb],
        configurable={"thread_id": "0"},  # single thread
    )
    if cb.has_streamed_final_answer and cb.final_stream is not None:
        await cb.final_stream.update()
    else:
        await cl.Message(content=res).send()


@cl.on_chat_end
async def end() -> None:
    llm_graph: RAGGraphWithMemory | None = cl.user_session.get("llm_graph")
    if llm_graph is not None:
        await llm_graph.close_conn_pool()
