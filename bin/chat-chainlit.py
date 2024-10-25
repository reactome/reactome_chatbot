#!/usr/bin/env python

import logging
import logging.config
import os

import chainlit as cl

from dotenv import load_dotenv
from retreival_chain import initialize_retrieval_chain
from util.embedding_environment import EmbeddingEnvironment

load_dotenv()

default_log_level = os.getenv("LOG_LEVEL", "INFO").upper()

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
            "level": default_log_level,  # Change to WARNING, ERROR, or CRITICAL
        },
    },
    "root": {
        "handlers": ["console"],
        "level": default_log_level,  # Set the default log level for all loggers
    },
}

logging.config.dictConfig(LOGGING_CONFIG)

selected_env = os.getenv("CHAT_ENV", "reactome")
logging.info(f"Selected environment: {selected_env}")

env = os.getenv("CHAT_ENV", "reactome")


@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="React-to-me",
            markdown_description="An AI assistant specialized in exploring **Reactome** biological pathways and processes.",
            icon="https://reactome.org/templates/favourite/favicon.ico"
        )
    ]


@cl.on_chat_start
async def quey_llm() -> None:
    chat_profile = cl.user_session.get("chat_profile")
    embeddings_directory = EmbeddingEnvironment.get_dir(env)
    llm_chain = initialize_retrieval_chain(
        env,
        embeddings_directory,
        False,
        False,
        hf_model=EmbeddingEnvironment.get_model(env),
    )
    cl.user_session.set("llm_chain", llm_chain)

    initial_message: str = f"""Welcome to {chat_profile} your interactive chatbot for exploring Reactome!
        Ask me about biological pathways and processes"""
    await cl.Message(content=initial_message).send()


@cl.on_message
async def query_llm(message: cl.Message) -> None:
    llm_chain = cl.user_session.get("llm_chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True

    res = await llm_chain.ainvoke(message.content, callbacks=[cb])
    if cb.has_streamed_final_answer:
        await cb.final_stream.update()
    else:
        await cl.Message(content=res).send()
