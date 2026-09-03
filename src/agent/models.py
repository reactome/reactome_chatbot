from typing import Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_openai.chat_models.base import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings


def get_embedding(
    provider: (
        Literal[
            "openai",
            "huggingfacehub",
            "huggingfacelocal",
        ]
        | str
    ),
    model: str | None = None,
    *,
    device: str | None = "cpu",
    base_url: str | None = None,
) -> Embeddings:
    if model is None:
        provider, model = provider.split("/", 1)
    if provider == "openai":
        return OpenAIEmbeddings(model=model, base_url=base_url)
    elif provider == "huggingfacehub":
        return HuggingFaceEndpointEmbeddings(model=model)
    elif provider == "huggingfacelocal":
        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": device, "trust_remote_code": True},
            encode_kwargs={"batch_size": 12, "normalize_embeddings": False},
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_llm(
    provider: (
        Literal[
            "openai",
            "ollama",
        ]
        | str
    ),
    model: str | None = None,
    *,
    base_url: str | None = None,
    request_timeout: float | None = None,
) -> BaseChatModel:
    if model is None:
        provider, model = provider.split("/", 1)
    if provider == "openai":
        return ChatOpenAI(
            model=model,
            temperature=0.0,
            base_url=base_url,
            request_timeout=request_timeout,
        )
    elif provider == "ollama":
        return ChatOllama(
            model=model,
            temperature=0.0,
            base_url=base_url,
            request_timeout=request_timeout,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
