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
    if provider == "huggingfacehub":
        return HuggingFaceEndpointEmbeddings(model=model)
    if provider == "huggingfacelocal":
        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": device, "trust_remote_code": True},
            encode_kwargs={"batch_size": 12, "normalize_embeddings": False},
        )
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
    temperature: float = 0.0,
) -> BaseChatModel:
    """Build a chat model. See `agent.graph.resolve_temperature` for the value.

    0.0 is what this repository wants everywhere -- the graders, the intent
    classifier and the query expander should all give the same answer twice.
    Some models refuse it, which is why this is a parameter rather than the
    constant it used to be.

    There is no way to send *no* temperature on langchain-openai 0.2.14:
    omitting the argument makes ChatOpenAI send its own pydantic default of 0.7,
    which the models that refuse 0.0 refuse just as firmly. So the caller must
    pass a value the model accepts; it cannot opt out.
    """
    if model is None:
        provider, model = provider.split("/", 1)
    if provider == "openai":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url=base_url,
            request_timeout=request_timeout,  # type: ignore[call-arg]  # pydantic-generated __init__
        )
    if provider == "ollama":
        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=base_url,
            request_timeout=request_timeout,  # type: ignore[call-arg]  # pydantic-generated __init__
        )
    raise ValueError(f"Unknown provider: {provider}")
