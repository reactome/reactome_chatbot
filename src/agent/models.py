from dataclasses import dataclass
from typing import Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_openai.chat_models.base import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

#: Seconds for one embedding request.
EMBEDDING_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class ModelParameters:
    """What every call to one chat model sends, decided once.

    `None` means the key is left out of the request, so the model uses its own
    default. Decided by `agent.graph.resolve_model_parameters`, from the provider
    and config.yml -- never from the model name.
    """

    temperature: float | None
    reasoning_effort: str | None

    def describe(self) -> str:
        """For the startup log and refusal messages."""
        temperature = "none" if self.temperature is None else str(self.temperature)
        return (
            f"temperature {temperature}, "
            f"reasoning effort {self.reasoning_effort or 'none'}"
        )


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
        # With a timeout. The client default is none, and embedding calls run
        # in the shared thread pool: a stalled endpoint pinned ~5 workers per
        # question, which no cancellation could free, until every to_thread
        # in the process queued behind them (review, area 3).
        return OpenAIEmbeddings(
            model=model, base_url=base_url, timeout=EMBEDDING_TIMEOUT_SECONDS
        )
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
    parameters: ModelParameters,
    base_url: str | None = None,
    request_timeout: float | None = None,
) -> BaseChatModel:
    """Build a chat model that sends exactly `parameters`.

    Nothing is validated here; that is `agent.graph.resolve_model_parameters`,
    and callers go through `agent.graph.build_llm` rather than calling this.
    `parameters` has no default so that a caller that skips the resolver does
    not type-check.

    A `None` is left out of the request: on langchain-openai 1.x, ChatOpenAI's
    temperature defaults to None and is then omitted, so the model uses its own
    default. (On 0.2.x it was 0.7, and there was no way to send no temperature.)
    """
    if model is None:
        provider, model = provider.split("/", 1)
    if provider == "openai":
        return ChatOpenAI(
            model=model,
            temperature=parameters.temperature,
            reasoning_effort=parameters.reasoning_effort,
            base_url=base_url,
            request_timeout=request_timeout,  # type: ignore[call-arg]  # pydantic-generated __init__
        )
    if provider == "ollama":
        return ChatOllama(
            model=model,
            temperature=parameters.temperature,
            base_url=base_url,
            request_timeout=request_timeout,  # type: ignore[call-arg]  # pydantic-generated __init__
        )
    raise ValueError(f"Unknown provider: {provider}")
