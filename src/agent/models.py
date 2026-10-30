from typing import Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_openai.chat_models.base import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from pydantic import SecretStr

#: Seconds for one embedding request.
EMBEDDING_TIMEOUT_SECONDS = 30.0


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
    api_key: str | None = None,
) -> Embeddings:
    if model is None:
        provider, model = provider.split("/", 1)
    if provider == "openai":
        # With a timeout. The client default is none, and embedding calls run
        # in the shared thread pool: a stalled endpoint pinned ~5 workers per
        # question, which no cancellation could free, until every to_thread
        # in the process queued behind them (review, area 3).
        return OpenAIEmbeddings(
            model=model,
            base_url=base_url,
            timeout=EMBEDDING_TIMEOUT_SECONDS,
            # SecretStr, not str: langchain-openai 1.x types api_key as SecretStr,
            # and the wrapper also keeps the key out of repr(), so a future
            # logging of the constructed client cannot write a key to the log.
            api_key=SecretStr(api_key) if api_key else None,
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
            "openai-responses",
            "ollama",
        ]
        | str
    ),
    model: str | None = None,
    *,
    base_url: str | None = None,
    request_timeout: float | None = None,
    temperature: float = 0.0,
    api_key: str | None = None,
    api_key_header: str | None = None,
) -> BaseChatModel:
    """Build a chat model. See `agent.graph.resolve_temperature` for the value.

    0.0 is what this repository wants everywhere -- the graders, the intent
    classifier and the query expander should all give the same answer twice.
    Some models refuse it, which is why this is a parameter rather than the
    constant it used to be.

    There is no way to send *no* temperature on ChatOpenAI: omitting the
    argument makes it send its own pydantic default of 0.7, which the models
    that refuse 0.0 refuse just as firmly. So the caller must pass a value the
    model accepts; it cannot opt out.

    `openai` and `openai-responses` are the same endpoint reached through the
    two API shapes -- Chat Completions and Responses. Which one works is decided
    by the endpoint, not by the model: an Azure APIM gateway that publishes only
    `POST /openai/responses` answers `POST /openai/chat/completions` with its own
    `{"statusCode": 404, "message": "Resource not found"}` before the deployment
    name in the body is ever read, so no value of `LLM_MODEL` can fix it. The
    Codex family compounds that -- it is Responses-only by design -- but the
    gateway is the binding constraint, and it is the one that is not visible in
    any error the model would have produced.

    `use_responses_api` needs langchain-openai >= 0.3.9. It is passed as None
    rather than False for the plain provider so that the library's own
    auto-routing (a Responses-only feature forces that API) still applies.
    """
    if model is None:
        provider, model = provider.split("/", 1)
    if provider in ("openai", "openai-responses"):
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url=base_url,
            request_timeout=request_timeout,  # type: ignore[call-arg]  # pydantic-generated __init__
            api_key=SecretStr(api_key) if api_key else None,
            use_responses_api=True if provider == "openai-responses" else None,
            # Azure API Management reads its subscription key from an `api-key`
            # header and answers the OpenAI SDK's default `Authorization: Bearer`
            # with 401 "Access denied due to missing subscription key" -- so the
            # key has to travel in the header the gateway looks at, not only the
            # one the SDK prefers. Measured against the endpoint: api-key 200,
            # bearer 401, both 200. An extra header is ignored by the
            # OpenAI-compatible servers that need only the bearer token, so this
            # stays off unless LLM_API_KEY_HEADER names it.
            default_headers=(
                {api_key_header: api_key} if api_key and api_key_header else None
            ),
        )
    if provider == "ollama":
        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=base_url,
            request_timeout=request_timeout,  # type: ignore[call-arg]  # pydantic-generated __init__
        )
    raise ValueError(f"Unknown provider: {provider}")
