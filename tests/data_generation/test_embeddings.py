"""The embedding model a bundle is built with is not optional.

A bundle generated with the wrong model is a valid Chroma database that answers
every query with nonsense: the documents were embedded by one model and the
query is embedded by another. Nothing reports it, because nothing is broken --
the numbers just do not mean the same thing.

There used to be four copies of this selection and they disagreed. Three fell
back to `OpenAIEmbeddings()`'s own default (text-embedding-ada-002, 1536
dimensions) and one hardcoded text-embedding-3-large (3072). None looked at the
path the bundle was being written to, which is the only thing that says which
model it is supposed to hold.
"""

from typing import cast

import pytest
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_openai import OpenAIEmbeddings

from data_generation.embeddings import build_embeddings


@pytest.fixture(autouse=True)
def _openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing OpenAIEmbeddings requires a key present, though it makes no
    call. CI has none, and these tests must not need one."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)


def test_no_model_is_an_error_not_a_default() -> None:
    """The case that produced an unusable bundle on 2026-09-14."""
    with pytest.raises(ValueError, match="no safe default"):
        build_embeddings(None)


def test_the_error_says_what_to_pass() -> None:
    with pytest.raises(ValueError, match="no safe default") as exc:
        build_embeddings(None)
    # Someone hitting this is mid-generation and needs the fix, not a diagnosis.
    assert "openai/text-embedding-3-large" in str(exc.value)


def test_an_openai_id_selects_that_exact_model() -> None:
    embeddings = cast(
        OpenAIEmbeddings, build_embeddings("openai/text-embedding-3-large")
    )
    # The prefix is stripped; the model must be what was asked for, never a
    # library default.
    assert embeddings.model == "text-embedding-3-large"


def test_a_different_openai_model_is_honoured() -> None:
    embeddings = cast(
        OpenAIEmbeddings, build_embeddings("openai/text-embedding-3-small")
    )
    assert embeddings.model == "text-embedding-3-small"


def test_chunk_size_is_passed_through() -> None:
    # reactome generates at 400, the others at 500. Losing this would change
    # how every bundle is built.
    at_400 = cast(
        OpenAIEmbeddings,
        build_embeddings("openai/text-embedding-3-large", chunk_size=400),
    )
    default = cast(OpenAIEmbeddings, build_embeddings("openai/text-embedding-3-large"))
    assert at_400.chunk_size == 400
    assert default.chunk_size == 500


def test_a_hugging_face_endpoint_is_used_when_a_token_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "x")
    embeddings = build_embeddings("BAAI/bge-m3")
    assert type(embeddings).__name__ == "HuggingFaceEndpointEmbeddings"
    assert cast(HuggingFaceEndpointEmbeddings, embeddings).model == "BAAI/bge-m3"
