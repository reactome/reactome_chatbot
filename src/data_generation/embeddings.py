"""One place to build the embedding model a bundle is generated with.

There were four copies of this, and they disagreed. Given `hf_model=None`,
three used `OpenAIEmbeddings()`'s own default -- `text-embedding-ada-002`, 1536
dimensions -- while `uniprot` hardcoded `text-embedding-3-large` at 3072. None
consulted the path the bundle was about to be written to, which is the only
thing that says which model the bundle is supposed to hold.

The consequence is not a crash. A bundle generated with the wrong model is a
valid Chroma database that answers every query with nonsense, because the query
is embedded by one model and the documents by another. That is the failure the
config schema already warns about:

    Do not add an embedding model here. It comes from the bundle that built the
    vectors; setting it to anything else makes retrieval silently meaningless.

Caught by accident on 2026-09-14: a user guide bundle generated this way raised
`InvalidDimensionException: Embedding dimension 3072 does not match collection
dimensionality 1536` at query time -- but only because the two models happen to
differ in size. Two 1536-dimension models would have installed cleanly and
returned noise.

So there is no default. The caller says which model, or this raises.
"""

import os

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain_openai import OpenAIEmbeddings

OPENAI_PREFIX = "openai/"


def build_embeddings(
    hf_model: str | None,
    device: str | None = None,
    chunk_size: int = 500,
) -> Embeddings:
    """The embedding model named by `hf_model`.

    `hf_model` is the model half of an embedding id -- `openai/text-embedding-3-large`,
    or a HuggingFace model name. It is not optional: the bundle is stored under a
    path that names the model, and a bundle whose contents disagree with its path
    is unusable in a way nothing reports.
    """
    if hf_model is None:
        raise ValueError(
            "No embedding model given. It is not optional and there is no safe "
            "default: the bundle is stored under a path naming the model, and one "
            "generated with a different model returns nonsense rather than an "
            "error. Pass the model from the embedding id being generated, e.g. "
            "'openai/text-embedding-3-large'."
        )

    if hf_model.startswith(OPENAI_PREFIX):
        return OpenAIEmbeddings(
            model=hf_model[len(OPENAI_PREFIX) :],
            chunk_size=chunk_size,
            show_progress_bar=True,
        )

    if "HUGGINGFACEHUB_API_TOKEN" in os.environ:
        return HuggingFaceEndpointEmbeddings(
            huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
            model=hf_model,
        )

    if device == "cuda":
        import torch

        torch.cuda.empty_cache()

    return HuggingFaceEmbeddings(
        model_name=hf_model,
        model_kwargs={"device": device, "trust_remote_code": True},
        encode_kwargs={"batch_size": 12, "normalize_embeddings": False},
    )
