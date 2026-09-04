import os
from pathlib import Path
from shutil import rmtree

import torch
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain_openai import OpenAIEmbeddings

from data_generation.userguide.fetch import fetch_userguide_pages
from data_generation.userguide.html_loader import UserGuideHTMLLoader
from data_generation.userguide.urls import USER_GUIDE_URLS

CHROMA_COLLECTION = "sections"
HTML_CACHE_DIR = "html_snapshots"


def upload_to_chromadb(
    embeddings_dir: str,
    docs: list[Document],
    embedding_table: str,
    hf_model: str | None = None,
    device: str | None = None,
) -> Chroma:
    embeddings_instance: Embeddings
    if hf_model is None:  # Use OpenAI
        embeddings_instance = OpenAIEmbeddings(
            chunk_size=500,
            show_progress_bar=True,
        )
    elif hf_model.startswith("openai/text-embedding-"):
        embeddings_instance = OpenAIEmbeddings(
            model=hf_model[len("openai/") :],
            chunk_size=500,
            show_progress_bar=True,
        )
    elif "HUGGINGFACEHUB_API_TOKEN" in os.environ:
        embeddings_instance = HuggingFaceEndpointEmbeddings(
            huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
            model=hf_model,
        )
    else:
        if device == "cuda":
            torch.cuda.empty_cache()
        embeddings_instance = HuggingFaceEmbeddings(
            model_name=hf_model,
            model_kwargs={"device": device, "trust_remote_code": True},
            encode_kwargs={"batch_size": 12, "normalize_embeddings": False},
        )

    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings_instance,
        persist_directory=os.path.join(embeddings_dir, embedding_table),
    )


def generate_userguide_embeddings(
    embeddings_dir: str,
    force: bool = False,
    hf_model: str | None = None,
    device: str | None = None,
    **_: object,
) -> None:
    embeddings_path = Path(embeddings_dir)
    chroma_dir = embeddings_path / CHROMA_COLLECTION
    if force and chroma_dir.exists():
        rmtree(chroma_dir)

    cache_dir = embeddings_path / HTML_CACHE_DIR
    html_paths = fetch_userguide_pages(
        USER_GUIDE_URLS,
        cache_dir=cache_dir,
        force=force,
    )

    loader = UserGuideHTMLLoader(html_paths)
    docs = loader.load()
    print(f"Loaded {len(docs)} user guide sections from {len(html_paths)} pages")

    if not docs:
        raise RuntimeError("No user guide documents were produced")

    db = upload_to_chromadb(embeddings_dir, docs, CHROMA_COLLECTION, hf_model, device)
    print(db._collection.count())
