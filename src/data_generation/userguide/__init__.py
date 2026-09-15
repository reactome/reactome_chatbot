import os
from pathlib import Path
from shutil import rmtree

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from data_generation.embeddings import build_embeddings
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
    embeddings_instance = build_embeddings(hf_model, device, chunk_size=500)

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
