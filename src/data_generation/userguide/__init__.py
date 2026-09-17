import os
from pathlib import Path
from shutil import rmtree

from chromadb.api.shared_system_client import SharedSystemClient
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
    # Unconditionally, not just under `force`. Chroma.from_documents appends, so
    # a plain rebuild used to add a second copy of every section to the existing
    # collection -- the same fault that left the shipped reactome bundle with
    # 33,498 documents for 16,749 rows. `force` governs whether the HTML pages
    # are re-fetched, which is a separate question from whether this collection
    # is rebuilt from them.
    if chroma_dir.exists():
        rmtree(chroma_dir)
        SharedSystemClient.clear_system_cache()

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
