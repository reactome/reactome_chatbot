from pathlib import Path

from langchain_chroma.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from retrievers.csv_chroma import chroma_settings

CHROMA_COLLECTION = "sections"
DEFAULT_SEARCH_K = 6


def create_userguide_retriever(
    embedding: Embeddings,
    embeddings_directory: Path | None,
    *,
    k: int = DEFAULT_SEARCH_K,
) -> BaseRetriever:
    if embeddings_directory is None:
        raise ValueError(
            "User guide embeddings are not configured. "
            "Run ./bin/embeddings_manager use <model>/userguide/<version>."
        )

    chroma_path = Path(embeddings_directory) / CHROMA_COLLECTION
    if not (chroma_path / "chroma.sqlite3").is_file():
        raise FileNotFoundError(
            f"User guide Chroma collection not found at {chroma_path}. "
            "Run ./bin/embeddings_manager make <model>/userguide/<version>."
        )

    vectordb = Chroma(
        persist_directory=str(chroma_path),
        embedding_function=embedding,
        client_settings=chroma_settings,
    )
    return vectordb.as_retriever(search_kwargs={"k": k})
