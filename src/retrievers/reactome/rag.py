from collections.abc import Callable
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

from retrievers.csv_chroma import create_bm25_chroma_ensemble_retriever
from retrievers.rag_chain import create_rag_chain
from retrievers.reactome.prompt import reactome_qa_prompt
from util.bundle_manifest import read_manifest


def show_fields(fields: tuple[str, ...]) -> Callable[[Document], str]:
    """Write a document as its text followed by the given metadata fields.

    Missing or blank fields are skipped.
    """

    def format_document(document: Document) -> str:
        shown = [
            f"{field}: {value}"
            for field in fields
            if (value := str(document.metadata.get(field) or "").strip())
        ]
        return "\n".join([document.page_content, *shown])

    return format_document


def create_reactome_rag(
    llm: BaseChatModel,
    embedding: Embeddings,
    embeddings_directory: Path,
    *,
    answer_llm: BaseChatModel | None = None,
    streaming: bool = False,
) -> Runnable:
    """`llm` expands the query; `answer_llm`, when given, writes the answer.

    For a bundle with a manifest, the model also sees each record's link and
    accession.
    """
    reactome_retriever = create_bm25_chroma_ensemble_retriever(
        llm,
        embedding,
        embeddings_directory,
    )

    answer_llm = answer_llm or llm
    if streaming:
        answer_llm = answer_llm.model_copy(update={"streaming": True})

    manifest = read_manifest(embeddings_directory)
    return create_rag_chain(
        answer_llm,
        reactome_retriever,
        reactome_qa_prompt,
        format_document=(
            None if manifest is None else show_fields(manifest.shown_to_model)
        ),
    )
