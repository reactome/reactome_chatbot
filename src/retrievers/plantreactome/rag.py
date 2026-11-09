from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

from retrievers.csv_chroma import create_bm25_chroma_ensemble_retriever
from retrievers.plantreactome.prompt import plantreactome_qa_prompt
from retrievers.rag_chain import create_rag_chain


def create_plantreactome_rag(
    llm: BaseChatModel,
    embedding: Embeddings,
    embeddings_directory: Path,
    *,
    answer_llm: BaseChatModel | None = None,
    streaming: bool = False,
) -> Runnable:
    """`llm` expands the query; `answer_llm`, when given, writes the answer."""
    plantreactome_retriever = create_bm25_chroma_ensemble_retriever(
        llm,
        embedding,
        embeddings_directory,
    )

    answer_llm = answer_llm or llm
    if streaming:
        answer_llm = answer_llm.model_copy(update={"streaming": True})

    return create_rag_chain(
        answer_llm, plantreactome_retriever, plantreactome_qa_prompt
    )
