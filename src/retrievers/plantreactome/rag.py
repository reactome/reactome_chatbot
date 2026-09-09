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
    streaming: bool = False,
) -> Runnable:
    plantreactome_retriever = create_bm25_chroma_ensemble_retriever(
        llm,
        embedding,
        embeddings_directory,
    )

    if streaming:
        llm = llm.model_copy(update={"streaming": True})

    return create_rag_chain(llm, plantreactome_retriever, plantreactome_qa_prompt)
