from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

from retrievers.csv_chroma import create_bm25_chroma_ensemble_retriever
from retrievers.rag_chain import create_rag_chain
from retrievers.reactome.metadata_info import (reactome_descriptions_info,
                                               reactome_field_info)
from retrievers.reactome.prompt import reactome_qa_prompt
from util.embedding_environment import EmbeddingEnvironment


def create_reactome_rag(
    llm: BaseChatModel,
    embedding: Embeddings,
    embeddings_directory: Path = EmbeddingEnvironment.get_dir("reactome"),
    *,
    streaming: bool = False,
) -> Runnable:
    reactome_retriever = create_bm25_chroma_ensemble_retriever(
        llm,
        embedding,
        embeddings_directory,
        descriptions_info=reactome_descriptions_info,
        field_info=reactome_field_info,
    )

    if streaming:
        llm = llm.model_copy(update={"streaming": True})

    return create_rag_chain(llm, reactome_retriever, reactome_qa_prompt)
