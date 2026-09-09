from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

from retrievers.rag_chain import create_rag_chain
from retrievers.userguide.prompt import userguide_qa_prompt
from retrievers.userguide.retriever import create_userguide_retriever


def create_userguide_rag(
    llm: BaseChatModel,
    embedding: Embeddings,
    embeddings_directory: Path,
    *,
    streaming: bool = False,
) -> Runnable:
    userguide_retriever = create_userguide_retriever(embedding, embeddings_directory)

    if streaming:
        llm = llm.model_copy(update={"streaming": True})

    return create_rag_chain(llm, userguide_retriever, userguide_qa_prompt)
