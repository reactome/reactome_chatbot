from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

from retrievers.rag_chain import create_rag_chain
from retrievers.userguide.prompt import userguide_qa_prompt
from retrievers.userguide.retriever import create_userguide_retriever
from util.embedding_environment import EmbeddingEnvironment


def create_userguide_rag(
    llm: BaseChatModel,
    embedding: Embeddings,
    # TODO(phase-2): resolved at import time, so importing this module requires an
    # installed embeddings bundle. Blocks unit-testing; fix with the agent-API refactor.
    embeddings_directory: Path = EmbeddingEnvironment.get_dir("userguide"),  # noqa: B008
    *,
    streaming: bool = False,
) -> Runnable:
    userguide_retriever = create_userguide_retriever(embedding, embeddings_directory)

    if streaming:
        llm = llm.model_copy(update={"streaming": True})

    return create_rag_chain(llm, userguide_retriever, userguide_qa_prompt)
