from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from retrievers.rag_chain import create_rag_chain
from retrievers.userguide.prompt import userguide_qa_prompt
from retrievers.userguide.retriever import create_userguide_retriever
from util.embedding_environment import EmbeddingEnvironment

USERGUIDE_DOCUMENT_PROMPT = PromptTemplate.from_template(
    "URL: {source}\n{page_content}"
)


def create_userguide_rag(
    llm: BaseChatModel,
    embedding: Embeddings,
    embeddings_directory: Path = EmbeddingEnvironment.get_dir("userguide"),
    *,
    streaming: bool = False,
) -> Runnable:
    userguide_retriever = create_userguide_retriever(embedding, embeddings_directory)

    if streaming:
        llm = llm.model_copy(update={"streaming": True})

    return create_rag_chain(
        llm,
        userguide_retriever,
        userguide_qa_prompt,
        document_prompt=USERGUIDE_DOCUMENT_PROMPT,
    )
