from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

from retrievers.rag_chain import create_advanced_rag_chain
from retrievers.reactome.prompt import reactome_system_prompt
from util.embedding_environment import EmbeddingEnvironment


def create_reactome_rag(
    llm: BaseChatModel,
    embedding: Embeddings,
    embeddings_directory: Path = EmbeddingEnvironment.get_dir("reactome"),
    *,
    streaming: bool = False,
) -> Runnable:
    return create_advanced_rag_chain(
        llm=llm,
        embedding=embedding,
        embeddings_directory=embeddings_directory,
        system_prompt=reactome_system_prompt,
        streaming=streaming,
    )
