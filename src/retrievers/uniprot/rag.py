from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

from retrievers.rag_chain import create_advanced_rag_chain
from retrievers.uniprot.prompt import uniprot_qa_prompt
from util.embedding_environment import EmbeddingEnvironment


def create_uniprot_rag(
    llm: BaseChatModel,
    embedding: Embeddings,
    embeddings_directory: Path = EmbeddingEnvironment.get_dir("uniprot"),
    *,
    streaming: bool = False,
) -> Runnable:
    """
    Create a UniProt-specific RAG chain with hybrid retrieval and query expansion.

    Args:
        llm: Language model for generation
        embedding: Embedding model for retrieval
        embeddings_directory: Directory containing UniProt embeddings and CSV files
        streaming: Whether to enable streaming responses

    Returns:
        Runnable RAG chain for UniProt queries
    """
    return create_advanced_rag_chain(
        llm=llm,
        embedding=embedding,
        embeddings_directory=embeddings_directory,
        system_prompt=uniprot_qa_prompt,
        streaming=streaming,
    )