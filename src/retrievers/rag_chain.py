from pathlib import Path
from typing import Any, Dict

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from retrievers.csv_chroma import create_hybrid_retriever


def create_advanced_rag_chain(
    llm: BaseChatModel,
    embedding: Embeddings,
    embeddings_directory: Path,
    prompt_template: ChatPromptTemplate,
    *,
    streaming: bool = False,
) -> Runnable:
    """
    Create an advanced RAG chain with hybrid retrieval, query expansion, and streaming support.

    Args:
        llm: Language model for generation
        embedding: Embedding model for retrieval
        embeddings_directory: Directory containing embeddings and CSV files
        prompt_template: Pre-built ChatPromptTemplate with system prompt and placeholders
        streaming: Whether to enable streaming responses

    Returns:
        Runnable RAG chain
    """
    retriever = create_hybrid_retriever(embedding, embeddings_directory)

    if streaming:
        llm = llm.model_copy(update={"streaming": True})

    async def rag_chain(inputs: Dict[str, Any]) -> Dict[str, Any]:
        user_input = inputs["input"]
        chat_history = inputs.get("chat_history", [])
        expanded_queries = inputs.get("expanded_queries", [])

        context = await retriever.ainvoke(
            {"input": user_input, "expanded_queries": expanded_queries}
        )

        response = await llm.ainvoke(
            prompt_template.format_messages(
                context=context, input=user_input, chat_history=chat_history
            )
        )

        return {"answer": response.content, "context": context}

    return RunnableLambda(rag_chain)