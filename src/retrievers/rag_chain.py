from pathlib import Path
from typing import Any, Dict

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda

from retrievers.csv_chroma import create_hybrid_retriever


def create_advanced_rag_chain(
    llm: BaseChatModel,
    embedding: Embeddings,
    embeddings_directory: Path,
    system_prompt: str,
    *,
    streaming: bool = False,
) -> Runnable:
    """
    Create an advanced RAG chain with hybrid retrieval, query expansion, and streaming support.

    Args:
        llm: Language model for generation
        embedding: Embedding model for retrieval
        embeddings_directory: Directory containing embeddings and CSV files
        system_prompt: System prompt for the LLM
        streaming: Whether to enable streaming responses

    Returns:
        Runnable RAG chain
    """
    retriever = create_hybrid_retriever(embedding, embeddings_directory)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "Context:\n{context}\n\nQuestion: {input}"),
        ]
    )

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
            prompt.format_messages(
                context=context, input=user_input, chat_history=chat_history
            )
        )

        return {"answer": response.content, "context": context}

    return RunnableLambda(rag_chain)