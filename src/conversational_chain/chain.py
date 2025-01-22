from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import \
    create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable

from system_prompt.reactome_prompt import contextualize_q_prompt, qa_prompt


def create_rag_chain(llm: BaseChatModel, retriever: BaseRetriever) -> Runnable:
    # Create the history-aware retriever
    history_aware_retriever: Runnable = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=contextualize_q_prompt,
    )

    # Create the documents chain
    question_answer_chain: Runnable = create_stuff_documents_chain(
        llm=llm.model_copy(update={"streaming": True}),
        prompt=qa_prompt,
    )

    # Create the retrieval chain
    rag_chain: Runnable = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=question_answer_chain,
    )

    return rag_chain
