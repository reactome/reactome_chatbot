from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable


def create_rag_chain(
    llm: BaseChatModel,
    retriever: BaseRetriever,
    qa_prompt: ChatPromptTemplate,
) -> Runnable:
    # Create the documents chain
    question_answer_chain: Runnable = create_stuff_documents_chain(
        llm=llm,
        prompt=qa_prompt,
    )

    # Create the retrieval chain
    rag_chain: Runnable = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain,
    )

    return rag_chain
