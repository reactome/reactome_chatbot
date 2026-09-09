# langchain_classic, not langchain: LangChain 1.0 moved the pre-LCEL chain
# builders out of the core package into langchain-classic, which is their
# supported home rather than a deprecation shim. Rewriting this as LCEL is a
# behaviour change and does not belong in a dependency upgrade.
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable


def create_rag_chain(
    llm: BaseChatModel,
    retriever: BaseRetriever,
    qa_prompt: ChatPromptTemplate,
    *,
    document_prompt: BasePromptTemplate | None = None,
) -> Runnable:
    # Create the documents chain
    question_answer_chain: Runnable = create_stuff_documents_chain(
        llm=llm,
        prompt=qa_prompt,
        document_prompt=document_prompt,
    )

    # Create the retrieval chain
    rag_chain: Runnable = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain,
    )

    return rag_chain
