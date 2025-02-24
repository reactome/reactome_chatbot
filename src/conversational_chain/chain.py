from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate


from src.system_prompt.prerocess_prompt import contextualize_q_prompt



def create_rephrase_chain(llm: BaseChatModel) -> Runnable:
    return (contextualize_q_prompt | llm | StrOutputParser()).with_config(
        run_name="rephrase_question"
    )


def create_rag_chain(llm: BaseChatModel, retriever: BaseRetriever, qa_prompt: ChatPromptTemplate ) -> Runnable:
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
