# langchain_classic, not langchain: LangChain 1.0 moved the pre-LCEL chain
# builders out of the core package into langchain-classic, which is their
# supported home rather than a deprecation shim. Rewriting this as LCEL is a
# behaviour change and does not belong in a dependency upgrade.
from collections.abc import Callable
from typing import Any

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.combine_documents.base import (
    DEFAULT_DOCUMENT_SEPARATOR,
    DOCUMENTS_KEY,
)
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough


def create_rag_chain(
    llm: BaseChatModel,
    retriever: BaseRetriever,
    qa_prompt: ChatPromptTemplate,
    *,
    document_prompt: BasePromptTemplate | None = None,
    format_document: Callable[[Document], str] | None = None,
) -> Runnable:
    """Retrieve, then answer from the documents written into `qa_prompt`.

    `format_document` writes each document into the prompt itself, for when a
    prompt template cannot: LangChain's `document_prompt` raises on any document
    missing a field it names, and collections in one bundle carry different
    fields. Give one or the other, not both.
    """
    if document_prompt is not None and format_document is not None:
        raise ValueError("Pass document_prompt or format_document, not both.")

    question_answer_chain: Runnable = (
        create_stuff_documents_chain(
            llm=llm,
            prompt=qa_prompt,
            document_prompt=document_prompt,
        )
        if format_document is None
        else _stuff_documents(llm, qa_prompt, format_document)
    )

    # Create the retrieval chain
    rag_chain: Runnable = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain,
    )

    return rag_chain


def _stuff_documents(
    llm: BaseChatModel,
    qa_prompt: ChatPromptTemplate,
    format_document: Callable[[Document], str],
) -> Runnable:
    """`create_stuff_documents_chain`, with each document written by a function.

    The same steps, separator, run names and output parser, so callbacks and
    streaming see the same chain; only how a document becomes text differs.
    """
    if DOCUMENTS_KEY not in qa_prompt.input_variables:
        raise ValueError(f"The answer prompt has no {{{DOCUMENTS_KEY}}} variable.")

    def format_docs(inputs: dict[str, Any]) -> str:
        return DEFAULT_DOCUMENT_SEPARATOR.join(
            format_document(document) for document in inputs[DOCUMENTS_KEY]
        )

    return (
        RunnablePassthrough.assign(**{DOCUMENTS_KEY: format_docs}).with_config(
            run_name="format_inputs"
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    ).with_config(run_name="stuff_documents_chain")
