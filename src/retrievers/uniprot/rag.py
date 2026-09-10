"""Retrieval over a UniProt embeddings bundle.

PARKED, NOT ABANDONED. Deliberately not deployed: every profile default is
React-to-Me, in `config_default.yml` and in `chat-chainlit.py`'s fallback, so this
runs only if a `config.yml` names it explicitly.

Do not delete it. Helia Mohammadi did the work to make UniProt integration possible, and it is kept so the capability can be resurrected
rather than rebuilt. Equally, do not invest in it while it is parked -- it does not
need new features, and a change that merely keeps it importable is enough.

If it is ever un-parked, note that it has not been exercised since 2026-09-10, so
its behaviour is unverified even where the code still type-checks.
"""

from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

from retrievers.csv_chroma import create_bm25_chroma_ensemble_retriever
from retrievers.rag_chain import create_rag_chain
from retrievers.uniprot.prompt import uniprot_qa_prompt


def create_uniprot_rag(
    llm: BaseChatModel,
    embedding: Embeddings,
    embeddings_directory: Path,
    *,
    streaming: bool = False,
) -> Runnable:
    reactome_retriever = create_bm25_chroma_ensemble_retriever(
        llm,
        embedding,
        embeddings_directory,
    )

    if streaming:
        llm = llm.model_copy(update={"streaming": True})

    return create_rag_chain(llm, reactome_retriever, uniprot_qa_prompt)
