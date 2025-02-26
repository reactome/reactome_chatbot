from pathlib import Path

import chromadb.config
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_chroma.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable

from retrievers.rag_chain import create_rag_chain
from retrievers.reactome.metadata_info import descriptions_info, field_info
from retrievers.reactome.prompt import qa_prompt
from util.embedding_environment import EmbeddingEnvironment

chroma_settings = chromadb.config.Settings(anonymized_telemetry=False)


def list_chroma_subdirectories(directory: Path) -> list[str]:
    subdirectories = list(
        chroma_file.parent.name for chroma_file in directory.glob("*/chroma.sqlite3")
    )
    return subdirectories


def create_reactome_rag(
    llm: BaseChatModel,
    embedding: Embeddings,
    embeddings_directory: Path = EmbeddingEnvironment.get_dir("reactome"),
    *,
    streaming: bool = False,
) -> Runnable:
    retriever_list: list[BaseRetriever] = []
    for subdirectory in list_chroma_subdirectories(embeddings_directory):
        # set up BM25 retriever
        csv_file_name = subdirectory + ".csv"
        reactome_csvs_dir: Path = embeddings_directory / "csv_files"
        loader = CSVLoader(file_path=reactome_csvs_dir / csv_file_name)
        data = loader.load()
        bm25_retriever = BM25Retriever.from_documents(data)
        bm25_retriever.k = 10

        # set up vectorstore SelfQuery retriever
        vectordb = Chroma(
            persist_directory=str(embeddings_directory / subdirectory),
            embedding_function=embedding,
            client_settings=chroma_settings,
        )

        selfq_retriever = SelfQueryRetriever.from_llm(
            llm=llm,
            vectorstore=vectordb,
            document_contents=descriptions_info[subdirectory],
            metadata_field_info=field_info[subdirectory],
            search_kwargs={"k": 10},
        )
        rrf_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, selfq_retriever], weights=[0.2, 0.8]
        )
        retriever_list.append(rrf_retriever)

    reactome_retriever = MergerRetriever(retrievers=retriever_list)

    if streaming:
        llm = llm.model_copy(update={"streaming": True})

    return create_rag_chain(llm, reactome_retriever, qa_prompt)
