from pathlib import Path

import chromadb.config
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_chroma.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from nltk.tokenize import word_tokenize

chroma_settings = chromadb.config.Settings(anonymized_telemetry=False)


def list_chroma_subdirectories(directory: Path) -> list[str]:
    subdirectories = list(
        chroma_file.parent.name for chroma_file in directory.glob("*/chroma.sqlite3")
    )
    return subdirectories


def create_bm25_chroma_ensemble_retriever(
    llm: BaseChatModel,
    embedding: Embeddings,
    embeddings_directory: Path,
    *,
    descriptions_info: dict[str, str],
    field_info: dict[str, list[AttributeInfo]],
) -> MergerRetriever:
    retriever_list: list[BaseRetriever] = []
    for subdirectory in list_chroma_subdirectories(embeddings_directory):
        # set up BM25 retriever
        csv_file_name = subdirectory + ".csv"
        reactome_csvs_dir: Path = embeddings_directory / "csv_files"
        loader = CSVLoader(file_path=reactome_csvs_dir / csv_file_name)
        data = loader.load()
        bm25_retriever = BM25Retriever.from_documents(
            data,
            preprocess_func=lambda text: word_tokenize(
                text.casefold(), language="english"
            ),
        )
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

    return reactome_retriever
