import os
from pathlib import Path
from typing import Callable, Optional

import chromadb.config
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_chroma.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import (HuggingFaceEmbeddings,
                                   HuggingFaceEndpointEmbeddings)
from langchain_ollama.chat_models import ChatOllama
from langchain_openai.chat_models.base import BaseChatOpenAI, ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from pydantic import SecretStr

from conversational_chain.graph import RAGGraphWithMemory
from reactome.metadata_info import descriptions_info, field_info

chroma_settings = chromadb.config.Settings(anonymized_telemetry=False)


def list_chroma_subdirectories(directory: Path) -> list[str]:
    subdirectories = list(
        chroma_file.parent.name for chroma_file in directory.glob("*/chroma.sqlite3")
    )
    return subdirectories


def get_embedding(
    hf_model: Optional[str] = None, device: str = "cpu"
) -> Callable[[], Embeddings]:
    if hf_model is None:
        return OpenAIEmbeddings
    elif hf_model.startswith("openai/text-embedding-"):
        return lambda: OpenAIEmbeddings(model=hf_model[len("openai/") :])
    elif "HUGGINGFACEHUB_API_TOKEN" in os.environ:
        return lambda: HuggingFaceEndpointEmbeddings(
            huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
            model=hf_model,
        )
    else:
        return lambda: HuggingFaceEmbeddings(
            model_name=hf_model,
            model_kwargs={"device": device, "trust_remote_code": True},
            encode_kwargs={"batch_size": 12, "normalize_embeddings": False},
        )


def create_retrieval_chain(
    env: str,
    embeddings_directory: Path,
    *,
    commandline: bool = False,
    verbose: bool = False,
    ollama_model: Optional[str] = None,
    ollama_url: str = "http://localhost:11434",
    hf_model: Optional[str] = None,
    ds_model: Optional[str] = None,
    device: str = "cpu",
) -> RAGGraphWithMemory:
    callbacks: list[BaseCallbackHandler] = []
    if commandline:
        callbacks = [StreamingStdOutCallbackHandler()]

    llm: BaseChatModel
    if ds_model and "DEEPSEEK_API_KEY" in os.environ:
        llm = BaseChatOpenAI(
            model=ds_model,
            api_key=SecretStr(os.environ["DEEPSEEK_API_KEY"]),
            base_url="https://api.deepseek.com",
            temperature=0.0,
            callbacks=callbacks,
            verbose=verbose,
        )
    elif ollama_model is None:  # Use OpenAI when Ollama not specified
        llm = ChatOpenAI(
            temperature=0.0,
            callbacks=callbacks,
            verbose=verbose,
            model="gpt-4o-mini",
        )
    else:  # Otherwise use Ollama
        llm = ChatOllama(
            temperature=0.0,
            callbacks=callbacks,
            verbose=verbose,
            model=ollama_model,
            base_url=ollama_url,
        )

    # Get OpenAIEmbeddings (or HuggingFaceEmbeddings model if specified)
    embedding_callable = get_embedding(hf_model, device)

    # Adjusted type for retriever_list
    retriever_list: list[BaseRetriever] = []
    for subdirectory in list_chroma_subdirectories(embeddings_directory):
        # set up BM25 retriever
        csv_file_name = subdirectory + ".csv"
        reactome_csvs_dir: Path = embeddings_directory / "csv_files"
        loader = CSVLoader(file_path=reactome_csvs_dir / csv_file_name)
        data = loader.load()
        bm25_retriever = BM25Retriever.from_documents(data)
        bm25_retriever.k = 15

        # set up vectorstore SelfQuery retriever
        embedding = embedding_callable()
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
            search_kwargs={"k": 15},
        )
        rrf_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, selfq_retriever], weights=[0.2, 0.8]
        )
        retriever_list.append(rrf_retriever)

    reactome_retriever = MergerRetriever(retrievers=retriever_list)

    qa = RAGGraphWithMemory(
        retriever=reactome_retriever,
        llm=llm,
    )

    return qa
