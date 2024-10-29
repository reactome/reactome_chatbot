import os
from pathlib import Path
from typing import AsyncGenerator, Callable, Optional

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_chroma.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_huggingface import (HuggingFaceEmbeddings,
                                   HuggingFaceEndpointEmbeddings)
from langchain_ollama.chat_models import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from conversational_chain.chain import RAGChainWithMemory
from conversational_chain.memory import ChatHistoryMemory
from reactome.metadata_info import descriptions_info, field_info


def list_chroma_subdirectories(directory: Path) -> list[str]:
    subdirectories = list(
        chroma_file.parent.name for chroma_file in directory.glob("*/chroma.sqlite3")
    )
    return subdirectories


async def invoke(self, query: str) -> AsyncGenerator[str, None]:
    async for message in self.astream(query):
        yield message


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


def initialize_retrieval_chain(
    env: str,
    embeddings_directory: Path,
    commandline: bool,
    verbose: bool,
    ollama_model: Optional[str] = None,
    ollama_url: str = "http://localhost:11434",
    hf_model: Optional[str] = None,
    device: str = "cpu",
) -> RAGChainWithMemory:
    memory = ChatHistoryMemory()

    callbacks: list[BaseCallbackHandler] = []
    if commandline:
        callbacks = [StreamingStdOutCallbackHandler()]

    # Define llm without redefinition
    llm: ChatOllama | ChatOpenAI

    if ollama_model is None:  # Use OpenAI when Ollama not specified
        llm = ChatOpenAI(
            temperature=0.0,
            streaming=commandline,
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
    retriever_list: list[SelfQueryRetriever] = []
    for subdirectory in list_chroma_subdirectories(embeddings_directory):
        embedding = embedding_callable()
        vectordb = Chroma(
            persist_directory=str(embeddings_directory / subdirectory),
            embedding_function=embedding,
        )

        retriever = SelfQueryRetriever.from_llm(
            llm=llm,
            vectorstore=vectordb,
            document_contents=descriptions_info[subdirectory],
            metadata_field_info=field_info[subdirectory],
            search_kwargs={"k": 10},
        )
        retriever_list.append(retriever)

    reactome_retriever = EnsembleRetriever(
        retrievers=retriever_list, weights=[0.25] * len(retriever_list)
    )

    RAGChainWithMemory.invoke = invoke
    qa = RAGChainWithMemory(
        memory=memory,
        retriever=reactome_retriever,
        llm=llm,
    )

    return qa
