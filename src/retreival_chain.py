import os
from typing import AsyncGenerator, Callable, List

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


from reactome.metadata_info import descriptions_info, field_info

from src.converesational_chain.memory import ChatHistoryMemory
from src.converesational_chain.chain import RAGChainWithMemory
from system_prompt.reactome_prompt import qa_prompt


def list_subdirectories(directory: str) -> List[str]:
    subdirectories = [
        f.name
        for f in os.scandir(directory)
        if f.is_dir() and f.name != "." and f.name != ".."
    ]
    return subdirectories


async def invoke(self, query: str) -> AsyncGenerator[str, None]:
    async for message in self.astream(query):
        yield message


def get_embedding(
    hf_model: str = None, device: str = "cpu"
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
    embeddings_directory: str,
    commandline: bool,
    verbose: bool,
    ollama_model: str = None,
    ollama_url: str = "http://localhost:11434",
    hf_model: str = None,
    device: str = "cpu",

)-> RAGChainWithMemory:
    memory = ChatHistoryMemory()

    callbacks: List[StreamingStdOutCallbackHandler] = []
    if commandline:
        callbacks = [StreamingStdOutCallbackHandler()]

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
            system=qa_prompt,
        )

    # Get OpenAIEmbeddings (or HuggingFaceEmbeddings model if specified)
    embedding_callable = get_embedding(hf_model, device)

    retriever_list: List[SelfQueryRetriever] = []
    for subdirectory in list_subdirectories(embeddings_directory):
        embedding = embedding_callable()
        vectordb = Chroma(
            persist_directory=os.path.join(embeddings_directory, subdirectory),
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

    reactome_retriever =  EnsembleRetriever(retrievers=retriever_list, weights=[0.25, 0.25, 0.25, 0.25]) 

    RAGChainWithMemory.invoke = invoke
    qa = RAGChainWithMemory(
        memory=memory,
        retriever=reactome_retriever,
        llm=llm,
    )

    return qa
