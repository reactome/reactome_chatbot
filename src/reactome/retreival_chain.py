import os
from typing import AsyncGenerator, Callable, List

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.retrievers import MergerRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.reactome.metadata_info import descriptions_info, field_info


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
    return lambda: HuggingFaceEmbeddings(
        model_name=hf_model,
        model_kwargs={"device": device, "trust_remote_code": True},
        encode_kwargs={"batch_size": 12, "normalize_embeddings": False},
    )


def initialize_retrieval_chain(
    embeddings_directory: str,
    commandline: bool,
    verbose: bool,
    ollama_model: str = None,
    ollama_url: str = "http://localhost:11434",
    hf_model: str = None,
    device: str = "cpu",
) -> ConversationalRetrievalChain:
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer",
    )

    system_prompt = r"""As a Reactome Curator with extensive knowledge in biological pathways, answer users' questions as comprehensively and accurately as possible based on the provided context. Provide any useful background information required to help users better understand the significance of the results based on the context provided.
    When providing answers, please adhere to the following guidelines:
  1. Given the user question and the provided context only, answer the question as comprehensively and accurately as possible and provide any useful background information to the user's question.
  2. If the answer cannot be derived from the provided context, do not ever provide a response, state that the information is not currently available in Reactome.
  3. If the user's question isn't a question or not related to Reactome, explain that you are an interactive chatbot designed to enhance their experience with Reactome.
  4. keep track of all the sources that are directly used to derive the final answer.
  5. Always keep track of all the sources that are directly used to derive the final answer, and return them along with the protien name according to the following format: <a href="url">display_name</a> as citations.
  6. Always provide the citations in the format requested, in point-form at the end of the response paragraph.

  Ensure your responses are detailed and informative, enhancing the user's understanding of biological pathways.
    """
    new_prompt = r"""context: {context}
    Question: {question}
    Helpful Answer:
    """

    # Combine system prompt for OpenAI
    if ollama_model is None:  # Use OpenAI when Ollama not specified
        new_prompt = system_prompt + new_prompt

    messages = [SystemMessagePromptTemplate.from_template(new_prompt)]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    callbacks: List[StreamingStdOutCallbackHandler] = []
    if commandline:
        callbacks = [StreamingStdOutCallbackHandler()]

    if ollama_model is None:  # Use OpenAI when Ollama not specified
        llm = ChatOpenAI(
            temperature=0.0,
            streaming=commandline,
            callbacks=callbacks,
            verbose=verbose,
            model="gpt-3.5-turbo-0125",
        )
    else:  # Otherwise use Ollama
        llm = ChatOllama(
            temperature=0.0,
            callbacks=callbacks,
            verbose=verbose,
            model=ollama_model,
            base_url=ollama_url,
            system=system_prompt,
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
            search_kwargs={"k": 15},
        )

        retriever = vectordb.as_retriever(
            search_type="similarity", search_kwargs={"k": 15}
        )

        retriever_list.append(retriever)

    lotr = MergerRetriever(retrievers=retriever_list)

    ConversationalRetrievalChain.invoke = invoke
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=lotr,
        verbose=verbose,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
    )

    return qa
