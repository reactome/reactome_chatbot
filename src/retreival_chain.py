import os

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import MergerRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate

from src.metadata_info import descriptions_info, field_info


def list_subdirectories(directory):
    subdirectories = [
        f.name
        for f in os.scandir(directory)
        if f.is_dir() and f.name != "." and f.name != ".."
    ]
    return subdirectories


def initialize_retrieval_chain(embeddings_directory, verbose):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    new_prompt = r"""Use the following pieces of context to answer the question at the end. Please follow the following rules:
    1. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    2. If you find the answer, answer the question and add the list of the sources that are **directly** used to derive the answer. Exclude the sources that are irrelevant to the final answer.
    3. Format the citations like this: https://reactome.org/content/detail/*Source_ID*
    {context}
    Question: {question}
    Helpful Answer:
    """
    messages = [SystemMessagePromptTemplate.from_template(new_prompt)]
    qa_prompt = ChatPromptTemplate.from_messages(messages)


    llm = ChatOpenAI(temperature=0.0,
                     streaming=True,
                     verbose=verbose,
                     model="gpt-3.5-turbo-0125")
    retriever_list = []
    for subdirectory in list_subdirectories(embeddings_directory):
        embedding = OpenAIEmbeddings()
        vectordb = Chroma(
            persist_directory=embeddings_directory + "/" + subdirectory,
            embedding_function=embedding,
        )
        retriever = SelfQueryRetriever.from_llm(
            llm=llm,
            vectorstore=vectordb,
            document_contents=descriptions_info[subdirectory],
            metadata_field_info=field_info,
            search_kwargs={"k": 15},
        )

        retriever = vectordb.as_retriever(
            search_type="similarity", search_kwargs={"k": 15}
        )
        retriever_list.append(retriever)

    lotr = MergerRetriever(retrievers=retriever_list)

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=lotr,
        verbose=verbose,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': qa_prompt}
    )
    return qa
