import os
from typing import List, Union

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.retrievers import MergerRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.metadata_info import descriptions_info, field_info


def list_subdirectories(directory: str) -> List[str]:
    subdirectories = [
        f.name
        for f in os.scandir(directory)
        if f.is_dir() and f.name != "." and f.name != ".."
    ]
    return subdirectories


async def invoke(self, query: str) -> Union[str, None]:
    async for message in self.astream(query):
        yield message


def initialize_retrieval_chain(
    embeddings_directory: str,
    commandline: bool,
    verbose: bool,
) -> ConversationalRetrievalChain:
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer",
    )

    new_prompt = r"""Use the following pieces of context to answer the question at the end. Please follow the following rules
    1.  As a Reactome curator with extensive knowledge in biological pathways, answer the user's question as comprehensively as possible.
    2. If you don't know the answer, or if the answer is not provided in the context, just say that you don't know, don't try to make up an answer.
    3. If you find the answer in the context provided, answer the question only based on the context, and add the list of the sources that are **directly** used to derive the answer.
    4. Format the citations with this template: <a href="https://reactome.org/content/detail/*Source_ID*">*Source_Name*<\a> where the Source_Name is populated with the display_name, reaction_name, complex_name, entity_name, or pathway_name depending on the type of entity. Place citations within the text close close to the sentences they are associatd to.
    5. Make sure to always provide the citation for the information you provide if they are from the Reactome Knowledgebase (they are from the context).
    {context}
    Question: {question}
    Helpful Answer:
    """
    messages = [SystemMessagePromptTemplate.from_template(new_prompt)]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    callbacks: List[StreamingStdOutCallbackHandler] = []
    if commandline:
        callbacks = [StreamingStdOutCallbackHandler()]

    llm = ChatOpenAI(
        temperature=0.0,
        streaming=commandline,
        callbacks=callbacks,
        verbose=verbose,
        model="gpt-3.5-turbo-0125",
    )

    retriever_list: List[SelfQueryRetriever] = []
    for subdirectory in list_subdirectories(embeddings_directory):
        embedding = OpenAIEmbeddings()
        vectordb = Chroma(
            persist_directory=os.path.join(embeddings_directory, subdirectory),
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
