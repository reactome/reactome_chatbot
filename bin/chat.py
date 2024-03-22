import os
import openai
import pprint as pp
import argparse
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain.vectorstores.chroma import Chroma

from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers import MergerRetriever

from langchain.memory import ConversationBufferMemory
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate


def main():
    parser = argparse.ArgumentParser(description="Reactome ChatBot")
    parser.add_argument("--openai-key", required=True, help="API key for OpenAI")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()
    os.environ['OPENAI_API_KEY'] = args.openai_key

    ### The directory where your embeddings are persisted
    persist_directory = 'embeddings'

    ### Initialize an OpenAIEmbeddings model. 



    ### Create a ConversationBufferMemory object.
    memory = ConversationBufferMemory(memory_key="chat_history",
                                      return_messages=True)

    ### Initialize a ChatOpenAI model.
    llm = ChatOpenAI(temperature=0.0,
                     model="gpt-3.5-turbo-0125")

    ### Create retrievers for each subfolder in the embeddings directory
    retriever_list = []
    for root, dirs, files in os.walk(persist_directory):
        embedding = OpenAIEmbeddings()
        vectordb = Chroma (persist_directory = root,
                  embedding_function = embedding
                  )
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={'k': 15})
        retriever_list.append(retriever)

    ### Create a MergerRetriever (LOTR) to join the retriever list together
    lotr = MergerRetriever(retrievers=retriever_list)

    ### Initialize a ConversationalRetrievalChain object with the MergerRetriever
    qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                               retriever=lotr,
                                               verbose=True,
                                               memory=memory)
    if args.interactive:
        interactive_mode(qa)
    else:
        query = "Provide a comprehensive list of all entities (including their names and IDs) where GTP is a component."
        print_results(qa, query)


def interactive_mode(qa):
    while True:
        query = input("Enter your query (or press Enter to exit): ")
        if not query:
            break
        print_results(qa, query)

def print_results(qa, query):
    # prints search VectorDB search results
    retriever_results = qa.retriever.invoke(query)
    #print("VectorDB search results:")
    #print(retriever_results)

    # prints LLM outputs
    qa_results = qa.invoke(query)
    pretty_print_results(qa_results)


def pretty_print_results(qa_results):
    print("Response")
    answer = qa_results['answer']
    # Remove the outer parentheses
    answer = answer.strip("('").rstrip("')")
    # Replace '\n' with actual new lines
    answer = answer.replace('\\n', '\n')
    print(answer)
    pp.pprint(qa_results['chat_history'])


if __name__ == "__main__":
    main()
