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

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers import MergerRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

from langchain.memory import ConversationBufferMemory
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate


def list_subdirectories(directory):
    subdirectories = [f.name for f in os.scandir(directory) if f.is_dir() and f.name != '.' and f.name != '..']
    return subdirectories


def main():
    parser = argparse.ArgumentParser(description="Reactome ChatBot")
    parser.add_argument("--openai-key", required=True, help="API key for OpenAI")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()
    os.environ['OPENAI_API_KEY'] = args.openai_key

    embeddings_directory = 'embeddings'

    memory = ConversationBufferMemory(memory_key="chat_history",
                                      return_messages=True)

    llm = ChatOpenAI(temperature=0.0,
                     model="gpt-3.5-turbo-0125")
    descriptions = {
        "ewas": "Contains data on proteins and nucleic acids with known sequences. Includes entity names, IDs, canonical and synonymous gene names, and functions.",
        "complexes": "Catalogs biological complexes, listing complex names and IDs along with the names and IDs of their components. ",
        "reactions": "Documents biological pathways and their constituent reactions, detailing pathway and reaction names and IDs. It includes information on the inputs, outputs, and catalysts for each reaction, emphasizing the interconnected nature of cellular processes. Inputs and outputs, critical to the initiation and conclusion of reactions, along with catalysts that facilitate these processes, are cataloged to highlight their roles across various reactions and pathways",
        "summations": "Enumerates biological reactions, accompanied by concise summaries ('summations') of each reaction. These summations encapsulate the essence and biochemical significance of the reactions, offering insights into their roles within cellular processes and pathways."}
    ### Define a list of AttributeInfo objects, each representing metadata about different fields in your dataset.
    ### This metadata includes names, descriptions, and types for each attribute, aiding in the understanding and processing of the data.
    field_info = [
        AttributeInfo(
            name="pathway_id",
            description=" A Reactome Identifier unique to each pathway. A pathway name may appear multiple times in the dataset\
                  This ID allows for the specific identification and exploration of each pathway's details within the Reactome Database.",
            type="string",
        ),
        AttributeInfo(
            name="pathway_name",
            description="The name of the biological pathway, indicating a specific series of interactions or processes within a cell.\
                A pathway name may appear multiple times in the dataset, reflecting the fact that several reactions (identified by 'reaction_name') contribute to a single pathway.\
                The relationship between 'reaction_name' and 'pathway_name' is foundational, with each reaction serving as a step or component within the overarching pathway, contributing to its completion and functional outcome.\
                This relationship is critical to understanding the biological processes and mechanisms within the Reactome Database.",
            type="string",
        ),
        AttributeInfo(
            name="reaction_id",
            description = "The Reactome Identifier (ID) for each biological reaction, serving as a unique key.\
                This ID allows for the specific identification and exploration of each reaction's details within the Reactome Database.",
            type="string",
        ),
        AttributeInfo(
            name="reaction_name",
            description= "The name of the biological reaction, encapsulating the interaction between proteins or molecules.\
                Each reaction name is a unique entry, reflecting a specific biological process.\
                These names provide insight into the dynamic processes within cellular functions, highlighting the roles of various proteins and molecules in biological mechanisms",
            type="string",
        ),
        AttributeInfo(
            name="input_id",
            description="The Reactome Identifier (ID) for each input.\
                Given that a single input can be involved in various reactionss, this ID may repeat across multiple rows, each associated with a different reaction.\
                This ID allows for the specific identification and exploration of each input's details within the Reactome Database.",
            type="string",
        ),
        AttributeInfo(
            name="input_name",
            description="Identifies the inputs of a biological reaction ('reaction_name'), which can be either entities or part of complexes.\
                Inputs are crucial for initiating reactions, acting as the reactants that drive the biochemical processes. \
                Given their fundamental role, inputs may repeat across multiple reactions, reflecting their involvement in various parts of the cellular machinery.",
            type="string",
        ),
        AttributeInfo(
            name="output_id",
            description=" A Reactome Identifier unique to each output of a reaction.\
                Given that a single input can be involved in various reactionss, this ID may repeat across multiple rows, each associated with a different reaction.\
                This ID allows for the specific identification and exploration of each output's details within the Reactome Database.",
                type="string",
        ),
        AttributeInfo(
            name="output_name",
            description="Represents the outputs of a biological reaction ('reaction_name'), denoting the products generated as a result of the biochemical interactions. \
                Outputs can be entities or complexes and may appear in multiple reactions, highlighting their multifunctional role in cellular pathways. \
                This repetition underscores the interconnected nature of biological processes, where one reaction's output can serve as another's input.",
            type="string",
        ),
        AttributeInfo(
            name="catalyst_id",
            description="The Reactome Identifier (ID) for each biological catalyst, serving as a unique key.\
                Given that a single catalyst can be involved in various reactions, this ID may repeat across multiple rows, each associated with a different reaction.\
                This ID allows for the specific identification and exploration of each catalyst's details within the Reactome Database.",
            type="string",
        ),
        AttributeInfo(
            name="catalyst_name",
            description="Specifies the catalysts that facilitate a biological reaction, potentially speeding up the process without being consumed.\
                Catalysts are crucial for modulating reaction rates and guiding the direction of the reaction, ensuring the efficient progression of biological pathways.\
                Catalysts can be proteins, enzymes, or molecular compounds, underscoring their vital role in cellular operations.",
            type="string",
        ),
        AttributeInfo(
            name="complex_id",
            description="The Reactome Identifier (ID) for each biological complex, serving as a unique key.\
                Given that a single complex can consist of various components, this ID may repeat across multiple rows, each associated with a different component of the same complex.",
            type="string",
        ),
        AttributeInfo(
            name="complex_name",
            description="The name of the biological complex.\
                  This field provides a reference to the complex itself, which may be listed across several rows to account for its multiple components.",
            type="string",
        ),
        AttributeInfo(
            name="component_id",
            description=" A Reactome Identifier unique to each component within a complex.\
                  This ID allows for the specific identification and exploration of each component's details within the Reactome Database.",
            type="string",
        ),
        AttributeInfo(
            name="component_name",
            description="The name of the individual component associated with the complex in that row.\
                  This reveals the specific protein or molecule constituting part of the complex, emphasizing the diversity of components within a single biological entity.",
            type="string",
        ),
        AttributeInfo(
            name="entity_id",
            description="The Reactome Identifier (ID) for each biological complex, serving as a unique key.\
                Given that a single complex can consist of various components, this ID may repeat across multiple rows, each associated with a different component of the same complex.",
            type="string",
        ),
        AttributeInfo(
            name="entity_name",
            description="The name of the biological complex.\
                  This field provides a reference to the complex itself, which may be listed across several rows to account for its multiple components.",
            type="string",
        ),
        AttributeInfo(
            name="canonical_geneName",
            description=" A Reactome Identifier unique to each component within a complex.\
                  This ID allows for the specific identification and exploration of each component's details within the Reactome Database.",
            type="string",
        ),
        AttributeInfo(
            name="synonyms_geneName",
            description="The name of the individual component associated with the complex in that row.\
                  This reveals the specific protein or molecule constituting part of the complex, emphasizing the diversity of components within a single biological entity.",
            type="string",
        )
        # Additional AttributeInfo objects are defined here...
    ]

    retriever_list = []
    for subdirectory in list_subdirectories(embeddings_directory):
        embedding = OpenAIEmbeddings()
        vectordb = Chroma (persist_directory = embeddings_directory + "/" + subdirectory,
                  embedding_function = embedding
                  )
        retriever = SelfQueryRetriever.from_llm(llm = llm,
                                                vectorstore = vectordb,
                                                document_contents = descriptions[subdirectory],
                                                metadata_field_info = field_info,
                                                search_kwargs = {'k' : 15})

        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={'k': 15})
        retriever_list.append(retriever)

    lotr = MergerRetriever(retrievers=retriever_list)

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
