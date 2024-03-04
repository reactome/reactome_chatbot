import os
import openai

from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain.vectorstores.chroma import Chroma

from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

from langchain.memory import ConversationBufferMemory
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate


os.environ['OPENAI_API_KEY'] = ''

### The directory where your embeddings are persisted
persist_directorys =''

### Initialize an OpenAIEmbeddings model. 
embedding = OpenAIEmbeddings()


vectordb = Chroma (persist_directory = persist_directorys,
                  embedding_function = embedding
                  )

### Define a list of AttributeInfo objects, each representing metadata about different fields in your dataset.
### This metadata includes names, descriptions, and types for each attribute, aiding in the understanding and processing of the data.
field_info = [
    AttributeInfo(
        name="Complex_ID",
        description="The Reactome Identifier (ID) for each biological complex, serving as a unique key.\
            Given that a single complex can consist of various components, this ID may repeat across multiple rows, each associated with a different component of the same complex.",
        type="string",
    ),
    AttributeInfo(
        name="Complex_name",
        description="The name of the biological complex.\
              This field provides a reference to the complex itself, which may be listed across several rows to account for its multiple components.",
        type="string",
    ),
    AttributeInfo(
        name="Component_ID",
        description=" A Reactome Identifier unique to each component within a complex.\
              This ID allows for the specific identification and exploration of each component's details within the Reactome Database.",
        type="string",
    ),
    AttributeInfo(
        name="Component_name",
        description="The name of the individual component associated with the complex in that row.\
              This reveals the specific protein or molecule constituting part of the complex, emphasizing the diversity of components within a single biological entity.",
        type="string",
    )
    # Additional AttributeInfo objects are defined here...
]


### A description of the document content, indicating it contains biological complexes and their components from the Reactome Database, aiding in the understanding and processing of the data.
document_content_description = "a tabulated file containing biological complexes names and IDs and their protein components' names and IDs  from the Reactome Database"

### either of these two retreivers can be used. i found 'SelfQueryRetriever' to be more restrictive. 
#retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={'k': 15})
#retriever = SelfQueryRetriever.from_llm (llm = llm, vectorstore = vectordb, document_contents = document_content_description, metadata_field_info = field_info, search_kwargs={'k': 15})

#### Initialize a ChatOpenAI model.
llm = ChatOpenAI (temperature = 0.0,
                 model = "gpt-3.5-turbo-0125")

### Create a ConversationBufferMemory object. This is used to store and retrieve conversation history, enhancing context understanding in conversational models.
memory = ConversationBufferMemory(memory_key="chat_history",
                                  return_messages=True)

### Create a retriever object from the vector database for similarity-based search. This enables querying the database based on semantic similarity with a specified number of results (k=15).
retriever = vectordb.as_retriever(search_type="similarity",
                                  search_kwargs={'k': 15})

### Initialize a ConversationalRetrievalChain object.
qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                           retriever = retriever,
                                           verbose = True,
                                           memory = memory )

query = "Provide a comprehensive list of all entities (including their names and IDs) where GTP is a component."

# prints search VectorDB search results 
print (retriever.invoke(query))

# prints LLM outputs
print (qa.invoke(query))
