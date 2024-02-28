import os 
import openai

from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings


from langchain.vectorstores.chroma import Chroma


os.environ['OPENAI_API_KEY'] = ""


file = '/Users/hmohammadi/Desktop/PathwayLLM/Neo4J_data_files/CSV_files/EWAS_NoDatabase.csv'
persist_directorys ='/Users/hmohammadi/Desktop/PathwayLLM/Neo4J_data_files/CSV_files/EWAS_Directory/'

### Load documents from the CSV file. 
loader = CSVLoader(file_path = file)
docs = loader.load()
#print (docs[0])

### Initialize OpenAIEmbeddings for generating embeddings of documents.
embeddings = OpenAIEmbeddings()

### Create a Chroma vector store from the documents and save to disk.
db = Chroma.from_documents(documents = docs, embedding = embeddings, persist_directory = persist_directorys)
db.persist()
print (db._collection.count())
#print (len(doc))
#print(doc)
