import os
import sys
import argparse
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.neo4j_connector import Neo4jConnector


def upload_to_chromadb(file, directory_name):
    embeddings = OpenAIEmbeddings()

    loader = CSVLoader(file_path=file)
    docs = loader.load()

    ### Initialize OpenAIEmbeddings for generating embeddings of documents.
    embeddings = OpenAIEmbeddings()

    ### Create a Chroma vector store from the documents and save to disk.
    db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="embeddings/" + directory_name)
    db.persist()

    return db

def main():
    parser = argparse.ArgumentParser(description="Generate and load CSV files from Reactome Neo4j for the LangChain application")
    parser.add_argument("--openai-key", required=True, help="API key for OpenAI")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="URI for Neo4j database connection")
    parser.add_argument("--neo4j-username", required=False, help="Username for Neo4j database connection")
    parser.add_argument("--neo4j-password", required=False, help="Password for Neo4j database connection")
    parser.add_argument("--force", action="store_true", help="Force regeneration of CSV files")
    args = parser.parse_args()

    os.environ['OPENAI_API_KEY'] = args.openai_key

    connector = Neo4jConnector(uri=ars.neo4j_uri, user=args.neo4j_password, password=args.neo4j_username)

    reactions_csv = generate_reactions_csv(connector, args.force)
    summations_csv = generate_summations_csv(connector, args.force)
    complexes_csv = generate_complexes_csv(connector, args.force)
    ewas_csv = generate_ewas_csv(connector, args.force)
    ewas_comments_csv = generate_ewas_comments_csv(connector, args.force)

    connector.close()


    db = upload_to_cromadb(reactions_csv, directory_name)
    print(db._collection.count())
    db = upload_to_cromadb(summations_csv, directory_name)
    print(db._collection.count())
    db = upload_to_cromadb(complexes_csv, directory_name)
    print(db._collection.count())
    db = upload_to_cromadb(ewas_csv, directory_name)
    print(db._collection.count())
    db = upload_to_cromadb(ewas_comments_csv, directory_name)
    print(db._collection.count())


if __name__ == "__main__":
    main()
