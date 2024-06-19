import argparse
import os
import sys

from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from csv_generators import (generate_complexes_csv, generate_ewas_csv,
                            generate_reactions_csv, generate_summations_csv)
from metadata_csv_loader import MetaDataCSVLoader
from neo4j_connector import Neo4jConnector


def upload_to_chromadb(file, embedding_table):
    embeddings = OpenAIEmbeddings()

    metadata_columns = {
        "reactions": [
            "pathway_id",
            "pathway_name",
            "reaction_id",
            "reaction_name",
            "input_id",
            "input_name",
            "output_id",
            "output_name",
            "catalyst_id",
            "catalyst_name",
        ],
        "summations": ["pathway_id", "pathway_name"],
        "complexes": ["complex_id", "complex_name", "component_id", "component_name"],
        "ewas": [
            "entity_id",
            "entity_name",
            "canonical_gene_name",
            "synonyms_gene_name",
            "uniprot_link",
        ],
    }

    loader = MetaDataCSVLoader(
        file_path=file, metadata_columns=metadata_columns[embedding_table]
    )
    docs = loader.load()

    ### Initialize OpenAIEmbeddings for generating embeddings of documents.
    embeddings = OpenAIEmbeddings()

    ### Create a Chroma vector store from the documents and save to disk.
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="embeddings/" + embedding_table,
    )
    db.persist()

    return db


def main():
    parser = argparse.ArgumentParser(
        description="Generate and load CSV files from Reactome Neo4j for the LangChain application"
    )
    parser.add_argument("--openai-key", required=True, help="API key for OpenAI")
    parser.add_argument(
        "--neo4j-uri",
        default="bolt://localhost:7687",
        help="URI for Neo4j database connection",
    )
    parser.add_argument(
        "--neo4j-username",
        required=False,
        help="Username for Neo4j database connection",
    )
    parser.add_argument(
        "--neo4j-password",
        required=False,
        help="Password for Neo4j database connection",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force regeneration of CSV files"
    )
    args = parser.parse_args()

    os.environ["OPENAI_API_KEY"] = args.openai_key

    connector = Neo4jConnector(
        uri=args.neo4j_uri, user=args.neo4j_password, password=args.neo4j_username
    )

    reactions_csv = generate_reactions_csv(connector, args.force)
    summations_csv = generate_summations_csv(connector, args.force)
    complexes_csv = generate_complexes_csv(connector, args.force)
    ewas_csv = generate_ewas_csv(connector, args.force)

    connector.close()

    db = upload_to_chromadb(reactions_csv, "reactions")
    print(db._collection.count())
    db = upload_to_chromadb(summations_csv, "summations")
    print(db._collection.count())
    db = upload_to_chromadb(complexes_csv, "complexes")
    print(db._collection.count())
    db = upload_to_chromadb(ewas_csv, "ewas")
    print(db._collection.count())


if __name__ == "__main__":
    main()
