import argparse
import os
import sys
from typing import Dict

import torch
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from src.metadata_csv_loader import MetaDataCSVLoader
from src.reactome.csv_generators import generate_all_csvs
from src.reactome.neo4j_connector import Neo4jConnector


def upload_to_chromadb(
    file: str, embedding_table: str, hf_model: str = None, device: str = "cpu"
) -> None:
    metadata_columns: Dict[str, list] = {
        "genes": [
            "st_id",
            "display_name",
            "pathway_id",
            "pathway_name",
            "input_id",
            "input_name",
            "output_id",
            "output_name",
            "catalyst_id",
            "catalyst_name",
        ]
    }

    loader = MetaDataCSVLoader(
        file_path=file,
        metadata_columns=metadata_columns[embedding_table],
        encoding="utf-8",
    )
    docs = loader.load()

    if device == "cuda":
        torch.cuda.empty_cache()

    if hf_model is None:  # Use OpenAI
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=hf_model,
            model_kwargs={"device": device, "trust_remote_code": True},
            encode_kwargs={"batch_size": 12, "normalize_embeddings": False},
        )

    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="embeddings/" + embedding_table,
    )

    return db


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and load CSV files from Reactome Neo4j for the LangChain application"
    )
    parser.add_argument("--openai-key", help="API key for OpenAI")
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
    parser.add_argument(
        "--hf-model",
        help="HuggingFace sentence_transformers model (alternative to OpenAI)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device to use when running HuggingFace model locally [cpu/cuda]",
    )
    args = parser.parse_args()

    if args.openai_key is not None:
        os.environ["OPENAI_API_KEY"] = args.openai_key

    connector = Neo4jConnector(
        uri=args.neo4j_uri, user=args.neo4j_password, password=args.neo4j_username
    )

    (reactions_csv, summations_csv, complexes_csv, ewas_csv) = generate_all_csvs(
        connector, args.force
    )
    connector.close()

    db = upload_to_chromadb(reactions_csv, "reactions", args.hf_model, args.device)
    print(db._collection.count())
    db = upload_to_chromadb(summations_csv, "summations", args.hf_model, args.device)
    print(db._collection.count())
    db = upload_to_chromadb(complexes_csv, "complexes", args.hf_model, args.device)
    print(db._collection.count())
    db = upload_to_chromadb(ewas_csv, "ewas", args.hf_model, args.device)
    print(db._collection.count())


if __name__ == "__main__":
    main()
