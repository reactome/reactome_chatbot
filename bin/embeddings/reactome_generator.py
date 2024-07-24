import os
import sys
from typing import Dict

import torch
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain_openai import OpenAIEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from src.metadata_csv_loader import MetaDataCSVLoader
from src.reactome.csv_generators import generate_all_csvs
from src.reactome.neo4j_connector import Neo4jConnector


def upload_to_chromadb(
    embeddings_dir: str,
    file: str,
    embedding_table: str,
    hf_model: str = None,
    device: str = None,
) -> None:
    metadata_columns: Dict[str, list] = {
        "reactions": [
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
        ],
        "summations": ["st_id", "display_name", "summation"],
        "complexes": ["st_id", "display_name", "component_id", "component_name"],
        "ewas": [
            "st_id",
            "display_name",
            "canonical_gene_name",
            "synonyms_gene_name",
            "uniprot_link",
        ],
    }

    loader = MetaDataCSVLoader(
        file_path=file,
        metadata_columns=metadata_columns[embedding_table],
        encoding="utf-8",
    )
    docs = loader.load()

    if hf_model is None:  # Use OpenAI
        embeddings = OpenAIEmbeddings()
    elif hf_model.startswith("openai/text-embedding-"):
        embeddings = OpenAIEmbeddings(model=hf_model[len("openai/"):])
    elif "HUGGINGFACEHUB_API_TOKEN" in os.environ:
        embeddings = HuggingFaceEndpointEmbeddings(
            huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
            model=hf_model,
        )
    else:
        if device == "cuda":
            torch.cuda.empty_cache()
        embeddings = HuggingFaceEmbeddings(
            model_name=hf_model,
            model_kwargs={"device": device, "trust_remote_code": True},
            encode_kwargs={"batch_size": 12, "normalize_embeddings": False},
        )

    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=os.path.join(embeddings_dir, embedding_table),
    )

    return db


def main(
    embeddings_dir: str,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_username: str = None,
    neo4j_password: str = None,
    force: bool = False,
    hf_model: str = None,
    device: str = None,
) -> None:
    connector = Neo4jConnector(
        uri=neo4j_uri, user=neo4j_username, password=neo4j_password
    )

    (reactions_csv, summations_csv, complexes_csv, ewas_csv) = generate_all_csvs(
        connector, force
    )
    connector.close()

    db = upload_to_chromadb(embeddings_dir, reactions_csv, "reactions", hf_model, device)
    print(db._collection.count())
    db = upload_to_chromadb(embeddings_dir, summations_csv, "summations", hf_model, device)
    print(db._collection.count())
    db = upload_to_chromadb(embeddings_dir, complexes_csv, "complexes", hf_model, device)
    print(db._collection.count())
    db = upload_to_chromadb(embeddings_dir, ewas_csv, "ewas", hf_model, device)
    print(db._collection.count())
