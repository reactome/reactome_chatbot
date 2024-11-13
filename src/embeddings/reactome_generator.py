import os
from typing import Optional

import torch
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain_openai import OpenAIEmbeddings

from csv_generator.reactome_generator import generate_all_csvs
from metadata_csv_loader import MetaDataCSVLoader
from reactome.neo4j_connector import Neo4jConnector


def upload_to_chromadb(
    embeddings_dir: str,
    file: str,
    embedding_table: str,
    hf_model: Optional[str] = None,
    device: Optional[str] = None,
) -> Chroma:
    metadata_columns: dict[str, list] = {
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
    embeddings_instance: Embeddings
    if hf_model is None:  # Use OpenAI
        embeddings_instance = OpenAIEmbeddings()
    elif hf_model.startswith("openai/text-embedding-"):
        embeddings_instance = OpenAIEmbeddings(model=hf_model[len("openai/") :])
    elif "HUGGINGFACEHUB_API_TOKEN" in os.environ:
        embeddings_instance = HuggingFaceEndpointEmbeddings(
            huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
            model=hf_model,
        )
    else:
        if device == "cuda":
            torch.cuda.empty_cache()
        embeddings_instance = HuggingFaceEmbeddings(
            model_name=hf_model,
            model_kwargs={"device": device, "trust_remote_code": True},
            encode_kwargs={"batch_size": 12, "normalize_embeddings": False},
        )

    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings_instance,
        persist_directory=os.path.join(embeddings_dir, embedding_table),
    )


def generate_reactome_embeddings(
    embeddings_dir: str,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_username: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    force: bool = False,
    hf_model: Optional[str] = None,
    device: Optional[str] = None,
) -> None:
    connector = Neo4jConnector(
        uri=neo4j_uri, user=neo4j_username, password=neo4j_password
    )

    (reactions_csv, summations_csv, complexes_csv, ewas_csv) = generate_all_csvs(
        connector, embeddings_dir, force
    )
    connector.close()

    db = upload_to_chromadb(
        embeddings_dir, reactions_csv, "reactions", hf_model, device
    )
    print(db._collection.count())
    db = upload_to_chromadb(
        embeddings_dir, summations_csv, "summations", hf_model, device
    )
    print(db._collection.count())
    db = upload_to_chromadb(
        embeddings_dir, complexes_csv, "complexes", hf_model, device
    )
    print(db._collection.count())
    db = upload_to_chromadb(embeddings_dir, ewas_csv, "ewas", hf_model, device)
    print(db._collection.count())
