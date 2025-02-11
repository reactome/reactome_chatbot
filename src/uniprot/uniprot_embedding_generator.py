import os
from typing import Optional
import torch
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain_openai import OpenAIEmbeddings
from metadata_csv_loader import MetaDataCSVLoader
from csv_generator.uniprot_generator import generate_uniprot_csv

def upload_to_chromadb(
    embeddings_dir: str,
    file: str,
    embedding_table: str,
    hf_model: Optional[str] = None,
    device: Optional[str] = None,
) -> Chroma:
    metadata_columns: dict[str, list] = {
        'uniprot_data':[
            "gene_names", 
            "short_protein_name", 
            "full_protein_name",
            "protein_family",
            "biological_pathways",
            ],
    }

    loader = MetaDataCSVLoader(
        file_path=file,
        metadata_columns=metadata_columns[embedding_table],
        encoding="utf-8",
    )
    
    docs = loader.load()
    print(f"Loaded {len(docs)} documents from {file}")

    embeddings_instance: Embeddings
    if hf_model is None:  # Use OpenAI
        print("Using OpenAI embeddings")
        embeddings_instance = OpenAIEmbeddings(model="text-embedding-3-large", chunk_size=800, show_progress_bar=True, )
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
def generate_uniprot_embeddings(
        embeddings_dir: str, 
        version: str, 
        download_url: str, 
        force: bool = False, 
        hf_model: Optional[str] = None, 
        device: Optional[str] = None
        ) -> None:
    csv_file = generate_uniprot_csv(version, download_url, force)
    db = upload_to_chromadb(embeddings_dir, csv_file,'uniprot_data',  hf_model, device)
    print(f"UniProt embeddings stored in {db._persist_directory}")
