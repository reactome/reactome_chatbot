import os
import requests
import sys
from typing import Dict

import torch
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain_openai import OpenAIEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from src.metadata_csv_loader import MetaDataCSVLoader

from src.alliance.csv_generators import generate_all_csvs

def get_release_version() -> str:
    url: str = "https://www.alliancegenome.org/api/releaseInfo"
    response = requests.get(url)
    if response.status_code == 200:
        response_json = response.json()
        release_version = response_json.get('releaseVersion')
        if release_version:
            return release_version
        else:
            raise ValueError('Release version not found in the response.')
    else:
        raise ConnectionError(f'Failed to get the response. Status code: {response.status_code}')


def upload_to_chromadb(
    embeddings_dir: str,
    file: str,
    embedding_table: str,
    hf_model: str = None,
    device: str = None,
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
    force: bool = False,
    hf_model: str = None,
    device: str = None,
    **kwargs
) -> None:
    release_version = get_release_version()
    print(f'Release Version: {release_version}')

    (gene_csv) = generate_all_csvs(
        release_version, force
    )

    #db = upload_to_chromadb(reactions_csv, "reactions", args.hf_model, args.device)
    #print(db._collection.count())
    #db = upload_to_chromadb(summations_csv, "summations", args.hf_model, args.device)
    #print(db._collection.count())
    #db = upload_to_chromadb(complexes_csv, "complexes", args.hf_model, args.device)
    #print(db._collection.count())
    #db = upload_to_chromadb(ewas_csv, "ewas", args.hf_model, args.device)
    #print(db._collection.count())
