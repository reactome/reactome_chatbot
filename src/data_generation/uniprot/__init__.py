import os
from pathlib import Path

from langchain_community.vectorstores import Chroma

from data_generation.embeddings import build_embeddings
from data_generation.metadata_csv_loader import MetaDataCSVLoader
from data_generation.uniprot.csv_generator import generate_uniprot_csv


def upload_to_chromadb(
    embeddings_dir: str,
    file: str,
    embedding_table: str,
    hf_model: str | None = None,
    device: str | None = None,
) -> Chroma:
    metadata_columns: dict[str, list] = {
        "uniprot_data": [
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

    embeddings_instance = build_embeddings(hf_model, device, chunk_size=500)

    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings_instance,
        persist_directory=os.path.join(embeddings_dir, embedding_table),
    )


def generate_uniprot_embeddings(
    embedding_path: Path,
    hf_model: str | None = None,
    device: str | None = None,
    **_: object,
) -> None:
    csv_path = generate_uniprot_csv(embedding_path)
    db = upload_to_chromadb(
        str(embedding_path), str(csv_path), "uniprot_data", hf_model, device
    )
    print(db._collection.count())
