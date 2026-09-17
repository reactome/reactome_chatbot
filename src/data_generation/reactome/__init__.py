import os
import shutil
from pathlib import Path

from chromadb.api.client import SharedSystemClient
from langchain_community.vectorstores import Chroma

from data_generation.disease_variant import generate_disease_variant_embeddings
from data_generation.embeddings import build_embeddings
from data_generation.metadata_csv_loader import MetaDataCSVLoader
from data_generation.reactome.csv_generator import generate_all_csvs
from data_generation.reactome.neo4j_connector import Neo4jConnector


def upload_to_chromadb(
    embeddings_dir: str,
    file: str,
    embedding_table: str,
    hf_model: str | None = None,
    device: str | None = None,
) -> Chroma:
    # `species` is deliberately absent from reactions and summations. Both
    # queries filter on speciesName = "Homo sapiens" and never return it, so
    # asking for it here killed generation on the first upload with
    # "Metadata column 'species' not found in CSV file" -- `make` has been
    # broken for the reactome bundle, and the Release95 bundle predates it.
    #
    # Adding it to the queries would be worse than removing it here: these
    # collections pass no `content_columns`, so every CSV column is embedded,
    # and a column that reads "species: Homo sapiens" in all 52,000 documents
    # is noise in every one of them.
    #
    # `complexes` does return it, and its value is constant too -- all 38,967
    # rows are Homo sapiens, because that query filters on it as well. That is
    # existing noise in an existing bundle rather than noise this adds, so it
    # is left alone here; removing it means changing the query and rebuilding,
    # which is a separate change with its own before-and-after measurement.
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
        "summations": ["st_id", "display_name", "labels", "summation"],
        "complexes": [
            "st_id",
            "display_name",
            "component_id",
            "component_name",
            "species",
        ],
        "ewas": [
            "st_id",
            "display_name",
            "canonical_gene_name",
            "synonyms_gene_name",
            "uniprot_link",
        ],
    }

    # Chroma.from_documents appends to whatever is already in the directory.
    # The shipped Release95 bundle has 33,498 reaction documents for 16,749 CSV
    # rows -- exactly 2.00x, every reaction stored twice, because generation ran
    # twice. Nothing reported it, and the vector retriever spends half its
    # overfetch on duplicates. Regeneration replaces.
    persist = Path(embeddings_dir) / embedding_table
    if persist.exists():
        shutil.rmtree(persist)
        # chromadb caches one system client per path; without this the next
        # write fails with "attempt to write a readonly database".
        SharedSystemClient.clear_system_cache()
        print(f"  replaced the previous {embedding_table} collection")

    loader = MetaDataCSVLoader(
        file_path=file,
        metadata_columns=metadata_columns[embedding_table],
        encoding="utf-8",
    )
    docs = loader.load()
    embeddings_instance = build_embeddings(hf_model, device, chunk_size=400)

    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings_instance,
        persist_directory=os.path.join(embeddings_dir, embedding_table),
    )


def generate_reactome_embeddings(
    embeddings_dir: str,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_username: str | None = None,
    neo4j_password: str | None = None,
    force: bool = False,
    hf_model: str | None = None,
    device: str | None = None,
    disease_variant_tsv: str | Path | None = None,
) -> None:
    csv_dir = Path(embeddings_dir) / "csv_files"
    reactions_csv = str(csv_dir / "reactions.csv")
    summations_csv = str(csv_dir / "summations.csv")
    complexes_csv = str(csv_dir / "complexes.csv")
    ewas_csv = str(csv_dir / "ewas.csv")

    all_exist = not force and all(
        Path(p).exists()
        for p in [reactions_csv, summations_csv, complexes_csv, ewas_csv]
    )

    if not all_exist:
        connector = Neo4jConnector(
            uri=neo4j_uri, user=neo4j_username, password=neo4j_password
        )
        reactions_csv, summations_csv, complexes_csv, ewas_csv = generate_all_csvs(
            connector, embeddings_dir, force
        )
        connector.close()
    else:
        print("Using existing CSV files. Skipping Neo4j.")

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

    # Built from a release file, not from Neo4j, so it is the one collection
    # that can be generated without database access. Skipped rather than
    # failed when no file is given: the other four are still a valid bundle.
    if disease_variant_tsv:
        # Its own name: this returns langchain_chroma's Chroma, while the four
        # above still come back as the deprecated langchain_community one.
        variants_db = generate_disease_variant_embeddings(
            embeddings_dir, disease_variant_tsv, hf_model, device
        )
        print(variants_db._collection.count())
    else:
        print(
            "No --disease-variant-tsv given; skipping the disease_variants "
            "collection. The chatbot will answer about disease at the level of "
            "a pathway and will not be able to name a variant."
        )
