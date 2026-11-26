"""Obsolete for the Reactome pipeline. To be moved out of `data_generation.reactome`.

TODO: move this, with the four old queries in `neo4j_connector` and
`generate_reactome_embeddings` in this package's `__init__`, to wherever Plant
Reactome's generation ends up -- or remove all three once Plant Reactome no
longer needs them.

Reactome's CSVs are now written by `data_generation.reactome.bundle_generator`,
from the queries in `neo4j_connector.SOURCE_QUERIES`. This module is still
here only because `generate_reactome_embeddings` imports it, and
`bin/embeddings_manager make` still calls that function for Plant Reactome.
Plant Reactome's CSVs come from `export_csvs.py`, so this runs only when they
are missing, and then with queries that look for human records.

The branch in `generate_csv` that keeps a link the query returned was added for
the new queries and is unused now that `bundle_generator.py` writes their CSVs.
"""

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from data_generation.reactome.neo4j_connector import (
    REACTOME_DETAIL_URL,
    Neo4jConnector,
    Neo4jDict,
    get_complexes,
    get_ewas,
    get_reactions,
    get_summations,
)

CSV_GENERATION_MAP: dict[str, Callable[[Neo4jConnector], list[Neo4jDict]]] = {
    "reactions.csv": get_reactions,
    "summations.csv": get_summations,
    "complexes.csv": get_complexes,
    "ewas.csv": get_ewas,
}


def generate_csv(
    connector: Neo4jConnector,
    data_fetch_func: Callable[[Neo4jConnector], list[Neo4jDict]],
    file_name: str,
    csv_dir: Path,
    force: bool = False,
) -> str:
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_file_path = csv_dir / file_name
    if not force and csv_file_path.exists():
        return str(csv_file_path)

    data: list[Neo4jDict] = data_fetch_func(connector)
    df: pd.DataFrame = pd.DataFrame(data)
    # A query that returns its own link keeps it: proteins and disease terms
    # link to UniProt and the Disease Ontology, and have no `st_id` to build
    # one from. Only the original four queries, which return no link, still
    # have theirs built here.
    if "url" not in df.columns:
        df["url"] = REACTOME_DETAIL_URL + df["st_id"]
    df.to_csv(csv_file_path, index=False, lineterminator="\n")
    return str(csv_file_path)


def generate_all_csvs(
    connector: Neo4jConnector, parent_dir: str, force: bool = False
) -> tuple[str, ...]:
    csv_dir = Path(parent_dir) / "csv_files"
    csv_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = []
    for file_name, data_fetch_func in CSV_GENERATION_MAP.items():
        csv_path = generate_csv(connector, data_fetch_func, file_name, csv_dir, force)
        csv_paths.append(csv_path)
    return tuple(csv_paths)
