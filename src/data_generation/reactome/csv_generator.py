from pathlib import Path
from typing import Callable

import pandas as pd

from data_generation.reactome.neo4j_connector import (Neo4jConnector,
                                                      Neo4jDict, get_complexes,
                                                      get_ewas, get_reactions,
                                                      get_summations)

CSV_GENERATION_MAP: dict[str, Callable[[Neo4jConnector], Neo4jDict]] = {
    "reactions.csv": get_reactions,
    "summations.csv": get_summations,
    "complexes.csv": get_complexes,
    "ewas.csv": get_ewas,
}


def generate_csv(
    connector: Neo4jConnector,
    data_fetch_func: Callable[[Neo4jConnector], Neo4jDict],
    file_name: str,
    csv_dir: Path,
    force: bool = False,
) -> str:
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_file_path = csv_dir / file_name
    if not force and csv_file_path.exists():
        return str(csv_file_path)

    data: Neo4jDict = data_fetch_func(connector)
    df: pd.DataFrame = pd.DataFrame(data)
    df["url"] = "https://reactome.org/content/detail/" + df["st_id"]
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
