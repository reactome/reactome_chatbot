from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import pandas as pd

from reactome.neo4j_connector import (
    Neo4jConnector,
    get_complexes,
    get_ewas,
    get_reactions,
    get_summations,
)

BASE_CSV_PATH = "csv_files/reactome"

CSV_GENERATION_MAP: Dict[str, Callable[[Neo4jConnector], Any]] = {
    "reactions.csv": get_reactions,
    "summations.csv": get_summations,
    "complexes.csv": get_complexes,
    "ewas.csv": get_ewas,
}


def generate_csv(
    connector: Neo4jConnector,
    data_fetch_func: Callable[[Neo4jConnector], Any],
    file_name: str,
    force: bool = False,
) -> str:
    csv_file_path = Path(BASE_CSV_PATH) / file_name
    if not force and csv_file_path.exists():
        return str(csv_file_path)

    data: Any = data_fetch_func(connector)
    df: pd.DataFrame = pd.DataFrame(data)
    df["url"] = "https://reactome.org/content/detail/" + df["st_id"]
    csv_file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_file_path, index=False)
    return str(csv_file_path)


def generate_all_csvs(
    connector: Neo4jConnector, force: bool = False
) -> Tuple[str, str, str, str]:
    Path(BASE_CSV_PATH).mkdir(parents=True, exist_ok=True)

    csv_paths = []
    for file_name, data_fetch_func in CSV_GENERATION_MAP.items():
        csv_path = generate_csv(connector, data_fetch_func, file_name, force)
        csv_paths.append(csv_path)
    return tuple(csv_paths)
