import os
from typing import Any

import pandas as pd

from neo4j_connector import (Neo4jConnector, get_complexes, get_ewas,
                             get_reactions, get_summations)


def generate_reactions_csv(connector: Neo4jConnector, force: bool = False) -> str:
    reactions_csv_path: str = "csv_files/reactions.csv"
    if not force and os.path.exists(reactions_csv_path):
        return reactions_csv_path

    reactions_data: Any = get_reactions(connector)
    reactions_df: pd.DataFrame = pd.DataFrame(reactions_data)
    reactions_df.to_csv(reactions_csv_path, index=False)
    return reactions_csv_path


def generate_summations_csv(connector: Neo4jConnector, force: bool = False) -> str:
    summations_csv_path: str = "csv_files/summations.csv"
    if not force and os.path.exists(summations_csv_path):
        return summations_csv_path

    summations_data: Any = get_summations(connector)
    summations_df: pd.DataFrame = pd.DataFrame(summations_data)
    summations_df.to_csv(summations_csv_path, index=False)
    return summations_csv_path


def generate_complexes_csv(connector: Neo4jConnector, force: bool = False) -> str:
    complexes_csv_path: str = "csv_files/complexes.csv"
    if not force and os.path.exists(complexes_csv_path):
        return complexes_csv_path

    complexes_data: Any = get_complexes(connector)
    complexes_df: pd.DataFrame = pd.DataFrame(complexes_data)
    complexes_df.to_csv(complexes_csv_path, index=False)
    return complexes_csv_path


def generate_ewas_csv(connector: Neo4jConnector, force: bool = False) -> str:
    ewas_csv_path: str = "csv_files/ewas.csv"
    if not force and os.path.exists(ewas_csv_path):
        return ewas_csv_path

    ewas_data: Any = get_ewas(connector)
    ewas_df: pd.DataFrame = pd.DataFrame(ewas_data)
    ewas_df.to_csv(ewas_csv_path, index=False)
    return ewas_csv_path
