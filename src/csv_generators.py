import os

import pandas as pd

from neo4j_connector import get_reactions
from neo4j_connector import get_summations
from neo4j_connector import get_complexes
from neo4j_connector import get_ewas


def generate_reactions_csv(connector, force=False):
    reactions_csv_path = "csv_files/reactions.csv"
    if not force and os.path.exists(reactions_csv_path):
        return reactions_csv_path

    reactions_data = get_reactions(connector)
    reactions_df = pd.DataFrame(reactions_data)
    reactions_df.to_csv(reactions_csv_path, index=False)
    return reactions_csv_path

def generate_summations_csv(connector, force=False):
    summations_csv_path = "csv_files/summations.csv"
    if not force and os.path.exists(summations_csv_path):
        return summations_csv_path

    summations_data = get_summations(connector)
    summations_df = pd.DataFrame(summations_data)
    summations_df.to_csv(summations_csv_path, index=False)
    return summations_csv_path

def generate_complexes_csv(connector, force=False):
    complexes_csv_path = "csv_files/complexes.csv"
    if not force and os.path.exists(complexes_csv_path):
        return complexes_csv_path

    complexes_data = get_complexes(connector)
    complexes_df = pd.DataFrame(complexes_data)
    complexes_df.to_csv(complexes_csv_path, index=False)
    return complexes_csv_path

def generate_ewas_csv(connector, force=False):
    ewas_csv_path = "csv_files/ewas.csv"
    if not force and os.path.exists(ewas_csv_path):
        return ewas_csv_path

    ewas_data = get_ewas(connector)
    ewas_df = pd.DataFrame(ewas_data)
    ewas_df.to_csv(ewas_csv_path, index=False)
    return ewas_csv_path
