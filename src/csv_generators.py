import os
import spacy
import pandas as pd
import en_core_web_sm

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


def split_sentences(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def split_summation_data(summation_data):
    split_summation_data = []
    for summation_record in summation_data:
        print("summation_record")
        print(summation_record)
        summation = summation_record["summation"]
        sentences = split_sentences(summation)
        for sentence in sentences:
            new_record = summation_record
            new_record["sentence"] = sentence
            del new_record["summation"]

            print("new_record")
            print(new_record)

            print("summation_record")
            print(summation_record)
            exit()
            sentence_summation_data.append(new_record)
    return sentence_summation_data


def generate_summations_csv(connector, force=False):
    nlp = en_core_web_sm.load()
    summations_csv_path = "csv_files/summations.csv"
    if not force and os.path.exists(summations_csv_path):
        return summations_csv_path

    split_summations_data = get_summations(connector)
  #  split_summation_data = split_summation_data(summation_data)
    summations_df = pd.DataFrame(split_summations_data)
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
