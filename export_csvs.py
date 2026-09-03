#!/usr/bin/env python3
import csv
from pathlib import Path

from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "react-app-user_pw"

OUTPUT_DIR = Path("./embeddings/openai/bge-m3/plantreactome/Release68/csv_files")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUERIES = {
    "reactions": """                                                                                                                                                                                                       
    MATCH (pathway:Pathway)-[:hasEvent]->(reaction:ReactionLikeEvent)                                                                                                                                                      
    OPTIONAL MATCH (reaction)-[:input]->(input:PhysicalEntity)                                                                                                                                                             
    OPTIONAL MATCH (reaction)-[:output]->(output:PhysicalEntity)                                                                                                                                                           
    OPTIONAL MATCH (reaction)-[:catalystActivity]->(cat:CatalystActivity)-[:physicalEntity]->(catalyst:PhysicalEntity)                                                                                                     
    RETURN reaction.stId AS st_id, reaction.displayName AS display_name,                                                                                                                                                   
      pathway.stId AS pathway_id, pathway.displayName AS pathway_name,                                                                                                                                                     
      pathway.speciesName AS species,                                                                                                                                                                                      
      COLLECT(DISTINCT input.stId) AS input_id,                                                                                                                                                                            
      COLLECT(DISTINCT input.displayName) AS input_name,                                                                                                                                                                   
      COLLECT(DISTINCT output.stId) AS output_id,                                                                                                                                                                          
      COLLECT(DISTINCT output.displayName) AS output_name,                                                                                                                                                                 
      COLLECT(DISTINCT catalyst.stId) AS catalyst_id,                                                                                                                                                                      
      COLLECT(DISTINCT catalyst.displayName) AS catalyst_name,
      "https://plantreactome.gramene.org/content/detail/" + reaction.stId AS url                                                                                                                                                       
    """,
    "summations": """                                                                                                                                                                                                      
    MATCH (e)-[:summation]->(s:Summation)                                                                                                                                                                                  
    WHERE (e:Pathway OR e:ReactionLikeEvent)                                                                                                                                                                               
    RETURN e.stId AS st_id, e.displayName AS display_name, labels(e) AS labels,                                                                                                                                            
      e.speciesName AS species,                                                                                                                                                                                            
      CASE WHEN size(s.text) > 10000 THEN LEFT(s.text, 10000) + '...' ELSE s.text END AS summation,
      "https://plantreactome.gramene.org/content/detail/" + e.stId AS url                                                                                                     
    """,
    "complexes": """                                                                                                                                                                                                       
    MATCH (complex:Complex)-[:hasComponent]->(component)                                                                                                                                                                   
    RETURN complex.speciesName AS species, complex.stId AS st_id,                                                                                                                                                          
      complex.name AS display_name, component.stId AS component_id, component.name AS component_name,
      "https://plantreactome.gramene.org/content/detail/" + complex.stId AS url                                                                                                                      
    """,
    "ewas": """                                                                                                                                                                                                            
    MATCH (db:ReferenceDatabase)<-[:referenceDatabase]-(gene:ReferenceEntity)<-[:referenceEntity]-(prot:PhysicalEntity)
    RETURN DISTINCT
      prot.stId AS st_id,
      prot.displayName AS display_name,
      gene.geneName AS canonical_gene_name,
      '' AS synonyms_gene_name,
      gene.url AS uniprot_link,
      "https://plantreactome.gramene.org/content/detail/" + prot.stId AS url
    """,
}


def clean_value(v):
    if v is None:
        return ""
    if isinstance(v, list):
        return "|".join(str(x) for x in v)
    s = str(v).strip()
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1]
        items = [i.strip().strip('"').strip("'") for i in inner.split(",") if i.strip()]
        return "|".join(items)
    return s


def run_query(driver, query):
    with driver.session() as session:
        result = session.run(query)
        records = [r.data() for r in result]
    cleaned = []
    for row in records:
        new_row = {}
        for k, v in row.items():
            if k is None:
                continue
            new_row[k.strip()] = clean_value(v)
        cleaned.append(new_row)
    return cleaned


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    for name, query in QUERIES.items():
        print(f"Exporting {name}...")
        rows = run_query(driver, query)
        if not rows:
            print(f"  WARNING: No rows returned for {name}")
            continue
        fieldnames = list(rows[0].keys())
        outfile = OUTPUT_DIR / f"{name}.csv"
        with open(outfile, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved {len(rows)} rows to {outfile}")
    driver.close()
    print("Done.")


if __name__ == "__main__":
    main()
