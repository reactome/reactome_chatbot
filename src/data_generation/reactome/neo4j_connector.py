from typing import Any, Optional

from neo4j import GraphDatabase

Neo4jDict = dict[str, Any]


class Neo4jConnector:
    def __init__(self, uri: str, user: Optional[str], password: Optional[str]):
        if user is None or password is None:
            self._driver = GraphDatabase.driver(uri)
        else:
            self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def execute_query(self, query: str) -> list[Neo4jDict]:
        with self._driver.session() as session:
            result = session.run(query)
            return result.data()


def get_reactions(connector: Neo4jConnector) -> list[Neo4jDict]:
    query = """
    MATCH (pathway:Pathway)-[:hasEvent]->(reaction:ReactionLikeEvent)
    WHERE pathway.speciesName = "Homo sapiens"
    OPTIONAL MATCH (reaction)-[:input]->(input:PhysicalEntity)
    OPTIONAL MATCH (reaction)-[:output]->(output:PhysicalEntity)
    OPTIONAL MATCH (reaction)-[:catalystActivity]->(catalystActivity:CatalystActivity)-[:physicalEntity]->(catalyst:PhysicalEntity)
    RETURN
     reaction.stId AS st_id,
     reaction.displayName AS display_name,
     pathway.stId AS pathway_id,
     pathway.displayName AS pathway_name,
     COLLECT(DISTINCT input.stId) AS input_id,
     COLLECT(DISTINCT input.displayName) AS input_name,
     COLLECT(DISTINCT output.stId) AS output_id,
     COLLECT(DISTINCT output.displayName) AS output_name,
     COLLECT(DISTINCT catalyst.stId) AS catalyst_id,
     COLLECT(DISTINCT catalyst.displayName) AS catalyst_name
    """
    return connector.execute_query(query)


def get_summations(connector: Neo4jConnector) -> list[Neo4jDict]:
    query = """
    MATCH (e)-[:summation]->(summation:Summation)
    WHERE (e:Pathway OR e:ReactionLikeEvent) and e.speciesName = "Homo sapiens"
    RETURN e.stId AS st_id,
    e.displayName AS display_name,
    labels(e) AS labels,
    CASE
        WHEN size(summation.text) > 10000 THEN LEFT(summation.text, 10000) + "..."
        ELSE summation.text
    END AS summation
    """
    return connector.execute_query(query)


def get_complexes(connector: Neo4jConnector) -> list[Neo4jDict]:
    query = """
    MATCH (complex:Complex)-[:hasComponent]->(component)
    WHERE complex.speciesName = "Homo sapiens"
    RETURN complex.speciesName as species,
     complex.stId as st_id,
     complex.name AS display_name,
     component.stId AS component_id,
     component.name AS component_name
    """
    return connector.execute_query(query)


def get_ewas(connector: Neo4jConnector) -> list[Neo4jDict]:
    query = """
     MATCH q1 =(database:ReferenceDatabase)<-[:referenceDatabase]-(entity1:ReferenceEntity)<--(gene:ReferenceEntity)<-[:referenceEntity]-(protein:PhysicalEntity)
      where database.displayName = "HGNC"
     RETURN DISTINCT protein.stId AS st_id,
      protein.displayName AS display_name,
      entity1.geneName AS canonical_gene_name,
      gene.geneName AS synonyms_gene_name,
      gene.url AS uniprot_link
            """
    return connector.execute_query(query)
