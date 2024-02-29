from neo4j import GraphDatabase

class Neo4jConnector:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def execute_query(self, query):
        with self._driver.session() as session:
            result = session.run(query)
            return result.data()

def get_reactions(connector):
    query = """
    MATCH (pathway:Pathway)-[:hasEvent]->(reaction:ReactionLikeEvent)
    WHERE pathway.speciesName = "Homo sapiens"
    OPTIONAL MATCH (reaction)-[:input]->(input:PhysicalEntity)
    OPTIONAL MATCH (reaction)-[:output]->(output:PhysicalEntity)
    OPTIONAL MATCH (reaction)-[:catalystActivity]->(catalystActivity:CatalystActivity)-[:physicalEntity]->(catalyst:PhysicalEntity)
    RETURN  pathway.stId AS pathway_ID,
     pathway.displayName AS pathway_name,
     reaction.stId AS reaction_ID,
     reaction.displayName AS reaction_name,
     COLLECT(DISTINCT input.stId) AS input_ID,
     COLLECT(DISTINCT input.displayName) AS input,
     COLLECT(DISTINCT output.stId) AS output_ID,
     COLLECT(DISTINCT output.displayName) AS output,
     COLLECT(DISTINCT catalyst.stId) AS catalyst_ID,
     COLLECT (DISTINCT catalyst.displayName) AS catalyst
    """
    return connector.execute_query(query)

def get_summations(connector):
    query = """
    MATCH (e)-[:summation]->(summation:Summation)
    WHERE (e:Pathway OR e:ReactionLikeEvent) and e.speciesName = "Homo sapiens"
    RETURN e.stId AS Pathway_ID,
    e.displayName AS Pathway_name,
    labels(e) AS labels,
    summation.text AS summation
    """
    return connector.execute_query(query)

def get_complexes(connector):
    query = """
    MATCH (complex:Complex)-[:hasComponent]->(component)
    WHERE complex.speciesName = "Homo sapiens"
    RETURN complex.speciesName,
     complex.stId as Complex_ID,
     complex.name AS Complex_name,
     component.stId AS Component_ID,
     component.name AS Component_name
    """
    return connector.execute_query(query)

def get_ewas(connector):
    query = """
    MATCH q1 =(database:ReferenceDatabase)<-[:referenceDatabase]-(entity1:ReferenceEntity)<--(gene:ReferenceEntity)<-[:referenceEntity]-(protein:PhysicalEntity)
    where database.displayName = "HGNC"
    RETURN DISTINCT protein.stId AS entity_ID,
     protein.displayName AS entity,
     entity1.geneName AS canonical_geneName,
     gene.geneName AS synonyms_geneName,
     gene.databaseName AS database,
     gene.url AS Uniprot_link
    """
    return connector.execute_query(query)

def get_ewas_comments(connector):
    query = """
    MATCH q1 =(database:ReferenceDatabase)<-[:referenceDatabase]-(entity1:ReferenceEntity)<--(gene:ReferenceEntity)<-[:referenceEntity]-(protein:PhysicalEntity)
    where database.displayName = "HGNC"
    RETURN DISTINCT protein.stId AS entity_ID,
     protein.displayName AS entity, 
     entity1.geneName AS canonical_geneName,
     gene.comment AS Function
    """
    return connector.execute_query(query)
