from typing import Any

from neo4j import GraphDatabase

Neo4jDict = dict[str, Any]


class Neo4jConnector:
    def __init__(self, uri: str, user: str | None, password: str | None) -> None:
        if user is None or password is None:
            self._driver = GraphDatabase.driver(uri)
        else:
            self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def execute_query(self, query: str) -> list[Neo4jDict]:
        with self._driver.session() as session:
            result = session.run(query)
            records: list[Neo4jDict] = result.data()
            return records


# --- The current sources -----------------------------------------------------
#
# The four queries today's bundle is built from. They stay until generation
# switches to SOURCE_QUERIES below, so `make` keeps working in the meantime.


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


# --- The new sources ---------------------------------------------------------
#
# Eight queries feeding three collections. Six come in normal and disease
# pairs. `ewas` covers each protein once, with its disease forms listed on
# the same row. `diseases` holds the disease terms themselves, which sit
# with the entities beside the mutant proteins and complexes that cause them:
#
#   pathways:   normal_pathways, disease_pathways
#   reactions:  normal_reactions, disease_reactions
#   entities:   normal_complexes, disease_complexes, ewas, and diseases
#
# Paste each query between its triple quotes once it is verified in Neo4j. They
# are raw strings, so a regular expression in Cypher keeps its backslashes.
#
# What every query should return:
#
# - `st_id` and `display_name`, spelled exactly so. The retriever de-duplicates
#   on `st_id` and citations are labelled with `display_name`, so
#   `fetch_source` refuses rows without them.
# - One row per entity. Gather pathways, components, participants and
#   summations into lists with COLLECT rather than returning a row for each;
#   repeated rows for one entity crowd each other out of the results.
# - The same column name for the same idea everywhere, not only within a
#   pair: `failure_reason` for a functional status, `disrupted_in_diseases`
#   for the diseases a normal record's disease forms belong to, and
#   `summation` for any descriptive text, whatever its source. Columns only
#   one half of a pair has are fine.
# - An entity's own summation, never its parent pathway's. Copied down, it
#   makes every reaction in a pathway read alike.
# - `url`, the page a citation links to. A row with an `st_id` links to
#   REACTOME_DETAIL_URL followed by that ID, and `fetch_source` checks every
#   one. `ewas` and `diseases` return the link stored on their record.
#   Generation writes the link as given and never builds one.
# - No trailing semicolon, so every query reaches the driver in the same form.
#
# The two halves of a pair must split one set between them with no overlap. An
# entity returned by both becomes two documents with one `st_id`.
#
# Disease terms are identified differently. They come from the Disease
# Ontology rather than being curated as Reactome events or entities, so
# they are not given Reactome stable IDs. The `diseases` query returns
# instead:
#
# - `display_name`, as everywhere else.
# - `url`, the link stored on the disease term. It is what identifies the
#   row and what a citation points at, so every disease needs its own.
#
# The Disease Ontology identifier is optional. Nothing reads it, and it
# cannot stand in for `st_id`: the website links every `st_id` to a
# Reactome detail page, so a disease identifier there is a dead link.
#
# Proteins are identified differently too. `ewas` has one row per UniProt
# reference record, whose stId is the UniProt accession, such as
# uniprot:Q13131, rather than an R-HSA identifier. It returns `protein_id`,
# `display_name` and the record's own `url`. The identifier stays out of
# `st_id` because the website links every `st_id` to a Reactome detail
# page, while a protein is cited by its UniProt page.

NORMAL_PATHWAYS = r"""
// One row per human pathway.
MATCH (p:Pathway)
WHERE p.speciesName = "Homo sapiens" and p.isInDisease = false

// Parent pathways
OPTIONAL MATCH (parent:Pathway)-[:hasEvent]->(p)
WITH p, collect(DISTINCT parent) AS parent_pathways

// Preview of directly contained subpathways and reactions
OPTIONAL MATCH (p)-[:hasEvent]->(child:Event)
WITH DISTINCT p, parent_pathways, child
ORDER BY child.displayName, child.stId
WITH p, parent_pathways, collect(child) AS all_child_events
WITH p, parent_pathways,
     all_child_events[0..5] AS child_events

// Preceding events
OPTIONAL MATCH (p)-[:precedingEvent]->(previous:Event)
WITH p, parent_pathways, child_events,
     collect(DISTINCT previous) AS preceding_events

// Linked disease pathways and their disease annotations
OPTIONAL MATCH (p)<-[:normalPathway]-(disease_pathway:Pathway)
OPTIONAL MATCH (disease_pathway)-[:disease]->(d:Disease)
WITH p, parent_pathways, child_events,preceding_events, 
     collect(DISTINCT disease_pathway) AS disease_pathways,
     collect(DISTINCT d.displayName) AS disrupted_in_diseases

// Summations
OPTIONAL MATCH (p)-[:summation]->(s:Summation)
WITH p, parent_pathways, child_events, preceding_events, disease_pathways, disrupted_in_diseases,
     collect(DISTINCT CASE
       WHEN size(s.text) > 10000 THEN left(s.text, 10000) + "..."
       ELSE s.text
     END) AS summation

RETURN p.stId AS st_id,
       p.displayName AS display_name,
       [parent IN parent_pathways | parent.stId] AS parent_pathway_ids,
       [parent IN parent_pathways | parent.displayName] AS parent_pathway_names,
       [e IN child_events | e.stId] AS child_event_ids,
       [e IN child_events | e.displayName] AS child_event_names,
       [e IN preceding_events | e.stId] AS preceding_event_ids,
       [e IN preceding_events | e.displayName] AS preceding_event_names,
       [dp IN disease_pathways | dp.stId] AS disease_pathway_ids,
       [dp IN disease_pathways | dp.displayName] AS disease_pathway_names,
       disrupted_in_diseases,
       summation,
       "https://reactome.org/content/detail/" + p.stId AS url
ORDER BY st_id
"""

DISEASE_PATHWAYS = r"""
// One row per human pathway marked isInDisease = true.
MATCH (p:Pathway)
WHERE p.speciesName = "Homo sapiens" AND p.isInDisease = true

// Parent pathways
OPTIONAL MATCH (parent:Pathway)-[:hasEvent]->(p)
WITH p, collect(DISTINCT parent) AS parent_pathways

// Preview of directly contained subpathways and reactions
OPTIONAL MATCH (p)-[:hasEvent]->(child:Event)
WITH DISTINCT p, parent_pathways, child
ORDER BY child.displayName, child.stId
WITH p, parent_pathways, collect(child) AS all_child_events
WITH p, parent_pathways, all_child_events[0..5] AS child_events

// Preceding events
OPTIONAL MATCH (p)-[:precedingEvent]->(previous:Event)
WITH p, parent_pathways, child_events,
     collect(DISTINCT previous) AS preceding_events

// Corresponding normal pathway, when annotated
OPTIONAL MATCH (p)-[:normalPathway]->(normal:Pathway)
WITH p, parent_pathways, child_events, preceding_events,
     collect(DISTINCT normal) AS normal_pathways

// Diseases annotated on this pathway
OPTIONAL MATCH (p)-[:disease]->(d:Disease)
WITH p, parent_pathways, child_events, preceding_events, normal_pathways,
     collect(DISTINCT d.displayName) AS disease

// Summations
OPTIONAL MATCH (p)-[:summation]->(s:Summation)
WITH p, parent_pathways, child_events, preceding_events, normal_pathways, disease,
     collect(DISTINCT CASE
       WHEN size(s.text) > 10000 THEN left(s.text, 10000) + "..."
       ELSE s.text
     END) AS summation

RETURN p.stId AS st_id,
       p.displayName AS display_name,
       [parent IN parent_pathways | parent.stId] AS parent_pathway_ids,
       [parent IN parent_pathways | parent.displayName] AS parent_pathway_names,
       [e IN child_events | e.stId] AS child_event_ids,
       [e IN child_events | e.displayName] AS child_event_names,
       [e IN preceding_events | e.stId] AS preceding_event_ids,
       [e IN preceding_events | e.displayName] AS preceding_event_names,
       [normal IN normal_pathways | normal.stId] AS normal_pathway_ids,
       [normal IN normal_pathways | normal.displayName] AS normal_pathway_names,
       disease,
       summation,
       "https://reactome.org/content/detail/" + p.stId AS url
ORDER BY st_id
"""

NORMAL_REACTIONS = r"""// One row per human reaction marked isInDisease = false.
MATCH (r:ReactionLikeEvent)
WHERE r.speciesName = "Homo sapiens" AND r.isInDisease = false

// Parent pathways
OPTIONAL MATCH (parent:Pathway)-[:hasEvent]->(r)
WITH r, collect(DISTINCT parent) AS parent_pathways

// Preceding events
OPTIONAL MATCH (r)-[:precedingEvent]->(previous:Event)
WITH r, parent_pathways, collect(DISTINCT previous) AS preceding_events

// Following events
OPTIONAL MATCH (r)<-[:precedingEvent]-(following:Event)
WITH r, parent_pathways, preceding_events,
     collect(DISTINCT following) AS following_events

// Inputs
OPTIONAL MATCH (r)-[:input]->(i:PhysicalEntity)
WITH r, parent_pathways, preceding_events, following_events,
     collect(DISTINCT i.stId) AS input_ids,
     collect(DISTINCT i.displayName) AS input_names

// Outputs
OPTIONAL MATCH (r)-[:output]->(o:PhysicalEntity)
WITH r, parent_pathways, preceding_events, following_events, input_ids, input_names,
     collect(DISTINCT o.stId) AS output_ids,
     collect(DISTINCT o.displayName) AS output_names

// Catalysts
OPTIONAL MATCH (r)-[:catalystActivity]->(:CatalystActivity)-[:physicalEntity]->(c:PhysicalEntity)
WITH r, parent_pathways, preceding_events, following_events,
     input_ids, input_names, output_ids, output_names,
     collect(DISTINCT c.stId) AS catalyst_ids,
     collect(DISTINCT c.displayName) AS catalyst_names

// Linked disease reactions and associated diseases
OPTIONAL MATCH (r)<-[:normalReaction]-(variant:ReactionLikeEvent)
OPTIONAL MATCH (variant)-[:disease]->(d:Disease)
WITH r, parent_pathways, preceding_events, following_events,
     input_ids, input_names, output_ids, output_names, catalyst_ids, catalyst_names,
     variant, collect(DISTINCT d.displayName) AS variant_diseases
ORDER BY variant.stId
WITH r, parent_pathways, preceding_events, following_events,
     input_ids, input_names, output_ids, output_names, catalyst_ids, catalyst_names,
     collect(variant {.stId, .displayName, diseases: variant_diseases}) AS variants

// Cellular compartments
OPTIONAL MATCH (r)-[:compartment]->(compartment:Compartment)
WITH r, parent_pathways, preceding_events, following_events,
     input_ids, input_names, output_ids, output_names, catalyst_ids, catalyst_names,
     variants, collect(DISTINCT compartment.displayName) AS cellular_compartment

// Summations
OPTIONAL MATCH (r)-[:summation]->(s:Summation)
WITH r, parent_pathways, preceding_events, following_events,
     input_ids, input_names, output_ids, output_names, catalyst_ids, catalyst_names,
     variants, cellular_compartment,
     collect(DISTINCT CASE
       WHEN size(s.text) > 10000 THEN left(s.text, 10000) + "..."
       ELSE s.text
     END) AS summation

OPTIONAL MATCH (r)-[:regulatedBy]->(reg:Regulation)
WITH r, parent_pathways, preceding_events, following_events,
     input_ids, input_names, output_ids, output_names, catalyst_ids, catalyst_names,
     variants, cellular_compartment, summation,
     collect(DISTINCT reg.displayName) as regulated_by     

RETURN r.stId AS st_id,
       r.displayName AS display_name,
       cellular_compartment,
       input_ids,
       input_names,
       output_ids,
       output_names,
       catalyst_ids,
       catalyst_names,
       regulated_by,
       [p IN parent_pathways | p.stId] AS parent_pathway_ids,
       [p IN parent_pathways | p.displayName] AS parent_pathway_names,
       [e IN preceding_events | e.stId] AS preceding_event_ids,
       [e IN preceding_events | e.displayName] AS preceding_event_names,
       [e IN following_events | e.stId] AS following_event_ids,
       [e IN following_events | e.displayName] AS following_event_names,
       [variant IN variants | variant.stId] AS disease_reaction_ids,
       [variant IN variants | variant.displayName] AS disease_reaction_names,
       reduce(diseases = [], variant IN variants |
         reduce(names = diseases, disease IN variant.diseases |
           CASE
             WHEN disease IN names THEN names
             ELSE names + [disease]
           END
         )
       ) AS disrupted_in_diseases,
       summation,
       "https://reactome.org/content/detail/" + r.stId AS url
ORDER BY st_id
"""

DISEASE_REACTIONS = r"""// One row per human reaction marked isInDisease = true.
MATCH (r:ReactionLikeEvent)
WHERE r.speciesName = "Homo sapiens" AND r.isInDisease = true

// Parent pathways
OPTIONAL MATCH (parent:Pathway)-[:hasEvent]->(r)
WITH r, collect(DISTINCT parent) AS parent_pathways

// Preceding events
OPTIONAL MATCH (r)-[:precedingEvent]->(previous:Event)
WITH r, parent_pathways,
     collect(DISTINCT previous) AS preceding_events

// Following events
OPTIONAL MATCH (r)<-[:precedingEvent]-(following:Event)
WITH r, parent_pathways, preceding_events,
     collect(DISTINCT following) AS following_events

// Inputs
OPTIONAL MATCH (r)-[:input]->(i:PhysicalEntity)
WITH r, parent_pathways, preceding_events, following_events,
     collect(DISTINCT i.stId) AS input_ids,
     collect(DISTINCT i.displayName) AS input_names

// Outputs
OPTIONAL MATCH (r)-[:output]->(o:PhysicalEntity)
WITH r, parent_pathways, preceding_events, following_events,
     input_ids, input_names,
     collect(DISTINCT o.stId) AS output_ids,
     collect(DISTINCT o.displayName) AS output_names

// Catalysts
OPTIONAL MATCH (r)-[:catalystActivity]->(:CatalystActivity)-[:physicalEntity]->(c:PhysicalEntity)
WITH r, parent_pathways, preceding_events, following_events,
     input_ids, input_names, output_ids, output_names,
     collect(DISTINCT c.stId) AS catalyst_ids,
     collect(DISTINCT c.displayName) AS catalyst_names

// Associated diseases
OPTIONAL MATCH (r)-[:disease]->(d:Disease)
WITH r, parent_pathways, preceding_events, following_events,
     input_ids, input_names, output_ids, output_names,
     catalyst_ids, catalyst_names,
     collect(DISTINCT d.displayName) AS disease

// Functional status annotations
OPTIONAL MATCH (r)-[:entityFunctionalStatus]->(efs:EntityFunctionalStatus)
WITH r, parent_pathways, preceding_events, following_events,
     input_ids, input_names, output_ids, output_names,
     catalyst_ids, catalyst_names, disease,
     collect(DISTINCT efs.displayName) AS failure_reason

// Cellular compartments
OPTIONAL MATCH (r)-[:compartment]->(compartment:Compartment)
WITH r, parent_pathways, preceding_events, following_events,
     input_ids, input_names, output_ids, output_names,
     catalyst_ids, catalyst_names, disease, failure_reason,
     collect(DISTINCT compartment.displayName) AS cellular_compartment

// Corresponding normal reaction
OPTIONAL MATCH (r)-[:normalReaction]->(normal:ReactionLikeEvent)
WITH r, parent_pathways, preceding_events, following_events,
     input_ids, input_names, output_ids, output_names,
     catalyst_ids, catalyst_names, disease, failure_reason, cellular_compartment,
     collect(DISTINCT normal.stId) AS normal_reaction_ids,
     collect(DISTINCT normal.displayName) AS normal_reaction_names

// Summations
OPTIONAL MATCH (r)-[:summation]->(s:Summation)
WITH r, parent_pathways, preceding_events, following_events,
     input_ids, input_names, output_ids, output_names,
     catalyst_ids, catalyst_names, disease, failure_reason, cellular_compartment,
     normal_reaction_ids, normal_reaction_names,
     collect(DISTINCT CASE
       WHEN size(s.text) > 10000 THEN left(s.text, 10000) + "..."
       ELSE s.text
     END) AS summation

// Regulation annotations
OPTIONAL MATCH (r)-[:regulatedBy]->(reg:Regulation)
WITH r, parent_pathways, preceding_events, following_events,
     input_ids, input_names, output_ids, output_names,
     catalyst_ids, catalyst_names, disease, failure_reason, cellular_compartment,
     normal_reaction_ids, normal_reaction_names, summation,
     collect(DISTINCT reg.displayName) AS regulated_by

RETURN r.stId AS st_id,
       r.displayName AS display_name,
       cellular_compartment,
       input_ids,
       input_names,
       output_ids,
       output_names,
       catalyst_ids,
       catalyst_names,
       regulated_by,
       disease,
       failure_reason,
       [p IN parent_pathways | p.stId] AS parent_pathway_ids,
       [p IN parent_pathways | p.displayName] AS parent_pathway_names,
       [e IN preceding_events | e.stId] AS preceding_event_ids,
       [e IN preceding_events | e.displayName] AS preceding_event_names,
       [e IN following_events | e.stId] AS following_event_ids,
       [e IN following_events | e.displayName] AS following_event_names,
       normal_reaction_ids,
       normal_reaction_names,
       summation,
       "https://reactome.org/content/detail/" + r.stId AS url
ORDER BY st_id
"""

NORMAL_COMPLEXES = r"""
MATCH (e:PhysicalEntity)
WHERE (e:Complex OR e:EntitySet)
  AND e.speciesName = "Homo sapiens"
  AND e.isInDisease = false

// Direct components, members, and candidates
OPTIONAL MATCH (e)-[link:hasComponent|hasMember|hasCandidate]->(part:PhysicalEntity)
WITH e,
     collect(DISTINCT CASE
       WHEN type(link) = "hasComponent" THEN part
     END) AS components,
     collect(DISTINCT CASE
       WHEN type(link) = "hasMember" THEN part
     END) AS members,
     collect(DISTINCT CASE
       WHEN type(link) = "hasCandidate" THEN part
     END) AS candidates

// Disease counterparts and their associated diseases
OPTIONAL MATCH (e)<-[:normalEntity]-(status:EntityFunctionalStatus)
OPTIONAL MATCH (status)-[:diseaseEntity]->(disease_entity:PhysicalEntity)
OPTIONAL MATCH (disease_entity)-[:disease]->(d:Disease)
WITH e, components, members, candidates,
     collect(DISTINCT disease_entity) AS disease_entities,
     collect(DISTINCT d.displayName) AS disrupted_in_diseases

RETURN e.stId AS st_id,
       e.displayName AS display_name,
       [entity IN disease_entities | entity.stId] AS disease_entity_ids,
       [entity IN disease_entities | entity.displayName] AS disease_entity_names,
       disrupted_in_diseases,
       [part IN components | part.stId] AS component_ids,
       [part IN components | part.displayName] AS component_names,
       [part IN members | part.stId] AS member_ids,
       [part IN members | part.displayName] AS member_names,
       [part IN candidates | part.stId] AS candidate_ids,
       [part IN candidates | part.displayName] AS candidate_names,
       "https://reactome.org/content/detail/" + e.stId AS url
ORDER BY st_id
"""

DISEASE_COMPLEXES = r"""
MATCH (e:PhysicalEntity)
WHERE (e:Complex OR e:EntitySet)
  AND e.speciesName = "Homo sapiens"
  AND e.isInDisease = true

// Direct components, members, and candidates
OPTIONAL MATCH (e)-[link:hasComponent|hasMember|hasCandidate]->(part:PhysicalEntity)
WITH e,
     collect(DISTINCT CASE
       WHEN type(link) = "hasComponent" THEN part
     END) AS components,
     collect(DISTINCT CASE
       WHEN type(link) = "hasMember" THEN part
     END) AS members,
     collect(DISTINCT CASE
       WHEN type(link) = "hasCandidate" THEN part
     END) AS candidates

// Diseases annotated on this entity
OPTIONAL MATCH (e)-[:disease]->(d:Disease)
WITH e, components, members, candidates,
     collect(DISTINCT d.displayName) AS disease

// Functional status annotations and normal counterparts
OPTIONAL MATCH (e)<-[:diseaseEntity]-(status:EntityFunctionalStatus)
OPTIONAL MATCH (status)-[:normalEntity]->(normal_entity:PhysicalEntity)
WITH e, components, members, candidates, disease,
     collect(DISTINCT status.displayName) AS failure_reason,
     collect(DISTINCT normal_entity) AS normal_entities

RETURN e.stId AS st_id,
       e.displayName AS display_name,
       disease,
       failure_reason,
       [entity IN normal_entities | entity.stId] AS normal_entity_ids,
       [entity IN normal_entities | entity.displayName] AS normal_entity_names,
       [part IN components | part.stId] AS component_ids,
       [part IN components | part.displayName] AS component_names,
       [part IN members | part.stId] AS member_ids,
       [part IN members | part.displayName] AS member_names,
       [part IN candidates | part.stId] AS candidate_ids,
       [part IN candidates | part.displayName] AS candidate_names,
       "https://reactome.org/content/detail/" + e.stId AS url
ORDER BY st_id
"""

EWAS = r"""// One row per human UniProt protein reference record.
MATCH (ref:ReferenceSequence)-[:species]->(sp:Species)
WHERE sp.displayName = "Homo sapiens"
  AND ref.databaseName CONTAINS "UniProt"
WITH DISTINCT ref

// Physical forms linked to this reference record
OPTIONAL MATCH (entity:EntityWithAccessionedSequence)-[:referenceEntity]->(ref)

// Disease counterparts of those physical forms
OPTIONAL MATCH
  (entity)<-[:normalEntity]-(status:EntityFunctionalStatus)
          -[:diseaseEntity]->(counterpart:PhysicalEntity)

WITH ref,
     collect(DISTINCT CASE
       WHEN entity.isInDisease = true THEN entity
     END) + collect(DISTINCT counterpart) AS disease_entities

// Preserve protein records that have no disease entities
UNWIND CASE
  WHEN size(disease_entities) = 0 THEN [null]
  ELSE disease_entities
END AS disease_entity

// Diseases annotated on the related disease entities
OPTIONAL MATCH (disease_entity)-[:disease]->(d:Disease)
WITH ref,
     collect(DISTINCT disease_entity) AS disease_entities,
     collect(DISTINCT d.displayName) AS disrupted_in_diseases

RETURN ref.stId AS protein_id,
       coalesce(head(ref.name), head(ref.geneName), ref.displayName) AS display_name,
       ref.geneName AS gene_names,
       ref.comment AS summation,
       [entity IN disease_entities | entity.stId] AS disease_entity_ids,
       [entity IN disease_entities | entity.displayName] AS disease_entity_names,
       disrupted_in_diseases,
       ref.url AS url
ORDER BY display_name, protein_id
"""

# The disease terms themselves: name, synonyms, definition and link.
DISEASES = r"""
// One row per disease annotation.
MATCH (d:Disease)
RETURN distinct d.databaseName + ":" + d.identifier AS disease_id,
       d.displayName AS display_name,
       coalesce(d.synonym, []) AS synonyms,
       d.definition AS summation,
       d.url AS url
ORDER BY display_name, disease_id
"""

SOURCE_QUERIES: dict[str, str] = {
    "normal_pathways": NORMAL_PATHWAYS,
    "disease_pathways": DISEASE_PATHWAYS,
    "normal_reactions": NORMAL_REACTIONS,
    "disease_reactions": DISEASE_REACTIONS,
    "normal_complexes": NORMAL_COMPLEXES,
    "disease_complexes": DISEASE_COMPLEXES,
    "ewas": EWAS,
    "diseases": DISEASES,
}

#: Where a Reactome stable ID's page lives. A row with an `st_id` must link
#: to exactly this followed by that ID.
REACTOME_DETAIL_URL = "https://reactome.org/content/detail/"

#: Every source must return these, with a value on every row.
REQUIRED_COLUMNS: tuple[str, ...] = ("st_id", "display_name", "url")

#: Sources identified some other way. These replace the default above.
REQUIRED_COLUMNS_BY_SOURCE: dict[str, tuple[str, ...]] = {
    "ewas": ("protein_id", "display_name", "url"),
    "diseases": ("display_name", "url"),
}


def fetch_source(connector: Neo4jConnector, name: str) -> list[Neo4jDict]:
    """Run one source's query and check its rows can become documents.

    Raises rather than returning something partial. An unwritten query, an
    empty result or a missing identifier would otherwise surface much later,
    as a collection whose documents cannot be cited or de-duplicated.
    """
    if name not in SOURCE_QUERIES:
        raise KeyError(
            f"No source named {name!r}. Known sources: {', '.join(SOURCE_QUERIES)}."
        )
    query = SOURCE_QUERIES[name]
    if not query.strip():
        raise ValueError(
            f"The {name} query has not been written yet. Paste it into "
            "src/data_generation/reactome/neo4j_connector.py."
        )

    rows = connector.execute_query(query)
    if not rows:
        raise ValueError(f"The {name} query returned no rows.")

    required = REQUIRED_COLUMNS_BY_SOURCE.get(name, REQUIRED_COLUMNS)
    # Neo4j gives every row the same keys, so the first row shows the columns.
    absent = [column for column in required if column not in rows[0]]
    if absent:
        raise ValueError(
            f"The {name} query does not return {', '.join(absent)}. This source "
            f"needs {', '.join(required)}, spelled exactly so; alias them in the "
            "RETURN clause."
        )
    for column in required:
        blank = sum(1 for row in rows if _is_blank(row[column]))
        if blank:
            raise ValueError(
                f"The {name} query returned {blank} of {len(rows)} rows with no "
                f"{column}."
            )
        # A list where one value belongs: `ref.name` is a list of names, and
        # passing it through made every protein's label a list.
        listed = [row[column] for row in rows if isinstance(row[column], list)]
        if listed:
            raise ValueError(
                f"The {name} query returned {len(listed)} of {len(rows)} rows "
                f"whose {column} is a list, such as {listed[0]!r}. It must be "
                "one value; take one with head() or coalesce()."
            )

    # The citation code drops any link that is not a web address, so a bad one
    # would only ever show up as a source that is never cited.
    not_web = [row["url"] for row in rows if not _is_web_address(row["url"])]
    if not_web:
        raise ValueError(
            f"The {name} query returned {len(not_web)} of {len(rows)} rows whose "
            f"url is not an http or https address, such as {not_web[0]!r}."
        )

    # Six queries build this link by hand. A typo in one, or a link built from
    # the wrong record's ID, would otherwise send every citation from that
    # source to the wrong page.
    if "st_id" in required:
        wrong = [
            row for row in rows if row["url"] != REACTOME_DETAIL_URL + str(row["st_id"])
        ]
        if wrong:
            raise ValueError(
                f"The {name} query returned {len(wrong)} of {len(rows)} rows whose "
                f"url is not {REACTOME_DETAIL_URL} followed by their st_id, such "
                f"as {wrong[0]['url']!r} for {wrong[0]['st_id']}."
            )
    return rows


def _is_blank(value: object) -> bool:
    """Missing, empty, whitespace, or a list of nothing but those."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list | tuple):
        return all(_is_blank(item) for item in value)
    return False


def _is_web_address(value: object) -> bool:
    return isinstance(value, str) and value.startswith(("https://", "http://"))


#: The release the connected graph holds, from its one database-info record.
#: Current graphs call it `releaseNumber`; `version` is kept for older ones.
RELEASE_QUERY = (
    "MATCH (d:DBInfo) RETURN coalesce(d.releaseNumber, d.version) AS version"
)


def fetch_release(connector: Neo4jConnector) -> int | None:
    """The Reactome release the connected graph holds, or None if it does not say.

    None rather than an error: the CSVs can still be written and read without
    it. Building a bundle from them is what refuses, because a bundle named for
    one release and built from another is wrong in a way nothing would report.
    """
    versions = {
        str(row["version"])
        for row in connector.execute_query(RELEASE_QUERY)
        if row.get("version") is not None
    }
    if not versions:
        return None
    if len(versions) > 1:
        raise ValueError(
            f"The graph reports more than one release: {', '.join(sorted(versions))}."
        )
    (version,) = versions
    return int(version)
