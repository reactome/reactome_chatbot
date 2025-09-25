from textwrap import dedent

Pathway = dedent("""
MATCH (pathway:Pathway)
WHERE pathway.speciesName = "Homo sapiens"

OPTIONAL MATCH (pathway)-[:summation]->(summation:Summation)
OPTIONAL MATCH (pathway)-[:comment]->(comment:Comment)

OPTIONAL MATCH (pathway)<-[:hasEvent]-(upstream_pathway:Pathway)
    WHERE upstream_pathway.speciesName = "Homo sapiens"

OPTIONAL MATCH (pathway)-[:hasEvent]->(sub_pathway:Pathway)
    WHERE sub_pathway.speciesName = "Homo sapiens"

OPTIONAL MATCH (pathway)-[:compartment]->(cellular_compartment:Compartment)

OPTIONAL MATCH (pathway)-[:disease]->(disease:Disease)

RETURN DISTINCT
    pathway.stId AS stable_id,
    pathway.displayName AS name,
    COLLECT(DISTINCT cellular_compartment.displayName) AS cellular_location,
    COLLECT(DISTINCT disease.displayName) AS associated_disease,
    COLLECT(DISTINCT upstream_pathway.displayName) AS upstream_pathway,
    COLLECT(DISTINCT sub_pathway.displayName) AS associated_pathway,
    COALESCE(summation.text, comment.text, "") AS description,
    'https://reactome.org/content/detail/' + pathway.stId AS url
""")

Reaction = dedent("""
MATCH (pathway:Pathway)-[:hasEvent]->(reaction:ReactionLikeEvent)
WHERE pathway.speciesName = "Homo sapiens"

OPTIONAL MATCH (reaction)-[:summation]->(summation:Summation)
OPTIONAL MATCH (reaction)-[:comment]->(comment:Comment)

OPTIONAL MATCH (reaction)-[:regulatedBy]->(regulator:Regulation)

OPTIONAL MATCH (reaction)-[:compartment]->(cellular_compartment:Compartment)
    WHERE cellular_compartment.speciesName = "Homo sapiens"

OPTIONAL MATCH (reaction)-[:disease]->(disease:Disease)

RETURN DISTINCT
    reaction.stId AS stable_id,
    reaction.displayName AS name,
    COLLECT(DISTINCT regulator.displayName) AS regulated_by,
    COLLECT(DISTINCT pathway.displayName) AS associated_pathway,
    COLLECT(DISTINCT disease.displayName) AS associated_disease,
    COLLECT(DISTINCT cellular_compartment.displayName) AS cellular_location,
    COALESCE(summation.text, comment.text, "") AS description,
    'https://reactome.org/content/detail/' + reaction.stId AS url
""")

MolecularComplex = dedent("""
MATCH (complex:Complex)
WHERE complex.speciesName = "Homo sapiens"

OPTIONAL MATCH (complex)-[:summation]->(summation:Summation)
OPTIONAL MATCH (complex)-[:comment]->(comment:Comment)

OPTIONAL MATCH (complex)-[:hasComponent]->(component:PhysicalEntity)

OPTIONAL MATCH (complex)-[:compartment]->(compartment:Compartment)

OPTIONAL MATCH (complex)-[:disease]->(disease:Disease)

RETURN DISTINCT
    complex.stId AS stable_id,
    complex.displayName AS name,
    COLLECT(DISTINCT component.displayName) AS components,
    COLLECT(DISTINCT disease.displayName) AS associated_disease,
    COLLECT(DISTINCT compartment.displayName) AS cellular_location,
    COALESCE(summation.text, comment.text, "") AS description,
    'https://reactome.org/content/detail/' + complex.stId AS url
""")

GeneProduct = dedent("""
MATCH (pep:EntityWithAccessionedSequence {speciesName:"Homo sapiens"})
OPTIONAL MATCH (pep)-[:referenceEntity]->(ref:ReferenceEntity)
OPTIONAL MATCH (pep)-[:compartment]->(comp:Compartment)
OPTIONAL MATCH (pep)-[:disease]->(d:Disease)
RETURN DISTINCT
  pep.stId AS stable_id,
  pep.displayName AS name,
  ref.geneName AS gene_symbol,
  COLLECT(DISTINCT comp.displayName) AS cellular_location,
  COLLECT(DISTINCT d.displayName) AS associated_disease,
  coalesce(ref.comment, "") AS description,
  'https://reactome.org/content/detail/' + pep.stId AS url;
""")

SmallMolecule = dedent("""
MATCH (molecule:SimpleEntity)
OPTIONAL MATCH (molecule)-[:referenceEntity]-(reference:ReferenceMolecule)
RETURN DISTINCT
    molecule.stId AS stable_id,
    molecule.displayName AS name,
    COALESCE(reference.displayName, "") AS official_name,
    COALESCE(reference.formula, "") AS formula,
    COALESCE('https://reactome.org/content/detail/' + molecule.stId) AS urls
""")

Disease = dedent("""
MATCH (disease:Disease)
OPTIONAL MATCH (disease)-[:summation]->(summation:Summation)
OPTIONAL MATCH (disease)-[:comment]->(comment:Comment)
CALL {
    WITH disease
    MATCH (disease)<-[:disease]-(pathway:Pathway)
    WHERE pathway.speciesName = "Homo sapiens"
    RETURN COLLECT(DISTINCT pathway.displayName)[..10] AS implicated_pathway
}
CALL {
    WITH disease
    MATCH (disease)<-[:disease]-(reaction:ReactionLikeEvent)
    WHERE reaction.speciesName = "Homo sapiens"
    RETURN COLLECT(DISTINCT reaction.displayName)[..10] AS implicated_reaction
}
CALL {
    WITH disease
    MATCH (disease)<-[:disease]-(complex:Complex)
    WHERE complex.speciesName = "Homo sapiens"
    RETURN COLLECT(DISTINCT complex.displayName)[..10] AS implicated_complex
}
CALL {
    WITH disease
    MATCH (disease)<-[:disease]-(variant:EntityWithAccessionedSequence)
    WHERE variant.speciesName = "Homo sapiens"
    RETURN COLLECT(DISTINCT variant.displayName)[..10] AS implicated_variant
}
CALL {
    WITH disease
    MATCH (disease)<-[:disease]-(drug:Drug)
    RETURN COLLECT(DISTINCT drug.displayName)[..10] AS implicated_drug
}

RETURN DISTINCT
    disease.identifier AS stable_id,
    disease.name AS name,
    implicated_pathway AS implicated_pathway,
    implicated_reaction AS implicated_reaction,
    implicated_complex AS implicated_complex,
    implicated_variant AS implicated_variant,
    implicated_drug AS implicated_drug,
    COALESCE(summation.text, comment.text, "") AS description,
    disease.url AS url
""")

Drug = dedent("""
MATCH (drug:Drug)
OPTIONAL MATCH (drug)-[:summation]->(summation:Summation)
OPTIONAL MATCH (drug)-[:comment]->(comment:Comment)
OPTIONAL MATCH (drug)<-[:hasMember|:hasCandidate]-(candidate:CandidateSet)
OPTIONAL MATCH (drug)<-[:hasMember]-(entity:EntitySet)
OPTIONAL MATCH (drug)<-[:input]-(input_reaction:ReactionLikeEvent)
OPTIONAL MATCH (drug)-[:disease]->(disease:Disease)
OPTIONAL MATCH (drug)-[:compartment]->(cellular_compartment:Compartment)
OPTIONAL MATCH (drug)-[:referenceEntity]->(ref:ReferenceTherapeutic)

RETURN DISTINCT
    drug.stId AS stable_id,
    drug.name AS name,
    CASE
        WHEN drug:ChemicalDrug THEN "ChemicalDrug"
        WHEN drug:ProteinDrug THEN "ProteinDrug"
        WHEN drug:RNADrug THEN "RNADrug"
        ELSE "Unknown"
    END AS drug_type,
    COLLECT(DISTINCT candidate.displayName) + COLLECT(DISTINCT input_reaction.displayName) + COLLECT(DISTINCT entity.displayName) AS mode_of_function,
    COLLECT(DISTINCT cellular_compartment.displayName) AS cellular_location,
    COLLECT(DISTINCT disease.displayName) AS associated_disease,
    COALESCE(summation.text, comment.text, "") AS description,
    COALESCE(ref.url, 'https://reactome.org/content/detail/' + drug.stId) AS url
""")

# ──────────────── ENHANCED EDGE QUERIES ────────────────

HasInput = dedent("""
MATCH (p:Pathway {speciesName: "Homo sapiens"})-[:hasEvent]->(rxn:ReactionLikeEvent)
CALL {
  WITH rxn
  MATCH (rxn)-[:input]->(inpe:PhysicalEntity)
  WHERE (inpe:EntityWithAccessionedSequence OR inpe:Complex OR inpe:ReferenceMolecule OR inpe:SimpleEntity)
    AND inpe.stId IS NOT NULL
  RETURN rxn AS r, inpe.stId AS stId, NULL AS set_stId, 'direct' AS membership

  UNION

  WITH rxn
  MATCH (rxn)-[:input]->(s:EntitySet)
  MATCH path = (s)-[:hasMember|hasCandidate*]->(m:PhysicalEntity)
  WHERE (m:EntityWithAccessionedSequence OR m:Complex OR m:ReferenceMolecule OR m:SimpleEntity)
    AND m.stId IS NOT NULL
    AND coalesce(m.speciesName, "Homo sapiens") = "Homo sapiens"
  RETURN rxn AS r, m.stId AS stId, s.stId AS set_stId,
         CASE WHEN ANY(rel IN relationships(path) WHERE type(rel) = 'hasCandidate')
              THEN 'candidate' ELSE 'member' END AS membership
}
RETURN DISTINCT 
  r.stId AS source, 
  stId AS target
""")

HasOutput = dedent("""
MATCH (p:Pathway {speciesName: "Homo sapiens"})-[:hasEvent]->(rxn:ReactionLikeEvent)
CALL {
  WITH rxn
  MATCH (rxn)-[:output]->(outpe:PhysicalEntity)
  WHERE (outpe:EntityWithAccessionedSequence OR outpe:Complex OR outpe:ReferenceMolecule OR outpe:SimpleEntity)
    AND outpe.stId IS NOT NULL
  RETURN rxn AS r, outpe.stId AS stId, NULL AS set_stId, 'direct' AS membership

  UNION

  WITH rxn
  MATCH (rxn)-[:output]->(s:EntitySet)
  MATCH path = (s)-[:hasMember|hasCandidate*]->(m:PhysicalEntity)
  WHERE (m:EntityWithAccessionedSequence OR m:Complex OR m:ReferenceMolecule OR m:SimpleEntity)
    AND m.stId IS NOT NULL
    AND coalesce(m.speciesName, "Homo sapiens") = "Homo sapiens"
  RETURN rxn AS r, m.stId AS stId, s.stId AS set_stId,
         CASE WHEN ANY(rel IN relationships(path) WHERE type(rel) = 'hasCandidate')
              THEN 'candidate' ELSE 'member' END AS membership
}
RETURN DISTINCT r.stId AS source, stId AS target
""")

HasCatalyst = dedent("""
MATCH (p:Pathway {speciesName: "Homo sapiens"})-[:hasEvent]->(rxn:ReactionLikeEvent)
CALL {
  WITH rxn
  MATCH (rxn)-[:catalystActivity]->(:CatalystActivity)-[:physicalEntity]->(pe:PhysicalEntity)
  WHERE (pe:EntityWithAccessionedSequence OR pe:Complex OR pe:ReferenceMolecule)
    AND pe.stId IS NOT NULL
  RETURN rxn AS r, pe.stId AS stId, NULL AS set_stId, 'direct' AS membership

  UNION

  WITH rxn
  MATCH (rxn)-[:catalystActivity]->(:CatalystActivity)-[:physicalEntity]->(s:EntitySet)
  MATCH path = (s)-[:hasMember|hasCandidate*]->(m:PhysicalEntity)
  WHERE (m:EntityWithAccessionedSequence OR m:Complex OR m:ReferenceMolecule)
    AND m.stId IS NOT NULL
    AND coalesce(m.speciesName, "Homo sapiens") = "Homo sapiens"
  RETURN rxn AS r, m.stId AS stId, s.stId AS set_stId,
         CASE WHEN ANY(rel IN relationships(path) WHERE type(rel) = 'hasCandidate')
              THEN 'candidate' ELSE 'member' END AS membership
}
RETURN DISTINCT r.stId AS source, stId AS target

""")

HasComponent = dedent("""
MATCH (cx:Complex {speciesName: "Homo sapiens"})
CALL {
  WITH cx
  MATCH (cx)-[:hasComponent*1..4]->(c:PhysicalEntity)   // bounded
  WHERE c.stId IS NOT NULL
    AND (c:EntityWithAccessionedSequence OR c:Complex OR c:ReferenceMolecule OR c:SimpleEntity)
    AND (c.speciesName IS NULL OR c.speciesName = "Homo sapiens")   // index-friendly
  RETURN cx AS complex, c.stId AS stId, NULL AS set_stId, 'component' AS membership

  UNION

  WITH cx
  MATCH (cx)-[:hasComponent]->(s:EntitySet)
  MATCH path = (s)-[:hasMember|hasCandidate*1..3]->(m:PhysicalEntity)  // bounded
  WHERE m.stId IS NOT NULL
    AND (m:EntityWithAccessionedSequence OR m:Complex OR m:ReferenceMolecule OR m:SimpleEntity)
    AND (m.speciesName IS NULL OR m.speciesName = "Homo sapiens")      // index-friendly
  RETURN cx AS complex, m.stId AS stId, s.stId AS set_stId,
         CASE WHEN ANY(rel IN relationships(path) WHERE type(rel) = 'hasCandidate')
              THEN 'candidate' ELSE 'member' END AS membership
}
RETURN DISTINCT complex.stId AS source, stId AS target

""")

HasVariant = dedent("""
MATCH (db:ReferenceDatabase)<-[:referenceDatabase]-(ref:ReferenceEntity)<--(gene:ReferenceGeneProduct)<-[:referenceEntity]-(protein:EntityWithAccessionedSequence)
WHERE db.displayName = "HGNC" AND coalesce(protein.isInDisease, false) = false
  AND protein.speciesName = "Homo sapiens"
WITH protein, gene

MATCH (variant:EntityWithAccessionedSequence)-[:referenceEntity]->(gene)
WHERE coalesce(variant.isInDisease, false) = true
  AND variant.speciesName = "Homo sapiens"

RETURN DISTINCT
    protein.stId AS source,
    variant.stId AS target
""")

Precedes = dedent(""" 
MATCH (pathway:Pathway)-[:hasEvent]->(reaction:ReactionLikeEvent)
WHERE pathway.speciesName = "Homo sapiens"
MATCH (reaction)<-[:precedingEvent]-(next_reaction:Reaction)

RETURN DISTINCT
    reaction.stId AS source,
    next_reaction.stId as target
""")

PartOf = dedent("""
MATCH (pathway:Pathway)-[:hasEvent]->(reaction:ReactionLikeEvent)
WHERE pathway.speciesName = "Homo sapiens"

RETURN DISTINCT
    reaction.stId AS source,
    pathway.stId as target
""")

SubPathwayOf = dedent("""
MATCH (pathway:Pathway)
WHERE pathway.speciesName = "Homo sapiens"
  AND NOT pathway:TopLevelPathway

MATCH (pathway)<-[:hasEvent]-(upstream_pathway:Pathway)
WHERE upstream_pathway.stId IS NOT NULL

RETURN DISTINCT
    pathway.stId AS source,
    upstream_pathway.stId as target
""")

ActsOn = dedent("""
MATCH (drug:Drug)
OPTIONAL MATCH (drug)<-[:hasMember|:hasCandidate|:hasComponent]-(entity:PhysicalEntity)
OPTIONAL MATCH (drug)<-[:regulator]-(:Regulation)<-[:regulatedBy]-(regulated_reaction:ReactionLikeEvent)
OPTIONAL MATCH (drug)<-[:input|:output]-(reaction:ReactionLikeEvent)
WHERE reaction.speciesName = "Homo sapiens" 
   OR entity.speciesName = "Homo sapiens" 
   OR regulated_reaction.speciesName = "Homo sapiens"
OPTIONAL MATCH (drug)-[:referenceEntity]->(ref:ReferenceTherapeutic)

WITH 
    drug,
    COLLECT(DISTINCT {name: entity.displayName, id: entity.stId}) AS entityTargets,
    COLLECT(DISTINCT {name: regulated_reaction.displayName, id: regulated_reaction.stId}) AS regulatedTargets,
    COLLECT(DISTINCT {name: reaction.displayName, id: reaction.stId}) AS reactionTargets

WITH 
    drug,
    [t IN (entityTargets + regulatedTargets + reactionTargets) WHERE t.name IS NOT NULL] AS allTargets
UNWIND allTargets AS target

RETURN DISTINCT
    drug.stId AS source,
    target.id AS target
""")

Treats = dedent("""
MATCH (drug:Drug)-[:disease]->(disease:Disease)

RETURN DISTINCT

    drug.stId AS source,
    disease.identifier as target
""")

AssociatedWith = dedent("""
MATCH (disease:Disease)
WITH disease

OPTIONAL MATCH (disease)<-[:disease]-(p:Pathway)
  WHERE p.speciesName = "Homo sapiens"
WITH disease,
     COLLECT(DISTINCT { name: p.displayName, id: p.stId }) AS pathways

OPTIONAL MATCH (disease)<-[:disease]-(r:ReactionLikeEvent)
  WHERE r.speciesName = "Homo sapiens"
WITH disease, pathways,
     COLLECT(DISTINCT { name: r.displayName, id: r.stId }) AS reactions

OPTIONAL MATCH (disease)<-[:disease]-(c:Complex)
  WHERE c.speciesName = "Homo sapiens"
WITH disease, pathways, reactions,
     COLLECT(DISTINCT { name: c.displayName, id: c.stId }) AS complexes

OPTIONAL MATCH (disease)<-[:disease]-(e:EntityWithAccessionedSequence)
  WHERE e.speciesName = "Homo sapiens"
WITH disease, pathways, reactions, complexes,
     COLLECT(DISTINCT { name: e.displayName, id: e.stId }) AS proteins

WITH disease,
     [t IN (pathways + reactions + complexes + proteins) WHERE t.name IS NOT NULL] AS all_implications
UNWIND all_implications AS imp

RETURN
    imp.id AS source,
    disease.identifier AS target
""")


# Query registries
NODE_QUERIES = {
    "Pathway": Pathway,
    "Reaction": Reaction,
    "MolecularComplex": MolecularComplex,
    "GeneProduct": GeneProduct,
    "SmallMolecule": SmallMolecule,
    "Disease": Disease,
    "Drug": Drug,
}

EDGE_QUERIES = {
    "HasInput": HasInput,
    "HasOutput": HasOutput,
    "HasCatalyst": HasCatalyst,
    "HasComponent": HasComponent,
    "HasVariant": HasVariant, 
    "Precedes": Precedes,
    "PartOf": PartOf,
    "SubPathwayOf": SubPathwayOf,
    "ActsOn": ActsOn,
    "Treats": Treats,
    "AssociatedWith": AssociatedWith,
}
