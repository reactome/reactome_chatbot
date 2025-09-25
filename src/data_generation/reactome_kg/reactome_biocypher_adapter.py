"""
Enhanced Reactome adapter that creates both KG nodes and embedding data
with proper ID management and comprehensive data extraction.
"""

import logging
import re
from typing import Dict, Any, Iterator, Tuple, Optional, List, Set
from dataclasses import dataclass
from enum import Enum

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neo4j import GraphDatabase
try:
    from queries import NODE_QUERIES, EDGE_QUERIES
except ImportError:
    # Handle import when used as a module
    from .queries import NODE_QUERIES, EDGE_QUERIES

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Entity types in the Reactome knowledge graph."""
    PATHWAY = "Pathway"
    REACTION = "Reaction"
    MOLECULAR_COMPLEX = "MolecularComplex"
    GENE_PRODUCT = "GeneProduct"
    SMALL_MOLECULE = "SmallMolecule"
    DISEASE = "Disease"
    DRUG = "Drug"


# Constants for text content limits
class TextLimits:
    """Limits for text content to prevent overly long embeddings."""
    MAX_DISEASES = 5
    MAX_PATHWAYS = 3
    MAX_COMPONENTS = 10
    MAX_VARIANTS = 5
    MAX_DRUGS = 5


@dataclass
class EmbeddingCandidate:
    """Represents an entity that can be embedded with its text content."""
    stable_id: str
    entity_type: str
    name: str
    text_content: str
    metadata: Dict[str, Any]


class EnhancedReactomeAdapter:
    """
    Enhanced adapter that extracts comprehensive Reactome data for both
    knowledge graph creation and vector database population.
    
    Key improvements:
    - Uses stable Reactome stId as primary identifier
    - Extracts comprehensive text content for embeddings
    - Maintains referential integrity between KG and vector DB
    - Handles both entities with and without descriptions
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        node_queries: Dict[str, str] = NODE_QUERIES,
        edge_queries: Dict[str, str] = EDGE_QUERIES,
        reactome_version: str = "2025-04-01",
        license: str = "CC BY 4.0",
        debug: bool = False
    ):
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        # Test connection
        try:
            self.driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j.")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
        
        self.node_queries = node_queries
        self.edge_queries = edge_queries
        self._source = "reactome"
        self._version = reactome_version
        self._license = license

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.driver.close()
        logger.debug("Neo4j driver closed.")

    def get_nodes(self) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
        """
        Yields (stable_id, input_label, properties_dict) for each node.
        
        Uses Reactome's stable stId as the primary identifier, ensuring
        consistency across KG rebuilds and proper linking to vector DB.
        """
        for stable_id, label, props in self._process_entities():
            # Use text_content from query (already comprehensive)
            # Only add generated text_content if it's missing from the query
            if "text_content" not in props or not props.get("text_content"):
                props.update({
                    "text_content": self._create_comprehensive_text_content(label, props),
                })

            # Convert keys to snake_case and add provenance
            normalized = self._normalize_keys(props)
            normalized.update(
                source=self._source,
                version=self._version,
                license=self._license
            )

            yield stable_id, label, normalized

    def get_edges(self) -> Iterator[Tuple[Optional[str], str, str, str, Dict[str, Any]]]:
        """
        Yields (edge_id, source_id, target_id, input_label, properties_dict).
        
        Uses stable IDs for source and target nodes to ensure proper linking.
        """
        with self.driver.session() as sess:
            for label, cypher in self.edge_queries.items():
                logger.debug("Processing edge query: %s", label)
                for record in sess.run(cypher):
                    raw = record.data()

                    src = raw.pop("source", None)
                    tgt = raw.pop("target", None)

                    if not src or not tgt:
                        logger.warning("Skipping %s (missing endpoints)", label)
                        continue

                    # Add provenance to edge
                    raw.update(
                        source=self._source,
                        version=self._version,
                        license=self._license
                    )

                    yield None, src, tgt, label, raw  # Set edge_id to None to match working format

    def get_embedding_candidates(self) -> Iterator[EmbeddingCandidate]:
        """
        Yields EmbeddingCandidate objects for all entities.
        
        This method provides the data needed for vector database population,
        using stable IDs for proper linking to KG nodes.
        """
        for stable_id, label, props in self._process_entities():
            metadata = self._create_metadata(stable_id, label, props)
            
            yield EmbeddingCandidate(
                stable_id=stable_id,
                entity_type=label,
                name=props.get("name", ""),
                text_content=self._create_comprehensive_text_content(label, props),
                metadata=metadata
            )

    def _process_entities(self) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
        """Common entity processing logic shared by get_nodes and get_embedding_candidates."""
        with self.driver.session() as sess:
            for label, cypher in self.node_queries.items():
                logger.debug("Processing node query: %s", label)
                for record in sess.run(cypher):
                    props = record.data()
                    
                    # Extract stable ID (Reactome stId)
                    stable_id = props.pop("stable_id", None)
                    if not stable_id:
                        logger.warning("Skipping %s (no stable ID)", label)
                        continue

                    # Add stId back as a property for the node
                    props["stId"] = stable_id

                    yield stable_id, label, props

    def _has_description(self, props: Dict[str, Any]) -> bool:
        """Check if entity has meaningful text content."""
        description = props.get("description", "")
        if isinstance(description, list):
            description = " ".join(description) if description else ""
        return bool(description and str(description).strip())

    def _create_metadata(self, stable_id: str, entity_type: str, props: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for vector DB entries."""
        return {
            "stable_id": stable_id,
            "entity_type": entity_type,
            "name": props.get("name", ""),
            "source": self._source,
            "version": self._version,
            "url": props.get("url", ""),
        }

    def _create_comprehensive_text_content(self, entity_type: str, props: Dict[str, Any]) -> str:
        """
        Create comprehensive text content for embedding that includes all node properties.
        
        This method creates a rich, structured text representation of the entire node
        that can be embedded to capture all the entity's information, not just descriptions.
        """
        text_parts = []
        
        # Basic entity information
        name = props.get("name", "")
        if name:
            text_parts.append(f"Entity: {name} (Type: {entity_type})")
        
        # Description (if available)
        description = props.get("description", "")
        if isinstance(description, list):
            description = " ".join(description) if description else ""
        if description and str(description).strip():
            text_parts.append(f"Description: {description}")
        
        # Entity-specific properties
        self._add_entity_specific_properties(text_parts, entity_type, props)
        
        # Common properties
        self._add_common_properties(text_parts, props)
        
        return " ".join(text_parts)

    def _add_entity_specific_properties(self, text_parts: List[str], entity_type: str, props: Dict[str, Any]) -> None:
        """Add entity-specific properties based on type."""
        property_handlers = {
            EntityType.PATHWAY.value: self._add_pathway_properties,
            EntityType.REACTION.value: self._add_reaction_properties,
            EntityType.MOLECULAR_COMPLEX.value: self._add_complex_properties,
            EntityType.GENE_PRODUCT.value: self._add_protein_properties,
            EntityType.SMALL_MOLECULE.value: self._add_molecule_properties,
            EntityType.DISEASE.value: self._add_disease_properties,
            EntityType.DRUG.value: self._add_drug_properties,
        }
        
        handler = property_handlers.get(entity_type)
        if handler:
            handler(text_parts, props)

    def _add_property_if_exists(self, text_parts: List[str], props: Dict[str, Any], 
                              key: str, label: str, limit: Optional[int] = None) -> None:
        """Helper method to add properties with optional limiting."""
        value = props.get(key, [])
        if value:
            if isinstance(value, list):
                if limit:
                    value = value[:limit]
                text_parts.append(f"{label}: {', '.join(value)}")
            else:
                text_parts.append(f"{label}: {value}")

    def _add_pathway_properties(self, text_parts: List[str], props: Dict[str, Any]) -> None:
        """Add pathway-specific properties to text content."""
        self._add_property_if_exists(text_parts, props, "cellular_location", "Cellular Location")
        self._add_property_if_exists(text_parts, props, "associated_disease", "Associated Diseases", TextLimits.MAX_DISEASES)
        self._add_property_if_exists(text_parts, props, "upstream_pathway", "Upstream Pathways", TextLimits.MAX_PATHWAYS)
        self._add_property_if_exists(text_parts, props, "associated_pathway", "Sub-pathways", TextLimits.MAX_PATHWAYS)

    def _add_reaction_properties(self, text_parts: List[str], props: Dict[str, Any]) -> None:
        """Add reaction-specific properties to text content."""
        self._add_property_if_exists(text_parts, props, "regulated_by", "Regulated by", TextLimits.MAX_DISEASES)
        self._add_property_if_exists(text_parts, props, "associated_pathway", "Part of Pathways", TextLimits.MAX_PATHWAYS)
        self._add_property_if_exists(text_parts, props, "associated_disease", "Associated Diseases", TextLimits.MAX_DISEASES)
        self._add_property_if_exists(text_parts, props, "cellular_location", "Cellular Location")

    def _add_complex_properties(self, text_parts: List[str], props: Dict[str, Any]) -> None:
        """Add molecular complex-specific properties to text content."""
        self._add_property_if_exists(text_parts, props, "components", "Components", TextLimits.MAX_COMPONENTS)
        self._add_property_if_exists(text_parts, props, "associated_disease", "Associated Diseases", TextLimits.MAX_DISEASES)
        self._add_property_if_exists(text_parts, props, "cellular_location", "Cellular Location")

    def _add_protein_properties(self, text_parts: List[str], props: Dict[str, Any]) -> None:
        """Add protein-specific properties to text content."""
        self._add_property_if_exists(text_parts, props, "gene_symbol", "Gene Symbol")
        self._add_property_if_exists(text_parts, props, "cellular_location", "Cellular Location")
        self._add_property_if_exists(text_parts, props, "disease_causing_variant", "Disease-causing Variants", TextLimits.MAX_VARIANTS)
        self._add_property_if_exists(text_parts, props, "associated_disease", "Associated Diseases", TextLimits.MAX_DISEASES)

    def _add_molecule_properties(self, text_parts: List[str], props: Dict[str, Any]) -> None:
        """Add small molecule-specific properties to text content."""
        self._add_property_if_exists(text_parts, props, "official_name", "Official Name")
        self._add_property_if_exists(text_parts, props, "formula", "Chemical Formula")

    def _add_disease_properties(self, text_parts: List[str], props: Dict[str, Any]) -> None:
        """Add disease-specific properties to text content."""
        self._add_property_if_exists(text_parts, props, "implicated_pathway", "Implicated Pathways", TextLimits.MAX_PATHWAYS)
        self._add_property_if_exists(text_parts, props, "implicated_reaction", "Implicated Reactions", TextLimits.MAX_DISEASES)
        self._add_property_if_exists(text_parts, props, "implicated_complex", "Implicated Complexes", TextLimits.MAX_DISEASES)
        self._add_property_if_exists(text_parts, props, "implicated_variant", "Implicated Variants", TextLimits.MAX_VARIANTS)
        self._add_property_if_exists(text_parts, props, "implicated_drug", "Implicated Drugs", TextLimits.MAX_DRUGS)

    def _add_drug_properties(self, text_parts: List[str], props: Dict[str, Any]) -> None:
        """Add drug-specific properties to text content."""
        self._add_property_if_exists(text_parts, props, "drug_type", "Drug Type")
        self._add_property_if_exists(text_parts, props, "mode_of_function", "Mode of Function", TextLimits.MAX_DISEASES)
        self._add_property_if_exists(text_parts, props, "cellular_location", "Cellular Location")
        self._add_property_if_exists(text_parts, props, "associated_disease", "Associated Diseases", TextLimits.MAX_DISEASES)

    def _add_common_properties(self, text_parts: List[str], props: Dict[str, Any]) -> None:
        """Add common properties to text content."""
        url = props.get("url", "")
        if url:
            text_parts.append(f"Reactome URL: {url}")

    def _normalize_keys(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert camelCase keys to snake_case, except for special properties."""
        normalized = {}
        # Special properties that should remain in camelCase
        preserve_keys = {"stId"}
        
        for key, val in data.items():
            if key in preserve_keys:
                normalized[key] = val
            else:
                # Convert camelCase to snake_case
                s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", key)
                snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
                normalized[snake] = val
        return normalized

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the data extraction."""
        stats = {
            "total_nodes": 0,
            "embedding_candidates": 0,
            "by_type": {},
        }
        
        # Count nodes by type
        with self.driver.session() as sess:
            for label in self.node_queries.keys():
                query = f"MATCH (n:{label}) RETURN count(n) as count"
                result = sess.run(query).single()
                count = result["count"] if result else 0
                stats["by_type"][label] = count
                stats["total_nodes"] += count
        
        # Count embedding candidates (now all entities)
        stats["embedding_candidates"] = stats["total_nodes"]
        
        return stats
