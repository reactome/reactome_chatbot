import logging
import os
from biocypher import BioCypher
from reactome_biocypher_adapter import EnhancedReactomeAdapter

DEFAULT_BATCH_SIZE = 100000
DEFAULT_NEO4J_URI = "bolt://localhost:7687"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASSWORD = "reactome"

# Configuration paths
BIOCYPHER_CONFIG = "config/biocypher_config.yaml"
SCHEMA_CONFIG = "config/schema_config.yaml"
KG_OUTPUT_DIR = "biocypher-out/enhanced_reactome_kg"

def _setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def _create_biocypher_instance() -> BioCypher:
    """Initialize and return BioCypher instance."""
    return BioCypher(
        biocypher_config_path=BIOCYPHER_CONFIG,
        schema_config_path=SCHEMA_CONFIG,
    )

def _create_adapter() -> EnhancedReactomeAdapter:
    """Create and return EnhancedReactomeAdapter instance."""
    return EnhancedReactomeAdapter(
        uri=os.getenv("NEO4J_URI", DEFAULT_NEO4J_URI),
        user=os.getenv("NEO4J_USER", DEFAULT_NEO4J_USER),
        password=os.getenv("NEO4J_PASSWORD", DEFAULT_NEO4J_PASSWORD),
        debug=True,
    )

def _process_nodes_batch(bc: BioCypher, adapter: EnhancedReactomeAdapter, stats: dict) -> int:
    """Process nodes in batches and return total count."""
    batch_size = DEFAULT_BATCH_SIZE
    node_batch = []
    node_count = 0
    
    for stable_id, label, properties in adapter.get_nodes():
        node_batch.append((stable_id, label, properties))
        node_count += 1
        
        if label not in stats["by_entity_type"]:
            stats["by_entity_type"][label] = 0
        stats["by_entity_type"][label] += 1
        
        if len(node_batch) >= batch_size:
            bc.write_nodes(node_batch)
            node_batch = []
    
    if node_batch:
        bc.write_nodes(node_batch)
    
    return node_count

def _process_edges_batch(bc: BioCypher, adapter: EnhancedReactomeAdapter) -> int:
    """Process edges in batches and return total count."""
    batch_size = DEFAULT_BATCH_SIZE
    edge_batch = []
    edge_count = 0
    
    for edge_id, source_id, target_id, label, properties in adapter.get_edges():
        edge_batch.append((edge_id, source_id, target_id, label, properties))
        edge_count += 1
        
        if len(edge_batch) >= batch_size:
            bc.write_edges(edge_batch)
            edge_batch = []
    
    if edge_batch:
        bc.write_edges(edge_batch)
    
    return edge_count

def _print_summary(stats: dict, import_script: str) -> None:
    """Print completion summary and next steps."""
    print("=" * 60)
    print("KNOWLEDGE GRAPH CREATION COMPLETED")
    print("=" * 60)
    print(f"Nodes created: {stats['kg_nodes_created']:,}")
    print(f"Edges created: {stats['kg_edges_created']:,}")
    print(f"Output directory: {KG_OUTPUT_DIR}")
    print(f"Neo4j import script: {import_script}")
    print()
    print("=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Review the CSV files in the output directory")
    print("2. Run the Neo4j import script to load the data")
    print("3. Validate the KG in Neo4j Browser (http://localhost:7474)")
    print("4. Once validated, run the full pipeline with embeddings")

def main():
    """Create only the Reactome knowledge graph."""
    logger = _setup_logging()
    
    logger.info("Creating Reactome knowledge graph (KG only, no embeddings)...")
    logger.info(f"KG output: {KG_OUTPUT_DIR}")
    
    try:
        logger.info("Initializing BioCypher...")
        bc = _create_biocypher_instance()
        
        logger.info("Connecting to Neo4j...")
        with _create_adapter() as adapter:
            stats = {
                "kg_nodes_created": 0,
                "kg_edges_created": 0,
                "by_entity_type": {},
            }
            
            logger.info("Creating KG nodes...")
            stats["kg_nodes_created"] = _process_nodes_batch(bc, adapter, stats)
            
            logger.info("Creating KG edges...")
            stats["kg_edges_created"] = _process_edges_batch(bc, adapter)
            
            import_script = bc.write_import_call()
            _print_summary(stats, import_script)
            logger.info("Neo4j import script: %s", import_script)
        
    except Exception as e:
        logger.error(f"Error creating knowledge graph: {e}")
        raise

if __name__ == "__main__":
    main()
