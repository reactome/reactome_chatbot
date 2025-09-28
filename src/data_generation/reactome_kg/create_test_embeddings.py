import logging
import os
import sys
import time
from typing import Any, Dict, List

import weaviate
from langchain_openai import OpenAIEmbeddings

# Test-specific configuration
TEST_NEO4J_URI = "bolt://localhost:7690"
TEST_NEO4J_USER = "neo4j"
TEST_NEO4J_PASSWORD = "reactome-test"
TEST_NEO4J_DATABASE = "reactome-kg-test"

TEST_WEAVIATE_HOST = "localhost"
TEST_WEAVIATE_PORT = 8081
TEST_WEAVIATE_CLASS_NAME = "TestReactomeKG"

# TP53-related terms for filtering
TP53_TERMS = [
    "TP53",
    "p53",
    "tumor protein p53",
    "cellular tumor antigen p53",
    "transformation-related protein 53",
    "tumor suppressor p53",
]

# Batch processing
BATCH_SIZE = 10
MIN_TEXT_LENGTH = 50


def _setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def _connect_to_services() -> tuple:
    """Connect to Neo4j and Weaviate services."""
    logger = logging.getLogger(__name__)

    # Connect to Neo4j with retry logic
    neo4j_driver = None
    for attempt in range(10):  # Try up to 10 times
        try:
            from neo4j import GraphDatabase

            neo4j_driver = GraphDatabase.driver(
                TEST_NEO4J_URI, auth=(TEST_NEO4J_USER, TEST_NEO4J_PASSWORD)
            )
            neo4j_driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {TEST_NEO4J_URI}")
            break
        except Exception as e:
            if attempt == 9:  # Last attempt
                logger.error(f"Failed to connect to Neo4j after 10 attempts: {e}")
                raise
            logger.info(
                f"Neo4j connection attempt {attempt + 1}/10 failed, retrying in 10 seconds..."
            )
            time.sleep(10)

    if neo4j_driver is None:
        raise Exception("Failed to establish Neo4j connection")

    # Connect to Weaviate
    try:
        weaviate_client = weaviate.WeaviateClient(
            connection_params=weaviate.connect.ConnectionParams.from_url(
                f"http://{TEST_WEAVIATE_HOST}:{TEST_WEAVIATE_PORT}", grpc_port=50052
            ),
            additional_headers={"X-OpenAI-Api-Key": os.environ.get("OPENAI_API_KEY")},
        )
        logger.info(
            f"Connected to Weaviate at {TEST_WEAVIATE_HOST}:{TEST_WEAVIATE_PORT}"
        )
    except Exception as e:
        logger.error(f"Failed to connect to Weaviate: {e}")
        raise

    return neo4j_driver, weaviate_client


def _is_tp53_related(text: str) -> bool:
    """Check if text content is related to TP53."""
    if not text:
        return False

    text_lower = str(text).lower()
    return any(term.lower() in text_lower for term in TP53_TERMS)


def _fetch_tp53_nodes(neo4j_driver) -> List[Dict[str, Any]]:
    """Fetch TP53-related nodes from Neo4j."""
    logger = logging.getLogger(__name__)

    query = """
    MATCH (n)
    WHERE n.name CONTAINS 'TP53' OR n.name CONTAINS 'p53' 
       OR n.displayName CONTAINS 'TP53' OR n.displayName CONTAINS 'p53'
       OR n.description CONTAINS 'TP53' OR n.description CONTAINS 'p53'
       OR n.text_content CONTAINS 'TP53' OR n.text_content CONTAINS 'p53'
       OR n.stableId CONTAINS 'TP53'
    RETURN n.stableId as stable_id, 
           labels(n)[0] as label,
           n.name as name,
           n.displayName as display_name,
           n.description as description,
           n.text_content as text_content
    LIMIT 200
    """

    try:
        with neo4j_driver.session(database=TEST_NEO4J_DATABASE) as session:
            result = session.run(query)
            nodes = []

            for record in result:
                node_data = {
                    "stable_id": record["stable_id"],
                    "label": record["label"],
                    "name": record["name"],
                    "display_name": record["display_name"],
                    "description": record["description"],
                    "text_content": record["text_content"],
                }

                # Additional TP53 filtering
                combined_text = " ".join(
                    [
                        node_data.get("name", ""),
                        node_data.get("description", ""),
                        node_data.get("text_content", ""),
                        node_data.get("stable_id", ""),
                    ]
                )

                if _is_tp53_related(combined_text):
                    nodes.append(node_data)

            logger.info(f"Found {len(nodes)} TP53-related nodes")
            return nodes

    except Exception as e:
        logger.error(f"Error fetching TP53 nodes: {e}")
        raise


def _create_embeddings_and_store(nodes: List[Dict[str, Any]], weaviate_client) -> int:
    """Create embeddings for nodes and store in Weaviate."""
    logger = logging.getLogger(__name__)

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Create Weaviate schema if it doesn't exist
    try:
        if weaviate_client.schema.exists(TEST_WEAVIATE_CLASS_NAME):
            logger.info(f"Deleting existing schema: {TEST_WEAVIATE_CLASS_NAME}")
            weaviate_client.schema.delete_class(TEST_WEAVIATE_CLASS_NAME)
    except Exception as e:
        logger.warning(f"Error checking/deleting schema: {e}")

    # Create new schema
    schema = {
        "class": TEST_WEAVIATE_CLASS_NAME,
        "description": "Test Reactome KG embeddings for TP53-related entities",
        "vectorizer": "text2vec-openai",
        "moduleConfig": {
            "text2vec-openai": {
                "model": "text-embedding-3-large",
                "modelVersion": "002",
                "dimensions": 3072,
                "type": "text",
            }
        },
        "properties": [
            {
                "name": "stable_id",
                "dataType": ["string"],
                "description": "Reactome stable identifier",
            },
            {"name": "label", "dataType": ["string"], "description": "Node label/type"},
            {"name": "name", "dataType": ["string"], "description": "Node name"},
            {
                "name": "display_name",
                "dataType": ["string"],
                "description": "Display name",
            },
            {
                "name": "description",
                "dataType": ["string"],
                "description": "Node description",
            },
            {
                "name": "text_content",
                "dataType": ["text"],
                "description": "Full text content for embedding",
            },
        ],
    }

    try:
        weaviate_client.schema.create_class(schema)
        logger.info(f"Created Weaviate schema: {TEST_WEAVIATE_CLASS_NAME}")
    except Exception as e:
        logger.error(f"Error creating schema: {e}")
        raise

    # Process nodes in batches
    processed_count = 0

    for i in range(0, len(nodes), BATCH_SIZE):
        batch = nodes[i : i + BATCH_SIZE]
        batch_objects = []

        for node in batch:
            # Create text content for embedding
            text_parts = []
            if node.get("name"):
                text_parts.append(f"Name: {node['name']}")
            if node.get("display_name"):
                text_parts.append(f"Display Name: {node['display_name']}")
            if node.get("description"):
                text_parts.append(f"Description: {node['description']}")
            if node.get("text_content"):
                text_parts.append(f"Content: {node['text_content']}")

            text_content = " | ".join(text_parts)

            if len(text_content) < MIN_TEXT_LENGTH:
                continue

            # Create Weaviate object
            obj = {
                "stable_id": node["stable_id"],
                "label": node["label"],
                "name": node.get("name", ""),
                "display_name": node.get("display_name", ""),
                "description": node.get("description", ""),
                "text_content": text_content,
            }

            batch_objects.append(obj)

        # Store batch in Weaviate
        if batch_objects:
            try:
                with weaviate_client.batch as batch:
                    for obj in batch_objects:
                        batch.add_data_object(
                            data_object=obj, class_name=TEST_WEAVIATE_CLASS_NAME
                        )

                processed_count += len(batch_objects)
                logger.info(
                    f"Processed batch {i//BATCH_SIZE + 1}: {len(batch_objects)} objects"
                )

            except Exception as e:
                logger.error(f"Error storing batch: {e}")
                continue

    logger.info(f"Total objects processed and stored: {processed_count}")
    return processed_count


def main():
    """Main function to create test embeddings."""
    logger = _setup_logging()

    try:
        logger.info("Starting TP53-focused test embedding creation...")
        logger.info(f"Neo4j: {TEST_NEO4J_URI} (database: {TEST_NEO4J_DATABASE})")
        logger.info(f"Weaviate: {TEST_WEAVIATE_HOST}:{TEST_WEAVIATE_PORT}")

        # Connect to services
        neo4j_driver, weaviate_client = _connect_to_services()

        # Fetch TP53 nodes
        nodes = _fetch_tp53_nodes(neo4j_driver)

        if not nodes:
            logger.warning("No TP53-related nodes found!")
            return

        # Create and store embeddings
        processed_count = _create_embeddings_and_store(nodes, weaviate_client)

        logger.info("=" * 60)
        logger.info("TEST EMBEDDING CREATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total TP53 nodes found: {len(nodes)}")
        logger.info(f"Total embeddings created: {processed_count}")
        logger.info(f"Weaviate class: {TEST_WEAVIATE_CLASS_NAME}")
        logger.info("=" * 60)

        logger.info("Test embedding creation completed successfully!")

    except Exception as e:
        logger.error(f"Error creating test embeddings: {e}")
        sys.exit(1)
    finally:
        if "neo4j_driver" in locals():
            neo4j_driver.close()


if __name__ == "__main__":
    main()
