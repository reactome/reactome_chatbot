import os
import logging
from typing import Dict, Any, Tuple, Optional

try:
    from dotenv import load_dotenv
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    env_path = os.path.join(project_root, '.env')
    load_dotenv(env_path)
except ImportError:
    pass

from .graphrag_retriever import GraphRAGRetriever
from .graph_clients import WeaviateVectorClient, BaseNeo4jGraphClient
from .graph_traversal_strategies import (
    OneHopStrategy, 
    SteinerTreeStrategy,
)
from .graph_traversal_strategies.one_hop import OneHopRenderer
from .graph_traversal_strategies.steiner_tree import SteinerTreeRenderer
from .retrieval_utils import ReactomeRetrievalConfig, UniProtRetrievalConfig, VectorSearchConfig, GraphTraversalConfig

from .uniprot_retriever import UniProtRetriever

logger = logging.getLogger(__name__)

# Weaviate Configuration
WEAVIATE_HOST = "WEAVIATE_HOST"
WEAVIATE_PORT = "WEAVIATE_PORT"
WEAVIATE_GRPC_PORT = "WEAVIATE_GRPC_PORT"
WEAVIATE_INDEX = "WEAVIATE_INDEX"
OPENAI_API_KEY = "OPENAI_API_KEY"
EMBEDDING_MODEL = "EMBEDDING_MODEL"

# Neo4j Configuration
NEO4J_URI = "NEO4J_URI"
NEO4J_USER = "NEO4J_USER"
NEO4J_PASSWORD = "NEO4J_PASSWORD"
NEO4J_DATABASE = "NEO4J_DATABASE"

# Graph RAG Configuration
GRAPH_RAG_STRATEGY = "GRAPH_RAG_STRATEGY"
GRAPH_RAG_USE_RRF = "GRAPH_RAG_USE_RRF"
GRAPH_RAG_RRF_PER_QUERY_K = "GRAPH_RAG_RRF_PER_QUERY_K"
GRAPH_RAG_RRF_FINAL_K = "GRAPH_RAG_RRF_FINAL_K"
GRAPH_RAG_RRF_LAMBDA = "GRAPH_RAG_RRF_LAMBDA"
GRAPH_RAG_RRF_ALPHA = "GRAPH_RAG_RRF_ALPHA"
GRAPH_RAG_RRF_CUTOFF_K = "GRAPH_RAG_RRF_CUTOFF_K"
GRAPH_RAG_ALPHA = "GRAPH_RAG_ALPHA"
GRAPH_RAG_MAX_NEIGHBORS_PER_TYPE = "GRAPH_RAG_MAX_NEIGHBORS_PER_TYPE"
GRAPH_RAG_MAX_TOTAL = "GRAPH_RAG_MAX_TOTAL"
GRAPH_RAG_SOURCE_ID = "GRAPH_RAG_SOURCE_ID"
GRAPH_RAG_GDS_GRAPH_NAME = "GRAPH_RAG_GDS_GRAPH_NAME"

# UniProt Configuration
UNIPROT_USE_RRF = "UNIPROT_USE_RRF"
UNIPROT_RRF_PER_QUERY_K = "UNIPROT_RRF_PER_QUERY_K"
UNIPROT_RRF_FINAL_K = "UNIPROT_RRF_FINAL_K"
UNIPROT_RRF_LAMBDA = "UNIPROT_RRF_LAMBDA"
UNIPROT_RRF_ALPHA = "UNIPROT_RRF_ALPHA"
UNIPROT_RRF_CUTOFF_K = "UNIPROT_RRF_CUTOFF_K"
UNIPROT_ALPHA = "UNIPROT_ALPHA"

STRATEGY_REGISTRY = {
    "one_hop": OneHopStrategy(),
    "steiner_tree": SteinerTreeStrategy(),
}

RENDERER_REGISTRY = {
    "one_hop": OneHopRenderer,
    "steiner_tree": SteinerTreeRenderer,
}

def get_weaviate_config() -> Dict[str, Any]:
    """
    Get Weaviate configuration from environment variables.
    
    Returns:
        Dictionary containing Weaviate connection and configuration parameters
    """
    return {
        "host": os.getenv(WEAVIATE_HOST, "localhost"),
        "port": int(os.getenv(WEAVIATE_PORT, "8080")),
        "grpc_port": int(os.getenv(WEAVIATE_GRPC_PORT, "50051")),
        "index_name": os.getenv(WEAVIATE_INDEX, "ReactomeKGNode"),
        "openai_api_key": os.getenv(OPENAI_API_KEY, ""),
        "embedding_model": os.getenv(EMBEDDING_MODEL, "text-embedding-3-large"),
    }


def get_neo4j_config() -> Dict[str, Any]:
    """
    Get Neo4j configuration from environment variables.
    
    Returns:
        Dictionary containing Neo4j connection parameters
    """
    return {
        "uri": os.getenv(NEO4J_URI, "bolt://localhost:7687"),
        "user": os.getenv(NEO4J_USER, "neo4j"),
        "password": os.getenv(NEO4J_PASSWORD, "reactome"),
        "default_db": os.getenv(NEO4J_DATABASE, "neo4j"),
    }


def _build_vector_config(
    prefix: str,
    use_rrf_default: str = "false",
    rrf_per_query_k_default: str = "20",
    rrf_final_k_default: str = "10",
    rrf_lambda_default: str = "60.0",
    rrf_alpha_default: str = "0.7",
    rrf_cutoff_k_default: str = "0",
    alpha_default: str = "0.7",
) -> VectorSearchConfig:
    """
    Build a VectorSearchConfig from environment variables with a given prefix.
    
    Args:
        prefix: Environment variable prefix (e.g., "GRAPH_RAG" or "UNIPROT")
        use_rrf_default: Default value for use_rrf
        rrf_per_query_k_default: Default value for rrf_per_query_k
        rrf_final_k_default: Default value for rrf_final_k
        rrf_lambda_default: Default value for rrf_lambda
        rrf_alpha_default: Default value for rrf_alpha
        rrf_cutoff_k_default: Default value for rrf_cutoff_k
        alpha_default: Default value for alpha
        
    Returns:
        Configured VectorSearchConfig instance
    """
    return VectorSearchConfig(
        use_rrf=os.getenv(f"{prefix}_USE_RRF", use_rrf_default).lower() == "true",
        rrf_per_query_k=int(os.getenv(f"{prefix}_RRF_PER_QUERY_K", rrf_per_query_k_default)),
        rrf_final_k=int(os.getenv(f"{prefix}_RRF_FINAL_K", rrf_final_k_default)),
        rrf_lambda=float(os.getenv(f"{prefix}_RRF_LAMBDA", rrf_lambda_default)),
        rrf_alpha=float(os.getenv(f"{prefix}_RRF_ALPHA", rrf_alpha_default)),
        rrf_cutoff_k=int(os.getenv(f"{prefix}_RRF_CUTOFF_K", rrf_cutoff_k_default)) or None,
        alpha=float(os.getenv(f"{prefix}_ALPHA", alpha_default)),
    )


def get_reactome_config() -> ReactomeRetrievalConfig:
    """
    Get Reactome retrieval configuration from environment variables.
    
    Returns:
        Configured ReactomeRetrievalConfig instance
    """
    # Parse strategy sequence (comma-separated list)
    strategy_env = os.getenv(GRAPH_RAG_STRATEGY, "one_hop")
    strategy_sequence = [s.strip() for s in strategy_env.split(",")]
    
    # Vector search configuration
    vector_config = _build_vector_config(
        prefix="GRAPH_RAG",
        rrf_final_k_default="5"
    )
    
    # Graph traversal configuration
    graph_config = GraphTraversalConfig(
        strategy_sequence=strategy_sequence,
        max_neighbors_per_type=int(os.getenv(GRAPH_RAG_MAX_NEIGHBORS_PER_TYPE, "5")),
        max_total=int(os.getenv(GRAPH_RAG_MAX_TOTAL, "10")),
        source_id=os.getenv(GRAPH_RAG_SOURCE_ID) or None,
        gds_graph_name=os.getenv(GRAPH_RAG_GDS_GRAPH_NAME) or None,
    )
    
    return ReactomeRetrievalConfig(
        vector_config=vector_config,
        graph_config=graph_config,
    )


def get_uniprot_config() -> UniProtRetrievalConfig:
    """
    Get UniProt retrieval configuration from environment variables.
    
    Returns:
        Configured UniProtRetrievalConfig instance
    """
    vector_config = _build_vector_config(prefix="UNIPROT")
    
    return UniProtRetrievalConfig(
        vector_config=vector_config,
    )


def get_uniprot_embeddings() -> Dict[str, Any]:
    """
    Get UniProt embeddings configuration from environment variables.
    
    Returns:
        Dictionary containing UniProt embeddings configuration
    """
    from src.util.embedding_environment import EmbeddingEnvironment
    
    embeddings_directory = EmbeddingEnvironment.get_dir("uniprot")
    embedding_model = EmbeddingEnvironment.get_model("uniprot")
    
    return {
        "embeddings_directory": embeddings_directory,
        "embedding_model": embedding_model,
    }

def create_vector_client() -> WeaviateVectorClient:
    """
    Create a Weaviate vector client with Reactome configuration from environment.
    
    Returns:
        Configured WeaviateVectorClient instance for Reactome data
        
    Raises:
        ValueError: If OPENAI_API_KEY environment variable is not set
    """
    config = get_weaviate_config()
    if not config["openai_api_key"]:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    logger.info("Creating Weaviate client for %s:%s", config["host"], config["port"])
    
    return WeaviateVectorClient(
        **config,
        text_key="text_content",  # Reactome uses combined text content
        attributes=[
            "stId",           # Reactome stable ID 
            "entity_type",    # Entity type (Pathway, Reaction, etc.)
            "name",           
            "url"             
        ],
        validate_stable_ids=False,  
    )

def create_graph_client() -> BaseNeo4jGraphClient:
    """
    Create a Neo4j graph client with configuration from environment.
    
    Returns:
        Configured BaseNeo4jGraphClient instance
    """
    config = get_neo4j_config()
    logger.info("Creating Neo4j client for %s", config["uri"])
    return BaseNeo4jGraphClient(**config)

def create_reactome_retriever(
    strategy_registry: Optional[Dict[str, Any]] = None,
    renderer_registry: Optional[Dict[str, Any]] = None,
) -> GraphRAGRetriever:
    """
    Create a Reactome Graph RAG retriever with default configuration.
    
    Args:
        strategy_registry: Optional custom strategy registry
        renderer_registry: Optional custom renderer registry
        
    Returns:
        Configured GraphRAGRetriever instance for Reactome data
    """
    strategies = strategy_registry or STRATEGY_REGISTRY
    renderers = renderer_registry or RENDERER_REGISTRY
    vector_client = create_vector_client()
    graph_client = create_graph_client()
    logger.info("Creating Reactome GraphRAGRetriever with default registries")
    return GraphRAGRetriever(
        vector_client=vector_client,
        graph_client=graph_client,
        strategy_registry=strategies,
        renderer_registry=renderers,
    )


def create_reactome_retriever_custom(
    retrieval_config: Optional[ReactomeRetrievalConfig] = None,
    strategy_registry: Optional[Dict[str, Any]] = None,
    renderer_registry: Optional[Dict[str, Any]] = None,
) -> Tuple[GraphRAGRetriever, ReactomeRetrievalConfig]:
    """
    Create a Reactome Graph RAG retriever with custom configuration.
    
    Args:
        retrieval_config: Optional retrieval configuration
        strategy_registry: Optional custom strategy registry
        renderer_registry: Optional custom renderer registry
        
    Returns:
        Tuple of (retriever, config) for Reactome data
    """
    strategies = strategy_registry or STRATEGY_REGISTRY
    renderers = renderer_registry or RENDERER_REGISTRY
    vector_client = create_vector_client()
    graph_client = create_graph_client()
    logger.info("Creating Reactome GraphRAGRetriever with default registries")
    retriever = GraphRAGRetriever(
        vector_client=vector_client,
        graph_client=graph_client,
        strategy_registry=strategies,
        renderer_registry=renderers,
    )
    config = retrieval_config or get_reactome_config()
    return retriever, config

def create_uniprot_retriever(
    embedding: Any,
    embeddings_directory: Optional[str] = None,
) -> UniProtRetriever:
    """
    Create a UniProt vector retriever with default configuration.
    
    Args:
        embedding: Embedding model for vector operations
        embeddings_directory: Optional path to embeddings directory
        
    Returns:
        Configured UniProtRetriever instance
    """
    if embeddings_directory is None:
        config = get_uniprot_embeddings()
        embeddings_directory = config["embeddings_directory"]
    
    logger.info("Creating UniProt vector retriever for directory: %s", embeddings_directory)
    return UniProtRetriever(
        embedding=embedding,
        embeddings_directory=embeddings_directory,
    )

def create_uniprot_retriever_custom(
    embedding: Any,
    retrieval_config: Optional[UniProtRetrievalConfig] = None,
    embeddings_directory: Optional[str] = None,
) -> Tuple[UniProtRetriever, UniProtRetrievalConfig]:
    """
    Create a UniProt vector retriever with custom configuration.
    
    Args:
        embedding: Embedding model for vector operations
        retrieval_config: Optional retrieval configuration
        embeddings_directory: Optional path to embeddings directory
        
    Returns:
        Tuple of (retriever, config)
    """
    retriever = create_uniprot_retriever(embedding, embeddings_directory)
    config = retrieval_config or get_uniprot_config()
    return retriever, config
