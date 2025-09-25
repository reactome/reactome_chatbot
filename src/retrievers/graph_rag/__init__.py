"""
Graph-RAG package: retriever, config, clients, models, utils, strategies.
"""

from .graphrag_retriever import GraphRAGRetriever
from .config import (
    create_reactome_retriever,
    STRATEGY_REGISTRY,
    RENDERER_REGISTRY,
)
from .types import NodeInfo, NeighborInfo, GraphContext, IVectorClient, IGraphClient
from .retrieval_utils import (
    reciprocal_rank_fusion, 
    ReactomeRetrievalConfig, 
    UniProtRetrievalConfig, 
    VectorSearchConfig, 
    GraphTraversalConfig,
)
from .graph_clients import (
    WeaviateVectorClient,
    BaseNeo4jGraphClient,
)
from .graph_traversal_strategies import GraphTraversalStrategy, OneHopStrategy, SteinerTreeStrategy

__all__ = [
    "GraphRAGRetriever",
    "create_reactome_retriever",
    "STRATEGY_REGISTRY",
    "RENDERER_REGISTRY",
    "NodeInfo",
    "NeighborInfo",
    "GraphContext",
    "IVectorClient",
    "IGraphClient",
    "reciprocal_rank_fusion",
    # Unified clients
    "WeaviateVectorClient",
    "BaseNeo4jGraphClient",
    # Strategy classes
    "GraphTraversalStrategy",
    "OneHopStrategy",
    "SteinerTreeStrategy",
    # Configuration classes
    "ReactomeRetrievalConfig",
    "UniProtRetrievalConfig",
    "VectorSearchConfig",
    "GraphTraversalConfig",
]


