from typing import List
from pydantic import BaseModel


def merge_contexts(left: str, right: str) -> str:
    """Reducer function to merge context strings from parallel searches."""
    if not left and not right:
        return ""
    elif not left:
        return right
    elif not right:
        return left
    else:
        return f"{left}\n\n{right}"


class VectorSearchConfig(BaseModel):
    """Configuration for vector search operations."""
    use_rrf: bool = True
    rrf_per_query_k: int = 20
    rrf_final_k: int = 15
    rrf_lambda: float = 60.0
    rrf_alpha: float = 0.8
    rrf_cutoff_k: int | None = None
    alpha: float = 0.8
    
    class Config:
        extra = "forbid"
        validate_assignment = True


class GraphTraversalConfig(BaseModel):
    """Configuration for graph traversal operations."""
    strategy_sequence: List[str] = ["one_hop"]
    max_neighbors_per_type: int = 2
    max_total: int = 7
    source_id: str | None = None
    gds_graph_name: str | None = None
    
    class Config:
        extra = "forbid"
        validate_assignment = True


class ReactomeRetrievalConfig(BaseModel):
    """Configuration for Reactome Graph RAG retrieval operations."""
    vector_config: VectorSearchConfig = VectorSearchConfig()
    graph_config: GraphTraversalConfig | None = None
    
    class Config:
        extra = "forbid"
        validate_assignment = True

