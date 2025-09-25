from collections import defaultdict
from typing import List, Any, Tuple, Dict, Callable
from pydantic import BaseModel

def reciprocal_rank_fusion(
    ranked_lists: List[List[Any]],
    final_k: int = 5,
    lambda_mult: float = 60.0,
    rrf_k: int | None = None,
    id_getter: Callable[[Any], str] = lambda doc: doc.metadata.get("stId") or doc.metadata.get("stable_id"),
) -> Tuple[List[Any], List[str], Dict[str, float]]:
    rrf_scores = defaultdict(float)
    doc_meta = {}

    for ranked in ranked_lists:
        considered = ranked[:rrf_k] if rrf_k else ranked
        for rank, doc in enumerate(considered):
            doc_id = id_getter(doc)
            if doc_id is not None:  # Skip documents without valid IDs
                rrf_scores[doc_id] += 1.0 / (lambda_mult + rank + 1)
                if doc_id not in doc_meta:
                    doc_meta[doc_id] = doc

    sorted_items = sorted(rrf_scores.items(), key=lambda x: (-x[1], x[0]))
    top_ids = [doc_id for doc_id, _ in sorted_items[:final_k]]
    top_docs = [doc_meta[doc_id] for doc_id in top_ids]
    return top_docs, top_ids, rrf_scores


class VectorSearchConfig(BaseModel):
    """Configuration for vector search operations."""
    use_rrf: bool = True            # Use Reciprocal Rank Fusion for multiple queries
    rrf_per_query_k: int = 20       # Number of results per query in RRF
    rrf_final_k: int = 10           # Final number of results after RRF
    rrf_lambda: float = 60.0        # RRF lambda parameter
    rrf_alpha: float = 0.8          # RRF alpha parameter
    rrf_cutoff_k: int | None = None # RRF cutoff (None = use all)
    
    alpha: float = 0.8  # Similarity threshold for simple search
    
    class Config:
        extra = "forbid"
        validate_assignment = True


class GraphTraversalConfig(BaseModel):
    """Configuration for graph traversal operations."""
    strategy_sequence: List[str] = ["one_hop"]  # List of strategies to execute
    max_neighbors_per_type: int = 2             # For one_hop strategy
    max_total: int = 7                          # For one_hop strategy
    source_id: str | None = None                # For steiner_tree strategy (None = use first seed)
    gds_graph_name: str | None = None           # For steiner_tree strategy (None = create temporary)
    
    class Config:
        extra = "forbid"
        validate_assignment = True


class ReactomeRetrievalConfig(BaseModel):
    """Configuration for Reactome Graph RAG retrieval operations."""
    vector_config: VectorSearchConfig = VectorSearchConfig()
    graph_config: GraphTraversalConfig | None = None  # Optional for simple vector-only searches
    
    class Config:
        extra = "forbid"
        validate_assignment = True


class UniProtRetrievalConfig(BaseModel):
    """Configuration for UniProt vector retrieval operations."""
    vector_config: VectorSearchConfig = VectorSearchConfig()
    
    class Config:
        extra = "forbid"
        validate_assignment = True