from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ..types import IGraphClient


class GraphTraversalStrategy(ABC):
    """
    Abstract base for graph traversal strategies.
    
    Subclasses:
        - Must set 'name'
        - Must implement 'traverse()'
    """

    name: str  # Unique identifier

    @abstractmethod
    def traverse(
        self,
        graph_client: IGraphClient,
        seed_ids: List[str],  # Changed from List[int] to List[str] for stable IDs
        cfg: Dict[str, Any],
    ) -> Any:
        """
        Run traversal starting from seed_ids, using any config.
        
        Args:
            graph_client: The graph database client
            seed_ids: List of seed node IDs to start traversal from
            cfg: Configuration dictionary for the traversal strategy
            
        Returns:
            Any structured, JSON-serializable object suitable for renderer.
        """
        raise NotImplementedError("traverse() must be implemented.")
