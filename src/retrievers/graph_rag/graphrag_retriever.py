from __future__ import annotations

import asyncio
from typing import List, Optional, Dict, Any

from .types import IVectorClient, IGraphClient
from .retrieval_utils import ReactomeRetrievalConfig, VectorSearchConfig
from .graph_traversal_strategies import GraphTraversalStrategy
from ..retrieval_utils import reciprocal_rank_fusion


class GraphRAGRetriever:
    """Graph-RAG retriever that orchestrates vector search and graph traversal strategies.    """
    
    def __init__(
        self,
        vector_client: IVectorClient,
        graph_client: IGraphClient,
        strategy_registry: Dict[str, GraphTraversalStrategy],
        renderer_registry: Dict[str, Any],
    ) -> None:
        """
        Initialize the Graph-RAG retriever.
        
        Args:
            vector_client: Client for vector similarity search operations
            graph_client: Client for graph traversal operations
            strategy_registry: Registry of available graph traversal strategies
            renderer_registry: Registry of result renderers for each strategy
        """
        self.vector_client = vector_client
        self.graph_client = graph_client
        self.strategies = strategy_registry
        self.renderers = renderer_registry

    async def ainvoke(
        self,
        query: str,
        cfg: ReactomeRetrievalConfig,
        expanded_queries: Optional[List[str]] = None,
        output_format: str = "llm_text",
    ) -> str:
        """
        Invoke the Graph-RAG retrieval pipeline.
        
        This is the main entry point for retrieval. It performs:
        1. Vector search to find relevant seed nodes
        2. Optional graph traversal to explore relationships
        3. Result formatting for consumption
        
        Args:
            query: The primary search query
            cfg: ReactomeRetrievalConfig with vector and graph configurations
            expanded_queries: Optional list of expanded queries for Reciprocal Rank Fusion (RRF)
            output_format: Output format - either "json" or "llm_text"
            
        Returns:
            Formatted retrieval results as a string
            
        Raises:
            ValueError: If output_format is invalid or strategy is unknown
        """
        vector_result: Dict[str, Any] = await self._search_vectors(query, cfg.vector_config, expanded_queries)
        seed_ids: List[str] = vector_result["seed_ids"]
        documents: List[Any] = vector_result["documents"]
        
        if not seed_ids:
            return "No relevant nodes found in the knowledge graph for this query."

        if cfg.graph_config is None:
            return self._format_documents(documents)
        
        final_result: Any = await self._traverse_graph(seed_ids, cfg.graph_config)
        return self._render_results(final_result, cfg.graph_config.strategy_sequence, output_format)

    async def _search_vectors(
        self,
        query: str,
        vector_config: VectorSearchConfig,
        expanded_queries: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Search vectors and return both seed node IDs and documents.
        
        Supports both simple similarity search and Reciprocal Rank Fusion (RRF)
        for multiple expanded queries.
        
        Args:
            query: The primary search query
            vector_config: Vector search configuration
            expanded_queries: Optional list of expanded queries for RRF
            
        Returns:
            Dictionary with 'seed_ids' and 'documents' keys
        """
        if vector_config.use_rrf and expanded_queries and len(expanded_queries) > 1:
            tasks: List[Any] = [
                asyncio.to_thread(
                    self.vector_client.search_similar,
                    q, 
                    k=vector_config.rrf_per_query_k, 
                    alpha=vector_config.rrf_alpha
                )
                for q in expanded_queries
            ]
            
            ranked_lists: List[List[Any]] = await asyncio.gather(*tasks)
            
            top_docs: List[Any]
            seed_ids: List[str]
            top_docs, seed_ids, _ = reciprocal_rank_fusion(
                ranked_lists=ranked_lists,
                final_k=vector_config.rrf_final_k,
                lambda_mult=vector_config.rrf_lambda,
                rrf_k=vector_config.rrf_cutoff_k,
                id_getter=lambda doc: self._extract_stable_id(doc),
            )
            
        else:
            # Simple similarity search
            docs: List[Any] = await asyncio.to_thread(
                self.vector_client.search_similar,
                query=query,
                k=vector_config.rrf_final_k,
                alpha=vector_config.alpha,
            )
            top_docs: List[Any] = docs[:vector_config.rrf_final_k]
            seed_ids: List[str] = [self._extract_stable_id(doc) for doc in top_docs]
            seed_ids = [sid for sid in seed_ids if sid is not None]  # Filter out None values
        
        return {
            "seed_ids": seed_ids,
            "documents": top_docs
        }

    def _extract_stable_id(self, doc: Any) -> Optional[str]:
        """
        Extract stable_id from document metadata.
        
        Args:
            doc: Document object with metadata containing stable ID
            
        Returns:
            Stable ID as string, or None if not found or invalid
        """
        try:
            stable_id = doc.metadata.get("stId") or doc.metadata.get("stable_id")
            if stable_id is None:
                return None
            return str(stable_id)
        except (ValueError, TypeError) as e:
            return None

    async def _traverse_graph(
        self,
        seed_ids: List[str],
        graph_config: Any,
    ) -> Any:
        """
        Traverse graph using the configured strategy sequence.
        
        Supports both single strategies and composite strategy sequences
        (e.g., steiner_tree followed by one_hop).
        
        Args:
            seed_ids: List of seed node IDs to start traversal from
            graph_config: Graph traversal configuration
            
        Returns:
            Graph traversal results
        """
        strategy_sequence: List[str] = graph_config.strategy_sequence
        current_seed_ids: List[str] = seed_ids.copy()
        
        if len(strategy_sequence) == 1:
            strategy_name: str = strategy_sequence[0]
            result: Any = await self._run_strategy(strategy_name, current_seed_ids, graph_config)
            return result
        
        results: List[Any] = []
        for i, strategy_name in enumerate(strategy_sequence):
            result: Any = await self._run_strategy(strategy_name, current_seed_ids, graph_config)
            results.append(result)
            
            if i < len(strategy_sequence) - 1:
                current_seed_ids = self._extract_node_ids(result, strategy_name)
                if not current_seed_ids:
                    break
        
        return results[-1] if results else None

    async def _run_strategy(
        self,
        strategy_name: str,
        seed_ids: List[str],
        graph_config: Any,
    ) -> Any:
        """
        Run a single graph traversal strategy.
        
        Args:
            strategy_name: Name of the strategy to run
            seed_ids: List of seed node IDs
            graph_config: Graph configuration object
            
        Returns:
            Strategy execution results
            
        Raises:
            ValueError: If strategy_name is not found in registry
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy '{strategy_name}'")

        strategy_cfg: Dict[str, Any] = self._build_config(strategy_name, graph_config)
        
        strategy: GraphTraversalStrategy = self.strategies[strategy_name]
        return await asyncio.to_thread(
            strategy.traverse,
            self.graph_client,
            seed_ids,
            strategy_cfg
        )

    def _build_config(self, strategy_name: str, graph_config: Any) -> Dict[str, Any]:
        """
        Build configuration dictionary for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
            graph_config: Graph configuration object
            
        Returns:
            Strategy-specific configuration dictionary
        """
        cfg: Dict[str, Any] = {}
        
        if strategy_name == "one_hop":
            cfg.update({
                "max_neighbors_per_type": graph_config.max_neighbors_per_type,
                "max_total": graph_config.max_total,
            })
        elif strategy_name == "steiner_tree":
            cfg.update({
                "source_id": graph_config.source_id,
                "gds_graph_name": graph_config.gds_graph_name,
            })
        
        return cfg

    def _extract_node_ids(self, result: Any, strategy_name: str) -> List[str]:
        """
        Extract node IDs from strategy result for use in subsequent strategies.
        
        Args:
            result: Strategy execution result
            strategy_name: Name of the strategy that produced the result
            
        Returns:
            List of node IDs extracted from the result
        """
        if strategy_name == "one_hop":
            node_ids: set[str] = set()
            if isinstance(result, dict):
                for seed_id, data in result.items():
                    node_ids.add(seed_id)
                    if "neighbors" in data:
                        for rel_type, neighbors in data["neighbors"].items():
                            for neighbor in neighbors:
                                node_ids.add(neighbor["node_id"])
            return list(node_ids)
        
        elif strategy_name == "steiner_tree":
            node_ids: set[str] = set()
            if isinstance(result, dict):
                node_ids.update(result.get("nodeIds", []))
            return list(node_ids)
        
        return []

    def _format_documents(self, documents: Any) -> str:
        """
        Format documents with full metadata and content.
        
        Args:
            documents: List of document objects with metadata and content
            
        Returns:
            Formatted string with document information
        """
        if not documents:
            return "No relevant Reactome entries found for this query."
        
        lines: List[str] = []
        for i, doc in enumerate(documents, 1):
            metadata: Dict[str, Any] = doc.metadata
            content: str = doc.page_content
            
            lines.append(f"## Reactome Entry {i}")
            
            for key, value in metadata.items():
                if value is not None and value != "":
                    lines.append(f"* {key}: {value}")
            
            if content:
                lines.append(f"* content: {content}")
            
            lines.append("")
        
        return "\n".join(lines).strip()

    def _render_results(
        self,
        result: Any,
        strategy_sequence: List[str],
        output_format: str,
    ) -> str:
        """
        Render the final results using the appropriate renderer.
        
        Args:
            result: Graph traversal result to render
            strategy_sequence: Sequence of strategies used
            output_format: Desired output format ("json" or "llm_text")
            
        Returns:
            Rendered result as string
            
        Raises:
            ValueError: If renderer not found or output format is invalid
        """
        final_strategy: str = strategy_sequence[-1]
        
        if final_strategy not in self.renderers:
            raise ValueError(f"No renderer registered for strategy '{final_strategy}'")

        renderer: Any = self.renderers[final_strategy]
        if output_format == "json":
            return renderer.to_json(result)
        elif output_format == "llm_text":
            return renderer.to_llm_text(result)
        else:
            raise ValueError(f"Unknown output format: {output_format}")

    def __enter__(self) -> "GraphRAGRetriever":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """
        Exit context manager and clean up resources.
        
        Args:
            exc_type: Exception type if any
            exc_val: Exception value if any  
            exc_tb: Exception traceback if any
            
        Returns:
            False to not suppress exceptions
        """
        try:
            self.vector_client.close()
        finally:
            self.graph_client.close()
        return False