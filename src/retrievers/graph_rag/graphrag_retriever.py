"""
Graph-RAG retriever orchestrator.
Moved from `src.retrievers.graphrag_retriever`.
"""

from __future__ import annotations

import asyncio
from typing import List, Optional, Dict, Any

from .types import IVectorClient, IGraphClient
from .retrieval_utils import ReactomeRetrievalConfig, UniProtRetrievalConfig, reciprocal_rank_fusion
from .graph_traversal_strategies import GraphTraversalStrategy


class GraphRAGRetriever:
    """Graph-RAG retriever that orchestrates vector search and graph traversal strategies."""
    
    # Strategy names
    ONE_HOP_STRATEGY = "one_hop"
    STEINER_TREE_STRATEGY = "steiner_tree"
    
    # Output formats
    OUTPUT_FORMAT_JSON = "json"
    OUTPUT_FORMAT_LLM_TEXT = "llm_text"
    
    # Default messages
    NO_RESULTS_MESSAGE = "No relevant nodes found in the knowledge graph for this query."
    NO_DOCUMENTS_MESSAGE = "No relevant Reactome entries found for this query."
    
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
        output_format: str = OUTPUT_FORMAT_LLM_TEXT,
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
        # Step 1: Vector Search
        vector_result = await self._search_vectors(query, cfg.vector_config, expanded_queries)
        seed_ids = vector_result["seed_ids"]
        documents = vector_result["documents"]
        
        if not seed_ids:
            return self.NO_RESULTS_MESSAGE

        # Step 2: Graph Traversal (only if graph_config is provided)
        if cfg.graph_config is None:
            # Vector-only search: return formatted vector results directly
            return self._format_documents(documents)
        
        # Perform graph traversal
        final_result = await self._traverse_graph(seed_ids, cfg.graph_config)

        # Step 3: Render results
        return self._render_results(final_result, cfg.graph_config.strategy_sequence, output_format)

    async def _search_vectors(
        self,
        query: str,
        vector_config,
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
            # Create tasks for parallel execution of all vector searches
            tasks = [
                asyncio.to_thread(
                    self.vector_client.search_similar,
                    q, 
                    k=vector_config.rrf_per_query_k, 
                    alpha=vector_config.rrf_alpha
                )
                for q in expanded_queries
            ]
            
            # Execute all vector searches in parallel
            ranked_lists = await asyncio.gather(*tasks)
            
            top_docs, seed_ids, _ = reciprocal_rank_fusion(
                ranked_lists=ranked_lists,
                final_k=vector_config.rrf_final_k,
                lambda_mult=vector_config.rrf_lambda,
                rrf_k=vector_config.rrf_cutoff_k,
                id_getter=lambda doc: self._extract_stable_id(doc),
            )
            
        else:
            # Simple similarity search
            docs = await asyncio.to_thread(
                self.vector_client.search_similar,
                query=query,
                k=vector_config.rrf_final_k,
                alpha=vector_config.alpha,
            )
            top_docs = docs[:vector_config.rrf_final_k]
            seed_ids = [self._extract_stable_id(doc) for doc in top_docs]
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
                print(f"WARNING: Document missing stId/stable_id - metadata: {doc.metadata}")
                return None
            return str(stable_id)  # Ensure it's a string
        except (ValueError, TypeError) as e:
            print(f"WARNING: Invalid stable_id '{stable_id}' - {e}")
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
        strategy_sequence = graph_config.strategy_sequence
        current_seed_ids = seed_ids.copy()
        
        # Handle single strategy
        if len(strategy_sequence) == 1:
            strategy_name = strategy_sequence[0]
            result = await self._run_strategy(strategy_name, current_seed_ids, graph_config)
            return result
        
        # Handle composite strategies (e.g., steiner_tree -> one_hop)
        results = []
        for i, strategy_name in enumerate(strategy_sequence):
            result = await self._run_strategy(strategy_name, current_seed_ids, graph_config)
            results.append(result)
            
            # For composite strategies, extract node IDs from result for next iteration
            if i < len(strategy_sequence) - 1:  # Not the last strategy
                current_seed_ids = self._extract_node_ids(result, strategy_name)
                if not current_seed_ids:
                    break  # No nodes to continue with
        
        # Return the final result or combine results as needed
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

        # Build strategy-specific configuration
        strategy_cfg = self._build_config(strategy_name, graph_config)
        
        strategy = self.strategies[strategy_name]
        # Use asyncio.to_thread for potentially blocking graph operations
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
        cfg = {}
        
        if strategy_name == self.ONE_HOP_STRATEGY:
            cfg.update({
                "max_neighbors_per_type": graph_config.max_neighbors_per_type,
                "max_total": graph_config.max_total,
            })
        elif strategy_name == self.STEINER_TREE_STRATEGY:
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
        if strategy_name == self.ONE_HOP_STRATEGY:
            # Extract all node IDs from one_hop result
            node_ids = set()
            if isinstance(result, dict):
                for seed_id, data in result.items():
                    node_ids.add(seed_id)  # seed_id is already a string (stable_id)
                    if "neighbors" in data:
                        for rel_type, neighbors in data["neighbors"].items():
                            for neighbor in neighbors:
                                node_ids.add(neighbor["node_id"])  # node_id is now a string
            return list(node_ids)
        
        elif strategy_name == self.STEINER_TREE_STRATEGY:
            # Extract all node IDs from steiner_tree result
            node_ids = set()
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
            return self.NO_DOCUMENTS_MESSAGE
        
        lines = []
        for i, doc in enumerate(documents, 1):
            metadata = doc.metadata
            content = doc.page_content
            
            lines.append(f"## Reactome Entry {i}")
            
            # Add all metadata properties
            for key, value in metadata.items():
                if value is not None and value != "":
                    lines.append(f"* {key}: {value}")
            
            # Add the page content
            if content:
                lines.append(f"* content: {content}")
            
            lines.append("")  # Empty line between entries
        
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
        # Use the renderer for the final strategy in the sequence
        final_strategy = strategy_sequence[-1]
        
        if final_strategy not in self.renderers:
            raise ValueError(f"No renderer registered for strategy '{final_strategy}'")

        renderer = self.renderers[final_strategy]
        if output_format == self.OUTPUT_FORMAT_JSON:
            return renderer.to_json(result)
        elif output_format == self.OUTPUT_FORMAT_LLM_TEXT:
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