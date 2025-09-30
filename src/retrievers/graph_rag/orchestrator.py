import asyncio
from typing import Callable, Awaitable, Dict, Any
from langchain_core.runnables import RunnableConfig

from agent.tasks.summarizers import create_reactome_summarizer
from retrievers.uniprot.rag import create_uniprot_rag
from retrievers.graph_rag.config import create_reactome_retriever_custom
from .retrieval_utils import ReactomeRetrievalConfig, VectorSearchConfig, GraphTraversalConfig


class GraphRAGOrchestrator:
    """Orchestrates parallel searches across Reactome and UniProt databases for GraphRAG."""
    
    def __init__(self, llm, embedding):
        """Initialize search components."""
        self.reactome_summarizer = create_reactome_summarizer(llm)
        self.uniprot_rag_chain = create_uniprot_rag(llm, embedding, streaming=False)
        self.reactome_retriever, _ = create_reactome_retriever_custom()
        
        from retrievers.csv_chroma import create_hybrid_retriever
        from util.embedding_environment import EmbeddingEnvironment
        self.uniprot_retriever = create_hybrid_retriever(
            embedding, EmbeddingEnvironment.get_dir("uniprot")
        )
        
        self._initialize_reactome_configurations()
    
    def _initialize_reactome_configurations(self):
        """Initialize Reactome retrieval configurations."""
        self.reactome_simple_config = ReactomeRetrievalConfig(
            vector_config=VectorSearchConfig(),
            graph_config=None
        )
        
        complex_vector_config = VectorSearchConfig(rrf_final_k=7)
        complex_graph_config = GraphTraversalConfig(
            strategy_sequence=["steiner_tree", "one_hop"],
            max_neighbors_per_type=2,
            max_total=5,
            source_id=None,
            gds_graph_name=None
        )
        self.reactome_complex_config = ReactomeRetrievalConfig(
            vector_config=complex_vector_config,
            graph_config=complex_graph_config
        )
    
    async def orchestrate_parallel_searches(
        self,
        state: Dict[str, Any],
        config: RunnableConfig,
        reactome_search_func: Callable[[], Awaitable[str]],
        uniprot_search_func: Callable[[], Awaitable[str]],
        search_type: str
    ) -> Dict[str, Any]:
        """Orchestrate Reactome and UniProt searches in parallel with unified error handling."""
        try:
            reactome_context, uniprot_context = await asyncio.gather(
                reactome_search_func(), uniprot_search_func(), return_exceptions=True
            )
            
            reactome_context = self._normalize_search_result(reactome_context, "Reactome", search_type)
            uniprot_context = self._normalize_search_result(uniprot_context, "UniProt", search_type)
            
            return self._merge_search_contexts_into_state(state, reactome_context, uniprot_context)
            
        except Exception as e:
            print(f"🔍 {search_type.title()} search error: {e}")
            return self._merge_search_contexts_into_state(state, "", "")
    
    def _normalize_search_result(self, result: str | Exception, source: str, search_type: str) -> str:
        """Normalize search results by converting exceptions to empty strings for graceful error handling."""
        if isinstance(result, Exception):
            print(f"🧬 {source} {search_type} search error: {result}")
            return ""
        return result
    
    def _merge_search_contexts_into_state(
        self, state: Dict[str, Any], reactome_context: str, uniprot_context: str
    ) -> Dict[str, Any]:
        """Merge search contexts into GraphRAG state."""
        new_state = dict(state)
        new_state.update({
            "reactome_context": reactome_context,
            "uniprot_context": uniprot_context
        })
        return new_state
    
    async def execute_uniprot_search(
        self, state: Dict[str, Any], config: RunnableConfig, search_type: str
    ) -> str:
        """Execute UniProt search with different strategies for simple vs complex queries."""
        expanded_queries = state.get("expanded_queries", [])
        
        if search_type == "simple":
            uniprot_context = await self.uniprot_retriever.ainvoke({
                "input": state.get("standalone_query", ""),
                "expanded_queries": expanded_queries
            })
            return uniprot_context
        
        else:
            rag_result = await self.uniprot_rag_chain.ainvoke({
                "input": state.get("standalone_query", ""),
                "expanded_queries": expanded_queries,
                "chat_history": state.get("chat_history", [])
            }, config)
            
            uniprot_context = rag_result.get("answer", "")
            return uniprot_context
    
