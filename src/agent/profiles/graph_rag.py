from typing import Literal, Annotated

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import StateGraph

from agent.profiles.base import BaseGraphBuilder, BaseState, SAFETY_SAFE, SAFETY_UNSAFE
from agent.tasks.final_answer_generation.expert import create_expert_answer_generator
from agent.tasks.final_answer_generation.non_expert import create_lay_answer_generator
from agent.tasks.final_answer_generation.unsafe_question import create_unsafe_answer_generator
from retrievers.graph_rag.retrieval_utils import merge_contexts
from retrievers.graph_rag.orchestrator import GraphRAGOrchestrator

# Configuration constants
DEFAULT_LANGUAGE = "English"
DEFAULT_USER_TONE = "lay"
DEFAULT_COMPLEXITY = "simple"


class GraphRAGState(BaseState):
    reactome_context: Annotated[str, merge_contexts]  
    uniprot_context: Annotated[str, merge_contexts]

class GraphRAGGraphBuilder(BaseGraphBuilder):
    """Graph builder for GraphRAG profile with Reactome and UniProt integration."""
    
    def __init__(self, llm: BaseChatModel, embedding: Embeddings) -> None:
        super().__init__(llm, embedding)
        
        self._setup_answer_generators(llm)
        self.search_orchestrator = GraphRAGOrchestrator(llm, embedding)
        self.uncompiled_graph = self._build_workflow_graph()
    
    def _setup_answer_generators(self, llm: BaseChatModel) -> None:
        """Initialize answer generation components."""
        streaming_llm = llm.model_copy(update={"streaming": True})
        self.lay_answer_generator = create_lay_answer_generator(streaming_llm)
        self.expert_answer_generator = create_expert_answer_generator(streaming_llm)
        self.unsafe_answer_generator = create_unsafe_answer_generator(streaming_llm)
    
    def _build_workflow_graph(self) -> StateGraph:
        """Build and configure the workflow graph."""
        state_graph = StateGraph(GraphRAGState)
        
        state_graph.add_node("preprocess", self.preprocess)
        state_graph.add_node("simple_search", self.simple_search)
        state_graph.add_node("complex_search", self.complex_search)
        state_graph.add_node("generate_final_response", self.generate_final_response)
        state_graph.add_node("postprocess", self.postprocess)
        state_graph.set_entry_point("preprocess")
        state_graph.add_conditional_edges(
            "preprocess",
            self.route_after_preprocessing,
            {
                "simple_search": "simple_search",
                "complex_search": "complex_search",
                "generate_final_response": "generate_final_response"
            },
        )
        state_graph.add_edge("simple_search", "generate_final_response")
        state_graph.add_edge("complex_search", "generate_final_response")
        state_graph.add_edge("generate_final_response", "postprocess")
        state_graph.set_finish_point("postprocess")
        
        return state_graph


    async def route_after_preprocessing(
        self, state: GraphRAGState
    ) -> Literal["simple_search", "complex_search", "generate_final_response"]:
        """Route to appropriate next step based on safety and complexity."""
        safety = state.get("safety", SAFETY_SAFE)
        complexity = state.get("complexity", DEFAULT_COMPLEXITY)
        
        if safety != SAFETY_SAFE:
            print("🔍 Finishing research - unsafe question")
            return "generate_final_response"
        
        print(f"🔍 Proceeding with research - complexity: {complexity}")
        return "complex_search" if complexity == "complex" else "simple_search"
    
    async def simple_search(self, state: GraphRAGState, config: RunnableConfig) -> GraphRAGState:
        """Execute vector-only search across Reactome and UniProt databases."""
        print("🔍 Executing simple search (vector-only) for both Reactome and UniProt")
        
        async def reactome_search() -> str:
            expanded_queries = state.get("expanded_queries", [])
            context_str = await self.search_orchestrator.reactome_retriever.ainvoke(
                query=state.get("standalone_query", ""),
                cfg=self.search_orchestrator.reactome_simple_config,
                expanded_queries=expanded_queries,
                output_format="llm_text"
            )
            print(f"🧬 Reactome simple search returned context of length: {len(context_str)}")
            return context_str
        
        async def uniprot_search() -> str:
            return await self.search_orchestrator.execute_uniprot_search(state, config, "simple")
        
        result = await self.search_orchestrator.orchestrate_parallel_searches(
            state, config, reactome_search, uniprot_search, "simple"
        )
        return GraphRAGState(**result)

    async def complex_search(self, state: GraphRAGState, config: RunnableConfig) -> GraphRAGState:
        """Execute advanced search with graph traversal and summarization across databases."""
        print("🔍 Executing complex search (vector + graph + RAG) for both Reactome and UniProt")
        
        async def reactome_search() -> str:
            expanded_queries = state.get("expanded_queries", [])
            context_str = await self.search_orchestrator.reactome_retriever.ainvoke(
                query=state.get("standalone_query", ""),
                cfg=self.search_orchestrator.reactome_complex_config,
                expanded_queries=expanded_queries,
                output_format="llm_text"
            )
            
            reactome_context = await self.search_orchestrator.reactome_summarizer.ainvoke(
                {"standalone_query": state.get("standalone_query", ""), "context": context_str}, config
            )
            print(f"🧬 Reactome complex search generated summarized context of length: {len(reactome_context)}")
            return reactome_context
        
        async def uniprot_search() -> str:
            return await self.search_orchestrator.execute_uniprot_search(state, config, "complex")
        
        result = await self.search_orchestrator.orchestrate_parallel_searches(
            state, config, reactome_search, uniprot_search, "complex"
        )
        return GraphRAGState(**result)


    async def generate_final_response(
        self, state: GraphRAGState, config: RunnableConfig
    ) -> GraphRAGState:
        """Generate final response based on safety status and user expertise level."""
        reactome_context = state.get("reactome_context", "")
        uniprot_context = state.get("uniprot_context", "")
        reason_unsafe = state.get("reason_unsafe", "")
        safety = state.get("safety", SAFETY_SAFE)
        
        print(f"🎯 Generating final response - Safety: {safety}, Reactome context: {len(reactome_context)} chars, UniProt context: {len(uniprot_context)} chars")

        if safety == SAFETY_UNSAFE and reason_unsafe:
            final_answer = await self._generate_unsafe_response(state, reason_unsafe, config)
            return self._create_final_state(state, final_answer, is_unsafe=True)
        else:
            final_answer = await self._generate_safe_response(state, reactome_context, uniprot_context, config)
            return self._create_final_state(state, final_answer, is_unsafe=False)
    
    async def _generate_unsafe_response(self, state: GraphRAGState, reason_unsafe: str, config: RunnableConfig) -> str:
        """Generate refusal response for unsafe queries."""
        response = await self.unsafe_answer_generator.ainvoke({
            "language": state.get("detected_language", DEFAULT_LANGUAGE),
            "user_input": state.get("standalone_query", ""),
            "reason_unsafe": reason_unsafe,
        }, config)
        
        # Extract content from AIMessage if needed
        if hasattr(response, 'content'):
            return response.content
        return str(response)
    
    async def _generate_safe_response(
        self, state: GraphRAGState, reactome_context: str, uniprot_context: str, config: RunnableConfig
    ) -> str:
        """Generate informative response for safe queries."""
        user_tone = state.get("user_tone", DEFAULT_USER_TONE)
        answer_generator = (self.expert_answer_generator if user_tone == "expert" 
                          else self.lay_answer_generator)
        
        response = await answer_generator.ainvoke({
            "language": state.get("detected_language", DEFAULT_LANGUAGE),
            "user_input": state.get("standalone_query", ""),
            "reactome_summary": reactome_context,
            "uniprot_summary": uniprot_context,
            "chat_history": state.get("chat_history", []),
        }, config)
        
        # Extract content from AIMessage if needed
        if hasattr(response, 'content'):
            return response.content
        return str(response)
    
    def _create_final_state(self, state: GraphRAGState, final_answer: str, is_unsafe: bool) -> GraphRAGState:
        """Create final state with generated response and updated chat history."""
        new_state = dict(state)
        new_state.update({
            "chat_history": [
                HumanMessage(state.get("standalone_query", "")),
                AIMessage(final_answer),
            ],
            "answer": final_answer,
        })
        
        if is_unsafe:
            new_state.update({
                "safety": SAFETY_UNSAFE,
                "additional_content": {"search_results": []},
            })
        
        return GraphRAGState(**new_state)


def create_graph_rag_graph(llm: BaseChatModel, embedding: Embeddings) -> StateGraph:
    """Create a GraphRAG workflow graph with Reactome and UniProt integration.
    
    Args:
        llm: Language model for text generation and processing
        embedding: Embedding model for vector similarity search
        
    Returns:
        Compiled StateGraph ready for execution
    """
    return GraphRAGGraphBuilder(llm, embedding).uncompiled_graph
